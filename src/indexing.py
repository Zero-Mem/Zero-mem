"""Graph and temporal hierarchy indexes, persistence and corpus isolation."""

from collections import Counter
from .embedding_store import EmbeddingStore
from collections import defaultdict
import hashlib
import json
from .utils import logger, safe_error_summary
import numpy as np
import os
import re
import time
from tqdm import tqdm


def _safe_dataset_namespace(name):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "dataset")).strip("._-")
    return text or "dataset"


class IsolationGuard:
    BOUNDARY_KEY_ALIASES = {
        "sampleid": "sample_id",
        "sample_id": "sample_id",
        "sample-id": "sample_id",
        "conversationid": "conversation_id",
        "conversation_id": "conversation_id",
        "conversation-id": "conversation_id",
        "convid": "conv_id",
        "conv_id": "conv_id",
        "conv-id": "conv_id",
        "documentid": "document_id",
        "document_id": "document_id",
        "document-id": "document_id",
        "docid": "doc_id",
        "doc_id": "doc_id",
        "doc-id": "doc_id",
        "storyid": "story_id",
        "story_id": "story_id",
        "story-id": "story_id",
        "articleid": "article_id",
        "article_id": "article_id",
        "article-id": "article_id",
        "contextid": "context_id",
        "context_id": "context_id",
        "context-id": "context_id",
        "sourceid": "source_id",
        "source_id": "source_id",
        "source-id": "source_id",
    }
    BOUNDARY_KEYS = (
        "sample_id",
        "conversation_id",
        "conv_id",
        "document_id",
        "doc_id",
        "story_id",
        "article_id",
        "context_id",
        "source_id",
    )
    CACHE_FILES = (
        "passage_embedding.parquet",
        "entity_embedding.parquet",
        "sentence_embedding.parquet",
        "window_embedding.parquet",
        "episode_embedding.parquet",
        "ner_results.json",
        "dual_index_metadata.json",
        "ZeroMem.graphml",
        "LinearRAG.graphml",  # Detect and reset legacy graph caches.
    )

    def __init__(self, rag):
        self.rag = rag

    def enabled(self):
        return bool(self.rag.config.enable_isolation_guard)

    def boundary_enabled(self, boundary=None):
        if not self.enabled():
            return False
        if boundary is not None and not str(boundary or "").strip():
            return False
        return bool(
            self.rag.config.isolation_strict_boundary
            or self.rag.config.sample_boundary_strict
        )

    def manifest_path(self):
        filename = str(self.rag.config.isolation_manifest_filename)
        return os.path.join(self.rag.config.working_dir, self.rag.dataset_name, filename)

    def cache_dir(self):
        return os.path.join(self.rag.config.working_dir, self.rag.dataset_name)

    @staticmethod
    def passage_hash_id(text):
        return "passage-" + hashlib.md5(str(text).encode()).hexdigest()

    @staticmethod
    def _canonical_boundary_key(key):
        raw = str(key or "").strip().lower().replace(" ", "_")
        compact = re.sub(r"[^a-z0-9_-]+", "", raw)
        return IsolationGuard.BOUNDARY_KEY_ALIASES.get(compact, compact)

    @staticmethod
    def _clean_boundary_value(value, key=""):
        raw = str(value or "").strip().strip("'\"")
        raw = re.sub(r"[\]\),.;]+$", "", raw).strip()
        if not raw:
            return ""
        extracted = IsolationGuard.extract_boundary_from_text(raw)
        if extracted and extracted != raw:
            return extracted
        canonical_key = IsolationGuard._canonical_boundary_key(key)
        if canonical_key in {"conversation_id", "conv_id"} and re.fullmatch(r"\d+", raw):
            return f"conv-{raw}"
        if canonical_key == "sample_id" and re.fullmatch(r"\d+", raw):
            return f"sample-{raw}"
        return raw

    @staticmethod
    def extract_boundary_from_text(text):
        txt = str(text or "")
        if not txt:
            return None
        key_pattern = (
            r"sample[_\s-]?id|conversation[_\s-]?id|conv[_\s-]?id|"
            r"document[_\s-]?id|doc[_\s-]?id|story[_\s-]?id|article[_\s-]?id|"
            r"context[_\s-]?id|source[_\s-]?id"
        )
        patterns = (
            rf"\[\s*(?P<key>{key_pattern})\s*[=:]\s*(?P<value>[^\]]+)\]",
            rf"\[\s*(?P<key>{key_pattern})\s*\]\s*(?P<value>[^\n\[]+)",
            rf"\b(?P<key>{key_pattern})\s*[=:]\s*(?P<value>[A-Za-z0-9_.:/-]+)",
        )
        for pat in patterns:
            m = re.search(pat, txt, flags=re.I)
            if m:
                key = IsolationGuard._canonical_boundary_key(m.group("key"))
                value = IsolationGuard._clean_boundary_value(m.group("value"), key=key)
                if value:
                    return value
        m = re.search(r"\b(conv|sample)[-_]?(\d+)\b", txt, flags=re.I)
        if m:
            return f"{m.group(1).lower()}-{m.group(2)}"
        return None

    @staticmethod
    def boundary_aliases(boundary):
        raw = str(boundary or "").strip()
        if not raw:
            return set()
        aliases = {raw, raw.lower()}
        extracted = IsolationGuard.extract_boundary_from_text(raw)
        if extracted:
            aliases.add(extracted)
            aliases.add(extracted.lower())
        m = re.fullmatch(r"(conv|sample)-(\d+)", raw, flags=re.I)
        if m:
            aliases.add(f"{m.group(1).lower()}_{m.group(2)}")
        return {str(a).strip().lower() for a in aliases if str(a or "").strip()}

    def boundary_from_question_info(self, question_info):
        if not isinstance(question_info, dict):
            return None
        for key in self.BOUNDARY_KEYS:
            if key in question_info and question_info.get(key) not in {None, ""}:
                value = self._clean_boundary_value(question_info.get(key), key=key)
                if value:
                    return value
        qid = question_info.get("qid") or question_info.get("id") or question_info.get("_id")
        extracted = self.extract_boundary_from_text(qid)
        if extracted:
            return extracted
        return None

    def corpus_fingerprint(self, passages):
        sha = hashlib.sha256()
        sha.update(str(self.rag.dataset_name).encode())
        sha.update(str(len(passages or [])).encode())
        for text in passages or []:
            sha.update(self.passage_hash_id(text).encode())
        return sha.hexdigest()

    def expected_passage_ids(self, passages):
        return {self.passage_hash_id(text) for text in passages or []}

    def _read_manifest(self):
        path = self.manifest_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _ner_passage_ids(self):
        path = os.path.join(self.cache_dir(), "ner_results.json")
        if not os.path.exists(path):
            return set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return set((data.get("passage_hash_id_to_entities") or {}).keys())
        except Exception:
            return set()

    def reset_cache(self, reason):
        if not bool(self.rag.config.isolation_reset_cache_on_mismatch):
            logger.warning("IsolationGuard detected cache mismatch but reset is disabled: %s", reason)
            return False
        cache_dir = self.cache_dir()
        filenames = list(self.CACHE_FILES)
        filenames.append(os.path.basename(self.manifest_path()))
        removed = []
        for filename in filenames:
            path = os.path.join(cache_dir, filename)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    removed.append(filename)
                except OSError as exc:
                    logger.warning("IsolationGuard could not remove a cache file: %s", safe_error_summary(exc))
        if removed:
            logger.info("IsolationGuard reset %s cache files for %s: %s", len(removed), self.rag.dataset_name, reason)
        self.rag.load_embedding_store()
        if hasattr(self.rag, "_reset_runtime_after_cache_clear"):
            self.rag._reset_runtime_after_cache_clear()
        return True

    def ensure_corpus_cache(self, passages):
        if not self.enabled():
            return
        os.makedirs(self.cache_dir(), exist_ok=True)
        expected_ids = self.expected_passage_ids(passages)
        expected_fp = self.corpus_fingerprint(passages)
        manifest = self._read_manifest()
        store_ids = set(getattr(self.rag.passage_embedding_store, "hash_ids", []) or [])
        ner_ids = self._ner_passage_ids()
        cache_exists = bool(store_ids or ner_ids or any(
            os.path.exists(os.path.join(self.cache_dir(), filename))
            for filename in self.CACHE_FILES
        ))
        if not manifest and cache_exists and bool(self.rag.config.isolation_rebuild_cache_without_manifest):
            self.reset_cache("existing cache has no isolation manifest")
            return
        if manifest and manifest.get("fingerprint") != expected_fp:
            self.reset_cache("corpus fingerprint changed")
            return
        stale_store = bool(store_ids - expected_ids)
        stale_ner = bool(ner_ids - expected_ids)
        if stale_store or stale_ner:
            self.reset_cache("stale passage ids found for current corpus")

    def write_manifest(self, passages):
        if not self.enabled():
            return
        path = self.manifest_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        boundary_counts = Counter()
        for text in passages or []:
            boundary = self.extract_boundary_from_text(text)
            if boundary:
                boundary_counts[boundary] += 1
        data = {
            "dataset_name": self.rag.dataset_name,
            "passage_count": len(passages or []),
            "fingerprint": self.corpus_fingerprint(passages),
            "boundary_count": sum(boundary_counts.values()),
            "unique_boundary_count": len(boundary_counts),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def corpus_has_boundary(self):
        return any(str(v or "").strip() for v in getattr(self.rag, "passage_id_to_sample_id", {}).values())

    def _id_matches_boundary(self, passage_id, query_boundary):
        pid_boundary = str(getattr(self.rag, "passage_id_to_sample_id", {}).get(passage_id, "") or "").strip()
        if not pid_boundary:
            return False
        return bool(self.boundary_aliases(pid_boundary) & self.boundary_aliases(query_boundary))

    def filter_passage_pairs(self, ids, scores, query_boundary, stage):
        ids = list(ids or [])
        scores = list(scores or [])
        trace = {"enabled": False, "stage": stage, "before": len(ids), "after": len(ids), "query_boundary": str(query_boundary or "")}
        if not self.boundary_enabled(query_boundary):
            return ids, scores, trace
        if not self.corpus_has_boundary():
            trace.update({"enabled": True, "reason": "corpus_has_no_boundary"})
            return ids, scores, trace
        kept = [(pid, score) for pid, score in zip(ids, scores) if self._id_matches_boundary(pid, query_boundary)]
        trace.update({"enabled": True, "after": len(kept), "removed": max(0, len(ids) - len(kept))})
        if not kept:
            trace["empty_after_filter"] = True
            return [], [], trace
        return [pid for pid, _ in kept], [score for _, score in kept], trace

    def evidence_matches_boundary(self, passage, meta, query_boundary):
        if not self.boundary_enabled(query_boundary):
            return True
        aliases = self.boundary_aliases(query_boundary)
        if not aliases:
            return True
        candidates = []
        if isinstance(meta, dict):
            for key in ("sample_id", "conversation_id", "conv_id", "document_id", "doc_id", "story_id", "article_id", "context_id", "source_id"):
                if meta.get(key):
                    candidates.append(str(meta.get(key)))
            if meta.get("text"):
                extracted = self.extract_boundary_from_text(meta.get("text"))
                if extracted:
                    candidates.append(extracted)
        extracted = self.extract_boundary_from_text(passage)
        if extracted:
            candidates.append(extracted)
        if not candidates:
            return not self.corpus_has_boundary()
        candidate_aliases = set()
        for value in candidates:
            candidate_aliases.update(self.boundary_aliases(value))
        return bool(candidate_aliases & aliases)

    def filter_reader_context(self, query_profile, selected_passages, selected_meta, stage="reader"):
        boundary = str((query_profile or {}).get("sample_id", "") or "").strip()
        passages = list(selected_passages or [])
        meta = list(selected_meta or [])
        if not self.boundary_enabled(boundary):
            return passages, meta, {"enabled": False, "stage": stage, "before": len(passages), "after": len(passages)}
        while len(meta) < len(passages):
            text = passages[len(meta)]
            meta.append({"unit_type": "passage", "source_id": "", "session_time": "", "text": str(text), "sample_id": self.extract_boundary_from_text(text) or ""})
        kept_passages = []
        kept_meta = []
        for idx, passage in enumerate(passages):
            item_meta = meta[idx] if idx < len(meta) and isinstance(meta[idx], dict) else {}
            if self.evidence_matches_boundary(passage, item_meta, boundary):
                kept_passages.append(passage)
                kept_meta.append(item_meta)
        trace = {"enabled": True, "stage": stage, "before": len(passages), "after": len(kept_passages), "query_boundary": boundary}
        if kept_passages:
            return kept_passages, kept_meta, trace
        trace["empty_after_filter"] = True
        return [], [], trace


class IndexingMixin:
    def load_existing_data(self,passage_hash_ids):
        self.ner_results_path = os.path.join(self.config.working_dir,self.dataset_name, "ner_results.json")
        if os.path.exists(self.ner_results_path):
            with open(self.ner_results_path, "r", encoding="utf-8") as f:
                existing_ner_reuslts = json.load(f)
            existing_passage_hash_id_to_entities = existing_ner_reuslts["passage_hash_id_to_entities"]
            existing_sentence_to_entities = existing_ner_reuslts["sentence_to_entities"]
            existing_passage_hash_ids = set(existing_passage_hash_id_to_entities.keys())
            new_passage_hash_ids = set(passage_hash_ids) - existing_passage_hash_ids
            return existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids
        else:
            return {}, {}, passage_hash_ids

    def _build_density_profiles(self, hash_id_to_passage, passage_hash_id_to_entities):
        profiles = {}
        passage_items = list(hash_id_to_passage.items())
        for pid, passage in tqdm(
            passage_items,
            desc="density profiles",
            unit="passage",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=len(passage_items) < 10000,
        ):
            body = re.sub(r"^\d+:", "", str(passage), count=1).strip()
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
            sentence_count = max(1, len(sents))
            entities = passage_hash_id_to_entities.get(pid, [])
            entity_count = len(entities)
            avg_entities_per_sentence = entity_count / sentence_count
            density = min(1.0, avg_entities_per_sentence / max(1.0, float(self.config.avg_entities_per_sentence_norm)))
            ratio = min(1.0, sum(1 for s in sents if any(e in s for e in entities)) / sentence_count) if entities else 0.0
            kw_density = len(self._extract_keywords(body, top_k=None)) / sentence_count
            norm_kw_density = min(1.0, kw_density / max(1.0, float(self.config.keyword_top_k)))
            score = float(self.config.entity_density_weight)*density + float(self.config.entity_sentence_ratio_weight)*ratio + float(self.config.keyword_density_weight)*norm_kw_density
            if ratio < float(self.config.entity_sentence_ratio_penalty_threshold):
                score *= float(self.config.entity_sentence_ratio_penalty_factor)
            bucket = "graph" if score >= float(self.config.density_threshold) else "hierarchy"
            if ratio <= float(self.config.force_hierarchy_ratio_threshold) and entity_count <= int(self.config.force_hierarchy_entity_count_threshold):
                bucket = "hierarchy"
            if ratio >= float(self.config.force_graph_ratio_threshold) and entity_count >= int(self.config.force_graph_entity_count_threshold):
                bucket = "graph"
            profiles[pid] = {"density_score": score, "entity_count": entity_count, "sentence_count": sentence_count, "avg_entities_per_sentence": avg_entities_per_sentence, "entity_sentence_ratio": ratio, "keyword_density": kw_density, "normalized_keyword_density": norm_kw_density, "bucket": bucket, "sample_id": self.passage_id_to_sample_id.get(pid), "session_ids": self.passage_id_to_session_ids.get(pid, [])}
        self.passage_profiles = profiles

    def _save_dual_index_metadata(self):
        out_path = os.path.join(self.config.working_dir, self.dataset_name, "dual_index_metadata.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"passage_profiles": self.passage_profiles, "hierarchy": {"turn_units": self.hierarchy_turn_units, "window_units": self.hierarchy_window_units, "episode_units": self.hierarchy_episode_units, "trigram_units": self.hierarchy_trigram_units, "turn_to_trigram_ids": dict(self.turn_to_trigram_ids), "passage_to_turn_ids": dict(self.passage_to_turn_ids), "passage_to_window_ids": dict(self.passage_to_window_ids), "window_id_to_episode_id": self.window_id_to_episode_id, "episode_id_to_window_ids": dict(self.episode_id_to_window_ids)}}, f, ensure_ascii=False, indent=2)

    def _get_passage_bucket_sets(self):
        graph_ids, hierarchy_ids = set(), set()
        for pid, prof in getattr(self, "passage_profiles", {}).items():
            if prof.get("bucket") == "graph":
                graph_ids.add(pid)
            else:
                hierarchy_ids.add(pid)
        return graph_ids, hierarchy_ids

    def _build_passage_metadata(self, hash_id_to_passage):
        for pid, passage in hash_id_to_passage.items():
            self.passage_id_to_sample_id[pid] = self._extract_sample_id(passage)
            self.passage_id_to_session_ids[pid] = self._extract_session_ids(passage)

    def index(self, passages):
        index_start = time.time()
        embed_device = getattr(self.config.embedding_model, "device", "unknown")
        logger.info(
            f"[index] Start build | dataset={self.dataset_name} | passages={len(passages)} | "
            f"embedding_device={embed_device} | batch_size={self.config.batch_size}"
        )
        self.node_to_node_stats = defaultdict(dict)
        self.entity_to_sentence_stats = defaultdict(dict)
        self._sparse_matrices_ready = False
        self.isolation_guard.ensure_corpus_cache(passages)
        logger.info("[index] Stage 1/10: passage embedding store")
        self.passage_embedding_store.insert_text(passages)
        self.isolation_guard.write_manifest(passages)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()
        logger.info("[index] Stage 2/10: passage metadata and BM25")
        self._build_passage_metadata(hash_id_to_passage)
        self._build_bm25_index(hash_id_to_passage)
        logger.info("[index] Stage 3/10: loading cached NER state")
        existing_passage_hash_id_to_entities,existing_sentence_to_entities, new_passage_hash_ids = self.load_existing_data(hash_id_to_passage.keys())
        if len(new_passage_hash_ids) > 0:
            logger.info(f"[index] Stage 4/10: NER for {len(new_passage_hash_ids)} new passages")
            new_hash_id_to_passage = {k : hash_id_to_passage[k] for k in new_passage_hash_ids}
            new_passage_hash_id_to_entities,new_sentence_to_entities = self.spacy_ner.batch_ner(new_hash_id_to_passage, self.config.max_workers)
            logger.info("[index] Stage 5/10: merging NER results")
            self.merge_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities)
        else:
            logger.info("[index] Stage 4/10: NER cache hit; no new passages")
        logger.info("[index] Stage 5/10: saving NER results")
        self.save_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        logger.info("[index] Stage 6/10: extracting entity/sentence nodes from NER results")
        entity_nodes, sentence_nodes,passage_hash_id_to_entities,self.entity_to_sentence,self.sentence_to_entity = self.extract_nodes_and_edges(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        logger.info(
            f"[index] NER graph nodes ready | entities={len(entity_nodes)} | "
            f"sentences={len(sentence_nodes)} | passages={len(passage_hash_id_to_entities)}"
        )
        if self.config.enable_dual_index:
            logger.info("[index] Stage 7/10: building hierarchy/window/episode structures")
            self._build_hierarchy_structures(hash_id_to_passage)
            logger.info("[index] Stage 8/10: building density profiles")
            self._build_density_profiles(hash_id_to_passage, passage_hash_id_to_entities)
        else:
            logger.info("[index] Stage 7/10: hierarchy disabled")
            logger.info("[index] Stage 8/10: density profiles disabled")
        logger.info("[index] Stage 9/10: sentence/entity embedding stores")
        self.sentence_embedding_store.insert_text(list(sentence_nodes))
        self.entity_embedding_store.insert_text(list(entity_nodes))
        self.entity_hash_id_to_sentence_hash_ids = {}
        entity_items = list(self.entity_to_sentence.items())
        for entity, sentence in tqdm(
            entity_items,
            desc="entity->sentence map",
            unit="entity",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=len(entity_items) < 10000,
        ):
            entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
            self.entity_hash_id_to_sentence_hash_ids[entity_hash_id] = [self.sentence_embedding_store.text_to_hash_id[s] for s in sentence]
        self.sentence_hash_id_to_entity_hash_ids = {}
        sentence_items = list(self.sentence_to_entity.items())
        for sentence, entities in tqdm(
            sentence_items,
            desc="sentence->entity map",
            unit="sentence",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=len(sentence_items) < 10000,
        ):
            sentence_hash_id = self.sentence_embedding_store.text_to_hash_id[sentence]
            self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id] = [self.entity_embedding_store.text_to_hash_id[e] for e in entities]
        logger.info("[index] Stage 10/10: graph edges and GraphML")
        self.add_entity_to_passage_edges(passage_hash_id_to_entities)
        self.add_adjacent_passage_edges()
        self.augment_graph()
        if self.config.enable_dual_index:
            self._save_dual_index_metadata()
        output_graphml_path = os.path.join(self.config.working_dir,self.dataset_name, "ZeroMem.graphml")
        os.makedirs(os.path.dirname(output_graphml_path),exist_ok=True)
        self.graph.write_graphml(output_graphml_path)
        self.isolation_guard.write_manifest(passages)
        self._sparse_matrices_ready = False
        self._refresh_retrieval_runtime_state()
        logger.info(f"[index] Index ready in {time.time() - index_start:.1f}s")

    def add_adjacent_passage_edges(self):
        passage_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        index_pattern = re.compile(r'^(\d+):')
        indexed_items = [
            (int(match.group(1)), node_key)
            for node_key, text in passage_id_to_text.items()
            if (match := index_pattern.match(text.strip()))
        ]
        indexed_items.sort(key=lambda x: x[0])
        for i in range(len(indexed_items) - 1):
            current_node = indexed_items[i][1]
            next_node = indexed_items[i + 1][1]
            self.node_to_node_stats[current_node][next_node] = 1.0

    def augment_graph(self):
        self.add_nodes()
        self.add_edges()

    def add_nodes(self):
        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()} 
        entity_hash_id_to_text = self.entity_embedding_store.get_hash_id_to_text()
        passage_hash_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text}
        
        passage_hash_ids = set(passage_hash_id_to_text.keys())
        
        for hash_id, text in all_hash_id_to_text.items():
            if hash_id not in existing_nodes:
                self.graph.add_vertex(name=hash_id, content=text)
        
        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}   
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[passage_id] 
            for passage_id in passage_hash_ids 
            if passage_id in self.node_name_to_vertex_idx
        ]

    def add_edges(self):
        edges = []
        weights = []
        
        for node_hash_id, node_to_node_stats in self.node_to_node_stats.items():
            for neighbor_hash_id, weight in node_to_node_stats.items():
                if node_hash_id == neighbor_hash_id:
                    continue
                edges.append((node_hash_id, neighbor_hash_id))
                weights.append(weight)
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights

    def add_entity_to_passage_edges(self, passage_hash_id_to_entities):
        passage_to_entity_count ={} 
        passage_to_all_score = defaultdict(int)
        passage_items = list(passage_hash_id_to_entities.items())
        for passage_hash_id, entities in tqdm(
            passage_items,
            desc="entity-passage edges",
            unit="passage",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=len(passage_items) < 10000,
        ):
            passage = self.passage_embedding_store.hash_id_to_text[passage_hash_id]
            for entity in entities:
                entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
                count = passage.count(entity)
                passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
                passage_to_all_score[passage_hash_id] += count
        edge_items = list(passage_to_entity_count.items())
        for (passage_hash_id, entity_hash_id), count in tqdm(
            edge_items,
            desc="entity-passage weights",
            unit="edge",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=len(edge_items) < 10000,
        ):
            score = count / passage_to_all_score[passage_hash_id]
            self.node_to_node_stats[passage_hash_id][entity_hash_id] = score

    def extract_nodes_and_edges(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        entity_nodes = set()
        sentence_nodes = set()
        passage_hash_id_to_entities = defaultdict(set)
        entity_to_sentence= defaultdict(set)
        sentence_to_entity = defaultdict(set)
        passage_items = list(existing_passage_hash_id_to_entities.items())
        for passage_hash_id, entities in tqdm(
            passage_items,
            desc="NER postprocess passages",
            unit="passage",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=len(passage_items) < 10000,
        ):
            for entity in entities:
                entity_nodes.add(entity)
                passage_hash_id_to_entities[passage_hash_id].add(entity)
        sentence_items = list(existing_sentence_to_entities.items())
        for sentence,entities in tqdm(
            sentence_items,
            desc="NER postprocess sentences",
            unit="sentence",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=len(sentence_items) < 10000,
        ):
            sentence_nodes.add(sentence)
            for entity in entities:
                entity_to_sentence[entity].add(sentence)
                sentence_to_entity[sentence].add(entity)
        return entity_nodes, sentence_nodes, passage_hash_id_to_entities, entity_to_sentence, sentence_to_entity

    def merge_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities):
        existing_passage_hash_id_to_entities.update(new_passage_hash_id_to_entities)
        existing_sentence_to_entities.update(new_sentence_to_entities)
        return existing_passage_hash_id_to_entities, existing_sentence_to_entities

    def save_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        with open(self.ner_results_path, "w", encoding="utf-8") as f:
            json.dump({"passage_hash_id_to_entities": existing_passage_hash_id_to_entities, "sentence_to_entities": existing_sentence_to_entities}, f, ensure_ascii=False)

    def _init_hierarchy_embedding_stores(self):
        if self.window_embedding_store is None:
            self.window_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "window_embedding.parquet"), batch_size=self.config.batch_size, namespace="window")
        if self.episode_embedding_store is None:
            self.episode_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "episode_embedding.parquet"), batch_size=self.config.batch_size, namespace="episode")

    def _ensure_hierarchy_runtime(self):
        if self.hierarchy_runtime_ready:
            return
        if not self.config.enable_dual_index:
            raise RuntimeError('Hierarchy/Fusion runtime requested, but enable_dual_index is False. Re-index with dual index enabled or run graph mode only.')
        self._init_dual_state()
        self._init_hierarchy_embedding_stores()
        meta_path = os.path.join(self.config.working_dir, self.dataset_name, "dual_index_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.passage_profiles = meta.get("passage_profiles", {})
            hierarchy = meta.get("hierarchy", {})
            self.hierarchy_turn_units = hierarchy.get("turn_units", {})
            self.hierarchy_window_units = hierarchy.get("window_units", {})
            self.hierarchy_episode_units = hierarchy.get("episode_units", {})
            self.passage_to_turn_ids = defaultdict(list, hierarchy.get("passage_to_turn_ids", {}))
            self.passage_to_window_ids = defaultdict(list, hierarchy.get("passage_to_window_ids", {}))
            self.window_id_to_episode_id = hierarchy.get("window_id_to_episode_id", {})
            self.episode_id_to_window_ids = defaultdict(list, hierarchy.get("episode_id_to_window_ids", {}))

            self.turn_id_to_text = {}
            self.window_id_to_text = {}
            self.episode_id_to_text = {}
            self.turn_text_to_id = {}
            self.window_text_to_id = {}
            self.episode_text_to_id = {}
            self.passage_id_to_sample_id = {}
            self.passage_id_to_session_ids = {}
            self.hierarchy_trigram_units = hierarchy.get("trigram_units", {})
            self.turn_to_trigram_ids = defaultdict(list, hierarchy.get("turn_to_trigram_ids", {}))
            self.trigram_id_to_text = {}

            for tid, t in self.hierarchy_turn_units.items():
                if "source_ref" not in t:
                    speaker = str(t.get("speaker", "")).strip()
                    if re.fullmatch(r"D\d+:\d+", speaker):
                        t["source_ref"] = speaker
                    else:
                        m = re.match(r"^(D\d+:\d+)\b", str(t.get("text", "")))
                        t["source_ref"] = m.group(1) if m else ""
                if "session_time" not in t:
                    pid = t.get("passage_hash_id")
                    passage_text = self.passage_embedding_store.hash_id_to_text.get(pid, "") if pid else ""
                    t["session_time"] = self._extract_session_datetime_value(passage_text)
                txt = t.get("text", "")
                self.turn_id_to_text[tid] = txt
                if txt:
                    self.turn_text_to_id[txt] = tid
                pid = t.get("passage_hash_id")
                if pid:
                    sample_id = t.get("sample_id")
                    if sample_id:
                        self.passage_id_to_sample_id[pid] = sample_id
                    sess = t.get("session_ids", [])
                    if sess:
                        self.passage_id_to_session_ids[pid] = sess
            for wid, w in self.hierarchy_window_units.items():
                txt = w.get("text", "")
                self.window_id_to_text[wid] = txt
                if txt:
                    self.window_text_to_id[txt] = wid
            for eid, e in self.hierarchy_episode_units.items():
                txt = e.get("text", "")
                self.episode_id_to_text[eid] = txt
                if txt:
                    self.episode_text_to_id[txt] = eid
            for gid, g in self.hierarchy_trigram_units.items():
                txt = g.get("text", "")
                self.trigram_id_to_text[gid] = txt
            if not self.hierarchy_trigram_units and self.hierarchy_turn_units:
                self._build_local_trigram_structures()
            else:
                self._refresh_trigram_embeddings()
            self._refresh_hierarchy_runtime_matrices()

        self.hierarchy_runtime_ready = True

    def _init_dual_state(self):
        self.passage_profiles = {}
        self.passage_id_to_sample_id = {}
        self.passage_id_to_session_ids = {}
        self.hierarchy_turn_units = {}
        self.hierarchy_window_units = {}
        self.hierarchy_episode_units = {}
        self.passage_to_turn_ids = defaultdict(list)
        self.passage_to_window_ids = defaultdict(list)
        self.window_id_to_episode_id = {}
        self.episode_id_to_window_ids = defaultdict(list)
        self.turn_id_to_text = {}
        self.window_id_to_text = {}
        self.episode_id_to_text = {}
        self.turn_text_to_id = {}
        self.window_text_to_id = {}
        self.episode_text_to_id = {}
        self.hierarchy_trigram_units = {}
        self.turn_to_trigram_ids = defaultdict(list)
        self.trigram_id_to_text = {}
        self.hierarchy_trigram_embeddings = None
        self.hierarchy_trigram_id_list = []
        self.hierarchy_trigram_id_to_embedding_idx = {}
        self.hierarchy_episode_id_list = []
        self.hierarchy_episode_id_to_matrix_idx = {}
        self.hierarchy_episode_embeddings = None
        self.hierarchy_window_id_list = []
        self.hierarchy_window_id_to_matrix_idx = {}
        self.hierarchy_window_embeddings = None
        self.hierarchy_window_signal_cache = {}
        self.hierarchy_episode_signal_cache = {}
        self.hierarchy_trigram_signal_cache = {}
        self.same_sample_turn_cache = {}

    def _ordered_passage_items(self, hash_id_to_passage):
        items = []
        for pid, passage in hash_id_to_passage.items():
            body = passage
            order = len(items)
            m = re.match(r"^(\d+):(.*)$", passage, re.S)
            if m:
                order = int(m.group(1))
                body = m.group(2).strip()
            items.append((order, pid, body))
        items.sort(key=lambda x: x[0])
        return items

    def _split_text_into_turns(self, body, passage_hash_id):
        text = str(body).strip()
        if not text:
            return []
        sample_id = self._extract_sample_id(body)
        passage_session_ids = self._extract_session_ids(body)
        passage_session_time = self._extract_session_datetime_value(body)

        document = re.search(r"(?m)^\[Document\][ \t]*([^\n]+)", text)
        source_chunks = re.search(r"(?m)^\[SourceChunks\][^\n]*(?:\n|$)", text)
        if document and source_chunks:
            title_match = re.search(r"(?m)^\[Title\][ \t]*([^\n]*)", text[:source_chunks.start()])
            title = title_match.group(1).strip() if title_match else ""
            body_text = text[source_chunks.end():].strip()
            sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", body_text) if part.strip()]
            turns = []
            for i, sentence in enumerate(sentences):
                sentence_text = title + "\n" + sentence if title else sentence
                turns.append({
                    "turn_id": f"{passage_hash_id}::turn::{i}", "passage_hash_id": passage_hash_id,
                    "speaker": None, "dia_id": "", "source_ref": f"{passage_hash_id}::sentence::{i}",
                    "session_time": "", "text": sentence_text,
                    "keywords": self._extract_keywords(sentence_text, top_k=self.config.keyword_top_k),
                    "sample_id": sample_id, "session_ids": [], "order_in_passage": i,
                    "document_id": document.group(1).strip(), "document_title": title,
                })
            return turns

        direct_segments = self._extract_dialogue_segments(body)
        if direct_segments:
            turns = []
            for i, seg in enumerate(direct_segments):
                dia_id = str(seg.get("dia_id", "")).strip()
                speaker = str(seg.get("speaker", "")).strip() or None
                text_part = str(seg.get("text", "")).strip()
                if not text_part:
                    continue
                turn_id = f"{passage_hash_id}::turn::{i}"
                turns.append({
                    "turn_id": turn_id,
                    "passage_hash_id": passage_hash_id,
                    "speaker": speaker,
                    "dia_id": dia_id,
                    "source_ref": dia_id,
                    "session_time": passage_session_time,
                    "text": text_part,
                    "keywords": self._extract_keywords(text_part, top_k=self.config.keyword_top_k),
                    "sample_id": sample_id,
                    "session_ids": list(passage_session_ids),
                    "order_in_passage": i,
                })
            if turns:
                return turns

        normalized = text
        if ("\n" not in normalized and "\r" not in normalized) and len(re.findall(r"\bD\d+:\d+\b", normalized)) >= 2:
            normalized = re.sub(r"\s+(?=D\d+:\d+\s+[A-Z][a-z]+\s*:)", "\n", normalized)
        lines = [ln.strip() for ln in re.split(r"[\n\r]+", normalized) if ln.strip()]
        chunks = []
        for ln in lines:
            if re.match(r"^(?:\[.*?\]|[A-Z][a-z]+|D\d+:\d+)\s*[:\-]", ln):
                chunks.append(ln)
            else:
                parts = [p.strip() for p in re.split(r"(?=(?:D\d+:\d+\s+[A-Z][a-z]+\s*:|D\d+:\d+|[A-Z][a-z]+\s*[:\-]))", ln) if p.strip()]
                if len(parts) <= 1:
                    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", ln) if p.strip()]
                chunks.extend(parts if parts else [ln])
        merged = []
        min_chars = int(self.config.hierarchy_min_turn_chars)
        for ch in chunks:
            if merged and len(ch) < min_chars and not re.match(r"^(?:\[.*?\]|[A-Z][a-z]+|D\d+:\d+)\s*[:\-]", ch):
                merged[-1] = merged[-1] + " " + ch
            else:
                merged.append(ch)
        turns = []
        for i, ch in enumerate(merged):
            speaker = None
            text_part = ch
            dia_id = ""
            m = re.match(r"^(D\d+:\d+)\s+([A-Z][a-z]+)\s*:\s*(.*)$", ch)
            if m:
                dia_id = m.group(1).strip()
                speaker = m.group(2).strip()
                text_part = m.group(3).strip()
            else:
                m = re.match(r"^(\[.*?\]|[A-Z][a-z]+|D\d+:\d+)\s*[:\-]\s*(.*)$", ch)
                if m:
                    speaker = m.group(1)
                    text_part = m.group(2).strip()
                    if re.fullmatch(r"D\d+:\d+", str(speaker or "")):
                        dia_id = str(speaker)
                if not dia_id:
                    m_dia = re.search(r"\b(D\d+:\d+)\b", ch)
                    if m_dia:
                        dia_id = m_dia.group(1)
                if not dia_id:
                    dia_id = self._fix_turn_dia_id(text_part, body, speaker=speaker, hint_index=i)
            if not text_part:
                continue
            turn_id = f"{passage_hash_id}::turn::{i}"
            turns.append({
                "turn_id": turn_id,
                "passage_hash_id": passage_hash_id,
                "speaker": speaker,
                "dia_id": dia_id,
                "source_ref": dia_id,
                "session_time": passage_session_time,
                "text": text_part,
                "keywords": self._extract_keywords(text_part, top_k=self.config.keyword_top_k),
                "sample_id": sample_id,
                "session_ids": list(passage_session_ids),
                "order_in_passage": i,
            })
        return turns

    def _ordered_turn_ids_for_passage(self, passage_hash_id):
        turn_ids = list(self.passage_to_turn_ids.get(passage_hash_id, []))
        turn_ids.sort(key=lambda tid: int(self.hierarchy_turn_units.get(tid, {}).get("order_in_passage", 0)))
        return turn_ids

    def _compose_trigram_source_id(self, source_ids):
        ids = [str(s).strip() for s in source_ids if str(s).strip()]
        if not ids:
            return ""
        if all(re.fullmatch(r"D\d+:\d+", sid) for sid in ids):
            sess = {sid.split(":")[0] for sid in ids}
            lines = [int(sid.split(":")[1]) for sid in ids]
            if len(sess) == 1:
                uniq_lines = sorted(set(lines))
                if uniq_lines == list(range(min(uniq_lines), max(uniq_lines) + 1)):
                    return f"{list(sess)[0]}:{min(uniq_lines)}-{max(uniq_lines)}"
        return "|".join(ids)

    def _build_local_trigram_structures(self):
        self.hierarchy_trigram_units = {}
        self.turn_to_trigram_ids = defaultdict(list)
        self.trigram_id_to_text = {}
        for passage_hash_id in list(self.passage_to_turn_ids.keys()):
            ordered_turn_ids = self._ordered_turn_ids_for_passage(passage_hash_id)
            if len(ordered_turn_ids) < 3:
                continue
            for start in range(len(ordered_turn_ids) - 2):
                tri_turn_ids = ordered_turn_ids[start:start+3]
                turns = [self.hierarchy_turn_units[tid] for tid in tri_turn_ids if tid in self.hierarchy_turn_units]
                if len(turns) != 3:
                    continue
                trigram_id = f"{passage_hash_id}::tri::{start}"
                source_ids = [self._get_turn_source_id(t) for t in turns]
                session_time = next((self._get_turn_session_time(t) for t in turns if self._get_turn_session_time(t)), "")
                lines = []
                for t, sid in zip(turns, source_ids):
                    prefix = f"[{sid}] " if sid else ""
                    lines.append(prefix + str(t.get("text", "")).strip())
                text = "\n".join(lines).strip()
                unit = {
                    "trigram_id": trigram_id,
                    "turn_ids": list(tri_turn_ids),
                    "center_turn_id": tri_turn_ids[1],
                    "passage_hash_id": passage_hash_id,
                    "sample_id": turns[1].get("sample_id"),
                    "session_ids": list(turns[1].get("session_ids", [])),
                    "session_time": session_time,
                    "source_ids": source_ids,
                    "source_ref": self._compose_trigram_source_id(source_ids),
                    "text": text,
                    "order_in_passage": int(turns[1].get("order_in_passage", start + 1)),
                }
                if turns[1].get("document_id"):
                    unit.update(document_id=turns[1]["document_id"], document_title=turns[1].get("document_title", ""))
                self.hierarchy_trigram_units[trigram_id] = unit
                self.trigram_id_to_text[trigram_id] = text
                for tid in tri_turn_ids:
                    self.turn_to_trigram_ids[tid].append(trigram_id)
        self._refresh_trigram_embeddings()

    def _refresh_trigram_embeddings(self):
        ids = list(self.hierarchy_trigram_units.keys())
        self.hierarchy_trigram_id_list = ids
        self.hierarchy_trigram_id_to_embedding_idx = {gid: i for i, gid in enumerate(ids)}
        texts = [self.hierarchy_trigram_units[gid].get("text", "") for gid in ids]
        if not texts:
            self.hierarchy_trigram_embeddings = np.zeros((0, 1), dtype=float)
            return
        embs = self._encode_embeddings(texts, "hierarchy trigram")
        self.hierarchy_trigram_embeddings = np.array(embs, dtype=np.float32)

    def _embedding_matrix_from_store(self, ids, text_lookup, embedding_store):
        matrix_rows = []
        valid_ids = []
        if not ids or embedding_store is None:
            return valid_ids, np.zeros((0, 1), dtype=np.float32), {}
        for unit_id in ids:
            text = str(text_lookup.get(unit_id, "")).strip()
            if not text:
                continue
            hash_id = embedding_store.text_to_hash_id.get(text)
            if hash_id is None:
                continue
            emb_idx = embedding_store.hash_id_to_idx.get(hash_id)
            if emb_idx is None:
                continue
            valid_ids.append(unit_id)
            matrix_rows.append(embedding_store.embeddings[emb_idx])
        if not matrix_rows:
            return [], np.zeros((0, 1), dtype=np.float32), {}
        id_to_idx = {unit_id: idx for idx, unit_id in enumerate(valid_ids)}
        return valid_ids, np.asarray(matrix_rows, dtype=np.float32), id_to_idx

    def _refresh_hierarchy_runtime_matrices(self):
        self.hierarchy_episode_id_list, self.hierarchy_episode_embeddings, self.hierarchy_episode_id_to_matrix_idx = self._embedding_matrix_from_store(
            list(self.hierarchy_episode_units.keys()),
            self.episode_id_to_text,
            self.episode_embedding_store,
        )
        self.hierarchy_window_id_list, self.hierarchy_window_embeddings, self.hierarchy_window_id_to_matrix_idx = self._embedding_matrix_from_store(
            list(self.hierarchy_window_units.keys()),
            self.window_id_to_text,
            self.window_embedding_store,
        )
        self.hierarchy_window_signal_cache = {}
        for wid in self.hierarchy_window_units.keys():
            raw_text = str(self.window_id_to_text.get(wid, ""))
            text = raw_text.lower()
            self.hierarchy_window_signal_cache[wid] = {
                "raw": raw_text,
                "lower": text,
                "tokens": set(self._normalize_bm25_tokens(text)),
            }
        self.hierarchy_episode_signal_cache = {}
        for eid, episode in self.hierarchy_episode_units.items():
            raw_text = str(self.episode_id_to_text.get(eid, ""))
            text = raw_text.lower()
            kws = {str(k).lower() for k in episode.get("keywords", [])}
            self.hierarchy_episode_signal_cache[eid] = {
                "raw": raw_text,
                "lower": text,
                "tokens": set(self._normalize_bm25_tokens(text)) | kws,
            }
        self._refresh_hierarchy_trigram_signal_cache()

    def _refresh_hierarchy_trigram_signal_cache(self):
        self.hierarchy_trigram_signal_cache = {}
        for gid, tri in self.hierarchy_trigram_units.items():
            txt = str((tri or {}).get("text", "")).strip()
            low = txt.lower()
            member_turn_ids = list((tri or {}).get("turn_ids", []))
            low_value_count = 0
            supportive_count = 0
            question_like_count = 0
            for tid in member_turn_ids:
                turn_txt = str(self.hierarchy_turn_units.get(tid, {}).get("text", ""))
                if self._is_low_value_turn(turn_txt):
                    low_value_count += 1
                if self._is_supportive_reply_turn(turn_txt):
                    supportive_count += 1
                if self._is_question_like_turn(turn_txt):
                    question_like_count += 1
            where = {}
            for mode in ("where", "where_origin", "where_list"):
                where[mode] = {
                    "pattern": self._where_pattern_score(txt, mode),
                    "terms": self._where_location_term_hits(txt, mode),
                    "noise": self._where_noise_penalty(txt, mode),
                }
            self.hierarchy_trigram_signal_cache[gid] = {
                "raw": txt,
                "lower": low,
                "has_time_cue": self._contains_time_cue(txt),
                "has_year": bool(re.search(r"\b\d{4}\b", txt)),
                "has_duration_cue": self._contains_duration_cue(txt),
                "location_score": self._location_cue_score(txt),
                "enumeration_score": self._enumeration_shape_score(txt),
                "low_value_count": low_value_count,
                "member_turn_count": len(member_turn_ids),
                "supportive_count": supportive_count,
                "question_like_count": question_like_count,
                "generic_supportive": 1.0 if self._is_supportive_reply_turn(txt) else 0.0,
                "generic_question_like": 1.0 if self._is_question_like_turn(txt) else 0.0,
                "first_person": 1.0 if self._is_first_person_experience_turn(txt) else 0.0,
                "where": where,
            }

    def _build_episode_units(self, ordered_windows):
        if not ordered_windows:
            return []
        texts = [w["text"] for w in ordered_windows]
        embs = self._encode_embeddings(texts, "episode construction window")
        sim_th = float(self.config.hierarchy_episode_similarity_threshold)
        kw_th = float(self.config.hierarchy_episode_keyword_jaccard_threshold)
        max_windows = int(self.config.hierarchy_episode_max_windows)
        episodes = []
        def finalize(cur):
            if cur is None:
                return
            eid = f"episode::{len(episodes)}"
            episode = {"episode_id": eid, "window_ids": list(cur["window_ids"]), "turn_ids": list(dict.fromkeys(cur["turn_ids"])), "passage_hash_ids": list(dict.fromkeys(cur["passage_hash_ids"])), "sample_id": cur["sample_id"], "session_ids": list(cur["session_ids"]), "keywords": list(dict.fromkeys(cur["keywords"])), "text": "\n".join(cur["texts"])}
            if cur.get("document_id"):
                episode.update(document_id=cur["document_id"], document_title=cur.get("document_title", ""))
            episodes.append(episode)
        current = None
        for i, w in enumerate(ordered_windows):
            if current is None:
                current = {"window_ids":[w["window_id"]],"turn_ids":list(w["turn_ids"]),"passage_hash_ids":list(w["passage_hash_ids"]),"sample_id":w.get("sample_id"),"session_ids":list(w.get("session_ids", [])),"keywords":list(w["keywords"]),"texts":[w["text"]],"centroid":embs[i], "count":1}
                if w.get("document_id"):
                    current.update(document_id=w["document_id"], document_title=w.get("document_title", ""))
                continue
            jac = len(set(current["keywords"]) & set(w["keywords"])) / max(1, len(set(current["keywords"]) | set(w["keywords"])))
            sim = float(np.dot(current["centroid"], embs[i]))
            same_sample = (current["sample_id"] == w.get("sample_id")) or current["sample_id"] is None or w.get("sample_id") is None
            same_session = tuple(current.get("session_ids", [])) == tuple(w.get("session_ids", []))
            same_document = not (current.get("document_id") or w.get("document_id")) or set(current["passage_hash_ids"]) == set(w["passage_hash_ids"])
            if same_sample and same_session and same_document and current["count"] < max_windows and (sim >= sim_th or jac >= kw_th):
                current["window_ids"].append(w["window_id"])
                current["turn_ids"].extend(w["turn_ids"])
                current["passage_hash_ids"].extend(w["passage_hash_ids"])
                current["keywords"].extend([k for k in w["keywords"] if k not in current["keywords"]])
                current["texts"].append(w["text"])
                current["centroid"] = (current["centroid"] * current["count"] + embs[i]) / (current["count"] + 1)
                current["count"] += 1
            else:
                finalize(current)
                current = {"window_ids":[w["window_id"]],"turn_ids":list(w["turn_ids"]),"passage_hash_ids":list(w["passage_hash_ids"]),"sample_id":w.get("sample_id"),"session_ids":list(w.get("session_ids", [])),"keywords":list(w["keywords"]),"texts":[w["text"]],"centroid":embs[i], "count":1}
                if w.get("document_id"):
                    current.update(document_id=w["document_id"], document_title=w.get("document_title", ""))
        finalize(current)
        return episodes

    def _build_hierarchy_structures(self, hash_id_to_passage):
        self._init_dual_state()
        ordered_passages = self._ordered_passage_items(hash_id_to_passage)
        ordered_windows = []
        window_size = max(1, int(self.config.hierarchy_turn_window_size))
        window_counter = 0
        for _, pid, body in tqdm(
            ordered_passages,
            desc="hierarchy passages",
            unit="passage",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=len(ordered_passages) < 10000,
        ):
            sample_id = self._extract_sample_id(body)
            session_ids = self._extract_session_ids(body)
            self.passage_id_to_sample_id[pid] = sample_id
            self.passage_id_to_session_ids[pid] = session_ids
            turns = self._split_text_into_turns(body, pid)
            for turn in turns:
                self.hierarchy_turn_units[turn["turn_id"]] = turn
                self.turn_id_to_text[turn["turn_id"]] = turn["text"]
                self.turn_text_to_id[turn["text"]] = turn["turn_id"]
                self.passage_to_turn_ids[pid].append(turn["turn_id"])
            for start in range(len(turns)):
                end = min(len(turns), start + window_size)
                local_turns = turns[start:end]
                if not local_turns:
                    continue
                wid = f"window::{window_counter}"
                window_counter += 1
                text = "\n".join(t["text"] for t in local_turns)
                keywords = []
                for t in local_turns:
                    for kw in t["keywords"]:
                        if kw not in keywords:
                            keywords.append(kw)
                win = {"window_id": wid, "turn_ids": [t["turn_id"] for t in local_turns], "passage_hash_ids": [pid], "text": text, "keywords": keywords, "sample_id": sample_id, "session_ids": list(session_ids)}
                if local_turns[0].get("document_id"):
                    win.update(document_id=local_turns[0]["document_id"], document_title=local_turns[0].get("document_title", ""))
                self.hierarchy_window_units[wid] = win
                self.window_id_to_text[wid] = text
                self.window_text_to_id[text] = wid
                self.passage_to_window_ids[pid].append(wid)
                ordered_windows.append(win)
        self._build_local_trigram_structures()
        episodes = self._build_episode_units(ordered_windows)
        for ep in episodes:
            eid = ep["episode_id"]
            self.hierarchy_episode_units[eid] = ep
            self.episode_id_to_text[eid] = ep["text"]
            self.episode_text_to_id[ep["text"]] = eid
            for wid in ep["window_ids"]:
                self.window_id_to_episode_id[wid] = eid
                self.episode_id_to_window_ids[eid].append(wid)
        self._init_hierarchy_embedding_stores()
        if self.hierarchy_window_units:
            self.window_embedding_store.insert_text([w["text"] for w in self.hierarchy_window_units.values()])
        if self.hierarchy_episode_units:
            self.episode_embedding_store.insert_text([e["text"] for e in self.hierarchy_episode_units.values()])
        self._refresh_hierarchy_runtime_matrices()
        self.hierarchy_runtime_ready = True
