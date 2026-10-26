"""Graph and hierarchy retrieval, evidence selection, routing and fusion."""

from collections import Counter
from .indexing import IsolationGuard
from collections import defaultdict
from .utils import logger
import math
from .utils import min_max_normalize
import numpy as np
import re
import unicodedata
import torch
from tqdm import tqdm


class _DocumentTitleSupport:
    """Index explicit document titles once; find references within one sample."""

    @staticmethod
    def _tokens(text):
        return tuple(
            token for token in re.findall(r"\w+|[^\w\s]", str(text or "").casefold())
            if re.fullmatch(r"\w+", token) or unicodedata.category(token[0]).startswith("S")
        )

    def __init__(self, records, budget=2):
        self.records = records
        self.budget = max(0, min(2, int(budget)))
        self.index = {}
        self.order = {pid: rank for rank, pid in enumerate(records)}
        for pid, record in records.items():
            sample = str(record.get("sample_id", "")).strip()
            title = self._tokens(record.get("title", ""))
            if sample and title:
                self.index.setdefault(sample, {}).setdefault(title[0], {}).setdefault(title, []).append(pid)

    def support(self, primary_ids, ranked_ids, profile):
        sample = str((profile or {}).get("sample_id", "")).strip()
        index = self.index.get(sample, {})
        trace = {"matched_pages": 0, "ambiguous_titles": [], "selected_ids": []}
        if not index or not self.budget:
            return [], trace
        primary = set(primary_ids)
        ranks = {pid: rank for rank, pid in reversed(list(enumerate(ranked_ids)))}
        candidates = {}
        ambiguous = set()
        for primary_rank, source_id in enumerate(primary_ids):
            source = self.records.get(source_id, {})
            if str(source.get("sample_id", "")).strip() != sample:
                continue
            words = self._tokens(source.get("body", ""))
            for start, first in enumerate(words):
                for title, ids in index.get(first, {}).items():
                    if words[start:start + len(title)] != title or primary.intersection(ids):
                        continue
                    if len({str(self.records[pid].get("body", "")).strip() for pid in ids}) != 1:
                        ambiguous.add(" ".join(title))
                        continue
                    pid = min(ids, key=lambda key: (ranks.get(key, len(ranked_ids)), self.order[key]))
                    priority = (ranks.get(pid, len(ranked_ids)), primary_rank, self.order[pid])
                    if pid not in candidates or priority < candidates[pid]:
                        candidates[pid] = priority
        selected = sorted(candidates, key=candidates.get)[:self.budget]
        trace.update(matched_pages=len(candidates), ambiguous_titles=sorted(ambiguous), selected_ids=selected)
        return selected, trace


class RetrievalMixin:
    def _build_bm25_index(self, hash_id_to_passage):
        self._document_support_index = None
        self.bm25_doc_freq = defaultdict(int)
        self.bm25_doc_len = {}
        self.bm25_texts = {}
        self.bm25_token_cache = {}
        self.bm25_tf_cache = {}
        self.bm25_postings = defaultdict(dict)
        self.hierarchy_reranker = None
        self.hierarchy_reranker_failed = False
        total_len = 0
        for pid, passage in hash_id_to_passage.items():
            text = str(passage)
            toks = self._normalize_bm25_tokens(text)
            self.bm25_texts[pid] = text
            self.bm25_token_cache[pid] = toks
            tf = Counter(toks)
            self.bm25_tf_cache[pid] = tf
            self.bm25_doc_len[pid] = len(toks)
            total_len += len(toks)
            for term, count in tf.items():
                self.bm25_doc_freq[term] += 1
                self.bm25_postings[term][pid] = count
        self.bm25_avgdl = total_len / max(1, len(self.bm25_doc_len))

    def _bm25_score_map(self, question, candidate_ids=None):
        ids = list(candidate_ids) if candidate_ids is not None else list(self.bm25_texts.keys())
        query_terms = self._normalize_bm25_tokens(question)
        if not query_terms:
            return {pid: 0.0 for pid in ids}
        N = max(1, len(self.bm25_doc_len))
        k1 = float(self.config.graph_bm25_k1)
        b = float(self.config.graph_bm25_b)
        if candidate_ids is None:
            scores = defaultdict(float)
            for term in query_terms:
                postings = self.bm25_postings.get(term)
                if not postings:
                    continue
                df = self.bm25_doc_freq.get(term, 0)
                if df <= 0:
                    continue
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                for pid, f in postings.items():
                    if f <= 0:
                        continue
                    dl = self.bm25_doc_len.get(pid, 0)
                    denom = f + k1 * (1 - b + b * dl / max(1e-9, self.bm25_avgdl))
                    scores[pid] += idf * ((f * (k1 + 1)) / max(1e-9, denom))
            return {pid: float(score) for pid, score in scores.items()}
        scores = {}
        for pid in ids:
            toks = self.bm25_token_cache.get(pid, [])
            if not toks:
                scores[pid] = 0.0
                continue
            tf = self.bm25_tf_cache.get(pid)
            if tf is None:
                tf = Counter(toks)
                self.bm25_tf_cache[pid] = tf
            dl = self.bm25_doc_len.get(pid, len(toks))
            s = 0.0
            for term in query_terms:
                df = self.bm25_doc_freq.get(term, 0)
                if df <= 0:
                    continue
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                f = tf.get(term, 0)
                if f <= 0:
                    continue
                denom = f + k1 * (1 - b + b * dl / max(1e-9, self.bm25_avgdl))
                s += idf * ((f * (k1 + 1)) / max(1e-9, denom))
            scores[pid] = float(s)
        return scores

    def _graph_keyword_boost_rerank(self, question, ids, scores, query_profile=None):
        if not ids:
            return ids, scores
        dense = {pid: float(s) for pid, s in zip(ids, scores)}
        topk = int(self.config.graph_bm25_top_k)
        all_bm25 = self._bm25_score_map(question)
        bm25_top = sorted(all_bm25.items(), key=lambda x: x[1], reverse=True)[:topk]
        candidate_ids = list(dict.fromkeys(list(ids) + [pid for pid, _ in bm25_top]))
        dense_scores = np.array([dense.get(pid, 0.0) for pid in candidate_ids], dtype=float)
        bm25_scores = np.array([all_bm25.get(pid, 0.0) for pid in candidate_ids], dtype=float)

        def _norm(arr):
            if len(arr) == 0:
                return arr
            mn, mx = float(arr.min()), float(arr.max())
            if mx - mn < 1e-9:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        dense_n = _norm(dense_scores)
        bm25_n = _norm(bm25_scores)
        phrases = self._extract_query_phrases(question)
        phrase_boost = []
        for pid in candidate_ids:
            text = self.bm25_texts.get(pid, "").lower()
            boost = 0.0
            for ph in phrases:
                if ph and ph in text:
                    boost += 1.0
            phrase_boost.append(boost)
        phrase_n = _norm(np.array(phrase_boost, dtype=float))
        if self._task_type(query_profile) == "multi_evidence":
            embedding_weight = float(self.config.multi_evidence_graph_embedding_weight)
            bm25_weight = float(self.config.multi_evidence_graph_bm25_weight)
            total = max(1e-9, embedding_weight + bm25_weight)
            final = (embedding_weight / total) * dense_n + (bm25_weight / total) * bm25_n
        else:
            final = dense_n + float(self.config.graph_bm25_boost) * bm25_n + float(self.config.graph_phrase_boost) * phrase_n
        order = np.argsort(final)[::-1]
        return [candidate_ids[i] for i in order], [float(final[i]) for i in order]

    def _ensure_sparse_matrices_precomputed(self, log=True):
        if not self.config.use_vectorized_retrieval:
            return
        if getattr(self, "_sparse_matrices_ready", False):
            return
        if not hasattr(self, "entity_hash_ids") or not hasattr(self, "sentence_hash_ids"):
            self._refresh_retrieval_runtime_state()
        if not hasattr(self, "entity_hash_id_to_sentence_hash_ids"):
            self.entity_hash_id_to_sentence_hash_ids = {}
        if not hasattr(self, "sentence_hash_id_to_entity_hash_ids"):
            self.sentence_hash_id_to_entity_hash_ids = {}
        if log:
            logger.info("Precomputing sparse adjacency matrices for vectorized retrieval...")
        self._precompute_sparse_matrices()
        self._sparse_matrices_ready = True

    def _precompute_sparse_matrices(self):
        """
        Precompute and cache sparse adjacency matrices for efficient vectorized retrieval using PyTorch.
        This is called once at the beginning of retrieve() to avoid rebuilding matrices per query.
        """
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)
        
        # Build entity-to-sentence matrix (Mention matrix) using COO format
        entity_to_sentence_indices = []
        entity_to_sentence_values = []
        
        for entity_hash_id, sentence_hash_ids in self.entity_hash_id_to_sentence_hash_ids.items():
            entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
            for sentence_hash_id in sentence_hash_ids:
                sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
                entity_to_sentence_indices.append([entity_idx, sentence_idx])
                entity_to_sentence_values.append(1.0)
        
        # Build sentence-to-entity matrix
        sentence_to_entity_indices = []
        sentence_to_entity_values = []
        
        for sentence_hash_id, entity_hash_ids in self.sentence_hash_id_to_entity_hash_ids.items():
            sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
            for entity_hash_id in entity_hash_ids:
                entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
                sentence_to_entity_indices.append([sentence_idx, entity_idx])
                sentence_to_entity_values.append(1.0)
        
        # Convert to PyTorch sparse tensors (COO format, then convert to CSR for efficiency)
        if len(entity_to_sentence_indices) > 0:
            e2s_indices = torch.tensor(entity_to_sentence_indices, dtype=torch.long).t()
            e2s_values = torch.tensor(entity_to_sentence_values, dtype=torch.float32)
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                e2s_indices, e2s_values, (num_entities, num_sentences), device=self.device
            ).coalesce()
        else:
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_entities, num_sentences), device=self.device
            )
        
        if len(sentence_to_entity_indices) > 0:
            s2e_indices = torch.tensor(sentence_to_entity_indices, dtype=torch.long).t()
            s2e_values = torch.tensor(sentence_to_entity_values, dtype=torch.float32)
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                s2e_indices, s2e_values, (num_sentences, num_entities), device=self.device
            ).coalesce()
        else:
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_sentences, num_entities), device=self.device
            )

    def graph_search_with_seed_entities(self, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        if self.config.use_vectorized_retrieval:
            entity_weights, actived_entities = self.calculate_entity_scores_vectorized(question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores)
        else:
            entity_weights, actived_entities = self.calculate_entity_scores(question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores)
        passage_weights = self.calculate_passage_scores(question_embedding,actived_entities)
        node_weights = entity_weights + passage_weights
        ppr_sorted_passage_indices,ppr_sorted_passage_scores = self.run_ppr(node_weights)
        return ppr_sorted_passage_indices,ppr_sorted_passage_scores

    def run_ppr(self, node_weights):        
        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=self.config.damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )
        
        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_indices])
        sorted_indices_in_doc_scores = np.argsort(doc_scores)[::-1]
        sorted_passage_scores = doc_scores[sorted_indices_in_doc_scores]
        
        sorted_passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[i]] 
            for i in sorted_indices_in_doc_scores
        ]
        
        return sorted_passage_hash_ids, sorted_passage_scores.tolist()

    def calculate_entity_scores(self,question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
        actived_entities = {}
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        for seed_entity_idx,seed_entity,seed_entity_hash_id,seed_entity_score in zip(seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 1)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score    
        used_sentence_hash_ids = set()
        current_entities = actived_entities.copy()
        iteration = 1
        while len(current_entities) > 0 and iteration < self.config.max_iterations:
            new_entities = {}
            for entity_hash_id, (entity_id, entity_score, tier) in current_entities.items():
                if entity_score < self.config.iteration_threshold:
                    continue
                sentence_hash_ids = [sid for sid in list(self.entity_hash_id_to_sentence_hash_ids[entity_hash_id]) if sid not in used_sentence_hash_ids]
                if not sentence_hash_ids:
                    continue
                sentence_indices = [self.sentence_embedding_store.hash_id_to_idx[sid] for sid in sentence_hash_ids]
                sentence_embeddings = self.sentence_embeddings[sentence_indices]
                question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
                sentence_similarities = np.dot(sentence_embeddings, question_emb).flatten()
                top_sentence_indices = np.argsort(sentence_similarities)[::-1][:self.config.top_k_sentence]
                for top_sentence_index in top_sentence_indices:
                    top_sentence_hash_id = sentence_hash_ids[top_sentence_index]
                    top_sentence_score = sentence_similarities[top_sentence_index]
                    used_sentence_hash_ids.add(top_sentence_hash_id)
                    entity_hash_ids_in_sentence = self.sentence_hash_id_to_entity_hash_ids[top_sentence_hash_id]
                    for next_entity_hash_id in entity_hash_ids_in_sentence:
                        next_entity_score = entity_score * top_sentence_score
                        if next_entity_score < self.config.iteration_threshold:
                            continue
                        next_enitity_node_idx = self.node_name_to_vertex_idx[next_entity_hash_id]
                        entity_weights[next_enitity_node_idx] += next_entity_score
                        new_entities[next_entity_hash_id] = (next_enitity_node_idx, next_entity_score, iteration+1)
            actived_entities.update(new_entities)
            current_entities = new_entities.copy()
            iteration += 1
        return entity_weights, actived_entities

    def calculate_entity_scores_vectorized(self,question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
        """
        GPU-accelerated vectorized version using PyTorch sparse tensors.
        Uses sparse representation for both matrices and entity score vectors for maximum efficiency.
        Now includes proper dynamic pruning to match BFS behavior:
        - Sentence deduplication (tracks used sentences)
        - Per-entity top-k sentence selection
        - Proper threshold-based pruning
        """
        # Initialize entity weights
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)
        
        # Compute all sentence similarities with the question at once
        question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
        sentence_similarities_np = np.dot(self.sentence_embeddings, question_emb).flatten()
        
        # Convert to torch tensors and move to device
        sentence_similarities = torch.from_numpy(sentence_similarities_np).float().to(self.device)
        
        # Track used sentences for deduplication (like BFS version)
        used_sentence_mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.device)
        
        # Initialize seed entity scores as sparse tensor
        seed_indices = torch.tensor([[idx] for idx in seed_entity_indices], dtype=torch.long).t()
        seed_values = torch.tensor(seed_entity_scores, dtype=torch.float32)
        entity_scores_sparse = torch.sparse_coo_tensor(
            seed_indices, seed_values, (num_entities,), device=self.device
        ).coalesce()
        
        # Also maintain a dense accumulator for total scores
        entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.device)
        entity_scores_dense.scatter_(0, torch.tensor(seed_entity_indices, device=self.device), 
                                     torch.tensor(seed_entity_scores, dtype=torch.float32, device=self.device))
        
        # Initialize actived_entities
        actived_entities = {}
        for seed_entity_idx, seed_entity, seed_entity_hash_id, seed_entity_score in zip(
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
        ):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 0)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score
        
        current_entity_scores_sparse = entity_scores_sparse
        
        # Iterative matrix-based propagation using sparse matrices on GPU
        for iteration in range(1, self.config.max_iterations):
            # Convert sparse tensor to dense for threshold operation
            current_entity_scores_dense = current_entity_scores_sparse.to_dense()
            
            # Apply threshold to current scores
            current_entity_scores_dense = torch.where(
                current_entity_scores_dense >= self.config.iteration_threshold, 
                current_entity_scores_dense, 
                torch.zeros_like(current_entity_scores_dense)
            )
            
            # Get non-zero indices for sparse representation
            nonzero_mask = current_entity_scores_dense > 0
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(-1)
            
            if len(nonzero_indices) == 0:
                break
            
            # Extract non-zero values and create sparse tensor
            nonzero_values = current_entity_scores_dense[nonzero_indices]
            # Step 1: Sparse entity scores @ Sparse E2S matrix
            # Convert sparse vector to 2D for matrix multiplication
            current_scores_2d = torch.sparse_coo_tensor(
                torch.stack([nonzero_indices, torch.zeros_like(nonzero_indices)]),
                nonzero_values,
                (num_entities, 1),
                device=self.device
            ).coalesce()
            
            # E @ E2S -> sentence activation scores (sparse @ sparse = dense)
            sentence_activation = torch.sparse.mm(
                self.entity_to_sentence_sparse.t(),
                current_scores_2d
            )
            # Convert to dense before squeeze to avoid CUDA sparse tensor issues
            if sentence_activation.is_sparse:
                sentence_activation = sentence_activation.to_dense()
            sentence_activation = sentence_activation.squeeze()
            
            # Apply sentence deduplication: mask out used sentences
            sentence_activation = torch.where(
                used_sentence_mask,
                torch.zeros_like(sentence_activation),
                sentence_activation
            )
            
            # Step 2: Per-entity top-k sentence selection
            # This matches BFS behavior: each entity independently selects its top-k sentences
            selected_sentence_indices_list = []
            
            if self.config.top_k_sentence > 0:
                # Iterate through each active entity
                for entity_idx in nonzero_indices:
                    # Get sentences connected to this entity from the sparse matrix
                    # entity_to_sentence_sparse shape: (num_entities, num_sentences)
                    entity_row = self.entity_to_sentence_sparse[entity_idx].coalesce()
                    entity_sentence_indices = entity_row.indices()[0]  # Get column indices
                    
                    if len(entity_sentence_indices) == 0:
                        continue
                    
                    # Filter out already used sentences
                    sentence_mask = ~used_sentence_mask[entity_sentence_indices]
                    available_sentence_indices = entity_sentence_indices[sentence_mask]
                    
                    if len(available_sentence_indices) == 0:
                        continue
                    
                    # Get sentence similarities (for ranking)
                    sentence_sims = sentence_similarities[available_sentence_indices]
                    
                    # Select top-k sentences based ONLY on sentence similarity (matches BFS line 240)
                    # NOT weighted by entity_score at selection time
                    k = min(self.config.top_k_sentence, len(sentence_sims))
                    if k > 0:
                        top_k_values, top_k_local_indices = torch.topk(sentence_sims, k)
                        top_k_sentence_indices = available_sentence_indices[top_k_local_indices]
                        selected_sentence_indices_list.append(top_k_sentence_indices)
                
                # Merge all selected sentences (with deduplication via unique)
                if len(selected_sentence_indices_list) > 0:
                    all_selected_sentences = torch.cat(selected_sentence_indices_list)
                    unique_selected_sentences = torch.unique(all_selected_sentences)
                    
                    # Mark selected sentences as used
                    used_sentence_mask[unique_selected_sentences] = True
                    
                    # Compute weighted sentence scores for propagation
                    # weighted_score = sentence_activation * sentence_similarity
                    weighted_sentence_scores = sentence_activation * sentence_similarities
                    
                    # Zero out non-selected sentences
                    mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.device)
                    mask[unique_selected_sentences] = True
                    weighted_sentence_scores = torch.where(
                        mask,
                        weighted_sentence_scores,
                        torch.zeros_like(weighted_sentence_scores)
                    )
                else:
                    # No sentences selected, create zero vector
                    weighted_sentence_scores = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)
            else:
                # No active entities or top_k_sentence is 0
                weighted_sentence_scores = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)
            
            # Step 3: Weighted sentences @ S2E -> propagate to next entities
            # Convert to sparse for more efficient computation
            weighted_nonzero_mask = weighted_sentence_scores > 0
            weighted_nonzero_indices = torch.nonzero(weighted_nonzero_mask, as_tuple=False).squeeze(-1)
            
            if len(weighted_nonzero_indices) > 0:
                weighted_nonzero_values = weighted_sentence_scores[weighted_nonzero_indices]
                weighted_scores_2d = torch.sparse_coo_tensor(
                    torch.stack([weighted_nonzero_indices, torch.zeros_like(weighted_nonzero_indices)]),
                    weighted_nonzero_values,
                    (num_sentences, 1),
                    device=self.device
                ).coalesce()
                
                next_entity_scores_result = torch.sparse.mm(
                    self.sentence_to_entity_sparse.t(),
                    weighted_scores_2d
                )
                # Convert to dense before squeeze to avoid CUDA sparse tensor issues
                if next_entity_scores_result.is_sparse:
                    next_entity_scores_result = next_entity_scores_result.to_dense()
                next_entity_scores_dense = next_entity_scores_result.squeeze()
            else:
                next_entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.device)
            
            # Update entity scores (accumulate in dense format)
            entity_scores_dense += next_entity_scores_dense
            
            # Update actived_entities dictionary (record last trigger like BFS)
            # This matches BFS behavior: unconditionally update for entities above threshold
            next_entity_scores_np = next_entity_scores_dense.cpu().numpy()
            active_indices = np.where(next_entity_scores_np >= self.config.iteration_threshold)[0]
            for entity_idx in active_indices:
                score = next_entity_scores_np[entity_idx]
                entity_hash_id = self.entity_hash_ids[entity_idx]
                # Unconditionally update to record the last trigger (matches BFS line 252)
                actived_entities[entity_hash_id] = (entity_idx, float(score), iteration)
            
            # Prepare sparse tensor for next iteration
            next_nonzero_mask = next_entity_scores_dense > 0
            next_nonzero_indices = torch.nonzero(next_nonzero_mask, as_tuple=False).squeeze(-1)
            if len(next_nonzero_indices) > 0:
                next_nonzero_values = next_entity_scores_dense[next_nonzero_indices]
                current_entity_scores_sparse = torch.sparse_coo_tensor(
                    next_nonzero_indices.unsqueeze(0), next_nonzero_values, 
                    (num_entities,), device=self.device
                ).coalesce()
            else:
                break
        
        # Convert back to numpy for final processing
        entity_scores_final = entity_scores_dense.cpu().numpy()
        
        # Map entity scores to graph node weights (only for non-zero scores)
        nonzero_indices = np.where(entity_scores_final > 0)[0]
        for entity_idx in nonzero_indices:
            score = entity_scores_final[entity_idx]
            entity_hash_id = self.entity_hash_ids[entity_idx]
            entity_node_idx = self.node_name_to_vertex_idx[entity_hash_id]
            entity_weights[entity_node_idx] = float(score)
        
        return entity_weights, actived_entities

    def calculate_passage_scores(self, question_embedding, actived_entities):
        passage_weights = np.zeros(len(self.graph.vs["name"]))
        dpr_passage_indices, dpr_passage_scores = self.dense_passage_retrieval(question_embedding)
        dpr_passage_scores = min_max_normalize(dpr_passage_scores)
        for i, dpr_passage_index in enumerate(dpr_passage_indices):
            total_entity_bonus = 0
            passage_hash_id = self.passage_embedding_store.hash_ids[dpr_passage_index]
            dpr_passage_score = dpr_passage_scores[i]
            passage_text_lower = self.passage_embedding_store.hash_id_to_text[passage_hash_id].lower()
            for entity_hash_id, (entity_id, entity_score, tier) in actived_entities.items():
                entity_lower = self.entity_embedding_store.hash_id_to_text[entity_hash_id].lower()
                entity_occurrences = passage_text_lower.count(entity_lower)
                if entity_occurrences > 0:
                    denom = tier if tier >= 1 else 1
                    entity_bonus = entity_score * math.log(1 + entity_occurrences) / denom
                    total_entity_bonus += entity_bonus
            passage_score = self.config.passage_ratio * dpr_passage_score + math.log(1 + total_entity_bonus)
            passage_node_idx = self.node_name_to_vertex_idx[passage_hash_id]
            passage_weights[passage_node_idx] = passage_score * self.config.passage_node_weight
        return passage_weights

    def dense_passage_retrieval(self, question_embedding):
        question_emb = question_embedding.reshape(1, -1)
        question_passage_similarities = np.dot(self.passage_embeddings, question_emb.T).flatten()
        sorted_passage_indices = np.argsort(question_passage_similarities)[::-1]
        sorted_passage_scores = question_passage_similarities[sorted_passage_indices].tolist()
        return sorted_passage_indices, sorted_passage_scores

    def get_seed_entities(self, question):
        question_entities = list(self.spacy_ner.question_ner(question))
        if len(question_entities) == 0:
            return [],[],[],[]
        question_entity_embeddings = self.config.embedding_model.encode(question_entities,normalize_embeddings=True,show_progress_bar=False,batch_size=self.config.batch_size)
        similarities = np.dot(self.entity_embeddings, question_entity_embeddings.T)
        seed_entity_indices = []
        seed_entity_texts = []
        seed_entity_hash_ids = []
        seed_entity_scores = []       
        for query_entity_idx in range(len(question_entities)):
            entity_scores = similarities[:, query_entity_idx]
            best_entity_idx = np.argmax(entity_scores)
            best_entity_score = entity_scores[best_entity_idx]
            best_entity_hash_id = self.entity_hash_ids[best_entity_idx]
            best_entity_text = self.entity_embedding_store.hash_id_to_text[best_entity_hash_id]
            seed_entity_indices.append(best_entity_idx)
            seed_entity_texts.append(best_entity_text)
            seed_entity_hash_ids.append(best_entity_hash_id)
            seed_entity_scores.append(best_entity_score)
        return seed_entity_indices, seed_entity_texts, seed_entity_hash_ids, seed_entity_scores

    def _score_local_trigram(self, question, query_profile, trigram_unit, turn_score_map=None):
        trigram_id = str((trigram_unit or {}).get("trigram_id", ""))
        signal = self.hierarchy_trigram_signal_cache.get(trigram_id, {})
        txt = str(signal.get("raw", (trigram_unit or {}).get("text", ""))).strip()
        if not txt:
            return -1e9
        txt_lower = str(signal.get("lower", txt.lower()))
        member_turn_ids = list((trigram_unit or {}).get("turn_ids", []))
        member_scores = [float((turn_score_map or {}).get(tid, 0.0)) for tid in member_turn_ids]
        score = max(member_scores) if member_scores else 0.0

        focus_hits = self._focus_overlap_terms(question, txt, query_profile=query_profile)
        theme_hits = self._theme_overlap_terms(question, txt, query_profile=query_profile)
        anchor_hits = self._event_anchor_hit(question, txt, query_profile=query_profile)

        score += 0.18 * len(focus_hits)
        score += 0.06 * len(theme_hits)
        score += 0.28 * anchor_hits
        score += 0.12 * self._subject_gate_score(question, query_profile, {"text": txt, "speaker": ""})
        score += 0.08 * self._exact_phrase_bonus(question, txt)

        qtype = self._effective_question_type(question, query_profile=query_profile)
        where_mode = self._where_question_mode(question, query_profile=query_profile) or "where"
        where_info = (signal.get("where", {}) or {}).get(where_mode, {})
        if where_info:
            where_pattern = float(where_info.get("pattern", 0.0))
            where_terms = float(where_info.get("terms", 0.0))
            where_noise = float(where_info.get("noise", 0.0))
        else:
            where_pattern = self._where_pattern_score(txt, where_mode)
            where_terms = self._where_location_term_hits(txt, where_mode)
            where_noise = self._where_noise_penalty(txt, where_mode)
        strict_anchor = self._question_needs_strict_anchor(question)
        if qtype in {"when", "duration"}:
            if signal:
                has_time = bool(signal.get("has_time_cue", False)) or bool(signal.get("has_year", False)) or bool(str((trigram_unit or {}).get("session_time", "")).strip())
            else:
                has_time = self._contains_time_cue(txt) or bool(re.search(r"\b\d{4}\b", txt)) or bool(str((trigram_unit or {}).get("session_time", "")).strip())
            if qtype == "duration":
                has_time = has_time or (bool(signal.get("has_duration_cue", False)) if signal else self._contains_duration_cue(txt))
            if has_time:
                score += 0.30
            if anchor_hits == 0:
                score -= 0.55
        elif qtype == "where_origin":
            location_score = float(signal["location_score"]) if signal else self._location_cue_score(txt)
            score += float(self.config.hierarchy_where_pattern_bonus_weight) * where_pattern
            score += float(self.config.hierarchy_where_location_term_weight) * min(where_terms, 3.0)
            if location_score > 0:
                score += float(self.config.hierarchy_location_bonus_weight) * min(location_score, 2.0)
            elif where_pattern == 0.0:
                score -= 0.45
        elif qtype == "where_list":
            location_score = float(signal["location_score"]) if signal else self._location_cue_score(txt)
            score += float(self.config.hierarchy_where_pattern_bonus_weight) * where_pattern
            score += float(self.config.hierarchy_where_location_term_weight) * min(where_terms, 3.0)
            if location_score > 0:
                score += float(self.config.hierarchy_location_bonus_weight) * min(location_score, 2.0)
            enumeration_score = float(signal["enumeration_score"]) if signal else self._enumeration_shape_score(txt)
            score += 0.08 * enumeration_score
            if where_pattern == 0.0 and location_score == 0.0:
                score -= 0.30
        elif qtype == "where":
            location_score = float(signal["location_score"]) if signal else self._location_cue_score(txt)
            if location_score > 0:
                score += float(self.config.hierarchy_location_bonus_weight) * min(location_score, 2.0)
            score += 0.18 * where_pattern
            if location_score <= 0 and where_pattern <= 0:
                score -= 0.35
        elif qtype == "what_list":
            enumeration_score = float(signal["enumeration_score"]) if signal else self._enumeration_shape_score(txt)
            score += float(self.config.hierarchy_list_bonus_weight) * enumeration_score
        elif qtype == "what_static":
            subject_hits = [t for t in (query_profile or {}).get("subject_terms", []) if str(t).lower() in txt_lower]
            score += float(self.config.hierarchy_static_subject_bonus_weight) * min(len(subject_hits), 2)
        if strict_anchor:
            if anchor_hits > 0:
                score += float(self.config.hierarchy_strict_anchor_bonus) * min(anchor_hits, 2)
            else:
                score -= float(self.config.hierarchy_strict_anchor_miss_penalty)

        low_value_count = int(signal.get("low_value_count", 0))
        supportive_count = int(signal.get("supportive_count", 0))
        question_like_count = int(signal.get("question_like_count", 0))
        if not signal:
            for tid in member_turn_ids:
                turn = self.hierarchy_turn_units.get(tid, {})
                turn_txt = str(turn.get("text", ""))
                if self._is_low_value_turn(turn_txt):
                    low_value_count += 1
                if self._is_supportive_reply_turn(turn_txt):
                    supportive_count += 1
                if self._is_question_like_turn(turn_txt):
                    question_like_count += 1

        score -= float(self.config.hierarchy_trigram_low_value_penalty) * low_value_count
        score -= float(self.config.hierarchy_supportive_penalty) * supportive_count
        score -= 0.20 * question_like_count
        if supportive_count >= 2:
            score -= 0.35
        if low_value_count >= 2:
            score -= 0.20
        score -= where_noise
        score -= self._generic_cluster_penalty(question, txt, query_profile=query_profile, text_signal=signal)
        return score

    def _select_local_trigram_ids(self, question, query_profile, selected_turn_ids, ranked_turn_items, turn_score_map, sample_id=None):
        if not bool(self.config.enable_dual_view_evidence_closure):
            return []
        if not bool(self.config.hierarchy_use_local_trigrams):
            return []
        tri_top_k = max(0, int(self.config.hierarchy_local_trigram_top_k))
        if tri_top_k <= 0:
            return []
        seeds = list(selected_turn_ids)
        if len(seeds) < tri_top_k:
            for tid, _ in ranked_turn_items[:6]:
                if tid not in seeds:
                    seeds.append(tid)
        candidate_ids = []
        seen = set()
        for tid in seeds:
            for gid in self.turn_to_trigram_ids.get(tid, []):
                tri = self.hierarchy_trigram_units.get(gid, {})
                if self.isolation_guard.boundary_enabled(sample_id) and not (
                    IsolationGuard.boundary_aliases(tri.get("sample_id")) & IsolationGuard.boundary_aliases(sample_id)
                ):
                    continue
                if gid not in seen:
                    candidate_ids.append(gid)
                    seen.add(gid)
        if not candidate_ids:
            return []
        strict_anchor = self._question_needs_strict_anchor(question)
        scored = []
        for gid in candidate_ids:
            tri = self.hierarchy_trigram_units.get(gid, {})
            score = self._score_local_trigram(question, query_profile, tri, turn_score_map=turn_score_map)
            if strict_anchor and self._event_anchor_hit(question, tri.get("text", ""), query_profile=query_profile) == 0:
                score -= 0.35
            scored.append((gid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_ids = [gid for gid, _ in scored]
        ranked_ids = self._diversify_trigram_ids(question, ranked_ids, limit=tri_top_k)
        return ranked_ids[:tri_top_k]

    def _rank_parallel_trigram_items(self, question, question_embedding, query_profile, sample_id=None, top_k=None):
        if not bool(self.config.hierarchy_parallel_trigram_retrieval):
            return []
        trigram_ids = list(self.hierarchy_trigram_id_list or self.hierarchy_trigram_units.keys())
        if self.isolation_guard.boundary_enabled(sample_id):
            trigram_ids = [
                gid for gid in trigram_ids
                if IsolationGuard.boundary_aliases(self.hierarchy_trigram_units.get(gid, {}).get("sample_id")) & IsolationGuard.boundary_aliases(sample_id)
            ]
        if not trigram_ids:
            return []
        if self.hierarchy_trigram_embeddings is None or len(self.hierarchy_trigram_embeddings) != len(self.hierarchy_trigram_units):
            self._refresh_trigram_embeddings()
        if self.hierarchy_trigram_embeddings is None or len(self.hierarchy_trigram_embeddings) == 0:
            return []
        idx_lookup = self.hierarchy_trigram_id_to_embedding_idx
        use_ids = [gid for gid in trigram_ids if gid in idx_lookup]
        if not use_ids:
            return []

        strict_anchor = self._question_needs_strict_anchor(question)
        effective_qtype = self._effective_question_type(question, query_profile=query_profile)
        where_mode = self._where_question_mode(question, query_profile=query_profile) or "where"
        keep_ids = []
        local_scores = []
        for gid in use_ids:
            tri = self.hierarchy_trigram_units.get(gid, {})
            signal = self.hierarchy_trigram_signal_cache.get(gid, {})
            txt = str(signal.get("raw", tri.get("text", ""))).strip()
            if not txt:
                continue
            anchor_hits = self._event_anchor_hit(question, txt, query_profile=query_profile)
            supportive_count = int(signal.get("supportive_count", 0))
            if not signal:
                supportive_count = sum(
                    1 for tid in tri.get("turn_ids", [])
                    if self._is_supportive_reply_turn(self.hierarchy_turn_units.get(tid, {}).get("text", ""))
                )
            if strict_anchor and anchor_hits == 0 and supportive_count >= 1:
                continue
            if effective_qtype in {"where", "where_origin", "where_list"}:
                where_info = (signal.get("where", {}) or {}).get(where_mode, {})
                where_pattern = float(where_info.get("pattern", 0.0)) if where_info else self._where_pattern_score(txt, where_mode)
                location_score = float(signal["location_score"]) if signal else self._location_cue_score(txt)
                if where_pattern <= 0 and location_score <= 0:
                    continue
            keep_ids.append(gid)
            local_scores.append(self._score_local_trigram(question, query_profile, tri, turn_score_map={}))

        if not keep_ids:
            return []
        emb_idx = [idx_lookup[gid] for gid in keep_ids]
        trigram_embs = self.hierarchy_trigram_embeddings[emb_idx]
        semantic_scores = np.dot(trigram_embs, question_embedding.reshape(-1))
        semantic_n = min_max_normalize(np.array(semantic_scores, dtype=float)) if len(semantic_scores) > 0 else np.array([], dtype=float)
        local_scores = np.array(local_scores, dtype=float)
        local_n = min_max_normalize(local_scores) if len(local_scores) > 0 else local_scores
        fusion_w = float(self.config.hierarchy_trigram_fusion_weight)
        final_scores = (1.0 - fusion_w) * semantic_n + fusion_w * local_n
        ranked = sorted(zip(keep_ids, final_scores.tolist()), key=lambda x: x[1], reverse=True)
        ranked_ids = [gid for gid, _ in ranked]
        keep = int(top_k if top_k is not None else self.config.hierarchy_trigram_query_top_k)
        ranked_ids = self._diversify_trigram_ids(question, ranked_ids, limit=keep)
        score_map = {gid: sc for gid, sc in ranked}
        return [(gid, score_map.get(gid, 0.0)) for gid in ranked_ids[:max(1, keep)]]

    def _unit_cluster_key(self, question, text, session_ids=None, source_id=""):
        txt = str(text or "").lower()
        sess = tuple(session_ids or [])
        anchors = [a for a in self._event_anchor_terms(question) if a and a in txt]
        if anchors:
            return (sess, anchors[0])
        sid = str(source_id or "")
        if sid:
            return (sess, sid.split("|")[0])
        compact = re.sub(r"\s+", " ", txt)[:40]
        return (sess, compact)

    def _diversify_turn_ids(self, question, turn_ids, limit=None):
        if not turn_ids:
            return []
        return self._diversify_unit_ids(
            question, turn_ids, self.hierarchy_turn_units, self._get_turn_source_id, limit,
        )

    def _diversify_trigram_ids(self, question, trigram_ids, limit=None):
        if not trigram_ids:
            return []
        return self._diversify_unit_ids(
            question, trigram_ids, self.hierarchy_trigram_units,
            lambda unit: unit.get("source_ref", ""), limit,
        )

    def _diversify_unit_ids(self, question, unit_ids, units, source_id, limit):
        keep = int(limit or len(unit_ids))
        soft_cap = max(1, int(self.config.hierarchy_cluster_soft_cap))
        diversified = []
        cluster_counts = Counter()
        for unit_id in unit_ids:
            unit = units.get(unit_id, {})
            key = self._unit_cluster_key(question, unit.get("text", ""), unit.get("session_ids", []), source_id(unit))
            if cluster_counts[key] >= soft_cap:
                continue
            diversified.append(unit_id)
            cluster_counts[key] += 1
            if len(diversified) >= keep:
                break
        if len(diversified) < keep:
            for unit_id in unit_ids:
                if unit_id not in diversified:
                    diversified.append(unit_id)
                if len(diversified) >= keep:
                    break
        return diversified

    def _same_sample_turn_ids(self, sample_id):
        sample_id = str(sample_id or "").strip()
        if not sample_id:
            return []
        if sample_id in self.same_sample_turn_cache:
            return list(self.same_sample_turn_cache[sample_id])

        def source_order(turn):
            sid = self._get_turn_source_id(turn)
            m = re.fullmatch(r"D(\d+):(\d+)", str(sid or "").strip())
            if m:
                return int(m.group(1)), int(m.group(2))
            return 10**9, int(turn.get("order_in_passage", 0) or 0)

        items = []
        for tid, turn in getattr(self, "hierarchy_turn_units", {}).items():
            if str((turn or {}).get("sample_id", "") or "").strip() == sample_id:
                items.append((source_order(turn or {}), tid))
        items.sort(key=lambda item: item[0])
        self.same_sample_turn_cache[sample_id] = [tid for _, tid in items]
        return list(self.same_sample_turn_cache[sample_id])

    def _expand_dialogue_source_ids(self, source_ref):
        """Expand compact dialogue references without consulting benchmark data."""
        expanded = []
        seen = set()
        for raw_part in re.split(r"[|,;]", str(source_ref or "")):
            part = raw_part.strip().strip("[]")
            match = re.fullmatch(r"(D\d+):(\d+)(?:-(\d+))?", part)
            if not match:
                continue
            session = match.group(1)
            start = int(match.group(2))
            end = int(match.group(3) or start)
            if end < start:
                start, end = end, start
            for turn_no in range(start, end + 1):
                source_id = f"{session}:{turn_no}"
                if source_id not in seen:
                    seen.add(source_id)
                    expanded.append(source_id)
        return expanded

    def _same_session_neighbor_turn_ids(self, sample_id, source_ids, radius=1):
        """Return local graph neighbors without crossing sample or session boundaries."""
        sample_id = str(sample_id or "").strip()
        radius = max(0, int(radius))
        seeds = set()
        for source_id in source_ids or []:
            seeds.update(self._expand_dialogue_source_ids(source_id))
        if not sample_id or not seeds or radius <= 0:
            return []

        seed_positions = []
        for source_id in seeds:
            match = re.fullmatch(r"(D\d+):(\d+)", source_id)
            if match:
                seed_positions.append((match.group(1), int(match.group(2))))

        candidates = []
        for tid in self._same_sample_turn_ids(sample_id):
            turn = self.hierarchy_turn_units.get(tid, {})
            source_id = self._get_turn_source_id(turn)
            if source_id in seeds:
                continue
            match = re.fullmatch(r"(D\d+):(\d+)", str(source_id or "").strip())
            if not match:
                continue
            session = match.group(1)
            turn_no = int(match.group(2))
            distances = [
                abs(turn_no - seed_no)
                for seed_session, seed_no in seed_positions
                if seed_session == session
            ]
            if distances and 1 <= min(distances) <= radius:
                candidates.append((min(distances), session, turn_no, tid))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[3] for item in candidates]

    def _explicit_from_location_score(self, text):
        txt = str(text or "").strip()
        if not txt:
            return 0.0
        score = 0.0
        if re.search(r"\bfrom\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", txt):
            score += 1.2
        if self._extract_evidence_entities([txt], {"GPE", "LOC"}):
            score += 0.45
        if "home country" in txt.lower():
            score -= 0.25
        return score

    def _completion_candidate_score(self, question, query_profile, turn, existing_turn_ids=None):
        existing_turn_ids = list(existing_turn_ids or [])
        txt = str((turn or {}).get("text", "")).strip()
        if not txt:
            return -1e9
        effective_qtype = self._effective_question_type(question, query_profile=query_profile)
        base = self._answer_shape_score(question, query_profile, turn)
        source_id = self._get_turn_source_id(turn)
        source_ids = {self._get_turn_source_id(self.hierarchy_turn_units.get(tid, {})) for tid in existing_turn_ids}
        cluster_keys = {
            self._unit_cluster_key(
                question,
                self.hierarchy_turn_units.get(tid, {}).get("text", ""),
                self.hierarchy_turn_units.get(tid, {}).get("session_ids", []),
                self._get_turn_source_id(self.hierarchy_turn_units.get(tid, {})),
            )
            for tid in existing_turn_ids
        }
        score = float(base)
        if source_id and source_id not in source_ids:
            score += 0.18
        cluster_key = self._unit_cluster_key(question, txt, turn.get("session_ids", []), source_id)
        if cluster_key not in cluster_keys:
            score += 0.14
        if effective_qtype == "where_origin":
            score += 0.55 * self._explicit_from_location_score(txt)
        elif effective_qtype == "where_list":
            score += 0.15 * self._enumeration_shape_score(txt)
            if turn.get("session_ids") and tuple(turn.get("session_ids", [])) not in {
                tuple(self.hierarchy_turn_units.get(tid, {}).get("session_ids", [])) for tid in existing_turn_ids
            }:
                score += 0.16
        elif effective_qtype == "what_list":
            score += 0.20 * self._enumeration_shape_score(txt)
            if "," in txt or " and " in txt.lower():
                score += 0.10
        return score

    def _harvest_where_turns(self, question, query_profile, sample_id, mode):
        if not sample_id or mode not in {"where_origin", "where_list"}:
            return []
        harvested = []
        for tid in self._same_sample_turn_ids(sample_id):
            turn = self.hierarchy_turn_units.get(tid, {})
            txt = str(turn.get("text", "")).strip()
            if not txt or self._is_session_block_text(txt) or self._is_question_like_turn(txt) or self._is_low_value_turn(txt):
                continue
            if self._has_conflicting_subject(question, query_profile, turn):
                continue
            if not self._has_answer_like_payload(txt):
                continue
            txt_lower = txt.lower()
            if mode == "where_origin":
                if any(marker in txt_lower for marker in ("camp", "camping", "camped", "hike", "forest", "mountain", "beach")):
                    continue
                explicit_score = self._explicit_from_location_score(txt)
                if explicit_score < 0.9 and (self._where_pattern_score(txt, "where_origin") < 1.0 or self._location_cue_score(txt) <= 0.0):
                    continue
                score = self._answer_shape_score(question, query_profile, turn) + 0.65 * explicit_score + 0.20 * self._where_pattern_score(txt, "where_origin")
                harvested.append((tid, score))
            else:
                if not any(marker in txt_lower for marker in ("camp", "camping", "camped")):
                    continue
                location_hits = self._where_location_term_hits(txt, "where_list")
                pattern_score = self._where_pattern_score(txt, "where_list")
                if location_hits < 1.0 and pattern_score < 1.0:
                    continue
                score = self._answer_shape_score(question, query_profile, turn) + 0.30 * pattern_score + 0.18 * min(location_hits, 3.0)
                harvested.append((tid, score))
        if mode == "where_list":
            per_session_best = {}
            for tid, score in harvested:
                sess = tuple(self.hierarchy_turn_units.get(tid, {}).get("session_ids", []))
                key = sess if sess else (tid,)
                if key not in per_session_best or score > per_session_best[key][1]:
                    per_session_best[key] = (tid, score)
            harvested = list(per_session_best.values())
        harvested.sort(key=lambda x: x[1], reverse=True)
        limit = int(self.config.hierarchy_where_harvest_limit)
        if mode == "where_origin":
            limit = 1
        return harvested[:max(1, limit)]

    def _completion_candidate_ids(self, question, query_profile, ranked_turn_items, selected_turn_ids, sample_id=None):
        effective_qtype = self._effective_question_type(question, query_profile=query_profile)
        candidate_ids = []
        seen = set(selected_turn_ids)
        ranked_limit = int(self.config.hierarchy_completion_ranked_limit)
        for tid, _ in ranked_turn_items[:ranked_limit]:
            if tid in seen:
                continue
            turn = self.hierarchy_turn_units.get(tid, {})
            txt = str(turn.get("text", "")).strip()
            if not txt or self._is_session_block_text(txt) or self._is_question_like_turn(txt) or self._is_low_value_turn(txt):
                continue
            if self._has_conflicting_subject(question, query_profile, turn):
                continue
            if not self._has_answer_like_payload(txt):
                continue
            if effective_qtype == "where_origin":
                if self._where_pattern_score(txt, "where_origin") < 0.8 and self._explicit_from_location_score(txt) <= 0.0:
                    continue
            elif effective_qtype == "where_list":
                if self._where_pattern_score(txt, "where_list") < 0.7 and self._where_location_term_hits(txt, "where_list") <= 0.0:
                    continue
            candidate_ids.append(tid)
            seen.add(tid)
        if effective_qtype in {"where_origin", "where_list"}:
            selected_sessions = {
                tuple(self.hierarchy_turn_units.get(tid, {}).get("session_ids", []))
                for tid in selected_turn_ids
                if self.hierarchy_turn_units.get(tid, {}).get("session_ids")
            }
            for tid in self._same_sample_turn_ids(sample_id):
                if tid in seen:
                    continue
                turn = self.hierarchy_turn_units.get(tid, {})
                txt = str(turn.get("text", "")).strip()
                if not txt or self._is_session_block_text(txt) or self._is_question_like_turn(txt) or self._is_low_value_turn(txt):
                    continue
                if self._has_conflicting_subject(question, query_profile, turn):
                    continue
                if not self._has_answer_like_payload(txt):
                    continue
                txt_lower = txt.lower()
                sess = tuple(turn.get("session_ids", []))
                if effective_qtype == "where_origin":
                    if any(marker in txt_lower for marker in ("camp", "camping", "camped", "hike", "beach", "forest", "mountain")):
                        continue
                    explicit_score = self._explicit_from_location_score(txt)
                    if explicit_score < 0.9:
                        continue
                else:
                    if not any(marker in txt_lower for marker in ("camp", "camping", "camped")):
                        continue
                    if self._where_location_term_hits(txt, "where_list") < 1.0 and self._where_pattern_score(txt, "where_list") < 1.0:
                        continue
                    if sess and sess in selected_sessions:
                        continue
                    if not self._passes_relaxed_answer_slot_filter(question, query_profile, txt):
                        continue
                candidate_ids.append(tid)
                seen.add(tid)
                if sess:
                    selected_sessions.add(sess)
        return candidate_ids

    def _complete_selected_turn_ids(self, question, query_profile, ranked_turn_items, selected_turn_ids, top_turn_k, sample_id=None):
        if not bool(self.config.enable_dual_view_evidence_closure):
            return list(selected_turn_ids)
        effective_qtype = self._effective_question_type(question, query_profile=query_profile)
        if effective_qtype not in {"where_origin", "where_list", "what_list"}:
            return list(selected_turn_ids)
        completed = [tid for tid in selected_turn_ids if tid in self.hierarchy_turn_units]
        if not completed:
            return completed
        extra_budget = int(self.config.hierarchy_completion_extra_turns)
        if effective_qtype == "where_origin":
            extra_budget = 1
        elif effective_qtype == "where_list":
            extra_budget = min(extra_budget, 2)
        elif effective_qtype == "what_list":
            extra_budget = min(extra_budget, 2)
        candidate_ids = self._completion_candidate_ids(
            question, query_profile, ranked_turn_items, completed, sample_id=sample_id
        )
        scored = []
        for tid in candidate_ids:
            turn = self.hierarchy_turn_units.get(tid, {})
            txt = str(turn.get("text", "")).strip()
            if not txt:
                continue
            if effective_qtype == "where_origin":
                if not self._passes_relaxed_answer_slot_filter(question, query_profile, txt, turn=turn) and self._explicit_from_location_score(txt) <= 0.0:
                    continue
            elif effective_qtype == "where_list":
                if not self._passes_relaxed_answer_slot_filter(question, query_profile, txt, turn=turn) and self._where_pattern_score(txt, "where_list") <= 0.0:
                    continue
            elif effective_qtype == "what_list":
                if not self._passes_relaxed_answer_slot_filter(question, query_profile, txt, turn=turn):
                    continue
                if self._enumeration_shape_score(txt) <= 0.0 and len(self._focus_overlap_terms(question, txt, query_profile=query_profile)) == 0:
                    continue
            score = self._completion_candidate_score(question, query_profile, turn, existing_turn_ids=completed)
            if score > -0.75:
                scored.append((tid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        if effective_qtype in {"what_list", "where_list"}:
            clustered = []
            seen_clusters = {
                self._unit_cluster_key(
                    question,
                    self.hierarchy_turn_units.get(tid, {}).get("text", ""),
                    self.hierarchy_turn_units.get(tid, {}).get("session_ids", []),
                    self._get_turn_source_id(self.hierarchy_turn_units.get(tid, {})),
                )
                for tid in completed
            }
            seen_sessions = {
                tuple(self.hierarchy_turn_units.get(tid, {}).get("session_ids", []))
                for tid in completed
                if self.hierarchy_turn_units.get(tid, {}).get("session_ids")
            } if effective_qtype == "where_list" else set()
            for tid, score in scored:
                turn = self.hierarchy_turn_units.get(tid, {})
                key = self._unit_cluster_key(question, turn.get("text", ""), turn.get("session_ids", []), self._get_turn_source_id(turn))
                if key in seen_clusters:
                    continue
                if effective_qtype == "where_list":
                    sess = tuple(turn.get("session_ids", []))
                    if sess and sess in seen_sessions:
                        continue
                    if sess:
                        seen_sessions.add(sess)
                clustered.append((tid, score))
                seen_clusters.add(key)
            scored = clustered
        extras = []
        for tid, _ in scored:
            if tid in completed or tid in extras:
                continue
            extras.append(tid)
            if len(extras) >= extra_budget:
                break
        if not extras:
            return completed
        if effective_qtype == "where_origin":
            head = completed[:1]
            tail = [tid for tid in completed[1:] if tid not in extras]
            ordered = head + extras + tail
        elif effective_qtype in {"what_list", "where_list"}:
            head = completed[:2]
            tail = [tid for tid in completed[2:] if tid not in extras]
            ordered = head + extras + tail
        else:
            head = completed[:2]
            tail = [tid for tid in completed[2:] if tid not in extras]
            ordered = head + extras + tail
        limit = max(top_turn_k, len(ordered))
        return ordered[:limit]

    def _reader_priority_score(self, question, query_profile, turn):
        txt = str((turn or {}).get("text", "")).strip()
        if not txt:
            return -1e9
        effective_qtype = self._effective_question_type(question, query_profile=query_profile)
        score = self._answer_shape_score(question, query_profile, turn)
        if effective_qtype == "what_list":
            enumeration_score = self._enumeration_shape_score(txt)
            score += 0.30 * enumeration_score
            if "," in txt or " and " in txt.lower():
                score += 0.12
            score += 0.08 * min(len(self._focus_overlap_terms(question, txt, query_profile=query_profile)), 2)
        elif effective_qtype == "where_list":
            score += 0.45 * self._where_pattern_score(txt, "where_list")
            score += 0.20 * min(self._where_location_term_hits(txt, "where_list"), 3.0)
            if "camp" in txt.lower():
                score += 0.18
        elif effective_qtype == "where_origin":
            score += 0.55 * self._explicit_from_location_score(txt)
        elif effective_qtype == "what_static":
            slot_type = self._static_slot_type(question, query_profile=query_profile)
            score += 0.28 * min(self._static_slot_cue_score(txt, slot_type), 3.0)
            score += 0.20 * self._static_slot_alignment_score(question, query_profile, turn, slot_type)
            if self._is_short_fact_turn(txt):
                score += 0.14
            if slot_type in {"fields", "career"} and self._is_generic_static_support_turn(txt):
                score -= 0.60
        return score

    def _prioritize_reader_turn_ids(self, question, query_profile, ranked_turn_ids):
        effective_qtype = self._effective_question_type(question, query_profile=query_profile)
        if effective_qtype not in {"where_origin", "where_list", "what_list", "what_static"}:
            return list(ranked_turn_ids)
        scored = []
        for tid in ranked_turn_ids:
            turn = self.hierarchy_turn_units.get(tid, {})
            scored.append((tid, self._reader_priority_score(question, query_profile, turn)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [tid for tid, _ in scored]

    def _static_slot_reserve_turn_ids(self, question, query_profile, ranked_turn_ids, candidate_turn_ids, existing_source_ids):
        effective_qtype = self._effective_question_type(question, query_profile=query_profile)
        if effective_qtype != "what_static":
            return []
        slot_type = self._static_slot_type(question, query_profile=query_profile)
        if slot_type not in {"fields", "career"}:
            return []
        reserves = []
        seen = set(ranked_turn_ids)
        for tid in candidate_turn_ids:
            if tid in seen:
                continue
            turn = self.hierarchy_turn_units.get(tid)
            if not turn:
                continue
            txt = str(turn.get("text", "")).strip()
            if not txt or self._is_session_block_text(txt) or self._is_question_like_turn(txt):
                continue
            if self._is_generic_static_support_turn(txt):
                continue
            source_id = str(turn.get("source_id", "")).strip()
            if source_id and source_id in existing_source_ids:
                continue
            cue_score = self._static_slot_cue_score(txt, slot_type)
            if cue_score <= 0:
                continue
            score = self._reader_priority_score(question, query_profile, turn)
            if len(txt) <= 160:
                score += 0.25
            if len(txt) > 240:
                score -= 0.25
            reserves.append((tid, score))
        reserves.sort(key=lambda x: x[1], reverse=True)
        return [tid for tid, _ in reserves[:1]]

    def _generic_cluster_penalty(self, question, text, query_profile=None, text_signal=None):
        txt = str(text or "").strip()
        signal = text_signal or {}
        low = str(signal.get("lower", txt.lower()))
        if not low:
            return 0.0
        focus_hits = self._focus_overlap_terms(question, txt, query_profile=query_profile)
        anchor_hits = self._event_anchor_hit(question, txt, query_profile=query_profile)
        # strong factual / anchored snippets should not be punished
        if anchor_hits > 0 and len(focus_hits) >= 1:
            return 0.0
        has_time_cue = bool(signal.get("has_time_cue", False)) if signal else self._contains_time_cue(txt)
        has_year = bool(signal.get("has_year", False)) if signal else bool(re.search(r"\b\d{4}\b", txt))
        if has_time_cue or has_year:
            return 0.0
        supportive = float(signal.get("generic_supportive", 0.0)) if signal else (1.0 if self._is_supportive_reply_turn(txt) else 0.0)
        question_like = float(signal.get("generic_question_like", 0.0)) if signal else (1.0 if self._is_question_like_turn(txt) else 0.0)
        first_person = float(signal.get("first_person", 0.0)) if signal else (1.0 if self._is_first_person_experience_turn(txt) else 0.0)
        if signal:
            member_count = max(1, int(signal.get("member_turn_count", 1)))
            low_value_ratio = float(signal.get("low_value_count", 0)) / member_count
        else:
            low_value_ratio = 1.0 if self._is_low_value_turn(txt) else 0.0
        penalty = 0.18 * low_value_ratio + 0.22 * supportive + 0.12 * question_like
        if len(focus_hits) == 0:
            penalty += 0.12
        if first_person == 0.0:
            penalty += 0.08
        return float(self.config.hierarchy_generic_cluster_penalty) * penalty

    def _format_hierarchy_trigram_for_reader(self, trigram_unit):
        if not isinstance(trigram_unit, dict):
            return "", {"source_id": "", "session_time": "", "text": ""}
        source_id = str(trigram_unit.get("source_ref", "")).strip()
        session_time = str(trigram_unit.get("session_time", "")).strip()
        text = str(trigram_unit.get("text", "")).strip()
        parts = []
        if session_time:
            parts.append(f"[session_date_time={session_time}]")
        if source_id:
            parts.append(f"[{source_id}]")
        if text:
            parts.append(text)
        formatted = " ".join(parts).strip()
        sample_id = str(trigram_unit.get("sample_id", "") or "").strip()
        meta = {"unit_type": "trigram", "source_id": source_id, "session_time": session_time, "text": text, "sample_id": sample_id}
        return formatted, meta

    def _infer_query_sample_id(self, question_info, question_embedding):
        if isinstance(question_info, dict):
            sid = self.isolation_guard.boundary_from_question_info(question_info)
            if sid:
                return sid
        idxs, _ = self.dense_passage_retrieval(question_embedding)
        counts = Counter()
        for idx in idxs[:10]:
            pid = self.passage_hash_ids[idx]
            sid = self.passage_id_to_sample_id.get(pid)
            if sid:
                counts[sid] += 1
        return counts.most_common(1)[0][0] if counts else None

    def _multi_evidence_graph_sample_boundary_enabled(self, query_profile):
        if not bool(self.config.enable_multi_evidence_graph_sample_boundary):
            return False
        if self._task_type(query_profile) != "multi_evidence":
            return False
        return bool(str((query_profile or {}).get("sample_id", "") or "").strip())

    def _filter_graph_candidates_to_multi_evidence_sample(self, ids, scores, query_profile):
        ids = list(ids or [])
        scores = list(scores or [])
        if not self._multi_evidence_graph_sample_boundary_enabled(query_profile):
            return ids, scores, {}
        sample_id = str((query_profile or {}).get("sample_id", "") or "").strip()
        pairs = list(zip(ids, scores))
        kept = [(pid, score) for pid, score in pairs if self.passage_id_to_sample_id.get(pid) == sample_id]
        trace = {
            "enabled": True,
            "sample_id": sample_id,
            "before": len(pairs),
            "after": len(kept),
            "removed": max(0, len(pairs) - len(kept)),
            "fallback_to_unfiltered": False,
            "removed_sample_ids": sorted({
                str(self.passage_id_to_sample_id.get(pid) or "")
                for pid, _ in pairs
                if self.passage_id_to_sample_id.get(pid) not in {None, "", sample_id}
            })[:12],
        }
        if not kept:
            trace["fallback_to_unfiltered"] = True
            return ids, scores, trace
        return [pid for pid, _ in kept], [score for _, score in kept], trace

    def _event_overlap_count(self, query_keywords, text):
        return len(set(query_keywords) & set(self._extract_keywords(text, top_k=None)))

    def _subject_match(self, subject_terms, text):
        return 1 if any(t.lower() in str(text).lower() for t in subject_terms) else 0

    def _session_aware_rerank(self, question, query_profile, passage_ids, passage_scores, sample_id=None):
        reranked = []
        score_lookup = {pid: float(sc) for pid, sc in zip(passage_ids, passage_scores)}
        for pid in passage_ids:
            text = self.passage_embedding_store.hash_id_to_text.get(pid, "")
            s = score_lookup.get(pid, 0.0)
            if sample_id and self.passage_id_to_sample_id.get(pid) == sample_id:
                s += float(self.config.sample_match_bonus_weight)
            s += self._subject_match(query_profile.get("subject_terms", []), text) * float(self.config.subject_match_bonus_weight)
            s += self._event_overlap_count(query_profile.get("keywords", []), text) * float(self.config.event_match_bonus_weight)
            if query_profile.get("is_time_question") and self._contains_time_cue(text):
                s += float(self.config.time_cue_bonus_weight)
            reranked.append((pid, s))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in reranked], [score for _, score in reranked]

    def build_retrieval_schedule(self, query_profile):
        route = self._choose_fusion_route(query_profile)
        return ["graph", "hierarchy", f"fusion:{route}"]

    def _is_fusion_exact_support_question(self, query_profile):
        question = str((query_profile or {}).get("question", "")).strip().lower()
        if not question:
            return False
        if bool((query_profile or {}).get("is_time_question")) or bool((query_profile or {}).get("is_duration_question")):
            return False
        if question.startswith("would") or question.startswith("how long") or question.startswith("how many"):
            return False
        effective_qtype = self._effective_question_type(question, query_profile=query_profile or {})
        if effective_qtype in {"what_list", "where_list", "what_static", "static_slot"}:
            return True
        return bool((query_profile or {}).get("is_list_question"))

    def _is_direct_fact_graph_question(self, query_profile):
        profile = query_profile or {}
        if self._task_type(profile) != "factoid":
            return False
        question = str(profile.get("question", "")).strip().lower()
        if not question:
            return False
        effective_qtype = str(profile.get("effective_question_type") or self._effective_question_type(question, query_profile=profile))
        if effective_qtype in {"when", "duration", "where", "where_origin", "where_list", "what_list"}:
            return False
        if question.startswith(("how ", "why ")):
            return False
        if question.startswith(("what kind", "what type", "what kinds", "what types")):
            return False
        abstract_markers = (
            "discussed", "inspired", "outlook", "theme", "intention", "advice", "appreciate",
            "remind", "toughest part", "specific themes", "precautionary", "organization",
            "agree to try", "motivat", "mean to", "interested in getting involved",
        )
        if any(marker in question for marker in abstract_markers):
            return False
        graph_slot_markers = (
            "what book", "which book", "what project", "which project", "what hobby", "which hobby",
            "what show", "which show", "what movie", "which movie", "what city", "which city",
            "what country", "which country", "what language", "which language", "what pet",
            "which pet", "what game", "which game", "what event", "which event", "what instrument",
            "which instrument", "what activity", "which activity", "what company", "which company",
            "what flower", "which flower", "what kind of flowers", "which state", "what state",
            "which year", "what year", "who "
        )
        if question.startswith(graph_slot_markers):
            return True
        concrete_slot_terms = (
            " donate", " donated", " donation", " dish", " food", " injury", " course",
            " sign up for", " signed up for", " company", " flower", " flowers", " instrument",
            " book", " city", " state", " year", " wrote", " write", " written", " created",
            " creating", " made", " make", " painting", " photo of", " picture of"
        )
        if any(term in question for term in concrete_slot_terms):
            return True
        action_slot_patterns = (
            r"^what did .+\b(organize|acquire|make|play|ask .+ help with|do with|go to|donate|write|create|paint|show)\b",
            r"^what is .+\b(favorite|currently doing|new business venture|learning to play)\b",
            r"^what new hobby\b",
        )
        return any(re.search(pattern, question) for pattern in action_slot_patterns)

    def _choose_fusion_route(self, query_profile):
        question = str(query_profile.get("question", "")).lower()
        requires_temporal_route = query_profile.get("requires_temporal_route")
        if requires_temporal_route is None:
            requires_temporal_route = self._requires_temporal_route(question)
        plural_like = any(k in question for k in ["what events", "what activities", "what books", "where has", "which events"]) or (" and " in question and (question.startswith("what") or question.startswith("where")))
        if self._is_direct_fact_graph_question(query_profile):
            return "graph"
        if self._additional_hierarchy_support_budget(query_profile) > 0:
            return "graph"
        if bool(query_profile.get("is_static_question")):
            return "graph"
        if self._is_fusion_exact_support_question(query_profile):
            return "graph"
        if self.config.fusion_time_to_hierarchy and requires_temporal_route:
            return "hierarchy"
        if self.config.fusion_plural_to_hierarchy and plural_like:
            return "hierarchy"
        if self.config.fusion_unknown_to_hierarchy and len(query_profile.get("subject_terms", [])) == 0:
            return "hierarchy"
        return "graph"

    def _additional_hierarchy_support_budget(self, query_profile):
        if not bool(self.config.enable_dual_view_evidence_closure):
            return 0
        if self._task_type(query_profile) != "inferential":
            return 0
        return max(1, int(self.config.fusion_support_passage_budget))

    def retrieve_graph(self, question, question_embedding, question_info=None, query_profile=None):
        if query_profile is None:
            query_profile = self._build_query_profile(question, question_info)
        seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores = self.get_seed_entities(question)
        if len(seed_entities) != 0:
            ids, scores = self.graph_search_with_seed_entities(question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores)
            ids = list(ids)
            scores = list(scores)
        else:
            idxs, scores = self.dense_passage_retrieval(question_embedding)
            ids = [self.passage_hash_ids[i] for i in idxs]
            scores = list(scores)
        ids, scores = self._graph_keyword_boost_rerank(question, ids, scores, query_profile=query_profile)
        ids, scores, isolation_trace = self.isolation_guard.filter_passage_pairs(
            ids,
            scores,
            str((query_profile or {}).get("sample_id", "") or ""),
            "graph",
        )
        ids, scores, sample_trace = self._filter_graph_candidates_to_multi_evidence_sample(ids, scores, query_profile)
        ids = ids[:self.config.retrieval_top_k]
        scores = scores[:self.config.retrieval_top_k]
        result = {
            "sorted_passage_ids": ids,
            "scores": scores,
            "sorted_passage": [self.passage_embedding_store.hash_id_to_text[i] for i in ids],
            "sample_id": str((query_profile or {}).get("sample_id", "") or "") or None,
        }
        if isolation_trace.get("enabled"):
            result["isolation_trace"] = isolation_trace
        if sample_trace:
            result["multi_evidence_graph_sample_trace"] = sample_trace
        return result

    def retrieve_hierarchy(self, question, question_embedding, question_info=None, query_profile=None):
        self._ensure_hierarchy_runtime()
        if query_profile is None:
            query_profile = self._build_query_profile(question, question_info)
        effective_qtype = str(query_profile.get("effective_question_type", self._effective_question_type(question, query_profile=query_profile)))
        where_mode = str(query_profile.get("where_mode", "")) or ("where" if effective_qtype.startswith("where") else "")
        sample_id = self._infer_query_sample_id(question_info, question_embedding)
        if sample_id and not str(query_profile.get("sample_id", "") or "").strip():
            query_profile["sample_id"] = sample_id
        _, hierarchy_bucket_ids = self._get_passage_bucket_sets()
        episode_ids = list(self.hierarchy_episode_id_list or self.hierarchy_episode_units.keys())
        if self.isolation_guard.boundary_enabled(sample_id):
            sample_aliases = IsolationGuard.boundary_aliases(sample_id)
            episode_ids = [
                eid for eid in episode_ids
                if IsolationGuard.boundary_aliases(self.hierarchy_episode_units[eid].get("sample_id")) & sample_aliases
            ]

        selected_episode_ids = []
        if episode_ids:
            ep_indices = [self.hierarchy_episode_id_to_matrix_idx[eid] for eid in episode_ids if eid in self.hierarchy_episode_id_to_matrix_idx]
            if len(ep_indices) != len(episode_ids):
                episode_ids = [eid for eid in episode_ids if eid in self.hierarchy_episode_id_to_matrix_idx]
            ep_embs = self.hierarchy_episode_embeddings[ep_indices] if ep_indices else np.zeros((0, len(question_embedding.reshape(-1))), dtype=np.float32)
            semantic_scores = np.dot(ep_embs, question_embedding.reshape(-1))
            lexical_scores, phrase_scores, anchor_scores, subject_scores, template_scores, where_pattern_scores, where_location_scores = self._episode_match_signals(question, query_profile, episode_ids)
            semantic_n = min_max_normalize(np.array(semantic_scores, dtype=float))
            lexical_n = min_max_normalize(lexical_scores)
            phrase_n = min_max_normalize(phrase_scores)
            anchor_n = min_max_normalize(anchor_scores)
            subject_n = min_max_normalize(subject_scores)
            template_n = min_max_normalize(template_scores)
            if where_mode:
                where_pattern_n = min_max_normalize(where_pattern_scores)
                where_location_n = min_max_normalize(where_location_scores)
                ep_scores = (
                    0.40 * semantic_n
                    + 0.16 * lexical_n
                    + 0.08 * phrase_n
                    + 0.10 * anchor_n
                    + float(self.config.hierarchy_where_episode_pattern_weight) * where_pattern_n
                    + float(self.config.hierarchy_where_episode_location_weight) * where_location_n
                )
                episode_top_k = int(self.config.hierarchy_episode_top_k) + (2 if effective_qtype == "where_list" else 1)
            else:
                if effective_qtype in {"when", "what_event", "what_list"}:
                    ep_scores = (
                        float(self.config.hierarchy_event_episode_semantic_weight) * semantic_n
                        + float(self.config.hierarchy_event_episode_lexical_weight) * lexical_n
                        + float(self.config.hierarchy_event_episode_phrase_weight) * phrase_n
                        + float(self.config.hierarchy_event_episode_anchor_weight) * anchor_n
                        + float(self.config.hierarchy_event_episode_subject_weight) * subject_n
                    )
                    if query_profile.get("event_template_type"):
                        ep_scores = ep_scores + 0.22 * template_n + 0.05 * anchor_n
                elif effective_qtype == "what_static":
                    ep_scores = (
                        0.52 * semantic_n
                        + 0.16 * lexical_n
                        + 0.08 * phrase_n
                        + 0.06 * anchor_n
                        + 0.18 * subject_n
                    )
                else:
                    ep_scores = (
                        float(self.config.hierarchy_episode_semantic_weight) * semantic_n
                        + float(self.config.hierarchy_episode_lexical_weight) * lexical_n
                        + float(self.config.hierarchy_episode_phrase_weight) * phrase_n
                        + float(self.config.hierarchy_episode_anchor_weight) * anchor_n
                    )
                episode_top_k = int(self.config.hierarchy_episode_top_k)
                if effective_qtype in {"when", "what_event", "what_list"}:
                    episode_top_k += 1
                if query_profile.get("event_template_type") and effective_qtype in {"when", "what_event"}:
                    episode_top_k += 2
            order = np.argsort(ep_scores)[::-1][:episode_top_k]
            selected_episode_ids = [episode_ids[i] for i in order]

        candidate_window_ids = []
        for eid in selected_episode_ids:
            candidate_window_ids.extend(self.episode_id_to_window_ids.get(eid, []))
        if not candidate_window_ids:
            candidate_window_ids = list(self.hierarchy_window_id_list or self.hierarchy_window_units.keys())
            if self.isolation_guard.boundary_enabled(sample_id):
                sample_aliases = IsolationGuard.boundary_aliases(sample_id)
                candidate_window_ids = [
                    wid for wid in candidate_window_ids
                    if IsolationGuard.boundary_aliases(self.hierarchy_window_units[wid].get("sample_id")) & sample_aliases
                ]

        selected_window_ids = []
        selected_window_scores = []
        if candidate_window_ids:
            win_indices = [self.hierarchy_window_id_to_matrix_idx[wid] for wid in candidate_window_ids if wid in self.hierarchy_window_id_to_matrix_idx]
            if len(win_indices) != len(candidate_window_ids):
                candidate_window_ids = [wid for wid in candidate_window_ids if wid in self.hierarchy_window_id_to_matrix_idx]
            win_embs = self.hierarchy_window_embeddings[win_indices] if win_indices else np.zeros((0, len(question_embedding.reshape(-1))), dtype=np.float32)
            semantic_scores = np.dot(win_embs, question_embedding.reshape(-1))
            lexical_scores, phrase_scores, anchor_scores, subject_scores, template_scores, where_pattern_scores, where_location_scores = self._window_match_signals(question, candidate_window_ids, query_profile=query_profile)
            semantic_n = min_max_normalize(np.array(semantic_scores, dtype=float))
            lexical_n = min_max_normalize(lexical_scores)
            phrase_n = min_max_normalize(phrase_scores)
            anchor_n = min_max_normalize(anchor_scores)
            subject_n = min_max_normalize(subject_scores)
            template_n = min_max_normalize(template_scores)
            if where_mode:
                where_pattern_n = min_max_normalize(where_pattern_scores)
                where_location_n = min_max_normalize(where_location_scores)
                win_scores = (
                    0.46 * semantic_n
                    + 0.14 * lexical_n
                    + 0.06 * phrase_n
                    + float(self.config.hierarchy_where_window_pattern_weight) * where_pattern_n
                    + float(self.config.hierarchy_where_window_location_weight) * where_location_n
                )
                window_top_k = int(self.config.hierarchy_window_top_k) + (2 if effective_qtype == "where_list" else 1)
            else:
                if effective_qtype in {"when", "what_event", "what_list"}:
                    win_scores = (
                        float(self.config.hierarchy_event_window_semantic_weight) * semantic_n
                        + float(self.config.hierarchy_event_window_lexical_weight) * lexical_n
                        + float(self.config.hierarchy_event_window_phrase_weight) * phrase_n
                        + float(self.config.hierarchy_event_window_anchor_weight) * anchor_n
                        + float(self.config.hierarchy_event_window_subject_weight) * subject_n
                    )
                    if query_profile.get("event_template_type"):
                        win_scores = win_scores + 0.26 * template_n + 0.04 * anchor_n
                elif effective_qtype == "what_static":
                    win_scores = (
                        0.60 * semantic_n
                        + 0.12 * lexical_n
                        + 0.06 * phrase_n
                        + 0.04 * anchor_n
                        + 0.18 * subject_n
                    )
                else:
                    win_scores = (
                        float(self.config.hierarchy_window_semantic_weight) * semantic_n
                        + float(self.config.hierarchy_window_lexical_weight) * lexical_n
                        + float(self.config.hierarchy_window_phrase_weight) * phrase_n
                    )
                window_top_k = int(self.config.hierarchy_window_top_k)
                if effective_qtype in {"when", "what_event", "what_list"}:
                    window_top_k += 1
                if query_profile.get("event_template_type") and effective_qtype in {"when", "what_event"}:
                    window_top_k += 2
            order = np.argsort(win_scores)[::-1][:window_top_k]
            primary_window_ids = [candidate_window_ids[i] for i in order]
            primary_scores = [float(win_scores[i]) for i in order]

            expanded = {}
            closure_enabled = bool(self.config.enable_dual_view_evidence_closure)
            for wid, wscore in zip(primary_window_ids, primary_scores):
                expanded[wid] = max(expanded.get(wid, -1e9), float(wscore))
                if not closure_enabled:
                    continue
                eid = self.window_id_to_episode_id.get(wid)
                if not eid:
                    continue
                episode_window_ids = list(self.episode_id_to_window_ids.get(eid, []))
                if wid not in episode_window_ids:
                    continue
                pos = episode_window_ids.index(wid)
                focus_session = tuple(self.hierarchy_window_units[wid].get("session_ids", []))
                for offset in (-1, 1):
                    nptr = pos + offset
                    if 0 <= nptr < len(episode_window_ids):
                        nwid = episode_window_ids[nptr]
                        if tuple(self.hierarchy_window_units[nwid].get("session_ids", [])) == focus_session:
                            expanded[nwid] = max(
                                expanded.get(nwid, -1e9),
                                float(wscore) * float(self.config.hierarchy_neighbor_decay),
                            )

            session_vote = Counter()
            for wid, sc in expanded.items():
                sess = tuple(self.hierarchy_window_units[wid].get("session_ids", []))
                if sess:
                    session_vote[sess] += float(sc)
            keep_top_n = int(self.config.hierarchy_session_keep_top_n)
            if effective_qtype == "where_list":
                keep_top_n = max(keep_top_n, 3)
            top_sessions = {sess for sess, _ in session_vote.most_common(keep_top_n)} if session_vote else set()
            nondominant_penalty = float(self.config.hierarchy_non_dominant_session_penalty)
            filtered = []
            for wid, sc in expanded.items():
                sess = tuple(self.hierarchy_window_units[wid].get("session_ids", []))
                adjusted = float(sc)
                if top_sessions and sess and sess not in top_sessions:
                    adjusted *= nondominant_penalty
                filtered.append((wid, adjusted))
            filtered.sort(key=lambda x: x[1], reverse=True)
            topk = window_top_k
            selected_window_ids = [wid for wid, _ in filtered[:topk]]
            selected_window_scores = [float(sc) for _, sc in filtered[:topk]]

        turn_scores = {}
        for wid, wscore in zip(selected_window_ids, selected_window_scores):
            win = self.hierarchy_window_units[wid]
            for tid in win["turn_ids"]:
                t = self.hierarchy_turn_units[tid]
                bonus = 0.0
                bonus += self._subject_match(query_profile.get("subject_terms", []), t["text"]) * float(self.config.subject_match_bonus_weight)
                bonus += self._event_overlap_count(query_profile.get("keywords", []), t["text"]) * float(self.config.event_match_bonus_weight)
                if query_profile.get("is_time_question") and self._contains_time_cue(t["text"]):
                    bonus += float(self.config.time_cue_bonus_weight)
                if effective_qtype in {"where", "where_origin", "where_list"}:
                    location_score = self._location_cue_score(t["text"])
                    where_pattern = self._where_pattern_score(t["text"], where_mode or "where")
                    where_terms = self._where_location_term_hits(t["text"], where_mode or "where")
                    bonus += float(self.config.hierarchy_location_bonus_weight) * min(location_score, 2.0)
                    bonus += float(self.config.hierarchy_where_pattern_bonus_weight) * where_pattern
                    bonus += float(self.config.hierarchy_where_location_term_weight) * min(where_terms, 3.0)
                    bonus -= self._where_noise_penalty(t["text"], where_mode or "where")
                if effective_qtype in {"what_list", "where_list"}:
                    bonus += float(self.config.hierarchy_list_bonus_weight) * self._enumeration_shape_score(t["text"])
                if effective_qtype == "what_static":
                    subject_hits = [term for term in query_profile.get("subject_terms", []) if str(term).lower() in str(t.get("text", "")).lower()]
                    bonus += float(self.config.hierarchy_static_subject_bonus_weight) * min(len(subject_hits), 2)
                    bonus += 0.22 * min(self._static_slot_cue_score(t.get("text", ""), query_profile.get("static_slot_type", "")), 3.0)
                    bonus += 0.18 * self._static_slot_alignment_score(question, query_profile, t, query_profile.get("static_slot_type", ""))
                    if self._is_short_fact_turn(t.get("text", "")):
                        bonus += 0.10
                    if query_profile.get("static_slot_type", "") in {"fields", "career"} and self._is_generic_static_support_turn(t.get("text", "")):
                        bonus -= 0.60
                bonus += self._exact_phrase_bonus(question, t["text"])
                bonus += self._subject_gate_score(question, query_profile, t) * float(self.config.hierarchy_subject_gate_bonus_weight)
                bonus -= self._low_info_penalty(t["text"])
                bonus -= self._generic_cluster_penalty(question, t["text"], query_profile=query_profile)
                turn_scores[tid] = max(turn_scores.get(tid, -1e9), float(wscore) + bonus)
        if effective_qtype in {"where_origin", "where_list"}:
            harvested_turns = self._harvest_where_turns(question, query_profile, sample_id, effective_qtype)
            if harvested_turns:
                current_max = max(turn_scores.values()) if turn_scores else 0.0
                for rank, (tid, local_score) in enumerate(harvested_turns):
                    injected = max(current_max, 0.0) + 0.10 - 0.02 * rank + 0.03 * min(local_score, 2.0)
                    turn_scores[tid] = max(turn_scores.get(tid, -1e9), injected)

        ranked_turn_items = sorted(turn_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_trigram_items = self._rank_parallel_trigram_items(
            question, question_embedding, query_profile, sample_id=sample_id,
            top_k=int(self.config.hierarchy_trigram_query_top_k)
        )
        selected_turn_ids = []
        strict_candidates = []
        broad_candidates = []
        fallback_candidates = []
        top_turn_k = max(int(self.config.hierarchy_turn_top_k), int(self.config.hierarchy_candidate_turn_budget), int(self.config.reader_max_hierarchy_turns))
        if effective_qtype in {"what_list", "where_list"}:
            top_turn_k += 2
        if effective_qtype == "where_list":
            top_turn_k += 2
        target_candidate_count = max(top_turn_k, int(self.config.hierarchy_soft_filter_min_candidates))
        for tid, _ in ranked_turn_items:
            turn = self.hierarchy_turn_units.get(tid, {})
            turn_text = turn.get("text", "")
            if self._is_session_block_text(turn_text):
                continue
            if self._has_conflicting_subject(question, query_profile, turn):
                continue
            slot_ok = self._passes_answer_slot_filter(question, query_profile, turn_text, turn=turn)
            if self._is_strong_theme_turn(question, query_profile, turn_text) and slot_ok:
                strict_candidates.append(tid)
            elif self._passes_hierarchy_theme_filter(question, query_profile, turn_text) and slot_ok and not self._is_low_value_turn(turn_text):
                broad_candidates.append(tid)
            elif self._has_answer_like_payload(turn_text) and not self._is_question_like_turn(turn_text):
                fallback_candidates.append(tid)

        for tid in strict_candidates:
            if tid not in selected_turn_ids:
                selected_turn_ids.append(tid)
            if len(selected_turn_ids) >= target_candidate_count:
                break

        if len(selected_turn_ids) < target_candidate_count:
            for tid in broad_candidates:
                if tid not in selected_turn_ids:
                    selected_turn_ids.append(tid)
                if len(selected_turn_ids) >= target_candidate_count:
                    break

        if len(selected_turn_ids) < target_candidate_count:
            for tid in fallback_candidates:
                if tid not in selected_turn_ids:
                    selected_turn_ids.append(tid)
                if len(selected_turn_ids) >= target_candidate_count:
                    break

        # Turn-only soft fallback: if the strict/broad gates produce nothing,
        # relax the slot filter slightly but never fall back to whole-session passages.
        if not selected_turn_ids:
            soft_turn_ids = []
            weak_subject_focus_ids = []
            soft_limit = min(2, top_turn_k)
            weak_limit = 1
            if bool((query_profile or {}).get("is_time_question")):
                soft_turn_ids.extend(self._time_question_soft_candidates(question, query_profile, ranked_turn_items, top_turn_k))
            effective_qtype = self._effective_question_type(question, query_profile=query_profile)
            if effective_qtype in {"what_event", "other"}:
                for tid in self._event_question_soft_candidates(question, query_profile, ranked_turn_items, top_turn_k):
                    if tid not in soft_turn_ids:
                        soft_turn_ids.append(tid)
            for tid, _ in ranked_turn_items:
                if len(soft_turn_ids) >= soft_limit:
                    break
                turn = self.hierarchy_turn_units.get(tid, {})
                turn_text = turn.get("text", "")
                if self._is_session_block_text(turn_text):
                    continue
                if self._is_question_like_turn(turn_text):
                    continue
                if self._is_low_value_turn(turn_text):
                    continue
                if self._has_conflicting_subject(question, query_profile, turn):
                    continue
                if not self._has_answer_like_payload(turn_text):
                    continue
                txt_lower = str(turn_text).lower()
                focus_hits = self._focus_overlap_terms(question, turn_text, query_profile=query_profile)
                subject_hits = [t for t in (query_profile or {}).get("subject_terms", []) if str(t).lower() in txt_lower]
                if self._passes_relaxed_answer_slot_filter(question, query_profile, turn_text, turn=turn):
                    if tid not in soft_turn_ids:
                        soft_turn_ids.append(tid)
                elif subject_hits and focus_hits:
                    if tid not in weak_subject_focus_ids:
                        weak_subject_focus_ids.append(tid)
                    if len(weak_subject_focus_ids) >= weak_limit:
                        break

            selected_turn_ids.extend(soft_turn_ids[:soft_limit])
            if not selected_turn_ids:
                selected_turn_ids.extend(weak_subject_focus_ids[:weak_limit])

        selected_turn_ids = self._late_rerank_hierarchy_turn_ids(
            question, question_embedding, query_profile, ranked_turn_items, selected_turn_ids, top_turn_k
        )
        selected_turn_ids = self._diversify_turn_ids(question, selected_turn_ids, limit=top_turn_k)
        selected_turn_ids = self._complete_selected_turn_ids(
            question, query_profile, ranked_turn_items, selected_turn_ids, top_turn_k, sample_id=sample_id
        )
        local_trigram_ids = self._select_local_trigram_ids(
            question, query_profile, selected_turn_ids, ranked_turn_items, turn_scores, sample_id=sample_id
        )
        parallel_trigram_ids = [gid for gid, _ in ranked_trigram_items]
        trigram_score_map = {gid: float(sc) for gid, sc in ranked_trigram_items}
        for rank, gid in enumerate(local_trigram_ids):
            trigram_score_map[gid] = max(trigram_score_map.get(gid, -1e9), 1.0 - 0.05 * rank)
        merged_trigram_ids = []
        seen_gids = set()
        trigram_keep = max(int(self.config.hierarchy_local_trigram_top_k), int(self.config.reader_max_hierarchy_trigrams), int(self.config.hierarchy_candidate_trigram_budget))
        if effective_qtype not in {"what_list", "where_list", "when", "duration", "why", "how"}:
            trigram_keep = min(trigram_keep, max(0, int(self.config.hierarchy_singlehop_trigram_top_k)))
        if effective_qtype == "where_origin":
            trigram_keep = min(trigram_keep, 1)
        for gid in parallel_trigram_ids + local_trigram_ids:
            if gid in seen_gids:
                continue
            seen_gids.add(gid)
            merged_trigram_ids.append(gid)
        merged_trigram_ids.sort(key=lambda gid: trigram_score_map.get(gid, -1e9), reverse=True)
        merged_trigram_ids = self._diversify_trigram_ids(question, merged_trigram_ids, limit=max(1, trigram_keep))
        selected_trigram_ids = merged_trigram_ids[:max(1, trigram_keep)]

        passage_score_map = {}
        selected_turn_set = set(selected_turn_ids)
        for wid, wscore in zip(selected_window_ids, selected_window_scores):
            win = self.hierarchy_window_units[wid]
            has_selected_turn = any(tid in selected_turn_set for tid in win.get("turn_ids", []))
            for pid in win["passage_hash_ids"]:
                if hierarchy_bucket_ids and pid not in hierarchy_bucket_ids:
                    continue
                score = float(wscore) + (0.03 if has_selected_turn else 0.0)
                passage_score_map[pid] = max(passage_score_map.get(pid, 0.0), score)

        ids = list(passage_score_map.keys())
        scores = [passage_score_map[i] for i in ids]
        ids, scores, isolation_trace = self.isolation_guard.filter_passage_pairs(ids, scores, sample_id, "hierarchy")
        ids, scores = self._session_aware_rerank(question, query_profile, ids, scores, sample_id)
        ids = ids[:self.config.retrieval_top_k]
        scores = scores[:self.config.retrieval_top_k]
        valid_selected_turn_ids = []
        turn_texts = []
        turn_source_ids = []
        turn_session_times = []
        trigram_texts = []
        trigram_source_ids = []
        trigram_session_times = []
        for tid in selected_turn_ids:
            turn = self.hierarchy_turn_units[tid]
            txt = str(turn.get("text", "")).strip()
            if not txt or self._is_session_block_text(txt):
                continue
            valid_selected_turn_ids.append(tid)
            turn_texts.append(txt)
            turn_source_ids.append(self._get_turn_source_id(turn))
            turn_session_times.append(self._get_turn_session_time(turn))
        selected_turn_ids = valid_selected_turn_ids
        for gid in selected_trigram_ids:
            tri = self.hierarchy_trigram_units.get(gid, {})
            txt = str(tri.get("text", "")).strip()
            if not txt:
                continue
            trigram_texts.append(txt)
            trigram_source_ids.append(str(tri.get("source_ref", "")).strip())
            trigram_session_times.append(str(tri.get("session_time", "")).strip())
        return {
            "sorted_passage_ids": ids,
            "scores": scores,
            "sorted_passage": [self.passage_embedding_store.hash_id_to_text[i] for i in ids],
            "query_profile": query_profile,
            "selected_episode_ids": selected_episode_ids,
            "selected_window_ids": selected_window_ids,
            "candidate_turn_ids": [tid for tid, _ in ranked_turn_items[:target_candidate_count]],
            "selected_turn_ids": selected_turn_ids,
            "selected_turn_texts": turn_texts,
            "selected_turn_source_ids": turn_source_ids,
            "selected_turn_session_times": turn_session_times,
            "selected_trigram_ids": selected_trigram_ids,
            "selected_trigram_texts": trigram_texts,
            "selected_trigram_source_ids": trigram_source_ids,
            "selected_trigram_session_times": trigram_session_times,
            "parallel_trigram_ids": parallel_trigram_ids,
            "local_trigram_ids": local_trigram_ids,
            "sample_id": sample_id,
            "isolation_trace": isolation_trace if isolation_trace.get("enabled") else {},
        }

    def _fusion_exact_support_from_hierarchy(self, hierarchy_result, query_profile, route_hint):
        if route_hint != "graph":
            return [], []
        support_budget = self._additional_hierarchy_support_budget(query_profile)
        if not self._is_fusion_exact_support_question(query_profile) and support_budget <= 0:
            return [], []
        question = str((query_profile or {}).get("question", ""))
        candidates = []
        for tid in list(hierarchy_result.get("selected_turn_ids", []))[:4]:
            turn = self.hierarchy_turn_units.get(tid)
            if not turn:
                continue
            text = str(turn.get("text", "")).strip()
            if not text or self._is_session_block_text(text) or self._is_question_like_turn(text):
                continue
            formatted_text, formatted_meta = self._format_hierarchy_turn_for_reader(turn)
            if not formatted_text:
                continue
            score = self._reader_priority_score(question, query_profile, turn)
            if self._is_low_value_turn(text) or self._is_supportive_reply_turn(text):
                score -= 0.50
            if len(text) <= 180:
                score += 0.15
            candidates.append((score, formatted_text, formatted_meta))
        for gid in list(hierarchy_result.get("selected_trigram_ids", []))[:2]:
            trigram = self.hierarchy_trigram_units.get(gid)
            if not trigram:
                continue
            formatted_text, formatted_meta = self._format_hierarchy_trigram_for_reader(trigram)
            text = str(formatted_meta.get("text", "")).strip()
            if not formatted_text or not text:
                continue
            score = 0.0
            score += 0.10 * len(self._focus_overlap_terms(question, text, query_profile=query_profile))
            if "," in text or " and " in text.lower():
                score += 0.12
            if len(text) > 420:
                score -= 0.25
            candidates.append((score, formatted_text, formatted_meta))
        if not candidates:
            return [], []
        candidates.sort(key=lambda item: item[0], reverse=True)
        support_budget = max(
            1,
            int(self.config.fusion_support_passage_budget),
            support_budget,
        )
        selected = candidates[:support_budget]
        return [item[1] for item in selected], [item[2] for item in selected]

    def _explicit_document_support(self):
        if getattr(self, "_document_support_index", None) is None:
            records = {}
            for pid, passage in self.passage_embedding_store.hash_id_to_text.items():
                text = str(passage)
                marker = re.search(r"(?m)^\[SourceChunks\][^\n]*(?:\n|$)", text)
                if not marker:
                    continue
                prefix = text[:marker.start()]
                document = re.search(r"(?m)^\[Document\][ \t]*([^\n]+)", prefix)
                title = re.search(r"(?m)^\[Title\][ \t]*([^\n]*)", prefix)
                if document and title:
                    records[pid] = {
                        "title": title.group(1).strip(),
                        "body": text[marker.end():].strip(),
                        "sample_id": self.passage_id_to_sample_id.get(pid) or "",
                    }
            self._document_support_index = _DocumentTitleSupport(
                records, budget=self.config.fusion_support_passage_budget
            )
        return self._document_support_index

    def fuse_results(self, graph_result, hierarchy_result, query_profile=None):
        query_profile = query_profile or {}
        route_hint = self._choose_fusion_route(query_profile)
        graph_ids = list(graph_result.get("sorted_passage_ids", []))
        hierarchy_ids = list(hierarchy_result.get("sorted_passage_ids", []))
        sample_id = hierarchy_result.get("sample_id") or graph_result.get("sample_id") or query_profile.get("sample_id")
        document_support = self._explicit_document_support()
        candidate_ids = graph_ids + hierarchy_ids
        document_units = bool(sample_id and candidate_ids) and all(
            pid in document_support.records and self.isolation_guard._id_matches_boundary(pid, sample_id)
            for pid in candidate_ids
        )
        graph_guard = not document_units and self._is_direct_fact_graph_question(query_profile)
        if graph_guard:
            top_ids = graph_ids[:self.config.retrieval_top_k]
            top_scores = list(graph_result.get("scores", []))[:self.config.retrieval_top_k]
            sample_id = graph_result.get("sample_id") or query_profile.get("sample_id")
            top_ids, top_scores, isolation_trace = self.isolation_guard.filter_passage_pairs(top_ids, top_scores, sample_id, "fusion_graph_guard")
            top_texts = [self.passage_embedding_store.hash_id_to_text[i] for i in top_ids]
            return {
                "sorted_passage_ids": top_ids,
                "scores": top_scores,
                "sorted_passage": top_texts,
                "route": "graph",
                "query_profile": query_profile,
                "fusion_graph_guard": True,
                "support_passages": [],
                "support_exact_texts": [],
                "support_exact_meta": [],
                "selected_turn_ids": [],
                "selected_turn_texts": [],
                "selected_turn_source_ids": [],
                "selected_turn_session_times": [],
                "selected_trigram_ids": [],
                "selected_trigram_texts": [],
                "selected_trigram_source_ids": [],
                "selected_trigram_session_times": [],
                "isolation_trace": isolation_trace if isolation_trace.get("enabled") else {},
            }
        graph_scores = np.array(list(graph_result.get("scores", [])), dtype=float) if graph_ids else np.array([], dtype=float)
        hierarchy_scores = np.array(list(hierarchy_result.get("scores", [])), dtype=float) if hierarchy_ids else np.array([], dtype=float)
        graph_n = min_max_normalize(graph_scores) if len(graph_scores) > 0 else graph_scores
        hierarchy_n = min_max_normalize(hierarchy_scores) if len(hierarchy_scores) > 0 else hierarchy_scores
        primary_w = float(self.config.fusion_route_primary_weight)
        secondary_w = float(self.config.fusion_route_secondary_weight)
        if route_hint == "hierarchy":
            graph_w = secondary_w
            hierarchy_w = primary_w
        else:
            graph_w = primary_w
            hierarchy_w = secondary_w
        fused_score_map = {}
        for pid, score in zip(graph_ids, graph_n.tolist() if len(graph_n) > 0 else []):
            fused_score_map[pid] = fused_score_map.get(pid, 0.0) + graph_w * float(score)
        for pid, score in zip(hierarchy_ids, hierarchy_n.tolist() if len(hierarchy_n) > 0 else []):
            fused_score_map[pid] = fused_score_map.get(pid, 0.0) + hierarchy_w * float(score)
        combined_ids = list(dict.fromkeys(graph_ids + hierarchy_ids))
        combined_scores = [fused_score_map.get(pid, 0.0) for pid in combined_ids]
        sample_id = hierarchy_result.get("sample_id") or graph_result.get("sample_id") or query_profile.get("sample_id")
        combined_ids, combined_scores, isolation_trace = self.isolation_guard.filter_passage_pairs(
            combined_ids,
            combined_scores,
            sample_id,
            "fusion",
        )
        combined_ids, combined_scores = self._session_aware_rerank(
            query_profile.get("question", ""),
            query_profile,
            combined_ids,
            combined_scores,
            sample_id=sample_id,
        )
        top_ids = combined_ids[:self.config.retrieval_top_k]
        top_scores = combined_scores[:self.config.retrieval_top_k]
        top_texts = [self.passage_embedding_store.hash_id_to_text[i] for i in top_ids]
        support_passages = []
        support_exact_texts = []
        support_exact_meta = []
        document_reference_trace = {}
        if document_units and bool(self.config.enable_dual_view_evidence_closure):
            support_profile = {"sample_id": document_support.records[candidate_ids[0]]["sample_id"]}
            support_ids, document_reference_trace = document_support.support(top_ids, combined_ids, support_profile)
            support_ids, _, _ = self.isolation_guard.filter_passage_pairs(
                support_ids, [0.0] * len(support_ids), sample_id, "document_references"
            )
            support_passages = [self.passage_embedding_store.hash_id_to_text[pid] for pid in support_ids]
            support_exact_texts, support_exact_meta = self._fusion_exact_support_from_hierarchy(
                hierarchy_result, query_profile, route_hint,
            )
        elif bool(self.config.enable_dual_view_evidence_closure):
            support_budget = int(self.config.fusion_support_passage_budget)
            for pid in graph_ids:
                # The hierarchy reader consumes turns/trigrams plus support
                # passages, not the fused passage top-k. Preserve its strongest
                # graph evidence in the same single reader call.
                if route_hint != "hierarchy" and pid in top_ids:
                    continue
                if self.isolation_guard.boundary_enabled(sample_id) and not self.isolation_guard._id_matches_boundary(pid, sample_id):
                    continue
                text = self.passage_embedding_store.hash_id_to_text.get(pid, "")
                if text and text not in support_passages:
                    support_passages.append(text)
                if len(support_passages) >= support_budget:
                    break
            support_exact_texts, support_exact_meta = self._fusion_exact_support_from_hierarchy(
                hierarchy_result,
                query_profile,
                route_hint,
            )
        has_hierarchy_evidence = bool(hierarchy_result.get("selected_turn_ids") or hierarchy_result.get("selected_trigram_ids"))
        route = "hybrid" if has_hierarchy_evidence and route_hint == "hierarchy" else route_hint
        chosen_turns = hierarchy_result if route in {"hierarchy", "hybrid"} else graph_result
        result = {
            "sorted_passage_ids": top_ids,
            "scores": top_scores,
            "sorted_passage": top_texts,
            "route": route,
            "query_profile": query_profile,
            "support_passages": support_passages,
            "support_exact_texts": support_exact_texts,
            "support_exact_meta": support_exact_meta,
            "selected_turn_ids": list(chosen_turns.get("selected_turn_ids", [])),
            "selected_turn_texts": list(chosen_turns.get("selected_turn_texts", [])),
            "selected_turn_source_ids": list(chosen_turns.get("selected_turn_source_ids", [])),
            "selected_turn_session_times": list(chosen_turns.get("selected_turn_session_times", [])),
            "selected_trigram_ids": list(chosen_turns.get("selected_trigram_ids", [])),
            "selected_trigram_texts": list(chosen_turns.get("selected_trigram_texts", [])),
            "selected_trigram_source_ids": list(chosen_turns.get("selected_trigram_source_ids", [])),
            "selected_trigram_session_times": list(chosen_turns.get("selected_trigram_session_times", [])),
            "isolation_trace": isolation_trace if isolation_trace.get("enabled") else {},
        }

        if document_units:
            result.update(
                explicit_document_units=True,
                document_reference_trace=document_reference_trace,
                document_hierarchy_support_trace={
                    "stage": "fusion_before_reader_filter",
                    "source_ids": [str(meta.get("source_id", "")) for meta in support_exact_meta],
                },
            )
        return result

    def retrieve(self, questions, eval_mode="fusion", show_progress=True):
        self._refresh_retrieval_runtime_state()
        if self.config.use_vectorized_retrieval:
            self._ensure_sparse_matrices_precomputed(log=show_progress)
        retrieval_results = []
        mode = (eval_mode or "fusion").lower()
        if mode in {"hierarchy", "fusion", "all"}:
            self._ensure_hierarchy_runtime()
        iterator = tqdm(questions, total=len(questions), desc=f"Retrieving ({mode.upper()})") if show_progress else questions
        question_texts = [q["question"] for q in questions]
        if question_texts:
            question_embeddings = self.config.embedding_model.encode(
                question_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.config.batch_size,
            )
            question_embeddings = np.asarray(question_embeddings, dtype=np.float32)
        else:
            question_embeddings = np.zeros((0, 1), dtype=np.float32)
        for qidx, question_info in enumerate(iterator):
            question = question_info["question"]
            question_embedding = question_embeddings[qidx]
            query_profile = self._build_query_profile(question, question_info)
            graph_result = {"sorted_passage_ids": [], "scores": [], "sorted_passage": []}
            if mode in ["graph", "fusion", "all"]:
                graph_result = self.retrieve_graph(question, question_embedding, question_info, query_profile=query_profile)
            hierarchy_result = {"sorted_passage_ids": [], "scores": [], "sorted_passage": [], "selected_episode_ids": [], "selected_window_ids": [], "selected_turn_ids": [], "selected_turn_texts": [], "selected_turn_source_ids": [], "selected_turn_session_times": [], "sample_id": None}
            fusion_result = {"sorted_passage_ids": list(graph_result.get("sorted_passage_ids", [])), "scores": list(graph_result.get("scores", [])), "sorted_passage": list(graph_result.get("sorted_passage", [])), "route": "graph", "selected_turn_ids": [], "selected_turn_texts": [], "selected_turn_source_ids": [], "selected_turn_session_times": []}
            if mode in ["hierarchy", "fusion", "all"]:
                hierarchy_result = self.retrieve_hierarchy(question, question_embedding, question_info, query_profile=query_profile)
            query_profile = self._refine_query_profile_from_evidence(
                query_profile, graph_result, hierarchy_result
            )
            schedule = self.build_retrieval_schedule(query_profile)
            if mode in ["fusion", "all"]:
                fusion_result = self.fuse_results(graph_result, hierarchy_result, query_profile=query_profile)
            retrieval_results.append({
                "question": question,
                "gold_answer": question_info.get("answer", question_info.get("gold_answer", "")),
                "query_profile": query_profile,
                "schedule": schedule,
                "graph_retrieval": graph_result,
                "hierarchical_retrieval": hierarchy_result,
                "fusion": fusion_result,
                "sorted_passage": fusion_result["sorted_passage"] if mode in ["fusion", "all"] else (hierarchy_result["sorted_passage"] if mode == "hierarchy" else graph_result["sorted_passage"]),
                "sorted_passage_scores": fusion_result["scores"] if mode in ["fusion", "all"] else (hierarchy_result["scores"] if mode == "hierarchy" else graph_result["scores"]),
                "multi_evidence_graph_sample_trace": graph_result.get("multi_evidence_graph_sample_trace", {}),
            })
        return retrieval_results
