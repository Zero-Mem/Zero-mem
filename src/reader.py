"""Reader context, prompt construction and the single-call answer sequence."""

from .utils import CrossEncoder
from .indexing import IsolationGuard
from concurrent.futures import ThreadPoolExecutor
from .utils import logger, safe_error_summary
import re
from time import perf_counter
from tqdm import tqdm


class ReaderMixin:
    def _build_hierarchy_reader_context(self, branch, return_meta=False):
        ranked_turn_ids = list(branch.get("selected_turn_ids", []))
        candidate_turn_ids = list(branch.get("candidate_turn_ids", []))
        ranked_trigram_ids = list(branch.get("selected_trigram_ids", []))
        support_passages = list(branch.get("support_passages", []))
        query_profile = branch.get("query_profile", {}) if isinstance(branch, dict) else {}
        question = str(query_profile.get("question", ""))
        ranked_turn_ids = self._prioritize_reader_turn_ids(question, query_profile, ranked_turn_ids)
        max_turns = int(self.config.reader_max_hierarchy_turns)
        max_trigrams = int(self.config.reader_max_hierarchy_trigrams)
        min_total = int(self.config.hierarchy_reader_min_evidence)
        preserve_turns = max(0, int(self.config.hierarchy_exact_turn_preserve_top_k))
        effective_qtype = str(query_profile.get("effective_question_type", ""))
        if effective_qtype == "where_list":
            max_turns = max(max_turns, 10)
            preserve_turns = max(preserve_turns, 3)
            max_trigrams = min(max_trigrams, 2)
        elif effective_qtype == "where_origin":
            max_turns = min(max_turns, 6)
            max_trigrams = min(max_trigrams, 1)
        elif effective_qtype == "what_list":
            max_turns = max(max_turns, 10)
            preserve_turns = max(preserve_turns, 3)
            max_trigrams = min(max_trigrams, 1)
        elif effective_qtype == "what_static":
            slot_type = self._static_slot_type(question, query_profile=query_profile)
            if slot_type in {"relationship", "identity", "fields", "career"}:
                max_turns = min(max_turns, 4)
                preserve_turns = max(preserve_turns, 4)
            else:
                max_turns = min(max_turns, 6)
                preserve_turns = max(preserve_turns, 3)
            max_trigrams = 0
        if effective_qtype and effective_qtype not in {"what_list", "where_list", "when", "duration", "why", "how"}:
            max_trigrams = min(max_trigrams, max(0, int(self.config.hierarchy_singlehop_trigram_top_k)))
        collected = []
        meta = []
        covered_turn_ids = set()
        existing_turn_source_ids = set()
        for tid in ranked_turn_ids[:preserve_turns]:
            turn = self.hierarchy_turn_units.get(tid)
            if not turn:
                continue
            formatted_text, formatted_meta = self._format_hierarchy_turn_for_reader(turn)
            if formatted_text:
                collected.append(formatted_text)
                meta.append(formatted_meta)
                existing_turn_source_ids.add(formatted_meta.get("source_id", ""))
        reserve_turn_ids = self._static_slot_reserve_turn_ids(
            question,
            query_profile,
            ranked_turn_ids[:preserve_turns],
            candidate_turn_ids,
            existing_turn_source_ids,
        )
        for tid in reserve_turn_ids:
            turn = self.hierarchy_turn_units.get(tid)
            if not turn:
                continue
            formatted_text, formatted_meta = self._format_hierarchy_turn_for_reader(turn)
            if formatted_text and formatted_meta.get("source_id", "") not in existing_turn_source_ids:
                collected.append(formatted_text)
                meta.append(formatted_meta)
                existing_turn_source_ids.add(formatted_meta.get("source_id", ""))
        for gid in ranked_trigram_ids[:max_trigrams]:
            tri = self.hierarchy_trigram_units.get(gid)
            if tri:
                formatted_text, formatted_meta = self._format_hierarchy_trigram_for_reader(tri)
                if formatted_text:
                    collected.append(formatted_text)
                    meta.append(formatted_meta)
                    covered_turn_ids.update(tri.get("turn_ids", []))
        for tid in ranked_turn_ids[:max_turns]:
            if tid in covered_turn_ids and tid not in ranked_turn_ids[:preserve_turns]:
                continue
            turn = self.hierarchy_turn_units.get(tid)
            if turn:
                formatted_text, formatted_meta = self._format_hierarchy_turn_for_reader(turn)
                if formatted_text and formatted_meta.get("source_id", "") not in existing_turn_source_ids:
                    collected.append(formatted_text)
                    meta.append(formatted_meta)
                    existing_turn_source_ids.add(formatted_meta.get("source_id", ""))
        # If coverage is too sparse, keep a few covered answer-bearing turns too.
        if len(collected) < min_total:
            for tid in ranked_turn_ids[:max_turns]:
                turn = self.hierarchy_turn_units.get(tid)
                if not turn:
                    continue
                source_id = self._get_turn_source_id(turn)
                if source_id in existing_turn_source_ids:
                    continue
                formatted_text, formatted_meta = self._format_hierarchy_turn_for_reader(turn)
                if formatted_text:
                    collected.append(formatted_text)
                    meta.append(formatted_meta)
                    existing_turn_source_ids.add(source_id)
                if len(collected) >= min_total:
                    break
        for passage in support_passages[:int(self.config.fusion_support_passage_budget)]:
            text = str(passage).strip()
            if not text:
                continue
            collected.append(text)
            meta.append({"unit_type": "passage", "source_id": "", "session_time": "", "text": text, "sample_id": self._extract_sample_id(text) or ""})
        for unit_type, limit in (("trigram", max_trigrams), ("turn", max_turns)):
            if collected:
                break
            texts = list(branch.get(f"selected_{unit_type}_texts", []))[:limit]
            source_ids = list(branch.get(f"selected_{unit_type}_source_ids", []))[:limit]
            session_times = list(branch.get(f"selected_{unit_type}_session_times", []))[:limit]
            for idx, text in enumerate(texts):
                source_id = str(source_ids[idx]) if idx < len(source_ids) else ""
                session_time = str(session_times[idx]) if idx < len(session_times) else ""
                parts = []
                if session_time:
                    parts.append(f"[session_date_time={session_time}]")
                if source_id:
                    parts.append(f"[{source_id}]")
                parts.append(str(text))
                collected.append(" ".join(parts).strip())
                meta.append({"unit_type": unit_type, "source_id": source_id, "session_time": session_time, "text": str(text), "sample_id": self._extract_sample_id(text) or ""})
        return (collected, meta) if return_meta else collected

    def _select_passages_for_eval_mode(self, retrieval_result, eval_mode="fusion", return_meta=False):
        mode = (eval_mode or "fusion").lower()
        if mode == "graph":
            passages = list(retrieval_result.get("graph_retrieval", {}).get("sorted_passage", []))[:int(self.config.retrieval_top_k)]
            return (passages, []) if return_meta else passages
        branch = retrieval_result.get("hierarchical_retrieval" if mode == "hierarchy" else "fusion", {})
        document_units = mode == "fusion" and bool(branch.get("explicit_document_units"))
        if not document_units and (mode == "hierarchy" or branch.get("route") in {"hierarchy", "hybrid"} or branch.get("selected_turn_ids") or branch.get("selected_trigram_ids")):
            ctx, meta = self._build_hierarchy_reader_context(branch, return_meta=True)
            if ctx:
                return (ctx, meta) if return_meta else ctx
        passages = list(branch.get("sorted_passage", []))[:int(self.config.retrieval_top_k)]
        if not passages and mode == "fusion":
            passages = list(retrieval_result.get("graph_retrieval", {}).get("sorted_passage", []))[:int(self.config.retrieval_top_k)]
        meta = [{"unit_type": "passage", "source_id": "", "session_time": "", "text": str(p), "sample_id": self._extract_sample_id(p) or ""} for p in passages]
        support_budget = max(
            1,
            int(self.config.fusion_support_passage_budget),
            self._additional_hierarchy_support_budget(branch.get("query_profile", retrieval_result.get("query_profile", {}))),
        )
        support_texts = list(branch.get("support_exact_texts", []))[:support_budget]
        support_meta = list(branch.get("support_exact_meta", []))[:support_budget]
        if document_units:
            document_support = list(branch.get("support_passages", []))[:support_budget]
            support_texts = document_support + support_texts
            support_meta = [None] * len(document_support) + support_meta
        for idx, text in enumerate(support_texts):
            text = str(text).strip()
            if not text:
                continue
            passages.append(text)
            if idx < len(support_meta) and isinstance(support_meta[idx], dict):
                meta.append(support_meta[idx])
            else:
                meta.append({"unit_type": "support", "source_id": "", "session_time": "", "text": text, "sample_id": self._extract_sample_id(text) or ""})
        return (passages, meta) if return_meta else passages

    def _split_reader_text_into_evidence_units(self, text, meta=None):
        text = str(text or "")
        meta = meta if isinstance(meta, dict) else {}
        units = []
        unit_type = str(meta.get("unit_type", "") or "passage")
        source_id = str(meta.get("source_id", "") or "").strip()
        session_time = str(meta.get("session_time", "") or "").strip()
        sample_id = str(meta.get("sample_id", "") or self._extract_sample_id(text) or "").strip()
        speaker = str(meta.get("speaker", "") or "").strip()
        meta_text = str(meta.get("text", "") or "").strip()
        base_text = meta_text or text.strip()
        if unit_type in {"turn", "trigram", "support"} and base_text:
            units.append({
                "source_id": source_id,
                "session_time": session_time,
                "sample_id": sample_id,
                "speaker": speaker,
                "text": base_text,
                "unit_type": unit_type,
            })
            return units

        pattern = re.compile(r"(?:^|\n)\s*(?:\[[^\]]*session_date_time=([^\]]+)\]\s*)?\[(D\d+:\d+(?:-\d+)?)\]\s*(.*?)(?=(?:\n\s*(?:\[[^\]]*session_date_time=[^\]]+\]\s*)?\[D\d+:\d+(?:-\d+)?\])|\Z)", re.S)
        for match in pattern.finditer(text):
            sid = match.group(2).strip()
            body = re.sub(r"\s+", " ", match.group(3)).strip()
            if not body:
                continue
            units.append({
                "source_id": sid,
                "session_time": (match.group(1) or session_time or "").strip(),
                "sample_id": sample_id,
                "speaker": "",
                "text": body,
                "unit_type": "passage_turn",
            })
        if not units:
            for segment in self._extract_dialogue_segments(base_text):
                body = re.sub(r"\s+", " ", str(segment.get("text", "") or "")).strip()
                if not body:
                    continue
                units.append({
                    "source_id": str(segment.get("dia_id", "") or "").strip(),
                    "session_time": session_time,
                    "sample_id": sample_id,
                    "speaker": str(segment.get("speaker", "") or "").strip(),
                    "text": body,
                    "unit_type": "passage_turn",
                })
        if not units and base_text:
            units.append({
                "source_id": source_id,
                "session_time": session_time,
                "sample_id": sample_id,
                "speaker": speaker,
                "text": re.sub(r"\s+", " ", base_text).strip(),
                "unit_type": unit_type,
            })
        return units

    def _meta_source_ids(self, selected_meta):
        source_ids = set()
        for item in selected_meta or []:
            if not isinstance(item, dict):
                continue
            raw = str(item.get("source_id", "") or "")
            for part in raw.split("|"):
                part = part.strip()
                if part:
                    source_ids.add(part)
        return source_ids

    def _meta_expanded_source_ids(self, selected_meta):
        source_ids = set()
        for item in selected_meta or []:
            if not isinstance(item, dict):
                continue
            source_ids.update(self._expand_dialogue_source_ids(item.get("source_id", "")))
            for ref in re.findall(r"\bD\d+:\d+(?:-\d+)?\b", str(item.get("text", ""))):
                source_ids.update(self._expand_dialogue_source_ids(ref))
        return source_ids

    def _reader_sample_filter_enabled(self, query_profile):
        if not bool(self.config.enable_semantic_reader_sample_filter):
            return False
        if self._task_type(query_profile) not in {"multi_evidence", "inferential"}:
            return False
        return bool(str((query_profile or {}).get("sample_id", "") or "").strip())

    def _evidence_matches_query_sample(self, passage, meta, query_sample):
        query_sample = str(query_sample or "").strip()
        if not query_sample:
            return True
        explicit_ids = []
        if isinstance(meta, dict):
            meta_sid = str(meta.get("sample_id", "") or "").strip()
            if meta_sid:
                explicit_ids.append(meta_sid)
            meta_text = str(meta.get("text", "") or "")
            explicit_ids.extend(re.findall(r"\[sample_id=([^\]]+)\]", meta_text))
        explicit_ids.extend(re.findall(r"\[sample_id=([^\]]+)\]", str(passage or "")))
        explicit_ids = [sid.strip() for sid in explicit_ids if str(sid or "").strip()]
        if not explicit_ids:
            return True
        query_aliases = IsolationGuard.boundary_aliases(query_sample)
        evidence_aliases = set()
        for sid in explicit_ids:
            evidence_aliases.update(IsolationGuard.boundary_aliases(sid))
        return bool(query_aliases & evidence_aliases)

    def _filter_reader_context_to_query_sample(self, query_profile, selected_passages, selected_meta):
        if not self._reader_sample_filter_enabled(query_profile):
            return selected_passages, selected_meta
        query_sample = str((query_profile or {}).get("sample_id", "") or "").strip()
        passages = list(selected_passages or [])
        meta = list(selected_meta or [])
        while len(meta) < len(passages):
            text = passages[len(meta)]
            meta.append({"unit_type": "passage", "source_id": "", "session_time": "", "text": str(text), "sample_id": self._extract_sample_id(text) or ""})
        kept_passages = []
        kept_meta = []
        for idx, passage in enumerate(passages):
            item_meta = meta[idx] if idx < len(meta) and isinstance(meta[idx], dict) else {}
            if self._evidence_matches_query_sample(passage, item_meta, query_sample):
                kept_passages.append(passage)
                kept_meta.append(item_meta)
        if kept_passages:
            return kept_passages, kept_meta
        return selected_passages, selected_meta

    def _format_reader_context_for_prompt(self, query_profile, selected_passages, selected_meta):
        if self._task_type(query_profile) != "inferential":
            return "".join(f"{passage}\n" for passage in selected_passages)
        blocks = []
        primary_index = 0
        supplemental_index = 0
        context_index = 0
        for index, passage in enumerate(selected_passages):
            meta = selected_meta[index] if index < len(selected_meta) and isinstance(selected_meta[index], dict) else {}
            unit_type = str(meta.get("unit_type", "") or "passage")
            if unit_type == "focus":
                supplemental_index += 1
                label = f"Supplemental clue {supplemental_index}"
            elif unit_type in {"turn", "trigram", "support"}:
                primary_index += 1
                label = f"Primary evidence {primary_index}"
            else:
                context_index += 1
                label = f"Context session {context_index}"
            blocks.append(f"[{label}]\n{passage}\n")
        return "\n".join(blocks)

    def _reader_word_count(self, text):
        return len(re.findall(r"\S+", str(text or "")))

    def _truncate_to_reader_words(self, text, budget):
        words = re.findall(r"\S+", str(text or ""))
        if len(words) <= budget:
            return str(text or "")
        return " ".join(words[:max(0, int(budget))]).strip()

    def _apply_reader_prompt_budget(self, selected_passages, selected_meta, question, system_prompt, budget_override=None):
        budget = int(budget_override) if budget_override is not None else int(self.config.reader_prompt_token_budget)
        if budget <= 0:
            return selected_passages, selected_meta, {"enabled": False, "reason": "disabled", "unit": "whitespace_words"}
        overhead = self._reader_word_count(system_prompt) + self._reader_word_count(question) + 64
        remaining = max(256, budget - overhead)
        passages = list(selected_passages or [])
        meta = list(selected_meta or [])
        kept_passages = []
        kept_meta = []
        word_counts = [self._reader_word_count(p) for p in passages]
        kept_words = 0
        truncated = 0
        dropped = 0

        for idx, passage in enumerate(passages):
            if remaining <= 0:
                dropped += 1
                continue
            text = str(passage or "")
            word_count = word_counts[idx]
            if word_count > remaining:
                text = self._truncate_to_reader_words(text, remaining)
                word_count = self._reader_word_count(text)
                truncated += 1
            if not text.strip():
                dropped += 1
                continue
            kept_passages.append(text)
            item_meta = meta[idx] if idx < len(meta) and isinstance(meta[idx], dict) else {}
            item_meta = dict(item_meta)
            item_meta["text"] = text
            kept_meta.append(item_meta)
            remaining -= word_count
            kept_words += word_count

        return kept_passages, kept_meta, {
            "enabled": True,
            "unit": "whitespace_words",
            "scope": "evidence_text",
            "budget": budget,
            "overhead_words_heuristic": overhead,
            "before_passages": len(passages),
            "after_passages": len(kept_passages),
            "before_words": sum(word_counts),
            "after_words": kept_words,
            "truncated": truncated,
            "dropped": dropped,
        }

    def _reader_uses_explicit_documents(self, retrieval_result, eval_mode):
        mode = str(eval_mode or "fusion").lower()
        key = {"graph": "graph_retrieval", "hierarchy": "hierarchical_retrieval", "fusion": "fusion"}.get(mode)
        branch = retrieval_result.get(key) if key else None
        if not isinstance(branch, dict):
            return False
        if mode == "fusion":
            return bool(branch.get("explicit_document_units"))
        profile = retrieval_result.get("query_profile") or {}
        sample = profile.get("sample_id") or branch.get("sample_id")
        passages = list(branch.get("sorted_passage", []))[:max(0, int(self.config.retrieval_top_k))]
        if mode == "hierarchy" and not passages:
            passage_ids = []
            for field, units in (
                ("selected_turn_ids", getattr(self, "hierarchy_turn_units", {})),
                ("selected_trigram_ids", getattr(self, "hierarchy_trigram_units", {})),
            ):
                for unit_id in branch.get(field, []):
                    unit = units.get(unit_id, {})
                    sources = list(unit.get("passage_hash_ids") or [])
                    if unit.get("passage_hash_id"):
                        sources.append(unit["passage_hash_id"])
                    if not sources:
                        return False
                    passage_ids.extend(pid for pid in sources if pid not in passage_ids)
            store = getattr(self, "passage_embedding_store", None)
            source_texts = getattr(store, "hash_id_to_text", {})
            if any(pid not in source_texts for pid in passage_ids):
                return False
            passages = [source_texts[pid] for pid in passage_ids]
        if not sample or not passages:
            return False
        for passage in passages:
            text = str(passage)
            marker = re.search(r"(?m)^\[SourceChunks\][^\n]*(?:\n|$)", text)
            prefix = text[:marker.start()] if marker else ""
            boundary = IsolationGuard.extract_boundary_from_text(prefix)
            if not (
                marker and re.search(r"(?m)^\[Document\][ \t]*[^\n]+", prefix)
                and re.search(r"(?m)^\[Title\][ \t]*[^\n]*", prefix)
                and IsolationGuard.boundary_aliases(boundary) & IsolationGuard.boundary_aliases(sample)
            ):
                return False
        return True

    def _document_qa_instruction(self, question):
        instruction = (
            "Answer the question as written. Its wording determines the subject, relation, number of answers, "
            "and required detail. Answer each requested clause in order; multiple descriptions of one entity "
            "do not require separate answers. Include all requested values without repetition or unrelated facts, "
            "preserving necessary qualifiers, dates, ranges, and units. Connect complementary passages and use "
            "ordinary factual knowledge only for bridges required by their clues. Return the requested answer "
            "rather than an intermediate clue or broader category. The word 'or' does not by itself require an "
            "option: preserve the requested answer type. Give only the concise final answer."
        )
        if self._is_yes_no_question(question):
            return instruction + " " + self._qa_type_instruction(question)
        if self._is_choice_question(question):
            return instruction + (
                " If the question asks you to choose an option, return that option, not a truth value; "
                "answer 'Unknown' only if none of the requested options is supported."
            )
        return instruction

    def _build_reader_messages(self, retrieval_result, eval_mode="fusion"):
        question = retrieval_result["question"]
        query_profile = retrieval_result.get("query_profile", {})
        selected_passages, selected_meta = self._select_passages_for_eval_mode(retrieval_result, eval_mode=eval_mode, return_meta=True)
        selected_passages, selected_meta = self._filter_reader_context_to_query_sample(
            query_profile,
            selected_passages,
            selected_meta,
        )
        selected_passages, selected_meta, reader_isolation_trace = self.isolation_guard.filter_reader_context(
            query_profile,
            selected_passages,
            selected_meta,
            stage="reader_initial",
        )
        if reader_isolation_trace.get("enabled"):
            retrieval_result["reader_isolation_trace"] = reader_isolation_trace
        support_passages, support_meta, support_trace = self._build_multi_evidence_support_completion(
            question, query_profile, selected_meta
        )
        if support_trace.get("enabled"):
            retrieval_result["multi_evidence_support_completion_trace"] = support_trace
        if support_passages:
            selected_passages = list(support_passages) + list(selected_passages)
            selected_meta = list(support_meta) + list(selected_meta)
        completion_passages, completion_meta = self._build_inferential_evidence_completion(question, query_profile, selected_meta)
        if completion_passages:
            selected_passages = list(completion_passages) + list(selected_passages)
            selected_meta = list(completion_meta) + list(selected_meta)
        selected_passages, selected_meta = self._rerank_reader_context_for_answer_stage(
            question,
            query_profile,
            selected_passages,
            selected_meta,
        )
        spotlight_passages, spotlight_meta, spotlight_trace = self._build_inferential_reader_spotlights(
            question,
            query_profile,
            selected_meta,
        )
        if spotlight_passages:
            # Keep the retrieval ranking authoritative. Atomic spotlights are appended as
            # clarification so they cannot displace or dominate already selected evidence.
            selected_passages = list(selected_passages) + list(spotlight_passages)
            selected_meta = list(selected_meta) + list(spotlight_meta)
        if spotlight_trace.get("enabled"):
            retrieval_result["inferential_reader_spotlight_trace"] = spotlight_trace
        selected_passages, selected_meta, reader_final_isolation_trace = self.isolation_guard.filter_reader_context(
            query_profile,
            selected_passages,
            selected_meta,
            stage="reader_final",
        )
        if reader_final_isolation_trace.get("enabled"):
            retrieval_result["reader_final_isolation_trace"] = reader_final_isolation_trace
        if self._reader_uses_explicit_documents(retrieval_result, eval_mode):
            qa_instruction = self._document_qa_instruction(question)
            dynamic_target = ""
            dynamic_target_applied = False
            dynamic_target_guidance = ""
            task_guidance = ""
        else:
            qa_instruction = self._qa_type_instruction(question)
            dynamic_target = str(query_profile.get("dynamic_answer_target", "")).strip()
            if not dynamic_target:
                dynamic_target = self._dynamic_answer_target(question)
            dynamic_target_applied = self._should_apply_dynamic_answer_target(
                question,
                query_profile=query_profile,
                target=dynamic_target,
            )
            dynamic_target_guidance = (
                self._dynamic_answer_target_guidance(question, target=dynamic_target)
                if dynamic_target_applied
                else ""
            )
            task_guidance = self._task_reader_guidance(query_profile)
        retrieval_result["dynamic_answer_target"] = dynamic_target
        retrieval_result["dynamic_answer_target_applied"] = dynamic_target_applied
        system_prompt = (
            "As an advanced reading comprehension assistant, analyze the passages carefully. "
            "Your response must start after \"Thought: \", where you reason briefly. "
            "Conclude with \"Answer: \" and follow this output rule strictly: "
            f"{qa_instruction}{dynamic_target_guidance}{task_guidance}"
        )
        selected_passages, selected_meta, reader_budget_trace = self._apply_reader_prompt_budget(
            selected_passages,
            selected_meta,
            question,
            system_prompt,
        )
        if reader_budget_trace.get("enabled"):
            retrieval_result["reader_prompt_budget_trace"] = reader_budget_trace
        prompt_user = self._format_reader_context_for_prompt(query_profile, selected_passages, selected_meta)
        prompt_user += f"Question: {question}\n Thought: "
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_user},
        ]
        return messages, selected_passages, selected_meta

    def _ensure_answer_stage_reranker(self):
        if self.answer_stage_reranker_failed:
            return None
        if self.answer_stage_reranker is not None:
            return self.answer_stage_reranker
        model_name = str(self.config.answer_stage_reranker_model or "").strip()
        if not model_name or CrossEncoder is None:
            return None
        try:
            self.answer_stage_reranker = CrossEncoder(model_name, device=str(self.device))
        except Exception as e:
            logger.warning("Failed to load answer-stage reranker: %s", safe_error_summary(e))
            self.answer_stage_reranker_failed = True
            self.answer_stage_reranker = None
        return self.answer_stage_reranker

    def _is_inferential_question(self, question, query_profile=None):
        profile = query_profile or {}
        if self._task_type(profile) != "inferential":
            return False
        q = str(question or "").strip().lower()
        slot_type = self._static_slot_type(question, query_profile=profile)
        if slot_type and slot_type != "generic":
            return True
        if q.startswith(("would ", "could ", "might ")):
            return True
        return " likely " in f" {q} " or q.startswith(("what traits", "what degree", "what field"))

    def _should_apply_answer_stage_reranker(self, question, query_profile=None):
        if not bool(self.config.use_answer_stage_reranker):
            return False
        return self._task_type(query_profile) == "inferential"

    def _answer_stage_reranker_query(self, question, query_profile=None):
        question = str(question or "").strip()
        target = str((query_profile or {}).get("dynamic_answer_target", "")).strip()
        if not target:
            target = self._dynamic_answer_target(question)
        if not self._should_apply_dynamic_answer_target(
            question,
            query_profile=query_profile,
            target=target,
        ):
            return question
        return f"{question}\nAnswer target: {target}."

    def _rerank_reader_context_for_answer_stage(self, question, query_profile, selected_passages, selected_meta):
        if not self._should_apply_answer_stage_reranker(question, query_profile=query_profile):
            return selected_passages, selected_meta
        reranker = self._ensure_answer_stage_reranker()
        if reranker is None:
            return selected_passages, selected_meta
        passages = list(selected_passages or [])
        meta = list(selected_meta or [])
        if not passages or not meta or len(passages) != len(meta):
            return selected_passages, selected_meta
        top_n = max(1, int(self.config.answer_stage_reranker_top_n))
        texts = [
            str(meta[i].get("text", passages[i])) if isinstance(meta[i], dict) else str(passages[i])
            for i in range(len(passages))
        ]
        try:
            reranker_query = self._answer_stage_reranker_query(question, query_profile=query_profile)
            pairs = [[reranker_query, text] for text in texts]
            scores = list(
                reranker.predict(
                    pairs,
                    batch_size=min(16, max(1, len(pairs))),
                    show_progress_bar=False,
                )
            )
        except Exception as e:
            logger.warning("Answer-stage reranker scoring failed: %s", safe_error_summary(e))
            return selected_passages, selected_meta
        ranked = sorted(
            range(len(passages)),
            key=lambda i: (-float(scores[i]), i),
        )
        promoted = ranked[:min(top_n, len(ranked))]
        promoted_set = set(promoted)
        order = promoted + [i for i in range(len(passages)) if i not in promoted_set]
        ranked_meta = []
        target = str((query_profile or {}).get("dynamic_answer_target", "")).strip()
        if not target:
            target = self._dynamic_answer_target(question)
        target_applied = self._should_apply_dynamic_answer_target(
            question,
            query_profile=query_profile,
            target=target,
        )
        for new_rank, original_index in enumerate(order, start=1):
            item = dict(meta[original_index])
            item.update({
                "answer_reranker_score": float(scores[original_index]),
                "answer_reranker_original_rank": original_index + 1,
                "answer_reranker_rank": new_rank,
                "answer_reranker_promoted": original_index in promoted_set,
                "answer_reranker_target": target,
                "answer_reranker_target_applied": target_applied,
            })
            ranked_meta.append(item)
        return [passages[i] for i in order], ranked_meta

    def _qa_type_instruction(self, question):
        qtype, profile = self._classify_qa_question_type(question)
        slot_type = self._static_slot_type(question, query_profile=profile) if qtype == "static_slot" else ""
        q_lower = str(question or "").strip().lower()
        if q_lower.startswith("how many"):
            if re.match(r"^how many\s+(?:years|months|weeks|days|hours|minutes)\b", q_lower):
                return (
                    "Calculate the requested elapsed duration from the relevant dated evidence. Answer with only the "
                    "shortest number-and-unit expression, such as 'three months' or 'two weeks'. Always retain the "
                    "requested unit and do not explain."
                )
            return (
                "Reconcile all relevant evidence before answering this count question. Prefer an explicit cumulative "
                "total or ordinal milestone when one is supported; otherwise count distinct matching events or entities "
                "without treating repeated mentions as new instances. Do not infer zero merely because the evidence lacks "
                "one summary sentence. Answer concisely with the number and any unit required by the question. Do not explain."
            )
        multi_clause_question = qtype not in {"yes_no", "choice"} and bool(
            re.search(
                r"(?:;|\band\b)\s+(?:who|what|where|when|why|how(?:\s+(?:many|long))?)\b",
                q_lower,
            )
        )
        if multi_clause_question:
            return (
                "Answer every requested clause from its own supporting evidence and keep the answers in the same order "
                "as the question. Preserve each subject-to-value mapping. Return one concise coordinated answer; "
                "do not omit a clause or replace it with a merely related fact. Do not explain."
            )
        explicit_shared_relation = bool(
            (
                re.search(r"\b(?:both|in common|each of)\b", q_lower)
                and not re.search(r"\b(?:gift|present|recommendation|suggestion)\s+for\s+both\b", q_lower)
            )
            or re.search(
                r"\b(?:a|the)\s+shared\s+(?:activity|activities|interest|interests|frustration|frustrations|"
                r"experience|experiences|goal|goals|trait|traits|value|values|memory|memories)\b",
                q_lower,
            )
            or q_lower.startswith(("what shared ", "which shared "))
        )
        shared_question = qtype not in {"yes_no", "choice"} and explicit_shared_relation
        if shared_question:
            return (
                "Evaluate the evidence for each named subject separately, then return the shortest value or fact supported "
                "for every subject. Compute the intersection, not the union of unrelated subject-specific facts. Do not "
                "restate the subjects or the question. If the shared answer is a proposition, keep the complete concise "
                "common fact. Prefer the shortest wording copied from the supporting evidence; do not replace concrete "
                "nouns or relations with broader paraphrases. Do not explain."
            )
        if q_lower.startswith(("which countries", "what countries")):
            return (
                "Answer with only the country names supported by the evidence, not cities, regions, or landmarks. "
                "When the evidence names a city, map it to its country only when that mapping is unambiguous. "
                "Return a concise comma-separated list and do not explain."
            )
        if q_lower.startswith(("which cities", "what cities")):
            return "Answer with only the city names, not countries or regions. Return a concise comma-separated list and do not explain."
        if q_lower.startswith(("which states", "what states")):
            return "Answer with only the state names, not cities or countries. Return a concise comma-separated list and do not explain."
        if re.search(r"\bmediums?\b", q_lower) and re.search(r"\b(?:game|games|gaming|play|plays)\b", q_lower):
            return (
                "Answer with only the concrete platforms, devices, or media used to play the games. "
                "Do not replace named systems with a generic phrase such as 'computer gaming'. "
                "Return a concise comma-separated list and do not explain."
            )
        singular_activity_question = bool(
            re.match(r"^(?:what|which)\s+(?:sports\s+)?activit(?:y|ies)\s+(?:does|did|has|is|was)\b", q_lower)
        )
        aggregation_question = qtype == "list" and not singular_activity_question and bool(
            re.search(
                r"^(?:what|which)\s+(?:activities|events|changes|kinds|types|ways|symbols|items|places|"
                r"countries|cities|states|mediums|schools|games|movies|foods|books|sports)\b",
                q_lower,
            )
            or q_lower.startswith(("in what ways", "what are some"))
        )
        if aggregation_question:
            return (
                "Collect all distinct evidence-backed values that satisfy the subject, relation, event, and time "
                "constraints in the question. Exclude nearby topic-related values that do not satisfy every constraint, "
                "and do not repeat aliases or repeated mentions. Answer with only a concise comma-separated list. "
                "Do not explain."
            )
        if q_lower.startswith(("which country", "what country", "in what country")):
            return "Answer with only the country name, not a city, region, or full sentence. Do not explain."
        if q_lower.startswith(("which city", "what city")):
            return "Answer with only the city name, not a country, region, or full sentence. Do not explain."
        if q_lower.startswith(("which year", "what year")):
            return "Answer with only the four-digit year. Do not use relative phrases like 'five years ago'. Do not explain."
        if q_lower.startswith(("which state", "what state")):
            return "Answer with only the state name or comma-separated state names. Do not explain."
        if qtype == "time":
            return (
                "Answer with only the shortest final time expression, not a full sentence. "
                "Prefer a concise normalized date, month-year, year, or an anchored phrase such as 'the week before 9 June 2023' when it is clearly supported by the passages. "
                "Avoid vague deictic answers like 'yesterday', 'last week', 'next month', or 'this month' unless that exact form is the only supported answer. "
                "Do not add explanation."
            )
        if qtype == "duration":
            return (
                "Answer with only the duration expression, not a full sentence. "
                "Prefer a concise value such as '4 years', '10 years ago', or 'since 2016'. Do not explain."
            )
        if qtype == "yes_no":
            if self._is_modal_yes_no_question(question):
                return "Answer with only 'Likely yes' or 'Likely no'. Do not answer 'Unknown'. Do not add explanation."
            return "Answer with only 'Yes' or 'No'. Do not answer 'Unknown'. Do not add explanation."
        if qtype == "choice":
            return (
                "This is an A-or-B choice question. Answer with only the chosen option phrase from the question, "
                "not 'Yes' or 'No'. If neither option is supported, answer 'Unknown'. Do not explain."
            )
        if qtype == "static_slot":
            if slot_type in {"identity", "relationship", "fields", "career", "politics", "religion"}:
                return (
                    "Answer with only the complete short value phrase, not a full sentence. "
                    "Keep all necessary modifiers from the evidence and do not replace them with a broader category. Do not explain."
                )
            return (
                "Answer with only the complete short attribute value, not a full sentence. "
                "Keep necessary modifiers and do not over-compress the answer to a single word unless the evidence clearly supports that. Do not explain."
            )
        if qtype == "list":
            return "Answer with only a comma-separated list of concrete values found in the evidence. Do not explain."
        if qtype == "location":
            return "Answer with only the location name or comma-separated location names. Do not explain."
        if qtype == "person":
            return "Answer with only the person's name or a short noun phrase. Do not explain."
        if qtype == "event":
            return "Answer with only the short event, object, or phrase directly supported by the evidence. Do not add explanation."
        return "Answer concisely with only the final answer phrase. Do not explain."

    def _dynamic_answer_target(self, question):
        q = re.sub(r"\s+", " ", str(question or "").strip().lower()).rstrip("?").strip()
        if not q:
            return "answer matching the question wording"

        if "," in q:
            contextual_tail = q.split(",", 1)[1].strip()
            if re.match(
                r"^(?:what|which|who|whose|where|when|why|how|do|does|did|is|are|was|were|has|have|had|would|could|might|will|can|should)\b",
                contextual_tail,
            ):
                return self._dynamic_answer_target(contextual_tail)

        preposed_how = re.match(r"^(?:after|before|for|in|within)\s+how\s+(many|long)\s*(.*)$", q)
        if preposed_how:
            mode, remainder = preposed_how.groups()
            if mode == "long":
                return "duration"
            noun_phrase = re.split(
                r"\s+(?:do|does|did|is|are|was|were|has|have|had|would|could|might|will|can|should)\s+",
                remainder,
                maxsplit=1,
            )[0].strip(" ,.;:")
            return f"number of {noun_phrase}" if noun_phrase else "number"

        preposed_wh = re.match(
            r"^(?:around|in|during|after|before|on|at|from|to|for)\s+(?:what|which)\s+(.+?)(?:\s+(?:do|does|did|is|are|was|were|has|have|had|would|could|might|will|can|should)\s+)",
            q,
        )
        if preposed_wh:
            target = preposed_wh.group(1).strip(" ,.;:")
            possessive_head = re.match(r"^(.+?)(?:['\u2019]s)\s+", target)
            if possessive_head:
                target = possessive_head.group(1).strip()
            return " ".join(target.split()[:8])

        if q.startswith("whose "):
            return "person or owner"

        structural_targets = (
            ("who", "person or role"),
            ("where", "location"),
            ("when", "time"),
            ("why", "reason"),
        )
        for prefix, target in structural_targets:
            if q == prefix or q.startswith(prefix + " "):
                return target

        how_match = re.match(r"^how\s+(many|much|long|old|far|often)\s*(.*)$", q)
        if how_match:
            mode, remainder = how_match.groups()
            fixed_targets = {
                "much": "amount",
                "long": "duration",
                "old": "age",
                "far": "distance",
                "often": "frequency",
            }
            if mode != "many":
                return fixed_targets[mode]
            noun_phrase = re.split(
                r"\s+(?:do|does|did|is|are|was|were|has|have|had|would|could|might|will|can|should)\s+",
                remainder,
                maxsplit=1,
            )[0].strip(" ,.;:")
            return f"number of {noun_phrase}" if noun_phrase else "number"

        how_aux_match = re.match(
            r"^how\s+(do|does|did|is|are|was|were|has|have|had|would|could|might|will|can|should)\b",
            q,
        )
        if how_aux_match:
            auxiliary = how_aux_match.group(1)
            if auxiliary in {"is", "are", "was", "were"}:
                return "state or degree"
            if auxiliary in {"has", "have", "had"}:
                return "change or result"
            return "manner or result"
        if q.startswith("how "):
            return "degree, manner, or result"

        wh_match = re.match(r"^(?:what|which)\s+(.+)$", q)
        if wh_match:
            body = wh_match.group(1).strip()
            possessive_patterns = (
                r"^(?:might|would|could|will|can|should)\s+.+?['\u2019]s\s+(.+?)(?:\s+(?:likely|possibly|probably))?\s+be(?:\s+in)?$",
                r"^(?:is|was|are|were)\s+.+?['\u2019]s\s+(.+)$",
            )
            for pattern in possessive_patterns:
                match = re.match(pattern, body)
                if match:
                    target = re.sub(r"\s+", " ", match.group(1)).strip(" ,.;:")
                    if 1 <= len(target.split()) <= 10:
                        return target
            leading_aux = re.match(
                r"^(do|does|did|is|are|was|were|has|have|had|would|could|might|will|can|should)\s+(.+)$",
                body,
            )
            if leading_aux:
                auxiliary, remainder = leading_aux.groups()
                if auxiliary in {"is", "are", "was", "were"}:
                    possessive_parts = re.split(r"(?:['\u2019]s|s['\u2019])\s+", remainder)
                    if len(possessive_parts) > 1:
                        target = re.split(
                            r"\s+(?:to|for|from|with|during|after|before|in|on|at|by|that|who|which)\s+",
                            possessive_parts[-1],
                            maxsplit=1,
                        )[0].strip(" ,.;:")
                        if target and len(target.split()) <= 3:
                            return target
                    return "state, attribute, or description"
                if auxiliary in {"do", "does", "did"}:
                    return "object, action, or result"
                if auxiliary in {"has", "have", "had"}:
                    return "object, event, or result"
                return "expected value, action, or result"
            target = re.split(
                r"\s+(?:do|does|did|is|are|was|were|has|have|had|would|could|might|will|can|should)\s+",
                body,
                maxsplit=1,
            )[0].strip(" ,.;:")
            if target and not re.match(
                r"^(?:do|does|did|is|are|was|were|has|have|had|would|could|might|will|can|should)\b",
                target,
            ):
                return " ".join(target.split()[:10])
            return "answer type implied by the question wording"

        if re.match(
            r"^(?:do|does|did|is|are|was|were|has|have|had|would|could|might|will|can|should)\b",
            q,
        ):
            return "one of the stated alternatives" if " or " in f" {q} " else "truth value"
        return "answer type implied by the question wording"

    def _dynamic_answer_target_guidance(self, question, target=None):
        target = str(target or "").strip() or self._dynamic_answer_target(question)
        return (
            f" Answer target: {target}. Preserve the requested number, unit, and semantic granularity."
        )

    def _should_apply_dynamic_answer_target(self, question, query_profile=None, target=None):
        """Use target guidance only when the question exposes a compact entity slot."""
        profile = query_profile or {}
        q = str(question or "").strip().lower()
        if not (q.startswith("who ") or re.search(r"\bwhich\b", q)):
            return False
        target = str(target or profile.get("dynamic_answer_target", "")).strip()
        if not target:
            target = self._dynamic_answer_target(question)
        target_terms = re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", target.lower())
        return 1 <= len(target_terms) <= 3

    def _task_reader_guidance(self, query_profile):
        task_type = self._task_type(query_profile)
        if task_type == "multi_evidence":
            if not self._requires_answer_aggregation(
                (query_profile or {}).get("question", ""), query_profile=query_profile
            ):
                return (
                    " Inspect all supplied evidence and connect the clues needed to resolve the relation. "
                    "The evidence chain may span passages, but the requested answer is a single value: "
                    "return only that value at the requested semantic granularity. Do not enumerate nearby "
                    "examples, names, titles, or alternatives merely because they occur in the context."
                )
            return (
                " For a multi-evidence question, inspect all supplied evidence before answering "
                "and combine every distinct value that satisfies the requested subject, relation, "
                "time, and event-state constraints. For a list, return the complete deduplicated "
                "set of supported values rather than stopping after the first matching passage. "
                "For a count, count distinct completed matching events or objects only; do not "
                "count plans, repetitions, nearby examples, or other speakers' facts. Preserve the "
                "requested semantic level: return pet types rather than pet names, a category rather "
                "than arbitrary members, and an event rather than surrounding conversation. Return "
                "only the concise answer without dates, source identifiers, or explanation."
            )
        if task_type == "inferential":
            return (
                " For an evidence-grounded inference, first identify the requested subject and answer type, "
                "then connect the decisive clues across the focused evidence and context sessions. Compare "
                "plausible alternatives and check each conclusion against the subject, relation, time, and "
                "polarity in the question. Use ordinary, widely established background knowledge when an "
                "explicit clue requires a geographic, medical, occupational, cultural, category, alias, or "
                "implicit-relationship bridge. For questions using likely, might, or could, return the most "
                "plausible conclusion supported by the clues instead of refusing only because it is not stated "
                "verbatim. Match the requested semantic granularity: do not substitute a city for a state, a "
                "work for its creator, a venue for its region, or one example for its category. Do not invent "
                "an unsupported personal event or preference. Treat primary evidence as authoritative and use "
                "supplemental clues only to resolve a missing bridge. When the evidence supplies an anchor such as "
                "a date, landmark, or distinctive description instead of the requested name or category, resolve "
                "that anchor to the canonical target rather than copying the clue as the answer. Answer the requested "
                "relation or attribute, never merely repeat the subject. When one entity or option is requested, "
                "return only that one canonical answer without alternatives or a rationale."
            )
        scores = dict((query_profile or {}).get("need_scores") or {})
        if float(scores.get("inferential", 0.0)) >= 0.60:
            return (
                " If the answer is implicit, connect only the supplied clues and any ordinary factual bridge "
                "they require. Return the requested value or truth value directly, without replacing it with "
                "the subject's name or explaining the reasoning."
            )
        if float(scores.get("multi_evidence", 0.0)) >= 0.60:
            return (
                " Check the supplied evidence for complementary facts before answering, but preserve the "
                "question's answer cardinality and return only the requested concise value or values."
            )
        return ""

    def _safe_extract_answer_text(self, qa_result):
        text = "" if qa_result is None else str(qa_result).strip()
        if not text:
            return "Unknown"
        m = re.search(r"Answer:\s*(.*)", text, re.I | re.S)
        ans = m.group(1).strip() if m else text.strip()
        lines = [ln.strip() for ln in ans.splitlines() if ln.strip()]
        if not lines:
            return "Unknown"
        first = lines[0]
        return first if re.search(r"\w", first, flags=re.UNICODE) else "Unknown"

    def _run_reader_once(self, retrieval_result, eval_mode="fusion"):
        started = perf_counter()
        messages, selected_passages, selected_meta = self._build_reader_messages(retrieval_result, eval_mode=eval_mode)
        performance = {"prompt_build_seconds": perf_counter() - started}
        retrieval_result["performance"] = performance
        try:
            qa_result, call = self.llm_model.infer(messages, return_metadata=True)
        except Exception as error:
            performance["reader_call"] = getattr(error, "reader_call_metadata", None)
            raise
        performance["reader_call"] = call
        pred_ans = self._safe_extract_answer_text(qa_result)
        retrieval_result["reader_output"] = qa_result
        retrieval_result["reader_answer"] = pred_ans
        reader_pred_ans = pred_ans
        pred_ans = self._refine_choice_answer(
            retrieval_result.get("question", ""), pred_ans,
        )
        pred_ans = self._normalize_time_answer_format(
            retrieval_result.get("question", ""),
            retrieval_result.get("query_profile", {}),
            pred_ans,
            selected_passages,
            selected_meta,
        )
        pred_ans, postprocessor_guard_trace = self._guard_postprocessed_answer(
            retrieval_result.get("question", ""),
            retrieval_result.get("query_profile", {}),
            reader_pred_ans,
            pred_ans,
            selected_passages,
            selected_meta,
        )
        if postprocessor_guard_trace.get("enabled"):
            retrieval_result["postprocessor_expansion_guard_trace"] = postprocessor_guard_trace
        return pred_ans, selected_passages, selected_meta

    def read_one_from_retrieval_result(self, retrieval_result, eval_mode="fusion"):
        started = perf_counter()
        try:
            pred_ans, selected_passages, selected_meta = self._run_reader_once(retrieval_result, eval_mode=eval_mode)
        except Exception as error:
            performance = retrieval_result.get("performance", {})
            performance["reader_seconds"] = perf_counter() - started
            error.reader_performance = performance
            raise
        performance = retrieval_result["performance"]
        performance["reader_seconds"] = perf_counter() - started
        performance["answer_processing_seconds"] = max(
            0.0, performance["reader_seconds"] - performance["prompt_build_seconds"]
            - performance["reader_call"]["api_seconds"])
        final_result = dict(retrieval_result)
        final_result["pred_answer"] = pred_ans
        final_result["reader_context"] = selected_passages
        final_result["reader_context_meta"] = selected_meta
        final_result["selected_sorted_passage"] = selected_passages
        final_result["eval_mode"] = eval_mode
        return final_result

    def qa_one(self, question_item, eval_mode="fusion"):
        retrieval_result = self.retrieve([question_item], eval_mode=eval_mode, show_progress=False)[0]
        return self.read_one_from_retrieval_result(retrieval_result, eval_mode=eval_mode)

    def qa(self, questions, eval_mode="fusion"):
        retrieval_results = self.retrieve(questions, eval_mode=eval_mode, show_progress=True)
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            final_results = list(
                tqdm(
                    executor.map(lambda rr: self.read_one_from_retrieval_result(rr, eval_mode=eval_mode), retrieval_results),
                    total=len(retrieval_results),
                    desc=f"QA Reading ({eval_mode.upper()})",
                )
            )
        return final_results
