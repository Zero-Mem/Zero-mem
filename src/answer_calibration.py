"""Evidence support guards, answer surfaces and temporal calibration."""

from collections import Counter
from datetime import date, timedelta
import math
import re


class AnswerCalibrationMixin:
    def _is_unknown_like_answer(self, answer):
        low = str(answer or "").strip().lower()
        if not low:
            return True
        markers = (
            "unknown",
            "unclear",
            "indeterminable",
            "not determinable",
            "not possible to determine",
            "cannot determine",
            "can't determine",
            "not specified",
            "not mention",
            "does not specify",
            "does not provide",
            "not explicitly stated",
            "not enough information",
            "no information",
            "no specific",
            "none stated",
            "none specified",
            "n/a",
        )
        if any(m in low for m in markers):
            return True
        return bool(re.search(r"\bno\b.{0,40}\b(?:mention|mentioned|stated|provided|found|available)\b", low))

    def _context_texts_for_answer_refine(self, selected_passages=None, selected_meta=None):
        texts = []
        for passage in selected_passages or []:
            if passage:
                texts.append(str(passage))
        for item in selected_meta or []:
            if isinstance(item, dict) and item.get("text"):
                texts.append(str(item.get("text")))
        return texts

    def _normalize_support_text(self, text):
        text = str(text or "").lower()
        text = re.sub(r"[-_/]", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        return " ".join(text.split())

    def _answer_supported_by_context(self, answer, selected_passages, selected_meta=None):
        ans = self._normalize_support_text(answer)
        if not ans or ans in {"unknown", "none", "not specified", "not mentioned", "no information available", "indeterminable", "n/a"}:
            return False
        if len(ans.split()) == 1 and len(ans) <= 2:
            return False
        texts = self._context_texts_for_answer_refine(selected_passages, selected_meta)
        context = self._normalize_support_text(" \n ".join(texts))
        if not context:
            return False
        if ans in context:
            return True
        ans_tokens = ans.split()
        context_tokens = set(context.split())
        if len(ans_tokens) <= 2:
            return all(tok in context_tokens for tok in ans_tokens)
        hit_count = sum(1 for tok in ans_tokens if tok in context_tokens)
        return hit_count >= max(2, int(math.ceil(len(ans_tokens) * 0.75)))

    def _is_degenerate_factoid_answer(self, answer):
        ans = str(answer or "").strip()
        if not ans:
            return False
        tokens = self._normalize_support_text(ans).split()
        if len(tokens) > 60:
            return True
        comma_parts = [p.strip() for p in re.split(r"[,;]", ans) if p.strip()]
        if len(comma_parts) >= 10 and len(tokens) > 24:
            return True
        if len(tokens) >= 24:
            grams = [" ".join(tokens[i:i + 3]) for i in range(max(0, len(tokens) - 2))]
            counts = Counter(grams)
            if counts and max(counts.values()) >= 4:
                return True
        return False

    def _guard_postprocessed_answer(
        self,
        question,
        query_profile,
        reader_answer,
        postprocessed_answer,
        selected_passages,
        selected_meta,
    ):
        """Reject deterministic expansion that corrupts a supported reader answer."""
        trace = {"enabled": False}
        reader = str(reader_answer or "").strip()
        refined = str(postprocessed_answer or "").strip()
        if not reader or not refined or reader == refined:
            return postprocessed_answer, trace
        if self._is_unknown_like_answer(reader) or self._is_degenerate_factoid_answer(reader):
            return postprocessed_answer, trace
        if not self._is_degenerate_factoid_answer(refined):
            return postprocessed_answer, trace
        if not self._answer_supported_by_context(reader, selected_passages, selected_meta):
            return postprocessed_answer, trace
        reader_tokens = self._answer_token_count(reader)
        refined_tokens = self._answer_token_count(refined)
        cardinality = str((query_profile or {}).get("answer_cardinality", "single") or "single")
        contains_source_ids = bool(re.search(r"\bD\d+:\d+\b", refined))
        destructive_expansion = refined_tokens >= max(24, reader_tokens * 4)
        if cardinality != "single" and not contains_source_ids and not destructive_expansion:
            return postprocessed_answer, trace
        return reader, {
            "enabled": True,
            "reason": "degenerate_postprocessor_expansion",
            "before": refined[:500],
            "after": reader,
            "reader_tokens": reader_tokens,
            "postprocessed_tokens": refined_tokens,
            "contains_source_ids": contains_source_ids,
        }

    def _answer_token_count(self, answer):
        low = str(answer or "").strip().lower()
        low = re.sub(r"[^\w\s-]", " ", low)
        return len(low.split())

    def _parse_session_date_value(self, session_time):
        text = str(session_time or "").strip()
        if not text:
            return None
        m = re.search(r"\bon\s+(\d{1,2})\s+([A-Za-z]+),\s*((?:19|20)\d{2})\b", text, flags=re.I)
        if not m:
            m = re.search(r"\b(\d{1,2})\s+([A-Za-z]+),?\s*((?:19|20)\d{2})\b", text, flags=re.I)
        if not m:
            return None
        month_map = {
            "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
            "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
            "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
            "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
        }
        month = month_map.get(m.group(2).lower())
        if not month:
            return None
        try:
            return date(int(m.group(3)), month, int(m.group(1)))
        except ValueError:
            return None

    def _format_date_no_leading_zero(self, dt):
        if not dt:
            return ""
        return f"{dt.day} {dt.strftime('%B')} {dt.year}"



    def _iter_time_context_segments(self, selected_meta, texts):
        items = list(selected_meta or [])
        if not items:
            items = [{"text": text, "session_time": self._extract_session_datetime_value(text)} for text in texts or []]
        for item in items:
            if not isinstance(item, dict):
                continue
            raw = str(item.get("text", "")).strip()
            if not raw:
                continue
            session_time = str(item.get("session_time", "")).strip() or self._extract_session_datetime_value(raw)
            fallback_session_time = self._extract_session_datetime_value(raw) or session_time
            parts = re.split(r"\n+", raw)
            for part in parts:
                segment = re.sub(r"^\s*\[?D\d+:\d+\]?\s*", "", str(part)).strip()
                if not segment:
                    continue
                yield segment, session_time or fallback_session_time


    def _time_question_subject_terms(self, question):
        text = str(question or "").strip()
        text = re.sub(
            r"^(?:when|what|where|who|why|how)\s+(?:(?:did|does|do|was|were|is|are|will|would|has|have|had)\s+)?",
            "",
            text,
            flags=re.I,
        )
        prefix = re.split(
            r"\b(?:did|does|do|was|were|is|are|will|would|has|have|had|go|went|get|got|"
            r"start|started|join|joined|attend|attended|visit|visited|meet|met|see|saw|"
            r"make|made|buy|bought|adopt|adopted|return|returned|finish|finished|plan|planned)\b",
            text,
            maxsplit=1,
            flags=re.I,
        )[0]
        return {
            term.lower()
            for term in re.findall(r"\b[A-Z][a-z]+\b", prefix)
            if term.lower() not in {"the", "a", "an"}
        }

    def _strict_time_event_terms(self, question, query_profile=None):
        weak_terms = {
            "get", "got", "go", "went", "have", "had", "make", "made", "new", "old",
            "great", "will", "first", "start", "started", "find", "found", "out", "place",
            "event", "group", "friends", "partner", "attend", "attended", "join", "joined",
            "visit", "visited", "return", "returned", "receive", "received", "promote",
            "promoted", "planned", "complete", "completed", "buy", "bought", "move", "moved",
            "adopt", "adopted", "finish", "finished", "write", "wrote", "win", "won", "meet",
            "met", "see", "saw", "host", "hosted", "draw", "drew", "take", "took",
            "year", "month", "week", "weekend", "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday", "january", "february", "march", "april", "may",
            "june", "july", "august", "september", "october", "november", "december",
        }
        subjects = self._time_question_subject_terms(question)
        terms = []
        for term in self._normalize_bm25_tokens(question):
            normalized = str(term or "").lower().strip()
            if not normalized or normalized in weak_terms or normalized in subjects:
                continue
            if normalized not in terms:
                terms.append(normalized)
        return terms[:8]

    def _iter_local_relative_time_windows(self, text):
        cleaned = re.sub(r"\(img:[^)]*\)", " ", str(text or ""), flags=re.I)
        clauses = [
            clause.strip()
            for clause in re.split(r"(?<=[.!?])\s+|[;\n]+", cleaned)
            if clause.strip()
        ]
        cue_pattern = re.compile(
            r"\b(?:yesterday|two days ago|last night|last week|last weekend|last month|last year|"
            r"last monday|last tuesday|last wednesday|last thursday|last friday|last saturday|last sunday)\b",
            flags=re.I,
        )
        for idx, clause in enumerate(clauses):
            if not cue_pattern.search(clause):
                continue
            start = max(0, idx - 1)
            end = min(len(clauses), idx + 2)
            yield " ".join(clauses[start:end]), clause

    def _time_event_term_matches(self, term, text):
        normalized = str(term or "").lower().strip()
        low = str(text or "").lower()
        if not normalized:
            return False
        if normalized in low:
            return True
        aliases = {
            "competition": ("comp",),
        }
        if any(re.search(rf"\b{re.escape(alias)}\b", low) for alias in aliases.get(normalized, ())):
            return True
        stem = normalized
        for suffix in ("ing", "ed", "es", "s"):
            if stem.endswith(suffix) and len(stem) - len(suffix) >= 4:
                stem = stem[:-len(suffix)]
                break
        return len(stem) >= 4 and bool(re.search(rf"\b{re.escape(stem)}[a-z'-]*\b", low))

    def _time_candidate_respects_question_constraints(self, question, query_profile, local_text, session_date):
        q = str(question or "")
        q_lower = q.lower()
        local_lower = str(local_text or "").lower()
        month_names = (
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        )
        for month_index, month_name in enumerate(month_names, start=1):
            if re.search(rf"\bin\s+{month_name}\b", q_lower) and session_date.month != month_index:
                return False
        explicit_years = re.findall(r"\b(?:19|20)\d{2}\b", q)
        if explicit_years and str(session_date.year) not in explicit_years:
            return False
        quoted_values = re.findall(r'["“]([^"”]{2,80})["”]?', q)
        if quoted_values and not all(value.lower() in local_lower for value in quoted_values):
            return False
        for relation in ("mother", "father", "dad", "mom", "wife", "husband", "girlfriend", "boyfriend"):
            if re.search(rf"\b{relation}\b", q_lower) and not re.search(rf"\b{relation}\b", local_lower):
                return False
        for ordinal in ("first", "second", "third", "fourth"):
            if re.search(rf"\b{ordinal}\b", q_lower) and not re.search(rf"\b{ordinal}\b", local_lower):
                return False

        ignored_proper = {
            "when", "what", "where", "who", "why", "how", "the",
            *month_names,
        }
        proper_terms = [
            term for term in re.findall(r"\b[A-Z][a-z]+\b", q)
            if term.lower() not in ignored_proper
        ]
        subjects = self._time_question_subject_terms(q)
        required_proper = [
            term.lower()
            for term in proper_terms
            if term.lower() not in subjects and term.lower() not in ignored_proper
        ]
        if required_proper and not all(term in local_lower for term in required_proper):
            return False
        return True

    def _best_strict_event_matched_time(self, question, query_profile, selected_meta, texts):
        terms = self._strict_time_event_terms(question, query_profile=query_profile)
        if not terms:
            return "", 0.0
        candidates = []
        for segment, session_time in self._iter_time_context_segments(selected_meta, texts):
            session_date = self._parse_session_date_value(session_time)
            if not session_date:
                continue
            for local_text, cue_clause in self._iter_local_relative_time_windows(segment):
                local_lower = local_text.lower()
                hits = [term for term in terms if self._time_event_term_matches(term, local_lower)]
                required_hits = len(terms) if len(terms) <= 3 else (2 * len(terms) + 2) // 3
                if len(hits) < required_hits:
                    continue
                if not self._time_candidate_respects_question_constraints(
                    question,
                    query_profile,
                    local_text,
                    session_date,
                ):
                    continue
                cue_lower = cue_clause.lower()
                value = ""
                bonus = 0.0
                weekday = next((
                    day for day in (
                        "monday", "tuesday", "wednesday", "thursday",
                        "friday", "saturday", "sunday",
                    )
                    if re.search(rf"\blast\s+{day}\b", cue_lower)
                ), "")
                if weekday:
                    value = f"the {weekday} before {self._format_date_no_leading_zero(session_date)}"
                    bonus = 1.0
                elif "last weekend" in cue_lower:
                    value = f"the weekend before {self._format_date_no_leading_zero(session_date)}"
                    bonus = 1.0
                elif "last week" in cue_lower:
                    value = f"the week before {self._format_date_no_leading_zero(session_date)}"
                    bonus = 1.0
                elif "last month" in cue_lower:
                    value = f"the month before {self._format_date_no_leading_zero(session_date)}"
                    bonus = 0.8
                elif "last year" in cue_lower:
                    value = str(session_date.year - 1)
                    bonus = 1.0
                elif "two days ago" in cue_lower:
                    value = self._format_date_no_leading_zero(session_date - timedelta(days=2))
                    bonus = 0.85
                elif "yesterday" in cue_lower or "last night" in cue_lower:
                    value = self._format_date_no_leading_zero(session_date - timedelta(days=1))
                    bonus = 0.9
                if not value:
                    continue
                coverage = len(hits) / max(1.0, float(len(terms)))
                event_score = 2.0 * coverage + 0.2 * len(hits)
                score = event_score + bonus
                candidates.append((score, event_score, value))
        if not candidates:
            return "", 0.0
        candidates.sort(key=lambda item: item[0], reverse=True)
        top_score, top_event_score, top_value = candidates[0]
        top_anchor = self._relative_time_anchor_key(top_value)
        for _, competing_event_score, competing_value in candidates[1:]:
            if abs(competing_event_score - top_event_score) > 1e-9:
                continue
            competing_anchor = self._relative_time_anchor_key(competing_value)
            if top_anchor and competing_anchor and competing_anchor != top_anchor:
                return "", 0.0
        return top_value, top_score

    def _relative_time_anchor_key(self, answer):
        parsed = self._parse_session_date_value(answer)
        return parsed.isoformat() if parsed else ""

    def _normalize_time_answer_format(self, question, query_profile, pred_answer, selected_passages=None, selected_meta=None):
        effective_qtype = self._effective_question_type(question, query_profile=query_profile or {})
        q = str(question or "").strip().lower()
        is_year_question = q.startswith(("which year", "what year")) or "what year did" in q or "which year did" in q
        if effective_qtype not in {"when", "duration"} and not is_year_question:
            return pred_answer
        ans = str(pred_answer or "").strip()
        if not ans or self._is_unknown_like_answer(ans):
            return pred_answer
        months = (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
            "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec",
        )
        month_alt = "|".join(months)
        ans = re.sub(rf"\b(\d{{1,2}})-({month_alt})-(\d{{4}})\b", r"\1 \2 \3", ans, flags=re.I)
        ans = re.sub(rf"\b({month_alt})-(\d{{4}})\b", r"\1 \2", ans, flags=re.I)
        ans = re.sub(r"\s+", " ", ans).strip()
        texts = self._context_texts_for_answer_refine(selected_passages, selected_meta)
        if is_year_question:
            m = re.search(r"\b(?:19|20)\d{2}\b", ans)
            if m:
                return m.group(0)
        if effective_qtype == "when":
            ans_is_relative_expression = bool(re.search(
                r"\b(?:last|next|this)\s+(?:year|month|week|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|"
                r"\bthe\s+(?:week|weekend|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(?:before|after)\b|"
                r"\b(?:yesterday|today|tomorrow)\b",
                ans,
                flags=re.I,
            ))
            # An explicit calendar anchor belongs to the Reader answer;
            # only unresolved relative expressions need contextual anchoring.
            if ans_is_relative_expression and not re.search(r"\b\d{4}\b", ans):
                strict_relative, _ = self._best_strict_event_matched_time(
                    question,
                    query_profile,
                    selected_meta,
                    texts,
                )
                answer_anchor = self._relative_time_anchor_key(ans)
                strict_anchor = self._relative_time_anchor_key(strict_relative)
                if strict_anchor and strict_anchor != answer_anchor:
                    return strict_relative
        return ans
