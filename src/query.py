"""Question profiles and source, subject and dialogue signals."""

from collections import Counter
from .indexing import IsolationGuard
import re


class QueryMixin:
    def _effective_question_type(self, question, query_profile=None):
        qtype = self._detect_question_type(question)
        profile = query_profile or {}
        if qtype == "what":
            if bool(profile.get("is_static_question")):
                return "what_static"
            if bool(profile.get("is_list_question")):
                return "what_list"
            return "what_event"
        if qtype == "other" and bool(profile.get("is_list_question")):
            return "what_list"
        if qtype == "where":
            where_mode = self._where_question_mode(question, query_profile=profile)
            if where_mode in {"where_origin", "where_list"}:
                return where_mode
        return qtype

    def _detect_question_type(self, question):
        q = str(question or "").strip().lower()
        if q.startswith("how long "):
            return "duration"
        if q.startswith("in what country"):
            return "where"
        if q.startswith("when "):
            return "when"
        if q.startswith("where "):
            return "where"
        if q.startswith("who "):
            return "who"
        if q.startswith("why "):
            return "why"
        if q.startswith("how "):
            return "how"
        if q.startswith("what "):
            return "what"
        return "other"

    def _is_list_like_question(self, question):
        q = str(question or "").strip().lower()
        if not q:
            return False
        list_markers = (
            "what events", "what activities", "what books", "what are", "what has",
            "what kinds", "what types", "in what ways", "which events", "where has",
            "what places", "what countries", "which countries", "which geographical",
            "what music events", "what sports", "what areas", "what schools",
            "what professional activities", "what kind of indoor activities"
        )
        return any(marker in q for marker in list_markers) or (" and " in q and q.startswith(("what ", "where ")))

    def _is_static_attribute_question(self, question):
        q = str(question or "").strip().lower()
        if not q:
            return False
        static_markers = (
            "identity", "relationship status", "political leaning", "religious",
            "personality traits", "what traits", "career path", "relationship",
            "what country", "what fields", "what instruments", "what symbols",
            "pet name", "pets' names",
        )
        return any(marker in q for marker in static_markers)

    def _event_template_type(self, question, query_profile=None):
        q = str(question or "").strip().lower()
        if re.search(r"\b(?:plan|planning|intend|hope|want|will|would|could|might|going to)\b", q):
            return "planned_event"
        if self._asks_completed_relation(question):
            return "completed_event"
        effective_qtype = self._effective_question_type(question, query_profile=query_profile)
        return "generic_event" if effective_qtype in {"when", "what_event"} else ""

    def _event_template_terms(self, question, query_profile=None):
        stop = {
            "what", "when", "where", "which", "who", "why", "how", "did", "does", "do",
            "is", "are", "was", "were", "the", "a", "an", "to", "of", "for", "in", "on",
        }
        subjects = {
            token.lower()
            for subject in self._detect_primary_subjects(question, query_profile=query_profile)
            for token in re.findall(r"[a-z0-9']+", str(subject).lower())
        }
        terms = []
        for token in re.findall(r"[a-z0-9']+", str(question or "").lower()):
            if len(token) < 3 or token in stop or token in subjects:
                continue
            stem = self._multi_evidence_relation_term_stem(token)
            if stem and stem not in terms:
                terms.append(stem)
        return self._dedupe_preserve_order(terms)

    def _event_template_score(self, text, template_type):
        value = str(text or "").strip().lower()
        if not value or not template_type:
            return 0.0
        planned = bool(re.search(
            r"\b(?:plan|plans|planned|planning|intend|intends|intended|hope|hopes|"
            r"want|wants|wanted)\s+to\b|\b(?:will|would|could|might|going to)\b",
            value,
        ))
        negated = bool(re.search(
            r"\b(?:did not|has not|have not|had not|never|not yet|didn't|hasn't|haven't|hadn't)\b",
            value,
        ))
        actual = bool(re.search(
            r"\b(?:finished|completed|attended|visited|travelled|traveled|bought|made|painted|"
            r"played|watched|listened|joined|started|took|won|adopted|camped|hiked|went|read|"
            r"researched|organized|prepared)\b",
            value,
        ))
        if template_type == "planned_event":
            return 0.45 if planned and not negated else (-0.25 if actual and not planned else 0.0)
        if template_type == "completed_event":
            if negated or (planned and not actual):
                return -0.75
            return 0.45 if actual else 0.0
        return 0.15 if (planned or actual) and not negated else 0.0

    def _static_slot_type(self, question, query_profile=None):
        profile = query_profile or {}
        cached = str(profile.get("static_slot_type", "")).strip()
        if cached:
            return cached
        q = str(question or "").strip().lower()
        if not q:
            return ""
        if "relationship status" in q or ("relationship" in q and "status" in q):
            return "relationship"
        if "identity" in q:
            return "identity"
        if "what fields" in q or "field of study" in q or "education" in q:
            return "fields"
        if "career path" in q or "career" in q:
            return "career"
        if "political leaning" in q or "politically" in q:
            return "politics"
        if "religious" in q or "faith" in q or "spiritual" in q:
            return "religion"
        if "what kind of art" in q or "art does" in q:
            return "art"
        if "what instruments" in q or "instrument" in q:
            return "instruments"
        if "what symbols" in q or "symbols are important" in q:
            return "symbols"
        if "pets' names" in q or "pets names" in q:
            return "pets"
        return "generic"

    def _static_slot_terms(self, slot_type):
        slot = str(slot_type or "").strip().lower()
        mapping = {
            "relationship": (
                " relationship ", " status ", " partner ", " spouse ", " identifies as "
            ),
            "identity": (
                " identity ", " identifies as ", " pronoun ", " describes "
            ),
            "fields": (
                " degree ", " education ", " field ", " fields ", " study ", " studying ",
                " major ", " program ", " programme "
            ),
            "career": (
                " career ", " profession ", " job ", " work as ", " works as ", " working as ",
                " pursue ", " pursuing ", " become ", " becoming "
            ),
            "politics": (
                " political ", " politically ", " leaning ", " ideology "
            ),
            "religion": (
                " religious ", " religion ", " faith ", " spiritual ", " believes "
            ),
            "art": (
                " art ", " painting ", " paint ", " watercolor ", " watercolour ",
                " acrylic ", " sketch ", " mural ", " abstract "
            ),
            "instruments": (
                " instrument ", " instruments ", " play ", " plays "
            ),
            "symbols": (
                " symbol ", " symbols ", " represents ", " signifies "
            ),
            "pets": (
                " pet ", " pets ", " named ", " called "
            ),
            "generic": (
                " is ", " are ", " works as ", " identifies as ", " leans ",
                " believes ", " enjoys ", " plays "
            ),
        }
        return mapping.get(slot, mapping["generic"])

    def _build_query_profile(self, question, question_info=None):
        subject_terms = self._extract_subject_terms(question)
        qtype = self._detect_question_type(question)
        is_list_question = self._is_list_like_question(question)
        is_static_question = self._is_static_attribute_question(question)
        need_scores = self._question_need_scores(question)
        adaptive_task_type, adaptive_task_confidence = self._infer_task_type_from_scores(need_scores)
        answer_cardinality = self._answer_cardinality(question)
        profile = {
            "question": question,
            "dynamic_answer_target": self._dynamic_answer_target(question),
            "adaptive_task_type": adaptive_task_type,
            "adaptive_task_confidence": adaptive_task_confidence,
            "need_scores": need_scores,
            "answer_cardinality": answer_cardinality,
            "requires_answer_aggregation": answer_cardinality in {"multiple", "count"},
            "requires_evidence_closure": (
                adaptive_task_type == "multi_evidence"
                or float(need_scores.get("multi_evidence", 0.0)) >= 0.60
            ),
            "question_type": qtype,
            "keywords": self._extract_keywords(question, top_k=self.config.keyword_top_k),
            "subject_terms": subject_terms,
            "primary_subjects": [self._normalize_subject_name(s) for s in subject_terms if self._normalize_subject_name(s)],
            "is_duration_question": qtype == "duration",
            "is_time_question": self._contains_time_cue(question) or qtype in {"when", "duration"},
            "requires_temporal_route": self._requires_temporal_route(question),
            "is_list_question": is_list_question,
            "is_static_question": is_static_question,
        }
        profile["static_slot_type"] = self._static_slot_type(question, query_profile=profile) if is_static_question else ""
        profile["where_mode"] = self._where_question_mode(question, query_profile=profile)
        profile["effective_question_type"] = self._effective_question_type(question, query_profile=profile)
        profile["event_template_type"] = self._event_template_type(question, query_profile=profile)
        profile["event_template_terms"] = self._event_template_terms(question, query_profile=profile)
        if isinstance(question_info, dict):
            sid = self.isolation_guard.boundary_from_question_info(question_info)
            if sid:
                profile["sample_id"] = sid
            declared_task_type = self._task_type({"task_type": question_info.get("task_type", "")})
            if declared_task_type:
                profile["declared_task_type"] = declared_task_type
        if adaptive_task_type:
            profile["task_type"] = adaptive_task_type
            profile["task_type_source"] = "question_profile"
        else:
            profile["task_type"] = ""
            profile["task_type_source"] = "generic_pending_evidence"
        return profile

    def _requires_temporal_route(self, question):
        """Return whether answering the question requires temporal reasoning.

        A date mentioned in the question can be a retrieval constraint rather
        than the requested answer. Routing those constrained factoid questions
        to hierarchy was adding local timeline evidence even though the reader
        needed an entity or event value.
        """
        q = " ".join(str(question or "").strip().lower().split())
        if not q:
            return False
        if q.startswith("when "):
            if "," in q:
                tail = q.split(",", 1)[1].strip()
                if re.match(r"^(?:what|which|who|where|why|how(?!\s+long\b))\b", tail):
                    return False
            return True
        if re.search(r"(?:^|,\s*)how long\b", q):
            return True
        if re.match(r"^(?:(?:in|on|during|around|by)\s+)?(?:what|which)\s+(?:year|date|month|day|time)\b", q):
            return True
        if re.match(r"^how many\s+(?:years|months|weeks|days|hours|minutes)\b", q):
            return True
        comparison_markers = (
            "which happened first",
            "what happened first",
            "before or after",
            "earlier or later",
        )
        return any(marker in q for marker in comparison_markers)

    def _is_yes_no_question(self, question):
        q = str(question or "").strip().lower()
        if not q:
            return False
        if self._is_choice_question(question):
            return False
        if "answer in yes or no" in q or "answer yes or no" in q:
            return True
        starters = (
            "is ", "are ", "was ", "were ", "do ", "does ", "did ", "has ", "have ", "had ",
            "can ", "could ", "would ", "will ", "should "
        )
        if q.startswith(starters):
            return True
        if q.startswith(("what ", "when ", "where ", "who ", "which ", "why ", "how ")):
            return False
        preface_patterns = (
            r"^based on (?:the )?conversation,?\s+(?:is|are|was|were|do|does|did|has|have|had|can|could|would|will|should)\b",
            r"^from (?:the )?conversation,?\s+(?:is|are|was|were|do|does|did|has|have|had|can|could|would|will|should)\b",
            r"^according to (?:the )?conversation,?\s+(?:is|are|was|were|do|does|did|has|have|had|can|could|would|will|should)\b",
        )
        return any(re.search(pattern, q) for pattern in preface_patterns)

    def _is_modal_yes_no_question(self, question):
        q = str(question or "").strip().lower()
        if not q:
            return False
        if self._is_choice_question(question):
            return False
        return any(tok in q for tok in ("would ", "could ", "should ", "likely ", "still "))

    def _is_choice_question(self, question):
        q = str(question or "").strip().lower()
        if not q or " or " not in q:
            return False
        if "yes or no" in q or "yes/no" in q:
            return False
        choice_starters = (
            "would ", "could ", "should ", "does ", "do ", "did ", "is ", "are ", "was ", "were ",
            "which ", "what ", "based on ", "considering "
        )
        if q.startswith(choice_starters):
            return True
        return bool(re.search(r"\b(?:prefer|interested in|choose|rather|close to|enjoy)\b", q))

    def _classify_qa_question_type(self, question):
        profile = self._build_query_profile(question)
        effective_qtype = self._effective_question_type(question, query_profile=profile)
        if self._is_choice_question(question):
            return "choice", profile
        if self._is_yes_no_question(question):
            return "yes_no", profile
        if effective_qtype == "when":
            if bool(profile.get("requires_temporal_route", True)):
                return "time", profile
            embedded = re.search(
                r"(?:,|\bthen\b)\s*(what|who|where|which|how|why)\b",
                str(question or "").strip().lower(),
            )
            embedded_type = embedded.group(1) if embedded else ""
            if embedded_type == "what":
                if bool(profile.get("is_static_question")):
                    return "static_slot", profile
                if bool(profile.get("is_list_question")):
                    return "list", profile
                return "event", profile
            if embedded_type == "who":
                return "person", profile
            if embedded_type == "where":
                return "location", profile
            if embedded_type == "which" and bool(profile.get("is_list_question")):
                return "list", profile
            return "default", profile
        if effective_qtype == "duration":
            return "duration", profile
        if effective_qtype == "what_static":
            return "static_slot", profile
        if effective_qtype in {"what_list", "where_list"}:
            return "list", profile
        if effective_qtype in {"where", "where_origin"}:
            return "location", profile
        if effective_qtype == "who":
            return "person", profile
        if effective_qtype == "what_event":
            return "event", profile
        return "default", profile

    def _question_need_scores(self, question):
        """Build the paper's operational question profile without dataset labels."""
        q = " ".join(str(question or "").strip().lower().split())
        scores = {
            "temporal": 0.0,
            "multi_evidence": 0.0,
            "inferential": 0.0,
            "factoid": 0.0,
        }
        if not q:
            return scores
        temporal = (
            q.startswith("when ")
            or bool(re.match(r"^(?:in|on|during|around|by)\s+(?:what|which)\s+(?:year|date|month|day|time)\b", q))
            or bool(re.match(r"^how long\b", q))
            or bool(re.match(r"^how many\s+(?:years|months|weeks|days|hours|minutes)\b", q))
            or any(marker in q for marker in (
                "which happened first",
                "what happened first",
                "before or after",
                "earlier or later",
            ))
        )
        if temporal:
            scores["temporal"] = 1.0

        inferential = (
            bool(re.search(
                r"\b(?:likely|might|could|would|potentially|presumably|probably|underlying|imply|infer|suggest|considered)\b",
                q,
            ))
            or q.startswith(("based on ", "given "))
            or bool(re.search(r"\bif\b", q))
        )
        descriptive_identity = bool(re.match(
            r"^(?:what|which)\s+(?:is|was)\s+the\s+"
            r"(?:[a-z][a-z-]*\s+){0,2}[a-z][a-z-]*\s+(?:with|where)\b",
            q,
        ))
        bridge_target = (
            bool(re.match(r"^(?:in|from)\s+(?:what|which)\s+(?:country|state|region)\b", q))
            or bool(re.match(r"^(?:what|which)\s+(?:u\.?s\.?\s+)?(?:country|state|region)\b", q))
            or bool(re.match(r"^(?:what|which)\s+(?:popular|famous|well-known)\b", q))
            or bool(re.match(r"^what\s+(?:attributes?|challenges?|suspected|underlying)\b", q))
            or bool(re.match(r"^(?:what|which)\s+(?:board|card)\s+game\b", q))
            or bool(re.match(r"^which\s+(?:major\s+)?holiday\b", q))
            or bool(re.match(r"^how\s+(?:do|does|did)\b.+\b(?:cope|handle|address|overcome)\b", q))
            or bool(re.search(r"\bwhat\s+kind\s+of\s+(?:job|career|profession)\b.+\bbecause\b", q))
            or descriptive_identity
        )
        if inferential:
            scores["inferential"] = 0.95
        elif bridge_target:
            scores["inferential"] = 0.88
        elif (
            q.startswith(("around which ", "around what "))
            or bool(re.match(r"^what\s+(?:nickname|role|condition|career|profession|console|attributes?|challenges?)\b", q))
            or bool(re.match(r"^what\s+(?:other|additional|suspected)\b", q))
            or bool(re.match(r"^which\s+.+\s+(?:prefer|preferred)\b", q))
            or bool(re.match(r"^how old\b", q))
            or bool(re.match(r"^(?:does|do|is|are)\b.+\b(?:close to|near|closer to)\b", q))
            or bool(re.match(r"^who\s+(?:is|was|are|were)\b", q))
            or bool(re.match(r"^why\b", q))
            or bool(re.match(r"^(?:does|do|did|is|are|was|were|has|have)\b", q))
        ):
            scores["inferential"] = 0.62

        explicit_plural_target = self._has_explicit_plural_target(question)
        multi_evidence = (
            bool(re.search(r"\b(?:both|all|each|common|same|shared|various|multiple|several)\b", q))
            or bool(re.match(r"^how many\s+(?!years|months|weeks|days|hours|minutes)\b", q))
            or explicit_plural_target
        )
        if multi_evidence:
            scores["multi_evidence"] = 0.9
        elif (
            bool(re.match(r"^(?:what|which|where|who)\b.+\b(?:has|have)\b", q))
            or bool(re.match(r"^(?:what|which)\s+(?:kinds?|types?|ways?)\b", q))
            or bool(re.match(r"^(?:what|which)\s+.+(?:s|ies)\s+(?:did|does|has|have|are|were)\b", q))
            or bool(re.match(r"^what\s+(?:did|does)\b.+\b(?:research|study|collect|use|attend|visit|read|learn|play)\b", q))
        ):
            scores["multi_evidence"] = 0.66

        direct_question = bool(re.match(
            r"^(?:what|which|who|where|whose|did|does|do|is|are|was|were|how)\b",
            q,
        ))
        if direct_question and scores["temporal"] < 0.9:
            scores["factoid"] = 0.64
        if scores["multi_evidence"] >= 0.6 or scores["inferential"] >= 0.6:
            scores["factoid"] = min(scores["factoid"], 0.2)
        return scores

    def _has_explicit_plural_target(self, question):
        """Use the requested noun's grammatical number, not its subject matter."""
        return self.spacy_ner.question_analysis(question)[1] == "Plur"

    def _answer_cardinality(self, question):
        """Infer answer multiplicity independently from evidence-chain depth."""
        q = " ".join(str(question or "").strip().lower().split())
        if not q:
            return "single"
        if re.match(r"^how many\b", q):
            return "count"
        if re.search(r"\b(?:what|which|who|where)\b.+\b(?:and|or)\s+(?:what|which|who|where)\b", q):
            return "multiple"
        if self._has_explicit_plural_target(question):
            return "multiple"
        return "single"

    def _requires_answer_aggregation(self, question, query_profile=None):
        profile = query_profile or {}
        cardinality = str(profile.get("answer_cardinality", "") or "").strip().lower()
        if not cardinality:
            cardinality = self._answer_cardinality(question)
        return cardinality in {"multiple", "count"}

    def _infer_task_type_from_scores(self, scores):
        """Commit only high-confidence question shapes before retrieval."""
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        task_type, confidence = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        # Inference depth and answer multiplicity are orthogonal. A plural target may
        # require several values without weakening an explicit inferential cue.
        if task_type == "inferential" and confidence >= 0.90:
            return task_type, confidence
        if task_type != "factoid" and confidence >= 0.85 and confidence - runner_up >= 0.10:
            return task_type, confidence
        return "", 0.0

    def _adaptive_content_terms(self, text):
        stop = {
            "a", "an", "the", "and", "or", "but", "if", "then", "than", "to", "of", "in", "on",
            "at", "for", "from", "with", "about", "by", "as", "into", "around", "which", "what",
            "who", "whose", "where", "when", "why", "how", "did", "does", "do", "is", "are",
            "was", "were", "be", "been", "being", "has", "have", "had", "would", "could", "might",
            "should", "can", "will", "their", "his", "her", "its", "they", "them", "he", "she",
        }
        irregular = {
            "bought": "buy", "brought": "bring", "went": "go", "gone": "go", "seen": "see",
            "saw": "see", "made": "make", "took": "take", "given": "give", "gave": "give",
        }
        terms = []
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower()):
            if token in stop or len(token) <= 1:
                continue
            token = irregular.get(token, token)
            if token.endswith("ies") and len(token) > 4:
                token = token[:-3] + "y"
            elif token.endswith("ing") and len(token) > 5:
                token = token[:-3]
            elif token.endswith("ed") and len(token) > 4:
                token = token[:-2]
            elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
                token = token[:-1]
            terms.append(token)
        return set(terms)

    def _candidate_evidence_units(self, graph_result, hierarchy_result):
        units = []
        seen = set()

        def add(text, source, channel):
            normalized = " ".join(str(text or "").lower().split())
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            units.append({"text": str(text), "source": str(source), "channel": channel})

        turn_texts = list((hierarchy_result or {}).get("selected_turn_texts", []))
        turn_sources = list((hierarchy_result or {}).get("selected_turn_source_ids", []))
        for idx, text in enumerate(turn_texts[:8]):
            source = turn_sources[idx] if idx < len(turn_sources) and turn_sources[idx] else f"turn:{idx}"
            add(text, source, "hierarchy")
        trigram_texts = list((hierarchy_result or {}).get("selected_trigram_texts", []))
        trigram_sources = list((hierarchy_result or {}).get("selected_trigram_source_ids", []))
        for idx, text in enumerate(trigram_texts[:3]):
            source = trigram_sources[idx] if idx < len(trigram_sources) and trigram_sources[idx] else f"trigram:{idx}"
            add(text, source, "hierarchy")
        for idx, text in enumerate(list((graph_result or {}).get("sorted_passage", []))[:6]):
            add(text, f"graph:{idx}", "graph")
        return units

    def _evidence_need_signals(self, question, graph_result, hierarchy_result):
        query_terms = self._adaptive_content_terms(question)
        units = self._candidate_evidence_units(graph_result, hierarchy_result)
        if not query_terms or not units:
            return {
                "unit_count": len(units),
                "support_count": 0,
                "source_count": 0,
                "direct_coverage": 0.0,
                "complementary_coverage": 0.0,
                "hierarchy_support_count": 0,
            }
        overlaps = []
        union = set()
        sources = set()
        hierarchy_sources = set()
        graph_sources = set()
        hierarchy_support_count = 0
        minimum_hits = 1 if len(query_terms) <= 2 else 2
        for unit in units:
            overlap = query_terms & self._adaptive_content_terms(unit["text"])
            coverage = len(overlap) / max(1, len(query_terms))
            overlaps.append(coverage)
            if len(overlap) >= minimum_hits or coverage >= 0.5:
                union.update(overlap)
                if unit["channel"] == "hierarchy":
                    hierarchy_support_count += 1
                    hierarchy_sources.add(unit["source"])
                else:
                    graph_sources.add(unit["source"])
        sources = hierarchy_sources or graph_sources
        direct_coverage = max(overlaps, default=0.0)
        union_coverage = len(union) / max(1, len(query_terms))
        return {
            "unit_count": len(units),
            "support_count": len(sources),
            "source_count": len(sources),
            "direct_coverage": round(direct_coverage, 4),
            "complementary_coverage": round(max(0.0, union_coverage - direct_coverage), 4),
            "hierarchy_support_count": hierarchy_support_count,
        }

    def _refine_query_profile_from_evidence(self, query_profile, graph_result, hierarchy_result):
        """Calibrate an ambiguous question profile from already retrieved evidence."""
        profile = query_profile if isinstance(query_profile, dict) else {}
        signals = self._evidence_need_signals(
            profile.get("question", ""), graph_result, hierarchy_result
        )
        profile["evidence_need_signals"] = signals
        if profile.get("adaptive_task_type") and float(profile.get("adaptive_task_confidence", 0.0)) >= 0.85:
            return profile

        scores = dict(profile.get("need_scores") or self._question_need_scores(profile.get("question", "")))
        support_count = int(signals.get("support_count", 0))
        direct = float(signals.get("direct_coverage", 0.0))
        complementary = float(signals.get("complementary_coverage", 0.0))
        hierarchy_support = int(signals.get("hierarchy_support_count", 0))

        multi_score = float(scores.get("multi_evidence", 0.0))
        if hierarchy_support >= 2:
            multi_score += 0.10
        if complementary > 0.0:
            multi_score += 0.08

        inferential_score = float(scores.get("inferential", 0.0))
        if hierarchy_support > 0:
            inferential_score += 0.06
        if signals.get("unit_count", 0) and direct < 0.67:
            inferential_score += 0.06

        factoid_score = float(scores.get("factoid", 0.0))
        if direct >= 0.5:
            factoid_score += 0.12
        if support_count <= 1 and direct >= 0.5:
            factoid_score += 0.05

        selected = ""
        aggregation_required = self._requires_answer_aggregation(
            profile.get("question", ""), query_profile=profile
        )
        if (
            aggregation_required
            and multi_score >= 0.76
            and hierarchy_support >= 2
            and multi_score >= inferential_score + 0.06
        ):
            selected = "multi_evidence"
        elif inferential_score >= 0.67 and hierarchy_support > 0 and inferential_score >= multi_score:
            selected = "inferential"
        elif factoid_score >= 0.72:
            selected = "factoid"

        profile["calibrated_need_scores"] = {
            "temporal": round(float(scores.get("temporal", 0.0)), 4),
            "multi_evidence": round(multi_score, 4),
            "inferential": round(inferential_score, 4),
            "factoid": round(factoid_score, 4),
        }
        profile["requires_evidence_closure"] = bool(
            profile.get("requires_evidence_closure")
            or selected == "multi_evidence"
            or multi_score >= 0.76
        )
        profile["task_type"] = selected or "factoid"
        profile["task_type_source"] = "evidence_calibration" if selected else "safe_default"
        return profile

    def _task_type(self, query_profile):
        raw = (query_profile or {}).get("task_type")
        label = re.sub(r"[^a-z0-9]+", "_", str(raw or "").strip().lower()).strip("_")
        aliases = {
            "multi_evidence": "multi_evidence",
            "multi_hop": "multi_evidence",
            "multihop": "multi_evidence",
            "temporal": "temporal",
            "inferential": "inferential",
            "single_hop": "factoid",
            "singlehop": "factoid",
            "factoid": "factoid",
        }
        return aliases.get(label, "")

    def _extract_sample_id(self, text):
        if not text:
            return None
        boundary = IsolationGuard.extract_boundary_from_text(text)
        if boundary:
            return boundary
        m = re.search(r"(conv|sample)[-_]?(\d+)", str(text), re.I)
        if m:
            return f"{m.group(1).lower()}-{m.group(2)}"
        return None

    def _extract_session_ids(self, text):
        session_ids = []
        if not text:
            return session_ids
        for m in re.finditer(r"session[_\s-]?(\d+)", str(text), re.I):
            session_ids.append(int(m.group(1)))
        for m in re.finditer(r"\bD(\d+):(\d+)\b", str(text)):
            session_ids.append(int(m.group(1)))
        return sorted(set(session_ids))

    def _contains_time_cue(self, text):
        t = str(text).lower()
        cues = ["yesterday","today","tomorrow","last week","last month","last year","monday","tuesday","wednesday","thursday","friday","saturday","sunday","january","february","march","april","may","june","july","august","september","october","november","december"]
        return any(c in t for c in cues) or re.search(r"\b\d{4}\b", t) is not None

    def _extract_session_datetime_value(self, passage_text):
        m = re.search(r"\[session_date_time=([^\]]+)\]", str(passage_text or ""))
        return m.group(1).strip() if m else ""

    def _extract_subject_terms(self, text):
        tokens = re.findall(r"\b[A-Z][a-z]+\b", str(text))
        common = {"When","What","Where","Who","Why","How","Did","Does","Is","Are","The","A","An","And","Or","But","If","Then"}
        return [tok for tok in tokens if tok not in common]

    def _normalize_subject_name(self, name):
        n = str(name or "").strip()
        if not n:
            return ""
        return re.sub(r"\s+", " ", n).title()

    def _detect_primary_subjects(self, question, query_profile=None):
        if isinstance(query_profile, dict) and query_profile.get("primary_subjects"):
            return [self._normalize_subject_name(s) for s in query_profile.get("primary_subjects", []) if self._normalize_subject_name(s)]
        subjects = []
        for tok in self._extract_subject_terms(question):
            nm = self._normalize_subject_name(tok)
            if nm and nm not in subjects:
                subjects.append(nm)
        return subjects[:2]

    def _extract_explicit_speaker_name(self, text):
        txt = str(text or "").strip()
        m = re.match(r"^([A-Z][a-z]+)\s*:\s*", txt)
        return self._normalize_subject_name(m.group(1)) if m else ""

    def _extract_person_mentions(self, text):
        txt = str(text or "")
        tokens = re.findall(r"\b[A-Z][a-z]+\b", txt)
        stop = {"When","What","Where","Who","Why","How","The","A","An","And","Or","But","If","Then","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday","January","February","March","April","May","June","July","August","September","October","November","December"}
        out = []
        for tok in tokens:
            if tok in stop:
                continue
            nm = self._normalize_subject_name(tok)
            if nm and nm not in out:
                out.append(nm)
        return out

    def _has_conflicting_subject(self, question, query_profile, turn):
        targets = self._detect_primary_subjects(question, query_profile=query_profile)
        if not targets:
            return False
        txt = str((turn or {}).get("text", ""))
        mentions = set(self._extract_person_mentions(txt))
        speaker = self._normalize_subject_name((turn or {}).get("speaker", ""))
        explicit_speaker = self._extract_explicit_speaker_name(txt)
        if speaker:
            mentions.add(speaker)
        if explicit_speaker:
            mentions.add(explicit_speaker)
        targets_set = set(targets)
        if mentions & targets_set:
            return False
        other_people = {m for m in mentions if m and m not in targets_set}
        return bool(other_people)

    def _subject_gate_score(self, question, query_profile, turn):
        targets = self._detect_primary_subjects(question, query_profile=query_profile)
        if not targets:
            return 0.0
        txt = str((turn or {}).get("text", ""))
        mentions = set(self._extract_person_mentions(txt))
        speaker = self._normalize_subject_name((turn or {}).get("speaker", ""))
        explicit_speaker = self._extract_explicit_speaker_name(txt)
        if speaker:
            mentions.add(speaker)
        if explicit_speaker:
            mentions.add(explicit_speaker)
        targets_set = set(targets)
        if mentions & targets_set:
            return 1.0
        if mentions:
            return -1.0
        if re.search(r"\b(I|my|me|we|our|us)\b", txt, re.I):
            return 0.15
        return 0.0

    def _normalize_dialogue_text(self, text):
        txt = str(text or "").strip().lower()
        txt = re.sub(r"^([A-Z][a-z]+)\s*:\s*", "", txt)
        txt = re.sub(r"\[.*?\]", " ", txt)
        txt = re.sub(r"\s+", " ", txt)
        return txt.strip()

    def _extract_dialogue_segments(self, body):
        text = str(body or "")
        if not text:
            return []
        normalized = text.replace("\r", "\n")
        pattern = re.compile(r"(?:^|\n)\s*(D\d+:\d+)\s+([A-Z][a-z]+)\s*:\s*(.*?)(?=(?:\n\s*D\d+:\d+\s+[A-Z][a-z]+\s*:)|$)", re.S)
        segments = []
        for idx, m in enumerate(pattern.finditer(normalized)):
            dia_id = m.group(1).strip()
            speaker = m.group(2).strip()
            seg_text = m.group(3).strip()
            if seg_text:
                segments.append({
                    "dia_id": dia_id,
                    "speaker": speaker,
                    "text": seg_text,
                    "order": idx,
                    "norm_text": self._normalize_dialogue_text(seg_text),
                })
        return segments

    def _fix_turn_dia_id(self, text_part, body, speaker=None, hint_index=None):
        segments = self._extract_dialogue_segments(body)
        if not segments:
            return ""
        target = self._normalize_dialogue_text(text_part)
        if not target:
            return ""
        target_tokens = set(re.findall(r"\b[a-zA-Z']+\b", target))
        norm_speaker = self._normalize_subject_name(speaker)
        best_score, best_id = -1e9, ""
        for seg in segments:
            seg_text = str(seg.get("norm_text", ""))
            seg_speaker = self._normalize_subject_name(seg.get("speaker", ""))
            score = 0.0
            if norm_speaker and seg_speaker == norm_speaker:
                score += 1.5
            elif norm_speaker and seg_speaker and seg_speaker != norm_speaker:
                score -= 0.5
            if seg_text == target:
                score += 8.0
            elif target and target in seg_text:
                score += 6.0
            elif seg_text and seg_text in target:
                score += 4.0
            seg_tokens = set(re.findall(r"\b[a-zA-Z']+\b", seg_text))
            if target_tokens and seg_tokens:
                score += 3.0 * (len(target_tokens & seg_tokens) / max(1.0, len(target_tokens | seg_tokens)))
            if hint_index is not None and seg.get("order") is not None:
                distance = abs(int(seg.get("order")) - int(hint_index))
                score += max(0.0, 1.0 - 0.15 * distance)
            if score > best_score:
                best_score, best_id = score, str(seg.get("dia_id", ""))
        return best_id if best_score >= 2.0 else ""

    def _extract_keywords(self, text, top_k=None):
        counts = Counter(self._normalize_bm25_tokens(text))
        return [word for word, _ in counts.most_common(top_k or len(counts))]

    def _evidence_sample_id(self, text, meta=None):
        if isinstance(meta, dict):
            sample_id = str(meta.get("sample_id", "") or "").strip()
            if sample_id:
                return sample_id
        return self._extract_sample_id(text) or ""

    def _evidence_speaker_names(self, text, meta=None):
        names = []
        if isinstance(meta, dict):
            speaker = self._normalize_subject_name(meta.get("speaker", ""))
            if speaker:
                names.append(speaker)
        txt = str(text or "")
        for m in re.finditer(r"(?:^|\n|\|)\s*(?:D\d+:\d+\s+)?([A-Z][a-z]+)\s*:", txt):
            nm = self._normalize_subject_name(m.group(1))
            if nm and nm not in names:
                names.append(nm)
        explicit = self._extract_explicit_speaker_name(txt)
        if explicit and explicit not in names:
            names.append(explicit)
        return names

    def _target_subject_set(self, question, query_profile=None):
        return {s for s in self._detect_primary_subjects(question, query_profile=query_profile) if s}

    def _evidence_subject_binding_score(self, question, query_profile, text, meta=None):
        txt = str(text or "")
        if not txt.strip():
            return -10.0
        targets = self._target_subject_set(question, query_profile=query_profile)
        q_sample = str((query_profile or {}).get("sample_id", "") or "").strip()
        e_sample = self._evidence_sample_id(txt, meta=meta)
        speakers = self._evidence_speaker_names(txt, meta=meta)
        mentions = set(self._extract_person_mentions(txt))
        first_person = bool(re.search(r"\b(I|I'm|I am|I've|I'd|I'll|my|me|mine|we|our)\b", txt, re.I))
        lower = txt.lower()
        score = 0.0
        if q_sample and e_sample:
            score += 1.4 if q_sample == e_sample else -4.0
        elif q_sample:
            score -= 0.2
        if targets:
            speaker_hit = any(sp in targets for sp in speakers)
            mention_hit = bool(mentions & targets)
            if speaker_hit:
                score += 1.4
            if mention_hit:
                score += 0.8
            if first_person and speaker_hit:
                score += 0.6
            if speakers and not speaker_hit and first_person and not mention_hit:
                score -= 0.9
            if speakers and not speaker_hit and not mention_hit:
                score -= 0.25
        focus_hits = len(self._focus_overlap_terms(question, txt, query_profile=query_profile or {}))
        theme_hits = len(self._theme_overlap_terms(question, txt, query_profile=query_profile or {}))
        score += 0.08 * min(focus_hits, 4)
        score += 0.05 * min(theme_hits, 4)
        if self._is_low_value_turn(lower):
            score -= 0.25
        return score
