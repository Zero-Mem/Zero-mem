"""Evidence scoring, filtering, reranking and non-generative completion."""

from .utils import CrossEncoder
from .utils import logger, safe_error_summary
from .utils import min_max_normalize
import numpy as np
import re
import torch


class EvidenceMixin:
    def _normalize_bm25_tokens(self, text):
        words = re.findall(r"\b[a-zA-Z][a-zA-Z\-']+\b", str(text).lower())
        stop = {"the","a","an","and","or","but","if","then","to","of","in","on","at","for","with","from","by","is","are","was","were","be","been","being","it","this","that","these","those","he","she","they","them","his","her","their","as","about","into","over","after","before","during","what","when","where","who","why","how","did","does","do"}
        return [w for w in words if len(w) > 2 and w not in stop]

    def _extract_query_phrases(self, question, max_phrases=6):
        toks = self._normalize_bm25_tokens(question)
        phrases = []
        for n in (3, 2):
            for i in range(len(toks) - n + 1):
                phrase = " ".join(toks[i:i+n])
                if phrase not in phrases:
                    phrases.append(phrase)
        return phrases[:max_phrases]

    def _extract_theme_terms(self, question, query_profile=None):
        toks = self._normalize_bm25_tokens(question)
        generic = {
            "go","went","gone","going","did","does","done","make","made","making","have","has","had",
            "take","took","taking","attend","attended","attending","plans","plan","think","thinks","thinking",
            "decision","decide","decided","choose","chose","choosing","excited","kind","type","types",
            "long","much","many","more","most","recent","recently","currently","really","about",
            "based","conversation","according","considering","would","could","should","might","likely",
        }
        subject_terms = {str(t).lower() for t in (query_profile or {}).get("subject_terms", [])}
        out = []
        for tok in toks:
            if tok in generic:
                continue
            if tok not in out:
                out.append(tok)
        # keep subject terms available for light gating, but not as the only focus signal
        for subj in subject_terms:
            if subj and subj not in out:
                out.insert(0, subj)
        return out[:10]

    def _extract_focus_terms(self, question, query_profile=None):
        theme_terms = self._extract_theme_terms(question, query_profile=query_profile)
        subject_terms = {str(t).lower() for t in (query_profile or {}).get("subject_terms", [])}
        return [t for t in theme_terms if t not in subject_terms]

    def _theme_overlap_terms(self, question, text, query_profile=None):
        txt = str(text or "").lower()
        return [t for t in self._extract_theme_terms(question, query_profile=query_profile) if t and t in txt]

    def _focus_overlap_terms(self, question, text, query_profile=None):
        txt = str(text or "").lower()
        return [t for t in self._extract_focus_terms(question, query_profile=query_profile) if t and t in txt]

    def _is_low_value_turn(self, text):
        txt = str(text or "").strip().lower()
        toks = re.findall(r"[a-zA-Z]+", txt)
        if not toks:
            return True
        reaction_patterns = [
            r"^(yeah|yay|yep|yup|ok|okay|cool|nice|great|awesome|true|right|sure|exactly|totally)[!. ]*$",
            r"^(it was|that was)\s+(great|awesome|amazing|nice|cool|inspiring|fun|special|wonderful)[!. ]*$",
            r"^(sounds|seems)\s+(great|nice|good|awesome)[!. ]*$",
            r"^thanks[,! ]",
        ]
        if any(re.match(p, txt) for p in reaction_patterns):
            return True
        if self._is_supportive_reply_turn(txt):
            return True
        if len(toks) <= 3:
            return True
        return False

    def _is_question_like_turn(self, text):
        txt = str(text or "").strip()
        if not txt:
            return True
        if txt.endswith("?"):
            return True
        txt_lower = txt.lower()
        question_starts = (
            "what ", "when ", "where ", "who ", "why ", "how ",
            "did ", "do ", "does ", "is ", "are ", "was ", "were ", "can ", "could ",
            "would ", "will ", "have ", "has ", "had "
        )
        return txt_lower.startswith(question_starts)

    def _is_session_block_text(self, text):
        txt = str(text or "")
        if not txt:
            return False
        if "[sample_id=" in txt and "[session=" in txt:
            return True
        if len(re.findall(r"\bD\d+:\d+\b", txt)) >= 4:
            return True
        if txt.count("\n") >= 4 and len(re.findall(r"\bD\d+:\d+\b", txt)) >= 2:
            return True
        return False

    def _has_answer_like_payload(self, text):
        txt = str(text or "").strip()
        if not txt:
            return False
        if self._is_session_block_text(txt):
            return False
        if self._is_question_like_turn(txt):
            return False
        toks = re.findall(r"[A-Za-z0-9']+", txt)
        return len(toks) >= int(self.config.hierarchy_min_answer_tokens)

    def _contains_duration_cue(self, text):
        txt = str(text or "").lower()
        if not txt:
            return False
        duration_patterns = (
            r"\b\d+\s+(?:year|month|week|day|hour|minute)s?\b",
            r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:year|month|week|day)s?\b",
            r"\b(?:since|for)\s+\d{4}\b",
            r"\b(?:since|for)\s+\d+\s+(?:year|month|week|day)s?\b",
            r"\b\d+\s+years?\s+ago\b",
            r"\b(?:last|past)\s+\d+\s+(?:year|month|week|day)s?\b",
        )
        return any(re.search(p, txt) for p in duration_patterns)

    def _contains_location_cue(self, text):
        txt = str(text or "").strip()
        if not txt:
            return False
        txt_lower = f" {txt.lower()} "
        loc_markers = (
            " in ", " at ", " from ", " to ", " near ", " by ", " around ",
            " inside ", " outside ", " downtown ", " uptown ", " abroad ",
            " overseas ", " on the coast ", " in the mountains ", " at the beach ",
            " in the city ", " in town ", " moved from ", " moved to ",
            " lives in ", " live in ", " stayed in ", " camping in "
        )
        if any(marker in txt_lower for marker in loc_markers):
            return True
        place_suffixes = (
            "park", "beach", "forest", "mountains", "museum", "cafe", "campground",
            "lake", "river", "valley", "village", "town", "city", "country", "state"
        )
        return bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", txt)) or any(
            txt_lower.count(f" {suffix} ") > 0 for suffix in place_suffixes
        )

    def _location_cue_score(self, text):
        txt = str(text or "").strip()
        if not txt:
            return 0.0
        txt_lower = f" {txt.lower()} "
        loc_markers = (
            " in ", " at ", " from ", " to ", " near ", " by ", " around ",
            " inside ", " outside ", " moved from ", " moved to ",
            " lives in ", " live in ", " stayed in ", " camping in "
        )
        score = float(sum(1 for marker in loc_markers if marker in txt_lower))
        if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", txt):
            score += 0.5
        return score

    def _where_question_mode(self, question, query_profile=None):
        profile = query_profile or {}
        cached = str(profile.get("where_mode", "")).strip()
        if cached:
            return cached
        q = str(question or "").strip().lower()
        if not q.startswith("where "):
            return ""
        if any(marker in q for marker in ("move from", "moved from", "come from", "came from", "originally from")):
            return "where_origin"
        if any(marker in q for marker in ("where has", "camped", "camping", "gone camping", "went camping", "hiked", "traveled", "travelled", "visited", "stayed")):
            return "where_list"
        return "where"

    def _where_anchor_terms(self, question, query_profile=None):
        q = str(question or "").strip().lower()
        mode = self._where_question_mode(question, query_profile=query_profile)
        anchors = []
        seed_terms = []
        if mode == "where_origin":
            seed_terms.extend(["move", "moved", "from", "come", "came", "originally"])
        elif mode == "where_list":
            seed_terms.extend(["camp", "camped", "camping", "went camping", "hike", "hiked", "travel", "traveled", "travelled", "visited", "stayed"])
        else:
            seed_terms.extend(["in", "at", "from", "near"])
        seed_terms.extend(self._extract_focus_terms(question, query_profile=query_profile))
        for term in seed_terms:
            t = str(term or "").lower().strip()
            if not t or t in anchors:
                continue
            if t in q or len(t) > 3:
                anchors.append(t)
        return anchors[:6]

    def _where_location_term_hits(self, text, where_mode="where"):
        txt = str(text or "").strip()
        if not txt:
            return 0.0
        txt_lower = f" {txt.lower()} "
        place_terms = (
            " beach ", " beaches ", " mountain ", " mountains ", " forest ", " forests ",
            " park ", " parks ", " campground ", " campgrounds ", " campsite ", " campsites ",
            " lake ", " lakes ", " river ", " rivers ", " valley ", " valleys ",
            " village ", " villages ", " town ", " towns ", " city ", " cities ",
            " country ", " countries ", " coast ", " coastline ", " trail ", " trails "
        )
        hits = float(sum(1 for term in place_terms if term in txt_lower))
        if re.search(r"\b(?:in|at|from|near|by|to)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", txt):
            hits += 1.0
        if where_mode == "where_origin" and re.search(r"\bfrom\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", txt):
            hits += 1.0
        return hits

    def _where_pattern_score(self, text, where_mode="where"):
        txt = str(text or "").strip()
        if not txt:
            return 0.0
        txt_lower = f" {txt.lower()} "
        score = 0.0
        if where_mode == "where_origin":
            if any(marker in txt_lower for marker in (" moved from ", " move from ", " came from ", " come from ", " originally from ")):
                score += 2.0
            if re.search(r"\bfrom\s+(?:the\s+)?(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}|[a-z]+(?:\s+[a-z]+){0,2})\b", txt):
                score += 1.2
            if re.search(r"\b(?:moved|move|came|come)\b", txt_lower) and self._location_cue_score(txt) > 0:
                score += 0.8
            if " moved to " in txt_lower and " moved from " not in txt_lower:
                score -= 0.25
        elif where_mode == "where_list":
            action_markers = (
                " camped ", " camping ", " went camping ", " gone camping ",
                " stayed in ", " stayed at ", " hiked ", " hiking ",
                " traveled ", " travelled ", " visited "
            )
            score += float(sum(1 for marker in action_markers if marker in txt_lower)) * 0.65
            if re.search(r"\b(?:camped|camping|went camping|gone camping|hiked|visited|stayed|traveled|travelled)\b.*\b(?:in|at|near|by|on)\b", txt_lower):
                score += 1.0
            if self._enumeration_shape_score(txt) > 0 and self._location_cue_score(txt) > 0:
                score += 0.55
        else:
            if re.search(r"\b(?:in|at|from|near|to)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", txt):
                score += 0.9
            if self._location_cue_score(txt) > 0:
                score += 0.4
        return max(score, 0.0)

    def _where_noise_penalty(self, text, where_mode="where"):
        txt = str(text or "").strip()
        if not txt:
            return 0.0
        txt_lower = f" {txt.lower()} "
        pattern_score = self._where_pattern_score(txt, where_mode)
        location_score = self._location_cue_score(txt)
        penalty = 0.0
        generic_noise = (" home ", " house ", " houses ", " apartment ", " neighborhood ", " area ", " place ", " spot ")
        if location_score > 0 and pattern_score <= 0.0:
            if where_mode == "where_origin":
                penalty += float(self.config.hierarchy_where_noise_penalty) + 0.08
            elif where_mode == "where_list":
                penalty += float(self.config.hierarchy_where_noise_penalty)
        if any(term in txt_lower for term in generic_noise) and self._where_location_term_hits(txt, where_mode) <= 0:
            penalty += 0.12
        return penalty

    def _enumeration_shape_score(self, text):
        txt = str(text or "").strip()
        if not txt:
            return 0.0
        txt_lower = txt.lower()
        score = 0.0
        if "," in txt:
            score += 1.0
        if " and " in txt_lower or " or " in txt_lower:
            score += 0.8
        if len(re.findall(r"\b(?:also|plus|including|such as)\b", txt_lower)) > 0:
            score += 0.6
        if len(re.findall(r"\b[A-Za-z][A-Za-z'-]+\b", txt)) >= 10:
            score += 0.25
        return score

    def _static_slot_cue_score(self, text, slot_type):
        txt = str(text or "").strip()
        if not txt:
            return 0.0
        txt_lower = f" {txt.lower()} "
        score = 0.0
        for term in self._static_slot_terms(slot_type):
            if term in txt_lower:
                score += 1.0
        if slot_type in {"relationship", "identity"} and (" is " in txt_lower or " identifies as " in txt_lower):
            score += 0.6
        if slot_type in {"fields", "career"} and re.search(r"\b(?:study|studying|pursu\w+|career|degree|certification|program)\b", txt_lower):
            score += 0.6
        if slot_type in {"art", "instruments", "symbols"} and self._enumeration_shape_score(txt) > 0:
            score += 0.3
        return score

    def _is_short_fact_turn(self, text):
        txt = str(text or "").strip()
        if not txt or self._is_question_like_turn(txt):
            return False
        token_count = len(re.findall(r"[A-Za-z0-9']+", txt))
        return 3 <= token_count <= 22

    def _speaker_matches_target_subject(self, question, query_profile, turn):
        targets = set(self._detect_primary_subjects(question, query_profile=query_profile))
        if not targets:
            return False
        speaker = self._normalize_subject_name((turn or {}).get("speaker", ""))
        explicit_speaker = self._extract_explicit_speaker_name((turn or {}).get("text", ""))
        return bool((speaker and speaker in targets) or (explicit_speaker and explicit_speaker in targets))

    def _static_slot_alignment_score(self, question, query_profile, turn, slot_type):
        txt = str((turn or {}).get("text", "")).strip()
        if not txt:
            return -1.0
        txt_lower = txt.lower()
        targets = set(self._detect_primary_subjects(question, query_profile=query_profile))
        mentions = set(self._extract_person_mentions(txt))
        explicit_subject = bool(mentions & targets) if targets else False
        speaker_match = self._speaker_matches_target_subject(question, query_profile, turn)
        first_person = bool(re.search(r"\b(I|I'm|I am|I've|my|me)\b", txt, re.I))
        slot = str(slot_type or "").strip().lower()
        score = 0.0
        if explicit_subject:
            score += 1.0
        if speaker_match:
            score += 0.8
        if first_person and speaker_match:
            score += 0.6 if slot in {"relationship", "identity"} else 0.25
        if slot in {"relationship", "identity"}:
            rel_terms = (" husband ", " wife ", " partner ", " boyfriend ", " girlfriend ", " spouse ", " single ", " married ")
            if any(term in f" {txt_lower} " for term in rel_terms) and not explicit_subject and not speaker_match:
                score -= 1.0
            if first_person and not explicit_subject and not speaker_match:
                score -= 1.0
        elif slot in {"fields", "career"}:
            if not explicit_subject and not speaker_match and not first_person:
                score -= 0.45
        return score

    def _is_generic_static_support_turn(self, text):
        txt = str(text or "").strip().lower()
        if not txt:
            return False
        supportive = bool(re.search(
            r"\b(?:glad|proud|kind|caring|support|encourag|inspir|difference|means? a lot)\b",
            txt,
        ))
        second_person = bool(re.search(r"\b(?:you|your|you're|you've)\b", txt))
        factual_slot = bool(re.search(
            r"\b(?:is|am|are|identif(?:y|ies)|works? as|career|degree|major(?:ed)?|"
            r"stud(?:y|ies|ied)|believes?|member|partner|spouse|husband|wife)\b",
            txt,
        ))
        return supportive and second_person and not factual_slot

    def _event_anchor_terms(self, question, query_profile=None):
        generic = {
            "go","went","gone","going","did","does","done","have","has","had","make","made","making",
            "when","what","where","who","why","how","long","ago","recently","current","current",
            "would","likely","still","really","kind","type","types","some","many","much"
        }
        anchors = []
        for term in self._extract_focus_terms(question, query_profile=query_profile):
            t = str(term or "").lower().strip()
            if not t or len(t) <= 2 or t in generic:
                continue
            if t not in anchors:
                anchors.append(t)
        qtype = self._effective_question_type(question, query_profile=query_profile)
        if qtype in {"where", "where_origin", "where_list"}:
            for term in self._where_anchor_terms(question, query_profile=query_profile):
                if term and term not in anchors:
                    anchors.append(term)
        return anchors[:4]

    def _event_anchor_hit(self, question, text, query_profile=None):
        txt = str(text or "").lower()
        if not txt:
            return 0
        return sum(1 for a in self._event_anchor_terms(question, query_profile=query_profile) if a and a in txt)

    def _question_needs_strict_anchor(self, question):
        q = str(question or "").strip().lower()
        if not q:
            return False
        if q.startswith(("when ", "where ", "how long ")):
            return True
        if self._is_static_attribute_question(q):
            return False
        event_phrases = (
            "joined", "applied", "passed", "attend", "attended", "meeting", "workshop",
            "speech", "support group", "conference", "parade", "picnic", "research",
            "researched", "read the book", "biking", "camping", "museum", "park"
        )
        return any(p in q for p in event_phrases)

    def _is_supportive_reply_turn(self, text):
        txt = str(text or "").strip().lower()
        if not txt:
            return True
        supportive_patterns = [
            r"^congrats\b",
            r"^wow[,! ]",
            r"^awesome[,! ]",
            r"^sounds (?:great|awesome|wonderful|special)\b",
            r"^glad to hear\b",
            r"^good on you\b",
            r"^keep it up\b",
            r"^proud of you\b",
            r"^that(?:'s| is) awesome\b",
            r"^that(?:'s| is) great\b",
            r"^so glad\b",
        ]
        if any(re.match(p, txt) for p in supportive_patterns):
            return True
        return bool(
            re.search(r"\b(?:you|your|you're|you've)\b", txt)
            and re.search(
                r"\b(?:proud|glad|awesome|great|amazing|support|encourag|impact|difference|"
                r"journey|kind words|well done)\b",
                txt,
            )
        )

    def _is_first_person_experience_turn(self, text):
        txt = str(text or "")
        return bool(re.search(r"\b(I|I've|I\'ve|I\'m|I am|My|We|We\'ve|We\'re)\b", txt, re.I))

    def _answer_shape_score(self, question, query_profile, turn):
        txt = str((turn or {}).get("text", "")).strip()
        if not txt:
            return -1.0
        txt_lower = txt.lower()
        qtype = self._effective_question_type(question, query_profile=query_profile)
        score = 0.0
        if self._is_question_like_turn(txt):
            score -= 1.2
        if self._is_low_value_turn(txt):
            score -= 1.0
        if self._is_supportive_reply_turn(txt):
            score -= 0.8
        if self._is_first_person_experience_turn(txt):
            score += 0.35
        if self._exact_phrase_bonus(question, txt) > 0:
            score += 0.45
        focus_hits = self._focus_overlap_terms(question, txt, query_profile=query_profile)
        theme_hits = self._theme_overlap_terms(question, txt, query_profile=query_profile)
        subject_hits = [t for t in (query_profile or {}).get("subject_terms", []) if str(t).lower() in txt_lower]
        location_score = self._location_cue_score(txt)
        enumeration_score = self._enumeration_shape_score(txt)
        where_mode = self._where_question_mode(question, query_profile=query_profile) or "where"
        where_pattern = self._where_pattern_score(txt, where_mode)
        where_terms = self._where_location_term_hits(txt, where_mode)
        where_noise = self._where_noise_penalty(txt, where_mode)
        score += 0.20 * len(focus_hits) + 0.08 * len(theme_hits)
        score += float(self.config.hierarchy_static_subject_bonus_weight) * min(len(subject_hits), 2)
        anchor_hits = self._event_anchor_hit(question, txt, query_profile=query_profile)
        score += 0.35 * anchor_hits
        if qtype == "when":
            has_time = self._contains_time_cue(txt) or bool(re.search(r"\b\d{4}\b", txt)) or bool(str((turn or {}).get("session_time", "")).strip())
            if has_time:
                score += 0.55
            if anchor_hits == 0:
                score -= 0.45
        elif qtype == "duration":
            has_duration = self._contains_duration_cue(txt) or bool(re.search(r"\b\d{4}\b", txt)) or bool(str((turn or {}).get("session_time", "")).strip())
            if has_duration:
                score += 0.60
            if anchor_hits == 0 and len(focus_hits) == 0:
                score -= 0.30
        elif qtype == "where_origin":
            score += float(self.config.hierarchy_where_pattern_bonus_weight) * where_pattern
            score += float(self.config.hierarchy_where_location_term_weight) * min(where_terms, 3.0)
            if location_score > 0:
                score += float(self.config.hierarchy_location_bonus_weight) * location_score
            if where_pattern == 0.0 and len(focus_hits) == 0:
                score -= 0.35
        elif qtype == "where_list":
            score += float(self.config.hierarchy_where_pattern_bonus_weight) * where_pattern
            score += float(self.config.hierarchy_where_location_term_weight) * min(where_terms, 3.0)
            score += 0.10 * enumeration_score
            if location_score > 0:
                score += float(self.config.hierarchy_location_bonus_weight) * location_score
            if where_pattern == 0.0 and location_score == 0.0:
                score -= 0.30
        elif qtype == "where":
            if location_score > 0:
                score += float(self.config.hierarchy_location_bonus_weight) * location_score
            score += 0.18 * where_pattern
            if len(focus_hits) == 0 and len(subject_hits) == 0:
                score -= 0.25
        elif qtype == "who":
            if bool(re.search(r"\b[A-Z][a-z]+\b", txt)):
                score += 0.35
        elif qtype in {"why", "how"}:
            if re.search(r"\b(because|since|so that|in order to|by|through|wanted to|decided to|trying to)\b", txt_lower):
                score += 0.45
        elif qtype == "what_static":
            slot_type = self._static_slot_type(question, query_profile=query_profile)
            slot_score = self._static_slot_cue_score(txt, slot_type)
            direct_fact = self._is_short_fact_turn(txt)
            alignment_score = self._static_slot_alignment_score(question, query_profile, turn, slot_type)
            if subject_hits:
                score += 0.18 * min(len(subject_hits), 2)
            score += 0.24 * min(slot_score, 3.0)
            score += 0.22 * alignment_score
            if direct_fact:
                score += 0.18
            if " is " in f" {txt_lower} " or " are " in f" {txt_lower} ":
                score += 0.18
            if slot_type in {"relationship", "identity"} and not subject_hits:
                score -= 0.25
            if slot_type in {"relationship", "identity"} and not self._speaker_matches_target_subject(question, query_profile, turn):
                if re.search(r"\b(my|i'm|i am|me)\b", txt, re.I):
                    score -= 0.55
            if slot_type in {"fields", "career"} and self._is_generic_static_support_turn(txt):
                score -= 0.85
            if slot_score <= 0.0 and len(focus_hits) == 0:
                score -= 0.35
            if slot_type in {"fields", "career", "relationship", "identity"} and len(re.findall(r"[A-Za-z0-9']+", txt)) > 28:
                score -= 0.12
        elif qtype == "what_list":
            score += float(self.config.hierarchy_list_bonus_weight) * enumeration_score
            if enumeration_score == 0.0 and len(focus_hits) == 0:
                score -= 0.22
        else:
            if enumeration_score > 0:
                score += 0.10 * enumeration_score
        if self._has_conflicting_subject(question, query_profile, turn):
            score -= 1.0
        score -= where_noise
        score -= self._generic_cluster_penalty(question, txt, query_profile=query_profile)
        return score

    def _low_info_penalty(self, text):
        txt = str(text or "").strip().lower()
        toks = re.findall(r"[a-zA-Z]+", txt)
        if txt in {"yay", "yeah", "okay", "ok", "nice", "cool", "great", "awesome"}:
            return float(self.config.hierarchy_reply_penalty)
        if len(toks) <= 2:
            return float(self.config.hierarchy_low_info_penalty)
        return 0.0

    def _exact_phrase_bonus(self, question, text):
        txt = str(text or "").lower()
        if not txt:
            return 0.0
        phrases = self._extract_query_phrases(question, max_phrases=8)
        hit_count = sum(1 for ph in phrases if ph and ph in txt)
        return hit_count * float(self.config.hierarchy_exact_phrase_bonus_weight)

    def _passes_answer_slot_filter(self, question, query_profile, turn_text, turn=None):
        txt = str(turn_text or "").strip()
        if not txt:
            return False
        if self._is_session_block_text(txt):
            return False
        if not self._has_answer_like_payload(txt):
            return False

        qtype = self._effective_question_type(question, query_profile=query_profile)
        txt_lower = txt.lower()
        pseudo_turn = dict(turn or {})
        pseudo_turn["text"] = txt
        pseudo_turn.setdefault("speaker", "")
        if self._has_conflicting_subject(question, query_profile, pseudo_turn):
            return False
        focus_hits = self._focus_overlap_terms(question, txt, query_profile=query_profile)
        theme_hits = self._theme_overlap_terms(question, txt, query_profile=query_profile)
        subject_hits = [t for t in (query_profile or {}).get("subject_terms", []) if str(t).lower() in txt_lower]

        # time questions need explicit or relative time evidence
        if qtype == "when":
            if self._contains_time_cue(txt):
                return True
            if re.search(r"\b\d{4}\b", txt) or re.search(r"\b(?:am|pm)\b", txt_lower):
                return True
            return False
        if qtype == "duration":
            if self._contains_duration_cue(txt):
                return True
            if self._contains_time_cue(txt) and (re.search(r"\bago\b", txt_lower) or "since " in txt_lower or "for " in txt_lower):
                return True
            return False

        # location questions should look locative
        if qtype == "where_origin":
            where_pattern = self._where_pattern_score(txt, "where_origin")
            if where_pattern >= 1.0:
                return True
            return self._location_cue_score(txt) >= 0.5 and bool(re.search(r"\b(?:move|moved|come|came|from)\b", txt_lower))
        if qtype == "where_list":
            where_pattern = self._where_pattern_score(txt, "where_list")
            if where_pattern >= 1.0:
                return True
            if self._enumeration_shape_score(txt) >= 0.5 and self._location_cue_score(txt) >= 0.5:
                return True
            return self._contains_location_cue(txt) and (len(focus_hits) >= 1 or len(subject_hits) >= 1)
        if qtype == "where":
            if self._contains_location_cue(txt):
                return True
            return len(focus_hits) >= 1 and self._location_cue_score(txt) >= 0.5

        # people/entity questions should mention a likely answer entity
        if qtype == "who":
            if bool(re.search(r"\b[A-Z][a-z]+\b", txt)):
                return True
            kin_markers = (" husband", " wife", " grandma", " mother", " father", " sister", " brother", " friend", " pet", " dog", " cat")
            return any(m in f" {txt_lower} " for m in kin_markers)

        # why/how questions should carry explanatory/process content
        if qtype in {"why", "how"}:
            markers = (" because ", " since ", " so that ", " in order to ", " by ", " through ", " so ", " decided to ", " wanted to ", " started to ", " trying to ")
            return any(m in f" {txt_lower} " for m in markers) or len(re.findall(r"[A-Za-z]+", txt)) >= 8

        if qtype == "what_static":
            slot_type = self._static_slot_type(question, query_profile=query_profile)
            slot_score = self._static_slot_cue_score(txt, slot_type)
            direct_fact = self._is_short_fact_turn(txt)
            alignment_score = self._static_slot_alignment_score(question, query_profile, pseudo_turn, slot_type)
            if slot_type in {"relationship", "identity"}:
                if slot_score >= 0.5 and alignment_score >= 0.8:
                    return True
                return bool(subject_hits) and slot_score >= 0.8 and alignment_score >= 0.3
            if slot_type in {"fields", "career"}:
                if self._is_generic_static_support_turn(txt):
                    return False
                if slot_score >= 0.8 and (alignment_score >= 0.3 or direct_fact):
                    return True
                return False
            if subject_hits and slot_score >= 0.8:
                return True
            if subject_hits and direct_fact and (focus_hits or theme_hits):
                return True
            return bool(subject_hits or focus_hits) and slot_score >= 0.6

        if qtype in {"what_list", "where_list"}:
            if self._enumeration_shape_score(txt) >= 0.8:
                return True
            if focus_hits or subject_hits or len(theme_hits) >= 1:
                return True
            return False

        # what_event/other questions: require real payload and at least one focus/thematic overlap or enumerative content
        if focus_hits or subject_hits or len(theme_hits) >= 1:
            return True
        if "," in txt or " and " in txt_lower:
            return True
        return len(re.findall(r"[A-Za-z]+", txt)) >= 8

    def _passes_relaxed_answer_slot_filter(self, question, query_profile, turn_text, turn=None):
        txt = str(turn_text or "").strip()
        if not txt:
            return False
        if self._is_session_block_text(txt):
            return False
        if self._is_question_like_turn(txt):
            return False
        if not self._has_answer_like_payload(txt):
            return False

        qtype = self._effective_question_type(question, query_profile=query_profile)
        txt_lower = txt.lower()
        pseudo_turn = dict(turn or {})
        pseudo_turn["text"] = txt
        pseudo_turn.setdefault("speaker", "")
        if self._has_conflicting_subject(question, query_profile, pseudo_turn):
            return False
        focus_hits = self._focus_overlap_terms(question, txt, query_profile=query_profile)
        theme_hits = self._theme_overlap_terms(question, txt, query_profile=query_profile)
        subject_hits = [t for t in (query_profile or {}).get("subject_terms", []) if str(t).lower() in txt_lower]
        has_time = self._contains_time_cue(txt)

        if qtype == "when":
            return (len(focus_hits) >= 1 and has_time) or bool(re.search(r"\b\d{4}\b", txt))
        if qtype == "duration":
            return len(focus_hits) >= 1 and (self._contains_duration_cue(txt) or has_time or bool(re.search(r"\b\d+\b", txt)))
        if qtype == "where_origin":
            return self._where_pattern_score(txt, "where_origin") >= 0.8 or (
                self._location_cue_score(txt) >= 0.5 and bool(re.search(r"\b(?:move|moved|come|came|from)\b", txt_lower))
            )
        if qtype == "where_list":
            return self._where_pattern_score(txt, "where_list") >= 0.6 or (
                self._contains_location_cue(txt) and (len(focus_hits) >= 1 or len(subject_hits) >= 1 or self._enumeration_shape_score(txt) >= 0.25)
            )
        if qtype == "where":
            return len(focus_hits) >= 1 or self._contains_location_cue(txt)
        if qtype == "who":
            return len(subject_hits) >= 1 or bool(re.search(r"\b[A-Z][a-z]+\b", txt))
        if qtype in {"why", "how"}:
            markers = (" because ", " since ", " so that ", " in order to ", " by ", " through ", " so ", " decided to ", " wanted to ", " started to ", " trying to ")
            return len(focus_hits) >= 1 or any(m in f" {txt_lower} " for m in markers)
        if qtype == "what_static":
            slot_type = self._static_slot_type(question, query_profile=query_profile)
            slot_score = self._static_slot_cue_score(txt, slot_type)
            alignment_score = self._static_slot_alignment_score(question, query_profile, pseudo_turn, slot_type)
            if slot_type in {"relationship", "identity"}:
                return slot_score >= 0.4 and alignment_score >= 0.5
            if slot_type in {"fields", "career"}:
                if self._is_generic_static_support_turn(txt):
                    return False
                return slot_score >= 0.6 and (alignment_score >= 0.2 or self._is_short_fact_turn(txt))
            return (slot_score >= 0.5 and (len(subject_hits) >= 1 or len(focus_hits) >= 1 or len(theme_hits) >= 1)) or (
                len(subject_hits) >= 1 and (len(focus_hits) >= 1 or len(theme_hits) >= 1)
            )
        if qtype in {"what_list", "where_list"}:
            return self._enumeration_shape_score(txt) >= 0.5 or len(focus_hits) >= 1 or len(subject_hits) >= 1
        return len(focus_hits) >= 1 or len(subject_hits) >= 1 or bool(re.search(r"\b\d{4}\b", txt))

    def _time_question_soft_candidates(self, question, query_profile, ranked_turn_items, top_turn_k):
        keep = []
        limit = min(int(self.config.hierarchy_time_soft_fallback_top_k), max(1, top_turn_k))
        for tid, _ in ranked_turn_items:
            if len(keep) >= limit:
                break
            turn = self.hierarchy_turn_units.get(tid, {})
            txt = str(turn.get("text", "")).strip()
            if not txt:
                continue
            if self._is_session_block_text(txt) or self._is_question_like_turn(txt) or self._is_low_value_turn(txt):
                continue
            if self._has_conflicting_subject(question, query_profile, turn):
                continue
            if not self._has_answer_like_payload(txt):
                continue
            txt_lower = txt.lower()
            focus_hits = self._focus_overlap_terms(question, txt, query_profile=query_profile)
            theme_hits = self._theme_overlap_terms(question, txt, query_profile=query_profile)
            target_subjects = self._detect_primary_subjects(question, query_profile=query_profile)
            subject_hits = [t for t in target_subjects if t.lower() in txt_lower]
            has_time = (
                self._contains_time_cue(txt)
                or self._contains_duration_cue(txt)
                or bool(re.search(r"\b\d{4}\b", txt))
                or bool(str(turn.get("session_time", "")).strip())
            )
            if not has_time:
                continue
            if focus_hits or subject_hits or len(theme_hits) >= 1:
                if tid not in keep:
                    keep.append(tid)
        return keep

    def _event_question_soft_candidates(self, question, query_profile, ranked_turn_items, top_turn_k):
        keep = []
        limit = min(int(self.config.hierarchy_event_soft_fallback_top_k), max(1, top_turn_k))
        for tid, _ in ranked_turn_items:
            if len(keep) >= limit:
                break
            turn = self.hierarchy_turn_units.get(tid, {})
            txt = str(turn.get("text", "")).strip()
            if not txt:
                continue
            if self._is_session_block_text(txt) or self._is_question_like_turn(txt) or self._is_low_value_turn(txt):
                continue
            if self._has_conflicting_subject(question, query_profile, turn):
                continue
            if not self._has_answer_like_payload(txt):
                continue
            focus_hits = self._focus_overlap_terms(question, txt, query_profile=query_profile)
            theme_hits = self._theme_overlap_terms(question, txt, query_profile=query_profile)
            anchor_hits = self._event_anchor_hit(question, txt, query_profile=query_profile)
            if anchor_hits <= 0 and len(focus_hits) == 0 and len(theme_hits) == 0:
                continue
            if not self._is_strong_theme_turn(question, query_profile, txt) and anchor_hits <= 0:
                continue
            if tid not in keep:
                keep.append(tid)
        return keep

    def _passes_hierarchy_theme_filter(self, question, query_profile, turn_text):
        txt = str(turn_text or "")
        txt_lower = txt.lower()
        qtype = self._effective_question_type(question, query_profile=query_profile)
        exact_bonus = self._exact_phrase_bonus(question, txt)
        if exact_bonus > 0:
            return True

        focus_hits = self._focus_overlap_terms(question, txt, query_profile=query_profile)
        theme_hits = self._theme_overlap_terms(question, txt, query_profile=query_profile)
        subject_hits = [t for t in (query_profile or {}).get("subject_terms", []) if str(t).lower() in txt_lower]
        has_time = self._contains_time_cue(txt)
        has_location = self._contains_location_cue(txt)
        where_pattern = self._where_pattern_score(txt, self._where_question_mode(question, query_profile=query_profile) or "where")
        low_value = self._is_low_value_turn(txt)
        enumeration_score = self._enumeration_shape_score(txt)

        focus_term_count = len(self._extract_focus_terms(question, query_profile=query_profile))
        required_focus_hits = 1 if focus_term_count <= 2 else 2

        # hard block low-value replies unless they also carry real focus terms
        if low_value and len(focus_hits) < required_focus_hits:
            return False

        if qtype == "what_static":
            slot_type = self._static_slot_type(question, query_profile=query_profile)
            slot_score = self._static_slot_cue_score(txt, slot_type)
            alignment_score = self._static_slot_alignment_score(question, query_profile, {"text": txt, "speaker": ""}, slot_type)
            if slot_type in {"relationship", "identity"}:
                return (slot_score >= 0.4 and alignment_score >= 0.3) or (bool(subject_hits) and slot_score >= 0.8)
            if slot_type in {"fields", "career"}:
                if self._is_generic_static_support_turn(txt):
                    return False
                return slot_score >= 0.6 and (bool(subject_hits) or self._is_short_fact_turn(txt))
            return bool(subject_hits) and (slot_score >= 0.5 or len(focus_hits) >= 1 or len(theme_hits) >= 1)

        if qtype in {"what_list", "where_list"}:
            if enumeration_score >= 0.8:
                return True
            if qtype == "where_list" and ((has_location and (focus_hits or subject_hits)) or where_pattern >= 0.9):
                return True
            if len(focus_hits) >= 1 and (len(subject_hits) >= 1 or enumeration_score >= 0.25):
                return True
            return False

        if qtype == "where_origin":
            if where_pattern >= 1.0:
                return True
            return has_location and (len(focus_hits) >= 1 or len(subject_hits) >= 1)

        if qtype == "where":
            return where_pattern >= 0.8 or (has_location and (len(focus_hits) >= 1 or len(subject_hits) >= 1))

        if len(focus_hits) >= required_focus_hits:
            return True

        # time questions can keep one focus hit if the sentence also contains an explicit time cue
        if bool((query_profile or {}).get("is_time_question")) and len(focus_hits) >= 1 and has_time:
            return True

        # subject-only fallback is intentionally disabled by default to avoid drifting into same-speaker chatter
        allow_subject_only = bool(self.config.hierarchy_allow_subject_only_fallback)
        if allow_subject_only and subject_hits and len(focus_hits) >= 1:
            return True

        return False

    def _is_strong_theme_turn(self, question, query_profile, turn_text):
        txt = str(turn_text or "")
        if not txt:
            return False
        qtype = self._effective_question_type(question, query_profile=query_profile)
        exact_bonus = self._exact_phrase_bonus(question, txt)
        focus_hits = self._focus_overlap_terms(question, txt, query_profile=query_profile)
        subject_hits = [t for t in (query_profile or {}).get("subject_terms", []) if str(t).lower() in txt.lower()]
        has_time = self._contains_time_cue(txt)
        has_location = self._contains_location_cue(txt)
        where_pattern = self._where_pattern_score(txt, self._where_question_mode(question, query_profile=query_profile) or "where")
        low_value = self._is_low_value_turn(txt)
        if low_value:
            return False
        if exact_bonus > 0:
            return True
        if qtype == "what_static":
            slot_type = self._static_slot_type(question, query_profile=query_profile)
            slot_score = self._static_slot_cue_score(txt, slot_type)
            if slot_type in {"relationship", "identity"}:
                return slot_score >= 0.6 and (bool(subject_hits) or self._speaker_matches_target_subject(question, query_profile, {"text": txt, "speaker": ""}))
            if slot_type in {"fields", "career"}:
                if self._is_generic_static_support_turn(txt):
                    return False
                return slot_score >= 0.8 and (bool(subject_hits) or self._is_short_fact_turn(txt))
            return bool(subject_hits) and (slot_score >= 0.8 or len(focus_hits) >= 1)
        if qtype in {"what_list", "where_list"}:
            return self._enumeration_shape_score(txt) >= 0.8 or where_pattern >= 1.0 or (len(focus_hits) >= 1 and len(subject_hits) >= 1)
        if qtype == "where_origin":
            return where_pattern >= 1.2 or (has_location and (len(focus_hits) >= 1 or len(subject_hits) >= 1))
        if qtype == "where":
            return where_pattern >= 0.9 or (has_location and (len(focus_hits) >= 1 or len(subject_hits) >= 1))
        if len(focus_hits) >= 2:
            return True
        if len(focus_hits) >= 1 and bool((query_profile or {}).get("is_time_question")) and has_time:
            return True
        return False

    def _get_turn_source_id(self, turn):
        if not isinstance(turn, dict):
            return ""
        dia_id = str(turn.get("dia_id", "")).strip()
        if re.fullmatch(r"D\d+:\d+", dia_id):
            return dia_id
        source_ref = str(turn.get("source_ref", "")).strip()
        if turn.get("document_id") and source_ref:
            return source_ref
        if re.fullmatch(r"D\d+:\d+", source_ref):
            return source_ref
        speaker = str(turn.get("speaker", "")).strip()
        if re.fullmatch(r"D\d+:\d+", speaker):
            return speaker
        m = re.search(r"\b(D\d+:\d+)\b", str(turn.get("text", "")))
        if m:
            return m.group(1)
        pid = turn.get("passage_hash_id")
        if pid:
            body = self.passage_embedding_store.hash_id_to_text.get(pid, "")
            fixed = self._fix_turn_dia_id(str(turn.get("text", "")), body, speaker=turn.get("speaker", ""), hint_index=turn.get("order_in_passage"))
            if re.fullmatch(r"D\d+:\d+", fixed):
                return fixed
        return ""

    def _get_turn_session_time(self, turn):
        if not isinstance(turn, dict):
            return ""
        session_time = str(turn.get("session_time", "")).strip()
        if session_time:
            return session_time
        pid = turn.get("passage_hash_id")
        if pid:
            passage_text = self.passage_embedding_store.hash_id_to_text.get(pid, "")
            return self._extract_session_datetime_value(passage_text)
        return ""

    def _ensure_hierarchy_reranker(self):
        if not bool(self.config.use_hierarchy_reranker):
            return None
        if self.hierarchy_reranker is not None:
            return self.hierarchy_reranker
        if self.hierarchy_reranker_failed:
            return None
        model_name = str(self.config.hierarchy_reranker_model or "").strip()
        if not model_name:
            return None
        if CrossEncoder is None:
            logger.warning("Hierarchy reranker requested but CrossEncoder is unavailable. Falling back to embedding rerank.")
            self.hierarchy_reranker_failed = True
            return None
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.hierarchy_reranker = CrossEncoder(model_name, device=device)
            logger.info("Loaded hierarchy reranker")
        except Exception as e:
            logger.warning("Failed to load hierarchy reranker: %s. Falling back to embedding rerank.", safe_error_summary(e))
            self.hierarchy_reranker_failed = True
            self.hierarchy_reranker = None
        return self.hierarchy_reranker

    def _score_hierarchy_candidates_with_reranker(self, question, question_embedding, query_profile, candidate_turn_ids):
        ids = [tid for tid in candidate_turn_ids if tid in self.hierarchy_turn_units]
        if not ids:
            return []
        texts = [str(self.hierarchy_turn_units[tid].get("text", "")) for tid in ids]
        shape_scores = np.array([self._answer_shape_score(question, query_profile, self.hierarchy_turn_units[tid]) for tid in ids], dtype=float)
        base_scores = np.array([
            self._event_anchor_hit(question, txt, query_profile=query_profile) + 0.5 * len(self._focus_overlap_terms(question, txt, query_profile=query_profile))
            for txt in texts
        ], dtype=float)
        penalty_scores = np.array([
            (
                0.8 if self._is_supportive_reply_turn(txt) else 0.0
            )
            + (
                0.5 if self._is_low_value_turn(txt) else 0.0
            )
            + (
                0.6 if self._is_question_like_turn(txt) else 0.0
            )
            for txt in texts
        ], dtype=float)
        reranker = self._ensure_hierarchy_reranker()
        rerank_scores = None
        if reranker is not None:
            try:
                pairs = [[question, text] for text in texts]
                rerank_scores = np.array(
                    reranker.predict(
                        pairs,
                        batch_size=min(32, max(1, len(pairs))),
                        show_progress_bar=False,
                    ),
                    dtype=float,
                )
            except Exception as e:
                logger.warning("Hierarchy reranker scoring failed: %s. Falling back to embedding rerank.", safe_error_summary(e))
                rerank_scores = None
        if rerank_scores is None:
            turn_embs = self.config.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=self.config.batch_size)
            rerank_scores = np.dot(np.array(turn_embs), question_embedding.reshape(-1))
        base_n = min_max_normalize(base_scores) if len(base_scores) > 0 else base_scores
        shape_n = min_max_normalize(shape_scores) if len(shape_scores) > 0 else shape_scores
        rerank_n = min_max_normalize(np.array(rerank_scores, dtype=float)) if len(rerank_scores) > 0 else rerank_scores
        base_w = float(self.config.hierarchy_reranker_base_weight)
        rerank_w = float(self.config.hierarchy_reranker_weight)
        shape_w = float(self.config.hierarchy_reranker_shape_weight)
        penalty_n = min_max_normalize(penalty_scores) if len(penalty_scores) > 0 else penalty_scores
        final_scores = base_w * base_n + rerank_w * rerank_n + shape_w * shape_n - 0.18 * penalty_n
        ranked = sorted(zip(ids, final_scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked

    def _late_rerank_hierarchy_turn_ids(self, question, question_embedding, query_profile, ranked_turn_items, selected_turn_ids, top_turn_k):
        if not bool(self.config.use_hierarchy_reranker):
            return list(selected_turn_ids)
        # Fusion computes both branches for every query. Avoid changing or
        # re-encoding hierarchy candidates when that branch cannot reach the
        # reader under the structural route policy.
        if self._choose_fusion_route(query_profile) != "hierarchy":
            return list(selected_turn_ids)
        candidate_limit = max(int(self.config.hierarchy_reranker_top_n), top_turn_k)
        candidate_ids = []
        seen = set()
        for tid in list(selected_turn_ids) + [tid for tid, _ in ranked_turn_items]:
            if tid in seen:
                continue
            turn = self.hierarchy_turn_units.get(tid, {})
            txt = str(turn.get("text", "")).strip()
            if not txt or self._is_session_block_text(txt):
                continue
            if self._has_conflicting_subject(question, query_profile, turn):
                continue
            if self._is_supportive_reply_turn(txt) or self._is_low_value_turn(txt) or self._is_question_like_turn(txt):
                continue
            if not self._has_answer_like_payload(txt):
                continue
            candidate_ids.append(tid)
            seen.add(tid)
            if len(candidate_ids) >= candidate_limit:
                break
        if len(candidate_ids) <= 1:
            return list(selected_turn_ids)
        ranked = self._score_hierarchy_candidates_with_reranker(question, question_embedding, query_profile, candidate_ids)
        reranked_ids = [tid for tid, _ in ranked[:top_turn_k]]
        if not reranked_ids:
            return list(selected_turn_ids)
        preserved = [tid for tid in selected_turn_ids[:1] if tid in self.hierarchy_turn_units]
        merged = []
        for tid in preserved + reranked_ids:
            if tid not in merged:
                merged.append(tid)
        return merged[:top_turn_k]

    def _format_hierarchy_turn_for_reader(self, turn):
        source_id = self._get_turn_source_id(turn)
        session_time = self._get_turn_session_time(turn)
        text = str((turn or {}).get("text", "")).strip()
        speaker = str((turn or {}).get("speaker", "") or "").strip()
        sample_id = str((turn or {}).get("sample_id", "") or "").strip()
        if not text or self._is_session_block_text(text):
            return "", {"source_id": source_id, "session_time": session_time, "text": text, "speaker": speaker, "sample_id": sample_id}
        parts = []
        if session_time:
            parts.append(f"[session_date_time={session_time}]")
        if source_id:
            parts.append(f"[{source_id}]")
        if text:
            parts.append(text)
        return " ".join(parts).strip(), {"unit_type": "turn", "source_id": source_id, "session_time": session_time, "text": text, "speaker": speaker, "sample_id": sample_id}

    def _window_match_signals(self, question, candidate_window_ids, query_profile=None):
        return self._hierarchy_match_signals(
            question, query_profile, candidate_window_ids,
            self.hierarchy_window_signal_cache, self.window_id_to_text,
        )

    def _episode_match_signals(self, question, query_profile, candidate_episode_ids):
        return self._hierarchy_match_signals(
            question, query_profile, candidate_episode_ids,
            self.hierarchy_episode_signal_cache, self.episode_id_to_text,
            keyword_units=self.hierarchy_episode_units,
        )

    def _hierarchy_match_signals(self, question, query_profile, candidate_ids, signal_cache, text_by_id, keyword_units=None):
        """Score cached windows and episodes with the same seven evidence signals."""
        query_terms = set(self._normalize_bm25_tokens(question))
        query_phrases = self._extract_query_phrases(question)
        anchors = self._event_anchor_terms(question, query_profile=query_profile)
        template_type = str((query_profile or {}).get("event_template_type", "")).strip()
        template_terms = [str(t).lower() for t in (query_profile or {}).get("event_template_terms", []) if str(t).strip()]
        subject_terms = [str(t).lower() for t in (query_profile or {}).get("subject_terms", []) if str(t).strip()]
        lexical_scores = []
        phrase_scores = []
        anchor_scores = []
        subject_scores = []
        template_scores = []
        where_pattern_scores = []
        where_location_scores = []
        where_mode = self._where_question_mode(question, query_profile=query_profile)
        for unit_id in candidate_ids:
            cached = signal_cache.get(unit_id)
            if cached:
                raw_text = cached["raw"]
                text = cached["lower"]
                toks = cached["tokens"]
            else:
                raw_text = str(text_by_id.get(unit_id, ""))
                text = raw_text.lower()
                toks = set(self._normalize_bm25_tokens(text))
                if keyword_units is not None:
                    unit = keyword_units.get(unit_id, {})
                    toks |= {str(k).lower() for k in unit.get("keywords", [])}
            lexical_scores.append(float(len(query_terms & toks)))
            phrase_scores.append(float(sum(1 for ph in query_phrases if ph and ph in text)))
            anchor_scores.append(float(sum(1 for anchor in anchors if anchor and anchor in text)))
            subject_scores.append(float(sum(1 for subj in subject_terms if subj and subj in text)))
            if template_type:
                template_hit = float(sum(1 for term in template_terms if term and term in text))
                template_scores.append(self._event_template_score(raw_text, template_type) + 0.35 * template_hit)
            else:
                template_scores.append(0.0)
            if where_mode:
                where_pattern_scores.append(self._where_pattern_score(raw_text, where_mode))
                where_location_scores.append(
                    self._where_location_term_hits(raw_text, where_mode)
                    + min(self._location_cue_score(raw_text), 2.0)
                    - self._where_noise_penalty(raw_text, where_mode)
                )
            else:
                where_pattern_scores.append(0.0)
                where_location_scores.append(0.0)
        return (
            np.array(lexical_scores, dtype=float),
            np.array(phrase_scores, dtype=float),
            np.array(anchor_scores, dtype=float),
            np.array(subject_scores, dtype=float),
            np.array(template_scores, dtype=float),
            np.array(where_pattern_scores, dtype=float),
            np.array(where_location_scores, dtype=float),
        )

    def _multi_evidence_count_evidence_cues(self, question, text, relation_hit=False):
        """Score quantities only when they are locally bound to the requested noun/relation."""
        q = str(question or "").strip().lower()
        value = str(text or "").strip().lower()
        if not q.startswith("how many") or not value:
            return 0.0, []
        target_match = re.match(
            r"^how many\s+(.+?)\s+(?:did|does|do|has|have|had|is|are|was|were|will|would|can|could)\b",
            q,
        )
        target_phrase = target_match.group(1) if target_match else ""
        def count_stem(token):
            token = re.sub(r"[^a-z0-9]", "", str(token or "").lower())
            if token.endswith("ies") and len(token) > 4:
                return token[:-3] + "y"
            if token.endswith("es") and len(token) > 4:
                return token[:-2]
            if token.endswith("s") and len(token) > 3:
                return token[:-1]
            return token

        target_stems = {
            count_stem(token)
            for token in re.findall(r"[a-z0-9]+", target_phrase)
            if token not in {"different", "distinct", "total", "number", "of"}
        }
        words = re.findall(r"[a-z]+|\d+(?:[.,]\d+)?", value)
        word_stems = [count_stem(token) for token in words]
        number_words = {
            "zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "once", "twice",
        }
        for idx, token in enumerate(words):
            numeric = token in number_words or bool(re.fullmatch(r"\d+(?:[.,]\d+)?", token))
            if not numeric or re.fullmatch(r"(?:19|20)\d{2}", token):
                continue
            window = set(word_stems[max(0, idx - 3):idx + 4])
            if (target_stems and target_stems & window) or (token in {"once", "twice"} and relation_hit):
                return 0.65, ["count_marker", "explicit_value"]
        if relation_hit and re.search(
            r"\b(?:total(?:ed|ing)?\s+(?:of\s+)?|in\s+total\s*[:=-]?\s*)"
            r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\b"
            r"|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+in\s+total\b",
            value,
        ):
            return 0.65, ["count_marker", "explicit_value"]
        if relation_hit and re.search(r"\b(?:first|second|third|fourth|fifth|another|again)\b", value):
            return 0.25, ["count_event_marker"]
        return 0.0, []

    def _multi_evidence_question_accepts_quoted_value(self, question):
        return bool(re.search(
            r"\b(?:book|novel|movie|film|show|series|song|album|title|story|poem|"
            r"play|game|quote|named|called)\b",
            str(question or "").lower(),
        ))

    def _multi_evidence_relation_term_stem(self, token):
        value = re.sub(r"[^a-z0-9]", "", str(token or "").lower())
        irregular = {
            "children": "child", "kids": "child", "kid": "child", "people": "person",
            "went": "go", "gone": "go", "made": "make", "wrote": "write",
            "written": "write", "bought": "buy", "taught": "teach", "read": "read",
        }
        if value in irregular:
            return irregular[value]
        if value.endswith("ies") and len(value) > 4:
            return value[:-3] + "y"
        if value.endswith("ing") and len(value) > 5:
            value = value[:-3]
            if len(value) > 3 and value[-1:] == value[-2:-1]:
                value = value[:-1]
            return value
        if value.endswith("ed") and len(value) > 4:
            base = value[:-2]
            return base[:-1] if base.endswith("i") else base
        if value.endswith("es") and len(value) > 4:
            return value[:-2]
        if value.endswith("s") and len(value) > 3:
            return value[:-1]
        return value

    def _multi_evidence_relation_constraint_terms(self, question, query_profile=None):
        stop = {
            "a", "an", "the", "what", "which", "who", "where", "when", "why", "how",
            "many", "much", "did", "does", "do", "has", "have", "had", "is", "are",
            "was", "were", "be", "been", "can", "could", "would", "will", "might",
            "should", "of", "to", "for", "in", "on", "at", "with", "from", "by",
            "about", "and", "or", "both", "all", "any", "some", "their", "his", "her",
            "its", "they", "them", "he", "she", "it", "recent", "recently", "latest",
            "new", "different", "distinct", "total", "number", "thing", "things", "item",
            "items", "event", "events", "activity", "activities", "subject", "type", "types",
            "kind", "kinds", "way", "ways",
        }
        target_tokens = set()
        for target in self._target_subject_set(question, query_profile=query_profile):
            target_tokens.update(re.findall(r"[a-z0-9]+", str(target).lower()))
        terms = []
        for token in re.findall(r"[a-z0-9]+", str(question or "").lower()):
            if token in stop or token in target_tokens:
                continue
            stem = self._multi_evidence_relation_term_stem(token)
            if len(stem) >= 2 and stem not in stop and stem not in target_tokens and stem not in terms:
                terms.append(stem)
        return terms

    def _multi_evidence_relation_constraint_hits(self, question, text, query_profile=None):
        required = set(self._multi_evidence_relation_constraint_terms(question, query_profile=query_profile))
        observed = {
            self._multi_evidence_relation_term_stem(token)
            for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        }
        hits = set()
        for need in required:
            for have in observed:
                common = min(len(need), len(have))
                if need == have or (common >= 5 and (need.startswith(have) or have.startswith(need))):
                    hits.add(need)
                    break
        return hits

    def _asks_completed_relation(self, question):
        q = str(question or "").strip().lower()
        if not q or re.search(r"\b(?:would|could|might|should|likely)\b", q):
            return False
        if re.search(r"\b(?:plan|planned|planning|intend|intended|hope|hoped|want|wanted)\b", q):
            return False
        return bool(re.search(r"\b(?:did|has|have|had)\b", q))

    def _completed_relation_evidence_state(self, question, text):
        q = str(question or "").strip().lower()
        value = str(text or "").strip().lower()
        if not q or not value or not self._asks_completed_relation(q):
            return 0
        non_actual = bool(re.search(
            r"\b(?:has not|have not|had not|did not|hasn't|haven't|hadn't|didn't|never)\b"
            r"|\bnot\s+yet\b"
            r"|\b(?:plan|plans|planned|planning|hope|hopes|hoped|hoping|want|wants|wanted|wanting)\s+to\b"
            r"|\b(?:will|might|could|would)\s+(?:try\s+to\s+)?\w+\b"
            r"|\b(?:going|intends?|intended|considering)\s+to\b"
            r"|\b(?:looking forward|can't wait|about|off)\s+to\b"
            r"|^\s*(?:please\s+)?(?:do|read|visit|watch|try|check|research|attend|join|buy|make|paint|play|listen)\b"
            r"|\byou\s+(?:should|could|might|can)\b",
            value,
        ))
        actual = bool(re.search(
            r"\b(?:finished|completed|attended|visited|travelled|traveled|bought|made|painted|"
            r"played|watched|listened|recommended|suggested|joined|started|took|won|adopted|"
            r"camped|hiked|went|read|researched|organized|prepared)\b",
            value,
        ))
        if non_actual:
            return 0 if actual and re.search(r"\b(?:but|however|instead)\b", value) else -1
        return 1 if actual else 0

    def _multi_evidence_requires_relation_constraint_gate(self, question, query_profile=None):
        q = str(question or "").strip().lower()
        if not q:
            return False
        if q.startswith("how many") or self._is_list_like_question(question):
            return True
        if re.match(r"^what\s+(?:did|does|do|has|have|had)\b", q) or self._asks_completed_relation(question):
            return True
        return len(self._target_subject_set(question, query_profile=query_profile)) > 1 and bool(
            re.search(r"\b(?:both|and|common|same|shared)\b", q)
        )

    def _multi_evidence_value_cue_score(self, question, text):
        q = str(question or "").strip().lower()
        value = str(text or "").strip()
        if not q or not value:
            return 0.0
        score = 0.18 * min(len(self._focus_overlap_terms(question, value, query_profile={})), 5)
        score += 0.10 * min(len(self._theme_overlap_terms(question, value, query_profile={})), 5)
        if self._is_list_like_question(question):
            score += 0.20 * self._enumeration_shape_score(value)
        if self._multi_evidence_question_accepts_quoted_value(question) and re.search(r'["\u201c\u201d][^"\u201c\u201d]{2,100}["\u201c\u201d]', value):
            score += 0.30
        return score

    def _multi_evidence_support_completion_enabled(self, question, query_profile):
        if not bool(self.config.enable_multi_evidence_support_completion):
            return False
        if (
            self._task_type(query_profile) != "multi_evidence"
            and not bool((query_profile or {}).get("requires_evidence_closure"))
        ):
            return False
        if not str((query_profile or {}).get("sample_id", "") or "").strip():
            return False
        q = str(question or "").strip().lower()
        return q.startswith(("what ", "which ", "who ", "where ", "how many", "in what "))

    def _multi_evidence_support_completion_shape_score(self, question, query_profile, meta):
        text = str((meta or {}).get("text", "") or "")
        q = str(question or "").strip().lower()
        score = self._multi_evidence_value_cue_score(question, text)
        reasons = ["multi_evidence_value_cue"] if score > 0 else []
        if q.startswith("how many"):
            relation_hit = bool(
                self._focus_overlap_terms(question, text, query_profile=query_profile)
                or self._theme_overlap_terms(question, text, query_profile=query_profile)
                or self._event_anchor_hit(question, text, query_profile=query_profile)
                or self._multi_evidence_relation_constraint_hits(question, text, query_profile=query_profile)
            )
            count_score, count_reasons = self._multi_evidence_count_evidence_cues(
                question, text, relation_hit=relation_hit
            )
            score += count_score
            reasons.extend(count_reasons)
        if q.startswith("who ") and self._extract_evidence_entities([text], {"PERSON"}):
            score += 0.45
            reasons.append("person_entity")
            reasons.append("explicit_value")
        if q.startswith(("where ", "which city", "what city", "in what ")) and self._extract_evidence_entities([text], {"GPE", "LOC", "FAC"}):
            score += 0.55
            reasons.append("location_entity")
            reasons.append("explicit_value")
        if self._is_list_like_question(question) and ("," in text or re.search(r"\band\b", text, re.I)):
            score += 0.30
            reasons.append("enumeration_shape")
        return score, self._dedupe_preserve_order(reasons)

    def _multi_evidence_support_completion_unit_score(self, question, query_profile, meta):
        text = str((meta or {}).get("text", "") or "").strip()
        if not text or self._is_session_block_text(text) or self._is_question_like_turn(text) or self._is_low_value_turn(text):
            return -1e9, []
        targets = self._target_subject_set(question, query_profile=query_profile)
        speakers = set(self._evidence_speaker_names(text, meta=meta))
        mentions = set(self._extract_person_mentions(text))
        first_person = bool(re.search(r"\b(I|I'm|I am|I've|I'd|I'll|my|me|mine|we|our)\b", text, re.I))
        if targets and speakers and not (speakers & targets) and first_person and not (mentions & targets):
            return -1e9, ["subject_mismatch"]
        binding = self._evidence_subject_binding_score(question, query_profile, text, meta=meta)
        if binding < -0.60:
            return -1e9, ["subject_mismatch"]
        relation_state = self._completed_relation_evidence_state(question, text)
        if relation_state < 0:
            return -1e9, ["relation_state_mismatch"]
        score = binding
        reasons = []
        if relation_state > 0:
            reasons.append("completed_relation")
        focus_hits = self._focus_overlap_terms(question, text, query_profile=query_profile)
        theme_hits = self._theme_overlap_terms(question, text, query_profile=query_profile)
        anchor_hits = self._event_anchor_hit(question, text, query_profile=query_profile)
        if focus_hits:
            score += 0.22 * min(len(focus_hits), 5)
            reasons.append("focus_overlap")
        if theme_hits:
            score += 0.12 * min(len(theme_hits), 5)
            reasons.append("theme_overlap")
        if anchor_hits:
            score += 0.45 * min(anchor_hits, 3)
            reasons.append("event_anchor")
        constraint_terms = self._multi_evidence_relation_constraint_terms(question, query_profile=query_profile)
        constraint_hits = self._multi_evidence_relation_constraint_hits(question, text, query_profile=query_profile)
        if constraint_hits:
            score += 0.20 * min(len(constraint_hits), 5)
            reasons.append("relation_constraint")
        shape_score, shape_reasons = self._multi_evidence_support_completion_shape_score(question, query_profile, meta)
        q = str(question or "").strip().lower()
        direct_slot_value = (
            "explicit_value" in shape_reasons
            and q.startswith(("who ", "where ", "which city", "what city", "in what ", "how many"))
        )
        if self._asks_completed_relation(question) and relation_state == 0 and not anchor_hits and not direct_slot_value:
            return -1e9, ["relation_anchor_missing"]
        if (
            self._multi_evidence_requires_relation_constraint_gate(question, query_profile=query_profile)
            and constraint_terms
            and not constraint_hits
            and not focus_hits
            and not theme_hits
            and not anchor_hits
            and not direct_slot_value
        ):
            return -1e9, ["relation_constraint_missing"]
        score += shape_score
        reasons.extend(shape_reasons)
        return score, self._dedupe_preserve_order(reasons)

    def _multi_evidence_support_session_key(self, meta):
        source_id = str((meta or {}).get("source_id", "") or "").strip()
        match = re.search(r"\b(D\d+):\d+", source_id)
        return match.group(1) if match else str((meta or {}).get("session_time", "") or "").strip().lower()

    def _multi_evidence_support_coverage_keys(self, question, query_profile, meta):
        text = str((meta or {}).get("text", "") or "")
        targets = self._target_subject_set(question, query_profile=query_profile)
        evidence_subjects = set(self._evidence_speaker_names(text, meta=meta))
        evidence_subjects.update(self._extract_person_mentions(text))
        keys = [f"subject:{target}" for target in sorted(targets & evidence_subjects)]
        keys.extend(
            f"constraint:{term}"
            for term in sorted(self._multi_evidence_relation_constraint_hits(question, text, query_profile=query_profile))
        )
        return self._dedupe_preserve_order(keys)

    def _select_diverse_multi_evidence_support_candidates(self, scored, top_k, question):
        ranked = list(scored or [])
        top_k = max(0, int(top_k))
        if top_k <= 0:
            return []
        q = str(question or "").strip().lower()
        list_like = bool(re.match(
            r"^(?:what|which)\s+(?:activities|events|items|places|countries|cities|books|"
            r"movies|games|recipes|equipment|mediums|ways|types|sports)\b",
            q,
        ))
        coverage_mode = q.startswith("how many") or list_like or any(
            (item[4] if len(item) > 4 and isinstance(item[4], dict) else {}).get("multi_evidence_support_coverage")
            for item in ranked
        )
        if not coverage_mode:
            return ranked[:top_k]
        chosen = []
        covered = set()
        seen_sessions = set()
        remaining = list(ranked)
        while remaining and len(chosen) < top_k:
            if not chosen:
                best_idx = 0
            else:
                def priority(item):
                    meta = item[4] if len(item) > 4 and isinstance(item[4], dict) else {}
                    new_keys = set(meta.get("multi_evidence_support_coverage", [])) - covered
                    session = self._multi_evidence_support_session_key(meta)
                    return (
                        sum(key.startswith("subject:") for key in new_keys),
                        sum(key.startswith("constraint:") for key in new_keys),
                        int(bool(session and session not in seen_sessions)),
                        len(new_keys),
                        int(bool(item[0])),
                        float(item[1]),
                    )
                best_idx = max(range(len(remaining)), key=lambda idx: priority(remaining[idx]))
            item = remaining.pop(best_idx)
            chosen.append(item)
            meta = item[4] if len(item) > 4 and isinstance(item[4], dict) else {}
            covered.update(meta.get("multi_evidence_support_coverage", []))
            session = self._multi_evidence_support_session_key(meta)
            if session:
                seen_sessions.add(session)
        return chosen

    def _build_multi_evidence_support_completion(self, question, query_profile, selected_meta):
        trace = {"enabled": False}
        if not self._multi_evidence_support_completion_enabled(question, query_profile):
            return [], [], trace
        query_sample = str((query_profile or {}).get("sample_id", "") or "").strip()
        existing_sources = self._meta_source_ids(selected_meta)
        if not getattr(self, "hierarchy_turn_units", None):
            trace = {"enabled": True, "sample_id": query_sample, "scanned_turns": 0, "selected": [], "reason": "hierarchy_runtime_unavailable"}
            return [], [], trace
        turn_ids = self._same_sample_turn_ids(query_sample)
        trace = {
            "enabled": True,
            "sample_id": query_sample,
            "scanned_turns": len(turn_ids),
            "threshold": float(self.config.multi_evidence_support_completion_min_score),
            "selected": [],
        }
        scored = []
        seen = set()
        for idx, tid in enumerate(turn_ids):
            turn = self.hierarchy_turn_units.get(tid)
            if not turn:
                continue
            formatted_text, formatted_meta = self._format_hierarchy_turn_for_reader(turn)
            if not formatted_text:
                continue
            source_id = str(formatted_meta.get("source_id", "") or "").strip()
            if source_id and source_id in existing_sources:
                continue
            key = (source_id, str(formatted_meta.get("text", ""))[:180].lower())
            if key in seen:
                continue
            seen.add(key)
            score, reasons = self._multi_evidence_support_completion_unit_score(question, query_profile, formatted_meta)
            explicit_value = "explicit_value" in reasons
            min_score = float(self.config.multi_evidence_support_completion_min_score)
            if score < (0.5 * min_score if explicit_value else min_score):
                continue
            formatted_meta = dict(formatted_meta)
            formatted_meta["unit_type"] = "multi_evidence_support"
            formatted_meta["multi_evidence_support_score"] = round(float(score), 4)
            formatted_meta["multi_evidence_support_reasons"] = reasons
            formatted_meta["multi_evidence_support_coverage"] = self._multi_evidence_support_coverage_keys(
                question, query_profile, formatted_meta
            )
            scored.append((explicit_value, score, -idx, formatted_text, formatted_meta, tid))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        # Share the hierarchy reader turn budget with evidence completion.
        top_k = max(1, int(self.config.reader_max_hierarchy_turns))
        if str(question or "").strip().lower().startswith("how many"):
            # Count questions need several distinct event mentions. Reuse the configured
            # evidence budget as the base instead of introducing another tuned constant.
            top_k *= 2
        chosen = self._select_diverse_multi_evidence_support_candidates(scored, top_k, question)
        trace["top_k"] = top_k
        trace["selected"] = [
            {
                "source_id": str(item[4].get("source_id", "")),
                "score": round(float(item[1]), 4),
                "reasons": list(item[4].get("multi_evidence_support_reasons", [])),
                "text": str(item[4].get("text", ""))[:260],
            }
            for item in chosen
        ]
        if not chosen:
            return [], [], trace
        return [item[3] for item in chosen], [item[4] for item in chosen], trace

    def _is_inferential_completion_question(self, question, query_profile):
        if not bool(self.config.enable_dual_view_evidence_closure):
            return False
        if not bool(self.config.enable_inferential_evidence_completion):
            return False
        if self._task_type(query_profile) != "inferential":
            return False
        return self._is_inferential_question(question, query_profile=query_profile)

    def _inferential_completion_unit_score(self, question, query_profile, unit):
        text = str((unit or {}).get("text", "") or "").strip()
        if not text or self._is_session_block_text(text) or self._is_question_like_turn(text) or self._is_low_value_turn(text):
            return -1e9
        binding = self._evidence_subject_binding_score(question, query_profile, text, meta=unit)
        if binding < -0.85:
            return -1e9
        score = binding
        score += 0.18 * min(len(self._focus_overlap_terms(question, text, query_profile=query_profile)), 6)
        score += 0.12 * min(len(self._theme_overlap_terms(question, text, query_profile=query_profile)), 6)
        score += 0.25 * min(self._event_anchor_hit(question, text, query_profile=query_profile), 3)
        if re.search(r"\b(?:because|since|therefore|likely|might|could|would|indicates?|suggests?)\b", text, re.I):
            score += 0.25
        if self._extract_evidence_entities([text]):
            score += 0.15
        return score

    def _inferential_completion_has_relation_cue(self, question, query_profile, unit):
        text = str((unit or {}).get("text", "") or "").strip()
        if not text:
            return False
        targets = self._target_subject_set(question, query_profile=query_profile)
        speakers = set(self._evidence_speaker_names(text, meta=unit))
        mentions = set(self._extract_person_mentions(text))
        subject_hit = not targets or bool(targets & (speakers | mentions))
        relation_hit = bool(
            self._focus_overlap_terms(question, text, query_profile=query_profile)
            or self._theme_overlap_terms(question, text, query_profile=query_profile)
            or self._event_anchor_hit(question, text, query_profile=query_profile)
        )
        return subject_hit and relation_hit

    def _build_inferential_evidence_completion(self, question, query_profile, selected_meta):
        if not self._is_inferential_completion_question(question, query_profile):
            return [], []
        query_sample = str((query_profile or {}).get("sample_id", "") or "").strip()
        if not query_sample:
            return [], []
        existing_sources = self._meta_source_ids(selected_meta)
        expanded_sources = self._meta_expanded_source_ids(selected_meta)
        neighbor_turn_ids = set(
            self._same_session_neighbor_turn_ids(query_sample, expanded_sources, radius=1)
        )
        scored = []
        seen = set()
        for idx, tid in enumerate(self._same_sample_turn_ids(query_sample)):
            if tid not in neighbor_turn_ids:
                continue
            turn = self.hierarchy_turn_units.get(tid)
            if not turn:
                continue
            formatted_text, formatted_meta = self._format_hierarchy_turn_for_reader(turn)
            if not formatted_text:
                continue
            source_id = str(formatted_meta.get("source_id", "") or "").strip()
            concrete_sources = set(self._expand_dialogue_source_ids(source_id))
            if source_id and source_id in existing_sources:
                continue
            if concrete_sources and concrete_sources <= expanded_sources:
                continue
            text = str(formatted_meta.get("text", "") or "").strip()
            key = (source_id, text[:180])
            if key in seen:
                continue
            seen.add(key)
            score = self._inferential_completion_unit_score(question, query_profile, formatted_meta)
            if score <= 0.0:
                continue
            scored.append((score, -idx, formatted_text, formatted_meta))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        top_k = max(1, int(self.config.inferential_evidence_completion_top_k))
        chosen = scored[:top_k]
        if not chosen:
            return [], []
        return [item[2] for item in chosen], [item[3] for item in chosen]

    def _build_inferential_reader_spotlights(self, question, query_profile, selected_meta):
        trace = {"enabled": False, "selected": []}
        if not bool(self.config.enable_dual_view_evidence_closure):
            return [], [], trace
        if self._task_type(query_profile) != "inferential":
            return [], [], trace

        existing_sources = set()
        for meta in selected_meta or []:
            if not isinstance(meta, dict):
                continue
            if str(meta.get("unit_type", "") or "passage") in {"turn", "trigram", "support", "focus"}:
                existing_sources.update(self._expand_dialogue_source_ids(meta.get("source_id", "")))

        scored = []
        for order, unit in enumerate(self._multi_evidence_refine_units(selected_meta)):
            source_id = str(unit.get("source_id", "") or "").strip()
            concrete_sources = set(self._expand_dialogue_source_ids(source_id))
            if concrete_sources and concrete_sources <= existing_sources:
                continue
            text = re.sub(r"\s+", " ", str(unit.get("text", "") or "")).strip()
            if not text or self._is_question_like_turn(text) or self._is_low_value_turn(text):
                continue
            binding = self._evidence_subject_binding_score(
                question,
                query_profile,
                text,
                meta=unit,
            )
            if binding < -0.60:
                continue
            focus_hits = self._focus_overlap_terms(question, text, query_profile=query_profile)
            theme_hits = self._theme_overlap_terms(question, text, query_profile=query_profile)
            anchor_hits = self._event_anchor_hit(question, text, query_profile=query_profile)
            relation_hit = self._inferential_completion_has_relation_cue(question, query_profile, unit)
            shape_score = self._answer_shape_score(question, query_profile, unit)
            direct_signal = bool(focus_hits or anchor_hits or shape_score >= 0.5)
            if not relation_hit or not direct_signal:
                continue
            score = float(binding)
            score += 0.28 * min(len(focus_hits), 6)
            score += 0.16 * min(len(theme_hits), 6)
            score += 0.30 * min(anchor_hits, 3)
            score += 0.35 if relation_hit else 0.0
            score += 0.16 * max(0.0, shape_score)
            if len(text) <= 260:
                score += 0.12
            scored.append((score, -order, source_id, text, dict(unit)))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        top_k = max(1, int(self.config.inferential_evidence_completion_top_k))
        chosen = []
        session_counts = {}
        seen = set()
        for item in scored:
            source_id = item[2]
            text = item[3]
            key = (source_id, text[:180].lower())
            if key in seen:
                continue
            session_match = re.match(r"(D\d+):\d+", source_id)
            session_key = session_match.group(1) if session_match else source_id
            if session_key and session_counts.get(session_key, 0) >= 2:
                continue
            chosen.append(item)
            seen.add(key)
            if session_key:
                session_counts[session_key] = session_counts.get(session_key, 0) + 1
            if len(chosen) >= top_k:
                break

        trace = {
            "enabled": True,
            "scanned": len(scored),
            "selected": [
                {"source_id": item[2], "score": round(float(item[0]), 4), "text": item[3][:220]}
                for item in chosen
            ],
        }
        passages = []
        metas = []
        for score, _, source_id, text, unit in chosen:
            speaker = str(unit.get("speaker", "") or "").strip()
            display_text = text
            if speaker and not re.match(rf"^{re.escape(speaker)}\s*:", display_text, re.I):
                display_text = f"{speaker}: {display_text}"
            source_label = f"[{source_id}] " if source_id else ""
            passages.append(f"[focused_evidence] {source_label}{display_text}".strip())
            meta = dict(unit)
            meta.update({"unit_type": "focus", "text": display_text, "focus_score": float(score)})
            metas.append(meta)
        return passages, metas, trace
