"""Shared evidence helpers and literal choice formatting."""

import re

class AnswerExtractionMixin:

    def _dedupe_preserve_order(self, values):
        out = []
        seen = set()
        for value in values or []:
            norm = re.sub(r"\s+", " ", str(value or "").strip().lower())
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append(str(value).strip())
        return out

    def _extract_evidence_entities(self, texts, allowed_labels=None):
        nlp = getattr(getattr(self, "spacy_ner", None), "spacy_model", None)
        if nlp is None:
            return []
        labels = set(allowed_labels or [])
        values = []
        try:
            docs = nlp.pipe([str(text or "") for text in texts if str(text or "").strip()])
            for doc in docs:
                for ent in doc.ents:
                    if labels and ent.label_ not in labels:
                        continue
                    value = re.sub(r"\s+", " ", ent.text).strip(" .,:;!?\"'")
                    if value:
                        values.append(value)
        except Exception:
            return []
        return self._dedupe_preserve_order(values)

    def _clean_choice_option(self, option, is_left=False):
        text = re.sub(r"\s+", " ", str(option or "").strip(" ?.;,:\"'"))
        if not text:
            return ""
        if is_left:
            cutters = (
                r".*\bbetween\s+",
                r".*\bprefer(?:s|red|ring)?\s+",
                r".*\binterested in going to\s+",
                r".*\benjoy reading books by\s+",
                r".*\bworking on\s+",
                r".*\bclose to\s+",
                r".*\bto\s+",
                r".*\bby\s+",
                r".*\bon\s+",
                r".*\bin\s+",
            )
            for pat in cutters:
                new_text = re.sub(pat, "", text, flags=re.I).strip()
                if new_text and new_text != text:
                    text = new_text
                    break
        text = re.sub(r"^(?:a|an|the)\s+", "", text, flags=re.I)
        if is_left and "," in text:
            text = text.split(",")[-1].strip()
        text = re.sub(r"\b(?:or|and)$", "", text, flags=re.I).strip()
        if len(text.split()) > 6:
            text = " ".join(text.split()[-6:])
        return text

    def _choice_options_from_question(self, question):
        q = str(question or "").strip()
        q = re.sub(r"\?.*$", "", q).strip()
        if " or " not in q.lower():
            return []
        m = re.search(r"\bor\b", q, re.I)
        if not m:
            return []
        left = self._clean_choice_option(q[:m.start()], is_left=True)
        right = self._clean_choice_option(q[m.end():], is_left=False)
        return [opt for opt in self._dedupe_preserve_order([left, right]) if opt]

    def _refine_choice_answer(self, question, pred_answer):
        if not self._is_choice_question(question):
            return pred_answer
        options = self._choice_options_from_question(question)
        pred = str(pred_answer or "").strip()
        pred_low = pred.lower().strip(" .")
        for opt in options:
            opt_low = opt.lower()
            if opt_low and re.search(rf"(?<!\w){re.escape(opt_low)}(?!\w)", pred_low):
                return opt
            if pred_low and re.search(rf"(?<!\w){re.escape(pred_low)}(?!\w)", opt_low):
                return pred_answer
        return pred_answer


    def _multi_evidence_refine_units(self, selected_meta):
        units = []
        seen = set()
        for item in selected_meta or []:
            if not isinstance(item, dict):
                continue
            base_text = str(item.get("text", "") or "").strip()
            if not base_text:
                continue
            sample_id = str(item.get("sample_id", "") or self._extract_sample_id(base_text) or "").strip()
            session_time = str(item.get("session_time", "") or "").strip()
            raw = re.sub(r"\s+\|\s+", "\n", base_text)
            raw_units = []
            for match in re.finditer(r"(D\d+:\d+)\s+([A-Z][A-Za-z]+):\s*(.*?)(?=(?:\n\s*D\d+:\d+\s+[A-Z][A-Za-z]+:)|\Z)", raw, re.S):
                sid = match.group(1).strip()
                speaker = match.group(2).strip()
                body = re.sub(r"\s+", " ", match.group(3)).strip()
                if body:
                    raw_units.append({
                        "unit_type": "turn",
                        "source_id": sid,
                        "session_time": session_time,
                        "sample_id": sample_id,
                        "speaker": speaker,
                        "text": f"{speaker}: {body}",
                    })
            for match in re.finditer(r"\[(D\d+:\d+)\]\s*(.*?)(?=(?:\s*\[D\d+:\d+\])|\Z)", raw, re.S):
                sid = match.group(1).strip()
                body = re.sub(r"\s+", " ", match.group(2)).strip()
                if body:
                    raw_units.append({
                        "unit_type": "passage_turn",
                        "source_id": sid,
                        "session_time": session_time,
                        "sample_id": sample_id,
                        "speaker": "",
                        "text": body,
                    })
            if not raw_units:
                raw_units = self._split_reader_text_into_evidence_units(base_text, meta=item)
            for unit in raw_units:
                text = re.sub(r"\s+", " ", str(unit.get("text", "") or "")).strip()
                if not text:
                    continue
                sid = str(unit.get("source_id", "") or "").strip()
                key = (sid, text[:180].lower())
                if key in seen:
                    continue
                seen.add(key)
                cleaned = dict(unit)
                cleaned["text"] = text
                units.append(cleaned)
        return units

