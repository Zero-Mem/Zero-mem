import spacy
from collections import defaultdict
from functools import lru_cache
from tqdm import tqdm


class SpacyNER:
    def __init__(self, spacy_model, device="cpu", batch_size=0):
        requested_device = str(device or "cpu").strip().lower()
        if requested_device == "cuda":
            spacy.require_gpu()
            self.device = "cuda"
        elif requested_device == "auto" and spacy.prefer_gpu():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.batch_size = max(0, int(batch_size or 0))
        self.spacy_model = spacy.load(spacy_model)

    def batch_ner(self, hash_id_to_passage, max_workers):
        passage_hash_ids = list(hash_id_to_passage.keys())
        passage_list = [
            hash_id_to_passage[passage_hash_id]
            for passage_hash_id in passage_hash_ids
        ]
        original_batch_size = max(
            1,
            len(passage_list) // max(1, int(max_workers or 1)),
        )
        effective_batch_size = self.batch_size or original_batch_size
        docs = self.spacy_model.pipe(
            passage_list,
            batch_size=effective_batch_size,
        )
        passage_hash_id_to_entities = {}
        sentence_to_entities = defaultdict(list)
        iterator = tqdm(
            zip(passage_hash_ids, docs),
            total=len(passage_hash_ids),
            desc=f"NER ({self.device})",
            unit="passage",
            dynamic_ncols=True,
            mininterval=1.0,
        )
        for passage_hash_id, doc in iterator:
            passage_entities, sentence_entities = (
                self.extract_entities_sentences(doc, passage_hash_id)
            )
            passage_hash_id_to_entities.update(passage_entities)
            for sentence, entities in sentence_entities.items():
                for entity in entities:
                    if entity not in sentence_to_entities[sentence]:
                        sentence_to_entities[sentence].append(entity)
        return passage_hash_id_to_entities, sentence_to_entities
            
    def extract_entities_sentences(self, doc,passage_hash_id):
        sentence_to_entities = defaultdict(list)
        unique_entities = set()
        passage_hash_id_to_entities = {}
        for ent in doc.ents:
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            sent_text = ent.sent.text
            ent_text = ent.text
            if ent_text not in sentence_to_entities[sent_text]:
                sentence_to_entities[sent_text].append(ent_text)
            unique_entities.add(ent_text)
        passage_hash_id_to_entities[passage_hash_id] = list(unique_entities)
        return passage_hash_id_to_entities,sentence_to_entities

    @lru_cache(maxsize=256)
    def question_analysis(self, question: str):
        doc = self.spacy_model(question)
        entities = tuple(
            ent.text.lower() for ent in doc.ents
            if ent.label_ not in {"ORDINAL", "CARDINAL"}
        )
        return entities, self._question_target_number(doc)

    @staticmethod
    def _question_target_number(doc):
        word = next((token for token in doc if not token.is_space), None)
        if word is None or word.lower_ not in {"what", "which"}:
            return ""
        target = word.head if word.dep_ == "det" else None
        if target is None and word.dep_ in {"attr", "nsubj"} and word.head.lemma_ == "be":
            targets = [
                child for child in word.head.children
                if child.i != word.i and child.dep_ in {"nsubj", "attr"}
                and child.pos_ in {"NOUN", "PROPN"}
            ]
            if len(targets) == 1:
                target = targets[0]
        if target is None or target.pos_ not in {"NOUN", "PROPN"}:
            return ""
        number = target.morph.get("Number")
        if "Plur" in number:
            return "Plur"
        if "Sing" in number:
            return "Sing"
        return {"NNS": "Plur", "NNPS": "Plur", "NN": "Sing", "NNP": "Sing"}.get(target.tag_, "")

    def question_ner(self, question: str):
        entities, _ = self.question_analysis(question)
        question_entities = set()
        for entity in entities:
            question_entities.add(entity)
        return question_entities
