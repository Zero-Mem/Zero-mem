"""Public ZeroMem entry point and shared runtime initialization."""

from .embedding_store import EmbeddingStore
from .indexing import IsolationGuard
from .ner import SpacyNER
from .indexing import _safe_dataset_namespace
from collections import defaultdict
import igraph as ig
from .utils import logger
import numpy as np
import os
import torch
from .indexing import IndexingMixin
from .query import QueryMixin
from .retrieval import RetrievalMixin
from .evidence import EvidenceMixin
from .reader import ReaderMixin
from .answer_calibration import AnswerCalibrationMixin
from .answer_extraction import AnswerExtractionMixin


class ZeroMem(
    IndexingMixin,
    QueryMixin,
    RetrievalMixin,
    EvidenceMixin,
    ReaderMixin,
    AnswerCalibrationMixin,
    AnswerExtractionMixin,
):
    """Shared runtime for graph/hierarchy retrieval and final-answer generation."""

    def __init__(self, global_config):
        self.config = global_config
        logger.info("Initializing ZeroMem")
        retrieval_method = "Vectorized Matrix-based" if self.config.use_vectorized_retrieval else "BFS Iteration"
        logger.info(f"Using retrieval method: {retrieval_method}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.config.use_vectorized_retrieval:
            logger.info(f"Using device: {self.device} for vectorized retrieval")
        self._sparse_matrices_ready = False

        self.raw_dataset_name = global_config.dataset_name
        self.dataset_name = _safe_dataset_namespace(global_config.dataset_name)
        self.load_embedding_store()
        self.isolation_guard = IsolationGuard(self)
        self.llm_model = self.config.llm_model
        self.spacy_ner = SpacyNER(
            self.config.spacy_model,
            device=self.config.spacy_ner_device,
            batch_size=self.config.spacy_ner_batch_size,
        )
        self.graph = ig.Graph(directed=False)

        # Graph branch must remain clean and isolated from hierarchy/fusion runtime state.
        # Common passage metadata is always needed by index()/retrieve(), even in graph-only mode.
        self.passage_id_to_sample_id = {}
        self.passage_id_to_session_ids = {}

        # Hierarchy state is initialized lazily only when hierarchy/fusion is explicitly used.
        self.hierarchy_runtime_ready = False
        self.window_embedding_store = None
        self.episode_embedding_store = None
        self.passage_profiles = {}
        self.bm25_doc_freq = {}
        self.bm25_doc_len = {}
        self.bm25_avgdl = 0.0
        self.bm25_texts = {}
        self.bm25_token_cache = {}
        self.bm25_tf_cache = {}
        self.bm25_postings = defaultdict(dict)
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
        self.hierarchy_reranker = None
        self.hierarchy_reranker_failed = False
        self.answer_stage_reranker = None
        self.answer_stage_reranker_failed = False
        self.same_sample_turn_cache = {}

    def load_embedding_store(self):
        # Keep graph branch identical to the original LinearRAG baseline: only passage/entity/sentence stores.
        self.passage_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "passage_embedding.parquet"), batch_size=self.config.batch_size, namespace="passage")
        self.entity_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "entity_embedding.parquet"), batch_size=self.config.batch_size, namespace="entity")
        self.sentence_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "sentence_embedding.parquet"), batch_size=self.config.batch_size, namespace="sentence")

    def _reset_runtime_after_cache_clear(self):
        self.graph = ig.Graph(directed=False)
        self._sparse_matrices_ready = False
        self.node_to_node_stats = defaultdict(dict)
        self.entity_to_sentence_stats = defaultdict(dict)
        self.bm25_doc_freq = {}
        self.bm25_doc_len = {}
        self.bm25_avgdl = 0.0
        self.bm25_texts = {}
        self.bm25_token_cache = {}
        self.bm25_tf_cache = {}
        self.bm25_postings = defaultdict(dict)
        self.hierarchy_runtime_ready = False
        self.window_embedding_store = None
        self.episode_embedding_store = None
        self._init_dual_state()

    def _refresh_retrieval_runtime_state(self):
        self.entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self.entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        self.passage_hash_ids = list(self.passage_embedding_store.hash_id_to_text.keys())
        self.passage_embeddings = np.array(self.passage_embedding_store.embeddings)
        self.sentence_hash_ids = list(self.sentence_embedding_store.hash_id_to_text.keys())
        self.sentence_embeddings = np.array(self.sentence_embedding_store.embeddings)
        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}
        self.vertex_idx_to_node_name = {v.index: v["name"] for v in self.graph.vs if "name" in v.attributes()}

    def prepare_retrieval_runtime(self, precompute_sparse=False, log=True):
        self._refresh_retrieval_runtime_state()
        if precompute_sparse and self.config.use_vectorized_retrieval:
            self._ensure_sparse_matrices_precomputed(log=log)

    def _encode_embeddings(self, texts, stage_name):
        texts = list(texts or [])
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        show_progress = len(texts) >= 1000
        if show_progress:
            logger.info(f"{stage_name}: encoding {len(texts)} texts with batch_size={self.config.batch_size}")
        return self.config.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            batch_size=self.config.batch_size,
        )
