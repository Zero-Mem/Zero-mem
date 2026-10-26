from dataclasses import dataclass
from src.utils import ZeroMemReaderClient

@dataclass
class ZeroMemConfig:
    """Runtime settings for indexing, retrieval and final-answer generation."""

    dataset_name: str
    embedding_model: str = "all-mpnet-base-v2"
    llm_model: ZeroMemReaderClient = None
    spacy_model: str = "en_core_web_trf"
    spacy_ner_device: str = "cpu"
    spacy_ner_batch_size: int = 64
    working_dir: str = "./import"
    batch_size: int = 128
    max_workers: int = 16
    retrieval_top_k: int = 5
    max_iterations: int = 3
    top_k_sentence: int = 1
    passage_ratio: float = 1.5
    passage_node_weight: float = 0.05
    damping: float = 0.5
    iteration_threshold: float = 0.5
    use_vectorized_retrieval: bool = False

    # dual branch / density
    enable_dual_index: bool = False
    density_threshold: float = 0.35
    min_entity_count_for_graph: int = 2
    keyword_top_k: int = 8
    graph_fusion_weight: float = 0.55
    hierarchy_fusion_weight: float = 0.45
    graph_bm25_top_k: int = 8
    graph_bm25_boost: float = 0.35
    graph_phrase_boost: float = 0.15
    multi_evidence_graph_embedding_weight: float = 0.05
    multi_evidence_graph_bm25_weight: float = 0.95
    graph_bm25_k1: float = 1.2
    graph_bm25_b: float = 0.75

    entity_density_weight: float = 0.45
    entity_sentence_ratio_weight: float = 0.35
    keyword_density_weight: float = 0.20
    avg_entities_per_sentence_norm: float = 8.0
    entity_sentence_ratio_penalty_threshold: float = 0.30
    entity_sentence_ratio_penalty_factor: float = 0.70
    force_hierarchy_ratio_threshold: float = 0.10
    force_hierarchy_entity_count_threshold: int = 6
    force_graph_ratio_threshold: float = 0.40
    force_graph_entity_count_threshold: int = 12

    # hierarchy construction / retrieval
    hierarchy_min_turn_chars: int = 12
    hierarchy_turn_window_size: int = 3
    hierarchy_episode_similarity_threshold: float = 0.72
    hierarchy_episode_keyword_jaccard_threshold: float = 0.18
    hierarchy_episode_max_windows: int = 4
    hierarchy_episode_top_k: int = 6
    hierarchy_window_top_k: int = 8
    hierarchy_turn_top_k: int = 8
    hierarchy_episode_semantic_weight: float = 0.58
    hierarchy_episode_lexical_weight: float = 0.20
    hierarchy_episode_phrase_weight: float = 0.10
    hierarchy_episode_anchor_weight: float = 0.12
    hierarchy_event_episode_semantic_weight: float = 0.46
    hierarchy_event_episode_lexical_weight: float = 0.18
    hierarchy_event_episode_phrase_weight: float = 0.10
    hierarchy_event_episode_anchor_weight: float = 0.18
    hierarchy_event_episode_subject_weight: float = 0.08
    hierarchy_window_semantic_weight: float = 0.72
    hierarchy_window_lexical_weight: float = 0.18
    hierarchy_window_phrase_weight: float = 0.10
    hierarchy_event_window_semantic_weight: float = 0.54
    hierarchy_event_window_lexical_weight: float = 0.12
    hierarchy_event_window_phrase_weight: float = 0.08
    hierarchy_event_window_anchor_weight: float = 0.18
    hierarchy_event_window_subject_weight: float = 0.08
    hierarchy_neighbor_decay: float = 0.92
    hierarchy_session_keep_top_n: int = 2
    hierarchy_non_dominant_session_penalty: float = 0.93
    hierarchy_exact_phrase_bonus_weight: float = 0.18
    hierarchy_low_info_penalty: float = 0.18
    hierarchy_reply_penalty: float = 0.25
    hierarchy_allow_subject_only_fallback: bool = False
    hierarchy_subject_gate_bonus_weight: float = 0.55
    hierarchy_time_soft_fallback_top_k: int = 3
    hierarchy_event_soft_fallback_top_k: int = 2

    # sample / rerank / prompt / fusion
    sample_boundary_strict: bool = False
    enable_isolation_guard: bool = True
    isolation_reset_cache_on_mismatch: bool = True
    isolation_rebuild_cache_without_manifest: bool = True
    isolation_strict_boundary: bool = True
    isolation_trace: bool = True
    isolation_manifest_filename: str = "isolation_manifest.json"
    sample_match_bonus_weight: float = 0.18
    subject_match_bonus_weight: float = 0.12
    event_match_bonus_weight: float = 0.12
    time_cue_bonus_weight: float = 0.10
    fusion_hierarchy_budget: int = 1

    # hierarchy evidence / answer-slot gating
    hierarchy_min_answer_tokens: int = 3

    # hierarchy late reranker
    use_hierarchy_reranker: bool = True
    hierarchy_reranker_model: str = ""
    hierarchy_reranker_top_n: int = 16
    hierarchy_reranker_weight: float = 0.45
    hierarchy_reranker_base_weight: float = 0.45
    hierarchy_reranker_shape_weight: float = 0.10

    # answer-stage reranker (post-retrieval, pre-reader)
    use_answer_stage_reranker: bool = False
    answer_stage_reranker_model: str = ""
    answer_stage_reranker_top_n: int = 8

    # Semantic task controls; option names do not depend on benchmark category IDs.
    enable_semantic_reader_sample_filter: bool = True
    enable_inferential_evidence_completion: bool = True
    enable_multi_evidence_graph_sample_boundary: bool = True
    enable_multi_evidence_support_completion: bool = True
    multi_evidence_support_completion_min_score: float = 2.25
    inferential_evidence_completion_top_k: int = 3

    # hierarchy local / parallel trigram retrieval
    hierarchy_use_local_trigrams: bool = True
    hierarchy_local_trigram_top_k: int = 3
    hierarchy_parallel_trigram_retrieval: bool = True
    hierarchy_trigram_query_top_k: int = 6
    hierarchy_trigram_fusion_weight: float = 0.22
    hierarchy_supportive_penalty: float = 0.55
    hierarchy_trigram_low_value_penalty: float = 0.35
    hierarchy_strict_anchor_bonus: float = 0.25
    hierarchy_strict_anchor_miss_penalty: float = 0.45
    hierarchy_candidate_turn_budget: int = 8
    hierarchy_candidate_trigram_budget: int = 6
    hierarchy_reader_min_evidence: int = 3
    hierarchy_generic_cluster_penalty: float = 0.55
    hierarchy_cluster_soft_cap: int = 2
    hierarchy_soft_filter_min_candidates: int = 12
    hierarchy_exact_turn_preserve_top_k: int = 2
    hierarchy_location_bonus_weight: float = 0.28
    hierarchy_list_bonus_weight: float = 0.24
    hierarchy_static_subject_bonus_weight: float = 0.18
    hierarchy_where_pattern_bonus_weight: float = 0.38
    hierarchy_where_location_term_weight: float = 0.16
    hierarchy_where_noise_penalty: float = 0.22
    hierarchy_where_episode_pattern_weight: float = 0.16
    hierarchy_where_episode_location_weight: float = 0.10
    hierarchy_where_window_pattern_weight: float = 0.22
    hierarchy_where_window_location_weight: float = 0.12
    hierarchy_where_harvest_limit: int = 3
    hierarchy_completion_extra_turns: int = 3
    hierarchy_completion_ranked_limit: int = 16
    hierarchy_singlehop_trigram_top_k: int = 1
    enable_dual_view_evidence_closure: bool = True

    # routing / reader
    reader_max_hierarchy_turns: int = 8
    reader_max_hierarchy_trigrams: int = 3
    reader_prompt_token_budget: int = 8000  # Legacy name: whitespace-word budget, not API tokens.
    fusion_time_to_hierarchy: bool = True
    # Hierarchy is the primary route only when the question supplies an
    # explicit temporal structure. Plural wording or a missing subject alone
    # is not sufficient evidence that local hierarchy units are reliable.
    fusion_plural_to_hierarchy: bool = False
    fusion_unknown_to_hierarchy: bool = False
    fusion_support_passage_budget: int = 2
    fusion_route_primary_weight: float = 0.78
    fusion_route_secondary_weight: float = 0.22
    enable_deterministic_evidence_calibration: bool = True

