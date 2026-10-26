"""Command-line entry point and execution loop for ZeroMem."""

import argparse
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data_io import (
    _category_is_skipped, append_jsonl, build_compact_result,
    ensure_output_dir_isolated, load_dataset, load_done_qids, make_qid,
    merge_jsonl_to_json, parse_skip_categories, prepare_output_dir,
    resolve_output_dir, save_run_metadata, summarize_reader_calls,
    snapshot_source_files, loaded_source_modules, _metadata_path,
    redact_diagnostic_text, safe_error_details, safe_error_summary, install_diagnostic_streams,
)

if __name__ == "__main__":
    install_diagnostic_streams()

try:
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
    
    from src.config import ZeroMemConfig
    from src.ZeroMem import ZeroMem
    from src.utils import ZeroMemReaderClient, setup_logging
except Exception as error:
    if __name__ != "__main__":
        raise
    print("[ERROR] Runtime import failed: " + safe_error_summary(error), file=sys.stderr)
    raise SystemExit(1) from None

warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse indexing, generation and provenance options; scoring is separate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf")
    parser.add_argument("--spacy_ner_device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--spacy_ner_batch_size", type=int, default=64)
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding_device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--embedding_batch_size", type=int, default=128)
    parser.add_argument("--dataset_name", type=str, default="locomo10")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--qa_workers", type=int, default=None)
    parser.add_argument("--retrieval_batch_size", type=int, default=32)
    parser.add_argument(
        "--skip_internal_evaluation",
        action="store_true",
        help="Compatibility flag; this runner generates answers without internal scoring.",
    )
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument("--iteration_threshold", type=float, default=0.35)
    parser.add_argument("--passage_ratio", type=float, default=2.0)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--top_k_sentence", type=int, default=3)
    parser.add_argument("--retrieval_top_k", type=int, default=5)
    parser.add_argument("--use_vectorized_retrieval", action="store_true")
    parser.add_argument("--enable_dual_index", action="store_true")

    parser.add_argument("--density_threshold", type=float, default=0.35)
    parser.add_argument("--min_entity_count_for_graph", type=int, default=2)
    parser.add_argument("--keyword_top_k", type=int, default=8)
    parser.add_argument("--entity_density_weight", type=float, default=0.45)
    parser.add_argument("--entity_sentence_ratio_weight", type=float, default=0.35)
    parser.add_argument("--keyword_density_weight", type=float, default=0.20)
    parser.add_argument("--avg_entities_per_sentence_norm", type=float, default=8.0)
    parser.add_argument("--entity_sentence_ratio_penalty_threshold", type=float, default=0.30)
    parser.add_argument("--entity_sentence_ratio_penalty_factor", type=float, default=0.70)
    parser.add_argument("--force_hierarchy_ratio_threshold", type=float, default=0.10)
    parser.add_argument("--force_hierarchy_entity_count_threshold", type=int, default=6)
    parser.add_argument("--force_graph_ratio_threshold", type=float, default=0.40)
    parser.add_argument("--force_graph_entity_count_threshold", type=int, default=12)

    parser.add_argument("--hierarchy_min_turn_chars", type=int, default=12)
    parser.add_argument("--hierarchy_turn_window_size", type=int, default=3)
    parser.add_argument("--hierarchy_episode_similarity_threshold", type=float, default=0.72)
    parser.add_argument("--hierarchy_episode_keyword_jaccard_threshold", type=float, default=0.18)
    parser.add_argument("--hierarchy_episode_max_windows", type=int, default=4)
    parser.add_argument("--hierarchy_episode_top_k", type=int, default=6)
    parser.add_argument("--hierarchy_window_top_k", type=int, default=8)
    parser.add_argument("--hierarchy_turn_top_k", type=int, default=8)
    parser.add_argument("--hierarchy_episode_semantic_weight", type=float, default=0.58)
    parser.add_argument("--hierarchy_episode_lexical_weight", type=float, default=0.20)
    parser.add_argument("--hierarchy_episode_phrase_weight", type=float, default=0.10)
    parser.add_argument("--hierarchy_episode_anchor_weight", type=float, default=0.12)

    parser.add_argument("--sample_boundary_strict", action="store_true")
    parser.add_argument("--sample_match_bonus_weight", type=float, default=0.18)
    parser.add_argument("--subject_match_bonus_weight", type=float, default=0.12)
    parser.add_argument("--event_match_bonus_weight", type=float, default=0.12)
    parser.add_argument("--time_cue_bonus_weight", type=float, default=0.10)
    parser.add_argument("--fusion_hierarchy_budget", type=int, default=1)

    parser.add_argument("--graph_fusion_weight", type=float, default=0.55)
    parser.add_argument("--hierarchy_fusion_weight", type=float, default=0.45)
    parser.add_argument("--graph_bm25_top_k", type=int, default=8)
    parser.add_argument("--graph_bm25_boost", type=float, default=0.35)
    parser.add_argument("--graph_phrase_boost", type=float, default=0.15)
    parser.add_argument("--multi_evidence_graph_embedding_weight", type=float, default=0.05)
    parser.add_argument("--multi_evidence_graph_bm25_weight", type=float, default=0.95)
    parser.add_argument("--graph_bm25_k1", type=float, default=1.2)
    parser.add_argument("--graph_bm25_b", type=float, default=0.75)

    parser.add_argument("--hierarchy_window_semantic_weight", type=float, default=0.72)
    parser.add_argument("--hierarchy_window_lexical_weight", type=float, default=0.18)
    parser.add_argument("--hierarchy_window_phrase_weight", type=float, default=0.10)
    parser.add_argument("--hierarchy_event_window_semantic_weight", type=float, default=0.54)
    parser.add_argument("--hierarchy_event_window_lexical_weight", type=float, default=0.12)
    parser.add_argument("--hierarchy_event_window_phrase_weight", type=float, default=0.08)
    parser.add_argument("--hierarchy_event_window_anchor_weight", type=float, default=0.18)
    parser.add_argument("--hierarchy_event_window_subject_weight", type=float, default=0.08)
    parser.add_argument("--hierarchy_neighbor_decay", type=float, default=0.92)
    parser.add_argument("--hierarchy_session_keep_top_n", type=int, default=2)
    parser.add_argument("--hierarchy_non_dominant_session_penalty", type=float, default=0.93)
    parser.add_argument("--hierarchy_exact_phrase_bonus_weight", type=float, default=0.18)
    parser.add_argument("--hierarchy_low_info_penalty", type=float, default=0.18)
    parser.add_argument("--hierarchy_reply_penalty", type=float, default=0.25)
    parser.add_argument("--hierarchy_subject_gate_bonus_weight", type=float, default=0.55)
    parser.add_argument("--hierarchy_time_soft_fallback_top_k", type=int, default=3)
    parser.add_argument("--hierarchy_event_soft_fallback_top_k", type=int, default=2)
    parser.add_argument("--hierarchy_min_answer_tokens", type=int, default=3)
    parser.add_argument("--use_hierarchy_reranker", dest="use_hierarchy_reranker", action="store_true")
    parser.add_argument("--no_use_hierarchy_reranker", dest="use_hierarchy_reranker", action="store_false")
    parser.set_defaults(use_hierarchy_reranker=None)
    parser.add_argument("--hierarchy_reranker_model", type=str, default="")
    parser.add_argument("--hierarchy_reranker_top_n", type=int, default=16)
    parser.add_argument("--hierarchy_reranker_weight", type=float, default=0.45)
    parser.add_argument("--hierarchy_reranker_base_weight", type=float, default=0.45)
    parser.add_argument("--hierarchy_reranker_shape_weight", type=float, default=0.10)
    parser.add_argument("--use_answer_stage_reranker", action="store_true")
    parser.add_argument("--answer_stage_reranker_model", type=str, default="")
    parser.add_argument("--answer_stage_reranker_top_n", type=int, default=8)
    parser.add_argument("--enable_semantic_reader_sample_filter", dest="enable_semantic_reader_sample_filter", action="store_true")
    parser.add_argument("--no_enable_semantic_reader_sample_filter", dest="enable_semantic_reader_sample_filter", action="store_false")
    parser.add_argument("--enable_inferential_evidence_completion", dest="enable_inferential_evidence_completion", action="store_true")
    parser.add_argument("--no_enable_inferential_evidence_completion", dest="enable_inferential_evidence_completion", action="store_false")
    parser.add_argument("--enable_multi_evidence_graph_sample_boundary", dest="enable_multi_evidence_graph_sample_boundary", action="store_true")
    parser.add_argument("--no_enable_multi_evidence_graph_sample_boundary", dest="enable_multi_evidence_graph_sample_boundary", action="store_false")
    parser.add_argument("--enable_multi_evidence_support_completion", dest="enable_multi_evidence_support_completion", action="store_true")
    parser.add_argument("--no_enable_multi_evidence_support_completion", dest="enable_multi_evidence_support_completion", action="store_false")
    parser.set_defaults(
        enable_semantic_reader_sample_filter=True,
        enable_inferential_evidence_completion=True,
        enable_multi_evidence_graph_sample_boundary=True,
        enable_multi_evidence_support_completion=True,
    )
    parser.add_argument("--multi_evidence_support_completion_min_score", type=float, default=2.25)
    parser.add_argument("--inferential_evidence_completion_top_k", type=int, default=3)
    parser.add_argument("--hierarchy_use_local_trigrams", action="store_true")
    parser.add_argument("--no_hierarchy_use_local_trigrams", dest="hierarchy_use_local_trigrams", action="store_false")
    parser.set_defaults(hierarchy_use_local_trigrams=True)
    parser.add_argument("--hierarchy_local_trigram_top_k", type=int, default=3)
    parser.add_argument("--hierarchy_parallel_trigram_retrieval", dest="hierarchy_parallel_trigram_retrieval", action="store_true")
    parser.add_argument("--no_hierarchy_parallel_trigram_retrieval", dest="hierarchy_parallel_trigram_retrieval", action="store_false")
    parser.set_defaults(hierarchy_parallel_trigram_retrieval=True)
    parser.add_argument("--hierarchy_trigram_query_top_k", type=int, default=6)
    parser.add_argument("--hierarchy_trigram_fusion_weight", type=float, default=0.22)
    parser.add_argument("--hierarchy_supportive_penalty", type=float, default=0.55)
    parser.add_argument("--hierarchy_trigram_low_value_penalty", type=float, default=0.35)
    parser.add_argument("--hierarchy_strict_anchor_bonus", type=float, default=0.25)
    parser.add_argument("--hierarchy_strict_anchor_miss_penalty", type=float, default=0.45)
    parser.add_argument("--hierarchy_candidate_turn_budget", type=int, default=8)
    parser.add_argument("--hierarchy_candidate_trigram_budget", type=int, default=6)
    parser.add_argument("--hierarchy_reader_min_evidence", type=int, default=3)
    parser.add_argument("--hierarchy_generic_cluster_penalty", type=float, default=0.55)
    parser.add_argument("--hierarchy_cluster_soft_cap", type=int, default=2)
    parser.add_argument("--hierarchy_soft_filter_min_candidates", type=int, default=12)
    parser.add_argument("--hierarchy_exact_turn_preserve_top_k", type=int, default=2)
    parser.add_argument("--hierarchy_location_bonus_weight", type=float, default=0.28)
    parser.add_argument("--hierarchy_list_bonus_weight", type=float, default=0.24)
    parser.add_argument("--hierarchy_static_subject_bonus_weight", type=float, default=0.18)
    parser.add_argument("--hierarchy_where_pattern_bonus_weight", type=float, default=0.38)
    parser.add_argument("--hierarchy_where_location_term_weight", type=float, default=0.16)
    parser.add_argument("--hierarchy_where_noise_penalty", type=float, default=0.22)
    parser.add_argument("--hierarchy_where_episode_pattern_weight", type=float, default=0.16)
    parser.add_argument("--hierarchy_where_episode_location_weight", type=float, default=0.10)
    parser.add_argument("--hierarchy_event_episode_semantic_weight", type=float, default=0.46)
    parser.add_argument("--hierarchy_event_episode_lexical_weight", type=float, default=0.18)
    parser.add_argument("--hierarchy_event_episode_phrase_weight", type=float, default=0.10)
    parser.add_argument("--hierarchy_event_episode_anchor_weight", type=float, default=0.18)
    parser.add_argument("--hierarchy_event_episode_subject_weight", type=float, default=0.08)
    parser.add_argument("--hierarchy_where_window_pattern_weight", type=float, default=0.22)
    parser.add_argument("--hierarchy_where_window_location_weight", type=float, default=0.12)
    parser.add_argument("--hierarchy_where_harvest_limit", type=int, default=3)
    parser.add_argument("--hierarchy_completion_extra_turns", type=int, default=3)
    parser.add_argument("--hierarchy_completion_ranked_limit", type=int, default=16)
    parser.add_argument("--hierarchy_singlehop_trigram_top_k", type=int, default=1)
    parser.add_argument("--enable_dual_view_evidence_closure", dest="enable_dual_view_evidence_closure", action="store_true")
    parser.add_argument("--no_enable_dual_view_evidence_closure", dest="enable_dual_view_evidence_closure", action="store_false")
    parser.set_defaults(enable_dual_view_evidence_closure=True)

    parser.add_argument("--reader_max_hierarchy_turns", type=int, default=8)
    parser.add_argument("--reader_max_hierarchy_trigrams", type=int, default=3)
    parser.add_argument("--reader_prompt_token_budget", type=int, default=8000,
                        help="历史参数名；按空白词数裁剪证据，非完整提示的token上限。真实token用量见API usage。")
    parser.add_argument("--fusion_time_to_hierarchy", dest="fusion_time_to_hierarchy", action="store_true")
    parser.add_argument("--no_fusion_time_to_hierarchy", dest="fusion_time_to_hierarchy", action="store_false")
    parser.set_defaults(fusion_time_to_hierarchy=None)
    parser.add_argument("--fusion_plural_to_hierarchy", dest="fusion_plural_to_hierarchy", action="store_true")
    parser.add_argument("--no_fusion_plural_to_hierarchy", dest="fusion_plural_to_hierarchy", action="store_false")
    parser.set_defaults(fusion_plural_to_hierarchy=None)
    parser.add_argument("--fusion_unknown_to_hierarchy", dest="fusion_unknown_to_hierarchy", action="store_true")
    parser.add_argument("--no_fusion_unknown_to_hierarchy", dest="fusion_unknown_to_hierarchy", action="store_false")
    parser.set_defaults(fusion_unknown_to_hierarchy=None)
    parser.add_argument("--fusion_support_passage_budget", type=int, default=2)
    parser.add_argument("--fusion_route_primary_weight", type=float, default=0.78)
    parser.add_argument("--fusion_route_secondary_weight", type=float, default=0.22)
    parser.add_argument("--enable_deterministic_evidence_calibration", dest="enable_deterministic_evidence_calibration", action="store_true")
    parser.add_argument("--no_enable_deterministic_evidence_calibration", dest="enable_deterministic_evidence_calibration", action="store_false")
    parser.set_defaults(enable_deterministic_evidence_calibration=True)

    parser.add_argument("--eval_mode", type=str, default="fusion", choices=["graph", "hierarchy", "fusion", "all"])
    parser.add_argument("--question_limit", type=int, default=0)
    parser.add_argument("--skip_categories", type=str, default="")

    # output / resume
    parser.add_argument("--output_dir", type=str, default="", help="结果输出目录。默认使用 results/<dataset>/default_run")
    parser.add_argument("--resume", action="store_true", help="从已有 output_dir 中断点续跑")
    parser.add_argument("--reset_output_dir", action="store_true", help="非 resume 模式下先清空 output_dir 再重跑")
    parser.add_argument("--expected_source_manifest", type=str, default="", help="启动前核对已冻结的源码与关键参数；不一致时停止")
    return parser.parse_args()


def _verify_expected_source_manifest(args, project):
    if not args.expected_source_manifest:
        return
    required = {"run.py"} | {
        f"src/{name}.py" for name in (
            "ZeroMem", "answer_calibration", "answer_extraction", "config", "data_io",
            "embedding_store", "evidence", "indexing", "ner", "query", "reader",
            "retrieval", "utils",
        )
    }
    try:
        manifest = json.loads(Path(args.expected_source_manifest).read_text(encoding="utf-8-sig"))
        expected = manifest["source_files_sha256"]
        if not isinstance(expected, dict) or set(expected) != required:
            raise ValueError("清单必须包含全部 14 个运行源码")
        for relative, digest in expected.items():
            actual = hashlib.sha256((project / relative).read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f"源码不匹配：{relative}")
        for name in ("damping", "fusion_route_primary_weight", "fusion_route_secondary_weight"):
            if getattr(args, name) != manifest["parameters"][name]:
                raise ValueError(f"启动参数不匹配：{name}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"ZeroMem 启动校验失败，尚未初始化模型或调用 API：{redact_diagnostic_text(exc)}") from None


def load_embedding_model(model_name, device="auto"):
    requested = str(device or "auto").lower()
    if requested == "auto":
        requested = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # SentenceTransformers 2.2.x may leave a locally loaded model on CPU even
    # when its constructor receives device="cuda". Move and verify explicitly.
    model = SentenceTransformer(model_name, device="cpu")
    model.to(requested)
    try:
        import torch
        model._target_device = torch.device(requested)
    except Exception:
        pass

    if requested == "cuda":
        actual = str(next(model.parameters()).device)
        if not actual.startswith("cuda"):
            raise RuntimeError(
                f"Embedding model was requested on CUDA but is on {actual}."
            )
    return model


def generate_answers_for_mode(
    rag_model,
    questions,
    out_dir,
    mode,
    max_workers,
    resume=False,
    qa_workers=None,
    retrieval_batch_size=32,
    run_metadata=None,
    performance=None,
):
    """Retrieve and generate one mode, retaining successes for optional resume.

    Retrieval and QA failures are recorded separately. This function exports
    predictions and timings; it does not compute benchmark scores.
    """
    mode_started = perf_counter()
    performance = performance if performance is not None else {}
    performance.update(questions_total=len(questions), qa_succeeded=0, qa_failed=0,
                       retrieval_failed=0)
    print(f"\n[INFO] Running evaluation mode: {mode.upper()}")
    os.makedirs(out_dir, exist_ok=True)
    raw_jsonl_path = os.path.join(out_dir, f"predictions_{mode}_raw.jsonl")
    raw_pred_path = os.path.join(out_dir, f"predictions_{mode}_raw.json")
    compact_jsonl_path = os.path.join(out_dir, f"predictions_{mode}.jsonl")
    compact_pred_path = os.path.join(out_dir, f"predictions_{mode}.json")
    failed_path = os.path.join(out_dir, f"failed_{mode}.jsonl")

    done_qids = load_done_qids(raw_jsonl_path) if resume else set()
    remaining = []
    for idx, q in enumerate(questions):
        qid = make_qid(q, idx)
        if qid not in done_qids:
            remaining.append((idx, qid, q))

    performance.update(skipped_existing=len(questions) - len(remaining),
                       questions_attempted=len(remaining))
    print(f"[INFO] Already done: {len(done_qids)}")
    print(f"[INFO] Remaining   : {len(remaining)}")
    effective_qa_workers = max(1, int(qa_workers or max_workers))
    effective_retrieval_batch_size = max(1, int(retrieval_batch_size))
    print(f"[INFO] QA workers   : {effective_qa_workers}")
    print(f"[INFO] Retrieval batch size: {effective_retrieval_batch_size}")

    retrieval_started = perf_counter()
    retrieval_jobs = []
    if rag_model.config.use_vectorized_retrieval:
        print("[INFO] Precomputing sparse adjacency matrices for vectorized retrieval...")
        rag_model.prepare_retrieval_runtime(precompute_sparse=True, log=False)

    with tqdm(total=len(remaining), desc=f"Retrieving ({mode.upper()})") as pbar:
        for start in range(0, len(remaining), effective_retrieval_batch_size):
            batch = remaining[start:start + effective_retrieval_batch_size]
            question_batch = [q for _, _, q in batch]
            try:
                batch_results = rag_model.retrieve(question_batch, eval_mode=mode, show_progress=False)
            except Exception:
                batch_results = []
                for idx, qid, q in batch:
                    try:
                        batch_results.append(rag_model.retrieve([q], eval_mode=mode, show_progress=False)[0])
                    except Exception as item_error:
                        append_jsonl(failed_path, {
                            "stage": "retrieval",
                            "run_info": run_metadata["run_info"] if run_metadata else None,
                            "invocation_id": run_metadata["invocation_id"] if run_metadata else None,
                            "qid_sha256": hashlib.sha256(qid.encode("utf-8")).hexdigest(),
                            "orig_idx": idx,
                            "error": safe_error_details(item_error),
                        })
                        print(f"[WARN] Retrieval failed at idx={idx}, error={safe_error_summary(item_error)}")
                        batch_results.append(None)
            for (idx, qid, q), retrieval_result in zip(batch, batch_results):
                if retrieval_result is not None:
                    retrieval_jobs.append((idx, qid, q, retrieval_result))
                pbar.update(1)

    performance["retrieval_seconds"] = perf_counter() - retrieval_started
    performance["retrieval_succeeded"] = len(retrieval_jobs)
    performance["retrieval_failed"] = len(remaining) - len(retrieval_jobs)
    qa_started = perf_counter()
    if retrieval_jobs:
        futures = {}
        with ThreadPoolExecutor(max_workers=effective_qa_workers) as executor:
            for idx, qid, q, retrieval_result in retrieval_jobs:
                future = executor.submit(
                    rag_model.read_one_from_retrieval_result,
                    retrieval_result,
                    mode,
                )
                futures[future] = (idx, qid, q)
            with tqdm(total=len(retrieval_jobs), desc=f"QA ({mode.upper()})") as pbar:
                for future in as_completed(futures):
                    idx, qid, q = futures[future]
                    try:
                        pred = future.result()
                        pred["qid"] = qid
                        pred["orig_idx"] = idx
                        pred["sample_id"] = q.get("sample_id", "") if isinstance(q, dict) else ""
                        pred["eval_mode"] = mode
                        if run_metadata is not None:
                            pred["run_info"] = run_metadata["run_info"]
                            pred["invocation_id"] = run_metadata["invocation_id"]
                        append_jsonl(raw_jsonl_path, pred)
                        performance["qa_succeeded"] += 1
                    except Exception as e:
                        performance["qa_failed"] += 1
                        append_jsonl(failed_path, {
                            "stage": "qa",
                            "run_info": run_metadata["run_info"] if run_metadata else None,
                            "invocation_id": run_metadata["invocation_id"] if run_metadata else None,
                            "performance": getattr(e, "reader_performance", None),
                            "qid_sha256": hashlib.sha256(qid.encode("utf-8")).hexdigest(),
                            "orig_idx": idx,
                            "error": safe_error_details(e),
                        })
                        print(f"[WARN] QA failed at idx={idx}, error={safe_error_summary(e)}")
                    finally:
                        pbar.update(1)

    performance["qa_wall_seconds"] = perf_counter() - qa_started
    export_started = perf_counter()
    raw_items = merge_jsonl_to_json(raw_jsonl_path, raw_pred_path)
    performance["predictions_total_in_output"] = len(raw_items)
    performance["output_model_consistency"] = summarize_reader_calls([
        (item.get("performance") or {}).get("reader_call") or {"status": "success"}
        for item in raw_items
    ])["model_consistency"]
    if not raw_items:
        print(f"[WARN] No successful predictions for mode={mode}. Skipping evaluation.")
        with open(compact_pred_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        with open(compact_jsonl_path, "w", encoding="utf-8") as f:
            pass
        performance["export_seconds"] = perf_counter() - export_started
        performance["mode_wall_seconds"] = perf_counter() - mode_started
        return

    print("[INFO] Predictions are ready for the benchmark-specific non-generative evaluator.")

    compact_items = []
    question_map = {make_qid(q, idx): (idx, q) for idx, q in enumerate(questions)}
    for raw in raw_items:
        qid = raw.get("qid", "")
        orig_idx, question_item = question_map.get(qid, (raw.get("orig_idx", 0), {}))
        compact_items.append(build_compact_result(raw, question_item, mode, qid, orig_idx))
    compact_items.sort(key=lambda x: x.get("orig_idx", 0))
    with open(compact_pred_path, "w", encoding="utf-8") as f:
        json.dump(compact_items, f, ensure_ascii=False, indent=2)
    with open(compact_jsonl_path, "w", encoding="utf-8") as f:
        for item in compact_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    performance["export_seconds"] = perf_counter() - export_started
    performance["mode_wall_seconds"] = perf_counter() - mode_started
    print(f"[INFO] Saved compact predictions: {_metadata_path(compact_pred_path)}")


def main():
    """Initialize the shared runtime, generate selected modes and save provenance."""
    started = perf_counter()
    started_at = datetime.now().isoformat()
    project = Path(__file__).resolve().parent
    source_hashes = snapshot_source_files(project)
    args = parse_arguments()
    _verify_expected_source_manifest(args, project)

    embedding_model = load_embedding_model(args.embedding_model, args.embedding_device)
    questions, passages = load_dataset(args.dataset_name)
    skip_categories = parse_skip_categories(args.skip_categories)
    if skip_categories:
        questions = [
            q for q in questions
            if not _category_is_skipped(q.get("category"), skip_categories)
        ]
    if args.question_limit and args.question_limit > 0:
        questions = questions[: args.question_limit]

    out_dir = resolve_output_dir(args)
    out_dir = prepare_output_dir(out_dir, resume=args.resume, reset_output_dir=args.reset_output_dir)
    dataset_fingerprint = ensure_output_dir_isolated(out_dir, args, questions, passages)
    setup_logging(f"{out_dir}/log.txt")
    llm_model = ZeroMemReaderClient(args.llm_model)

    enable_dual_index = args.enable_dual_index or args.eval_mode in {"hierarchy", "fusion", "all"}

    config_options = (
        'dataset_name', 'spacy_model', 'spacy_ner_device', 'spacy_ner_batch_size', 'max_workers',
        'max_iterations', 'iteration_threshold', 'passage_ratio', 'damping', 'top_k_sentence',
        'retrieval_top_k', 'use_vectorized_retrieval', 'density_threshold',
        'min_entity_count_for_graph', 'keyword_top_k', 'entity_density_weight',
        'entity_sentence_ratio_weight', 'keyword_density_weight', 'avg_entities_per_sentence_norm',
        'entity_sentence_ratio_penalty_threshold', 'entity_sentence_ratio_penalty_factor',
        'force_hierarchy_ratio_threshold', 'force_hierarchy_entity_count_threshold',
        'force_graph_ratio_threshold', 'force_graph_entity_count_threshold',
        'hierarchy_min_turn_chars', 'hierarchy_turn_window_size',
        'hierarchy_episode_similarity_threshold', 'hierarchy_episode_keyword_jaccard_threshold',
        'hierarchy_episode_max_windows', 'hierarchy_episode_top_k', 'hierarchy_window_top_k',
        'hierarchy_turn_top_k', 'hierarchy_episode_semantic_weight',
        'hierarchy_episode_lexical_weight', 'hierarchy_episode_phrase_weight',
        'hierarchy_episode_anchor_weight', 'sample_boundary_strict', 'sample_match_bonus_weight',
        'subject_match_bonus_weight', 'event_match_bonus_weight', 'time_cue_bonus_weight',
        'fusion_hierarchy_budget', 'graph_fusion_weight', 'hierarchy_fusion_weight',
        'graph_bm25_top_k', 'graph_bm25_boost', 'graph_phrase_boost',
        'multi_evidence_graph_embedding_weight', 'multi_evidence_graph_bm25_weight', 'graph_bm25_k1',
        'graph_bm25_b', 'hierarchy_window_semantic_weight', 'hierarchy_window_lexical_weight',
        'hierarchy_window_phrase_weight', 'hierarchy_neighbor_decay', 'hierarchy_session_keep_top_n',
        'hierarchy_non_dominant_session_penalty', 'hierarchy_exact_phrase_bonus_weight',
        'hierarchy_low_info_penalty', 'hierarchy_reply_penalty', 'hierarchy_subject_gate_bonus_weight',
        'hierarchy_time_soft_fallback_top_k', 'hierarchy_min_answer_tokens', 'use_hierarchy_reranker',
        'hierarchy_reranker_model', 'hierarchy_reranker_top_n', 'hierarchy_reranker_weight',
        'hierarchy_reranker_base_weight', 'hierarchy_reranker_shape_weight',
        'use_answer_stage_reranker', 'answer_stage_reranker_model', 'answer_stage_reranker_top_n',
        'enable_semantic_reader_sample_filter', 'enable_inferential_evidence_completion',
        'enable_multi_evidence_graph_sample_boundary',
        'enable_multi_evidence_support_completion',
        'multi_evidence_support_completion_min_score', 'inferential_evidence_completion_top_k',
        'hierarchy_use_local_trigrams', 'enable_dual_view_evidence_closure',
        'hierarchy_local_trigram_top_k', 'hierarchy_parallel_trigram_retrieval',
        'hierarchy_trigram_query_top_k', 'hierarchy_trigram_fusion_weight',
        'hierarchy_supportive_penalty', 'hierarchy_trigram_low_value_penalty',
        'hierarchy_strict_anchor_bonus', 'hierarchy_strict_anchor_miss_penalty',
        'hierarchy_candidate_turn_budget', 'hierarchy_candidate_trigram_budget',
        'hierarchy_reader_min_evidence', 'hierarchy_generic_cluster_penalty',
        'hierarchy_cluster_soft_cap', 'hierarchy_soft_filter_min_candidates',
        'hierarchy_exact_turn_preserve_top_k', 'hierarchy_location_bonus_weight',
        'hierarchy_list_bonus_weight', 'hierarchy_static_subject_bonus_weight',
        'hierarchy_singlehop_trigram_top_k', 'reader_max_hierarchy_turns',
        'reader_max_hierarchy_trigrams', 'reader_prompt_token_budget', 'fusion_support_passage_budget',
        'fusion_route_primary_weight', 'fusion_route_secondary_weight',
        'enable_deterministic_evidence_calibration',
        'fusion_time_to_hierarchy', 'fusion_plural_to_hierarchy', 'fusion_unknown_to_hierarchy',
    )
    argument_values = vars(args)
    config = ZeroMemConfig(
        **{name: argument_values[name] for name in config_options
           if argument_values[name] is not None},
        embedding_model=embedding_model,
        llm_model=llm_model,
        batch_size=args.embedding_batch_size,
        enable_dual_index=enable_dual_index,
    )

    run_metadata = save_run_metadata(
        out_dir, args, enable_dual_index, dataset_fingerprint=dataset_fingerprint,
        llm_settings=llm_model.llm_config,
        api_endpoint=str(llm_model.openai_client.base_url),
        started_at=started_at,
        project_root=project, source_hashes=source_hashes,
    )
    performance = {"schema_version": 1, "scope": "current_invocation",
                   "run_info": run_metadata["run_info"],
                   "invocation_id": run_metadata["invocation_id"],
                   "started_at": run_metadata["last_started_at"],
                   "source_files_sha256": source_hashes,
                   "timings": {}, "modes": {}, "status": "failed"}
    phase_started = started
    phase = "initialization_seconds"
    try:
        rag_model = ZeroMem(global_config=config)
        performance["timings"][phase] = perf_counter() - phase_started
        phase = "index_seconds"
        phase_started = perf_counter()
        rag_model.index(passages)
        performance["timings"][phase] = perf_counter() - phase_started

        phase = "reranker_initialization_seconds"
        phase_started = perf_counter()
        if config.use_answer_stage_reranker:
            answer_stage_reranker = rag_model._ensure_answer_stage_reranker()
            if answer_stage_reranker is None:
                raise RuntimeError(
                    "Answer-stage reranking was requested but the CrossEncoder could not be loaded."
                )
            print("[INFO] Answer-stage reranker ready")
        performance["timings"][phase] = perf_counter() - phase_started
        phase = None

        modes = ["graph", "hierarchy", "fusion"] if args.eval_mode == "all" else [args.eval_mode]
        for mode in modes:
            mode_performance = performance["modes"].setdefault(mode, {})
            call_start = len(llm_model.snapshot_call_records())
            mode_started = perf_counter()
            try:
                generate_answers_for_mode(
                    rag_model, questions, out_dir, mode, args.max_workers,
                    resume=args.resume, qa_workers=args.qa_workers,
                    retrieval_batch_size=args.retrieval_batch_size,
                    run_metadata=run_metadata, performance=mode_performance,
                )
            finally:
                mode_performance["mode_wall_seconds"] = perf_counter() - mode_started
                mode_performance["reader_api"] = summarize_reader_calls(
                    llm_model.snapshot_call_records()[call_start:])
        performance["status"] = "completed"
    finally:
        if phase is not None and phase not in performance["timings"]:
            performance["timings"][phase] = perf_counter() - phase_started
        performance["reader_api"] = summarize_reader_calls(llm_model.snapshot_call_records())
        audit_started = perf_counter()
        source_check = {"scope": "on_disk_start_to_finish", "project_root": "."}
        try:
            source_after = snapshot_source_files(project)
            changes = {name: {"before": source_hashes.get(name), "after": source_after.get(name)}
                       for name in source_hashes.keys() | source_after.keys()
                       if source_hashes.get(name) != source_after.get(name)}
            modules = loaded_source_modules(project)
            source_status = ("changed" if changes else "outside_project" if
                             any(not item["within_project"] for item in modules.values()) else "unchanged")
            source_check.update(status=source_status, changes=changes, loaded_modules=modules)
        except OSError as error:
            source_status = "unverified"
            source_check.update(status=source_status, error=safe_error_details(error))
        performance["source_consistency"] = source_check
        summaries = [performance["reader_api"]["model_consistency"]] + [
            item["output_model_consistency"] for item in performance["modes"].values()
            if "output_model_consistency" in item]
        states = {item["status"] for item in summaries}
        response_models = {name for item in summaries for name in item["response_models"]
                           if name != "unreported"}
        model_status = ("inconsistent" if "inconsistent" in states or len(response_models) > 1 else
                        "unverified" if "unverified" in states else
                        "reported_consistent" if "reported_consistent" in states else "not_checked")
        performance["model_consistency"] = {
            "scope": "current_calls_and_all_output_predictions", "status": model_status,
            "reported_models": sorted(response_models),
        }
        if performance["status"] == "completed":
            if source_status in {"changed", "outside_project"} or model_status == "inconsistent":
                performance["status"] = "completed_inconsistent"
            elif source_status == "unverified" or model_status != "reported_consistent":
                performance["status"] = "completed_unverified"
            elif any(item.get("predictions_total_in_output") != item.get("questions_total")
                     for item in performance["modes"].values()):
                performance["status"] = "completed_partial"
        performance["timings"]["consistency_record_seconds"] = perf_counter() - audit_started
        performance["finished_at"] = datetime.now().isoformat()
        performance["timings"]["total_wall_seconds"] = perf_counter() - started
        print(f"[INFO] Run consistency: status={performance['status']}, "
              f"reported_model={model_status}, source={source_status}")
        path = os.path.join(out_dir, "performance.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(performance, f, ensure_ascii=False, indent=2)
        append_jsonl(os.path.join(out_dir, "performance_history.jsonl"), performance)
        print(f"[INFO] Saved current invocation performance: {_metadata_path(path)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[STOPPED] Run interrupted; written results retained.")
        raise SystemExit(130) from None
    except Exception as error:
        print("[ERROR] " + safe_error_summary(error), file=sys.stderr)
        raise SystemExit(1) from None
