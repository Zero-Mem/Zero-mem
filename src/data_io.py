"""Dataset/prediction IO and the stdlib-only diagnostic bootstrap for ZeroMem.

Diagnostics must be available before the CLI imports model dependencies.
"""

import hashlib
import json
import logging
from uuid import uuid4
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path, PureWindowsPath


def redact_diagnostic_text(value):
    """Redact diagnostics only; never use this on model inputs or answers."""
    text = re.sub(r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]+", "[URL]", str(value))
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        secret = os.getenv(name)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)\b(?:authorization|proxy-authorization)\b[\"']?\s*[:=]\s*[^\n]+", "authorization=[REDACTED]", text)
    text = re.sub(r"(?i)\bbearer\s+[^\s,\"']+", "Bearer [REDACTED]", text)
    text = re.sub(r"(?i)\b(?:api[_-]?key|apikey|access[_-]?token|password|secret)\b[\"']?\s*[:=]\s*[^\n]+", "credential=[REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_.*-]+", "[REDACTED]", text)
    text = re.sub(r"(?:[A-Za-z]:[\\/]|\\\\)[^\n\"'<>]*", "[PATH]", text)
    text = re.sub(r"(?<![\w])/(?:[^\s/\"'<>]+/)*[^\s/\"'<>]*", "[PATH]", text)
    text = re.sub(r"(?i)\b(?:question|prompt|qid)\b[\"']?\s*[:=]\s*[^\n]+", "question=[REDACTED]", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    text = re.sub(r"\bwxid_[A-Za-z0-9_]{4,}\b", "[WXID]", text)
    return text


def _safe_diagnostic_label(value):
    if not isinstance(value, (str, int)):
        return None
    label = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", label):
        return None
    return label if redact_diagnostic_text(label) == label else None


def safe_error_details(error):
    """Keep status/type/code, never provider messages, URLs, bodies or arguments."""
    recorded = getattr(error, "zeromem_error_details", None)
    if isinstance(recorded, dict):
        details = {}
        for key in ("error_type", "provider_error_type", "provider_error_code", "source_module"):
            label = _safe_diagnostic_label(recorded.get(key))
            if label:
                details[key] = label
        for key in ("http_status", "errno", "source_line"):
            number = recorded.get(key)
            if isinstance(number, int) and not isinstance(number, bool):
                details[key] = number
        return details or {"error_type": "Exception"}
    details = {"error_type": _safe_diagnostic_label(type(error).__name__) or "Exception"}
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599:
        details["http_status"] = status
    number = getattr(error, "errno", None)
    if isinstance(number, int) and not isinstance(number, bool):
        details["errno"] = number
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        body = body.get("error", body)
    if isinstance(body, dict):
        for source, target in (("type", "provider_error_type"), ("code", "provider_error_code")):
            label = _safe_diagnostic_label(body.get(source))
            if label:
                details[target] = label
    project = Path(__file__).resolve().parent.parent
    trace = error.__traceback__
    while trace is not None:
        filename = Path(trace.tb_frame.f_code.co_filename)
        if not filename.is_absolute():
            filename = project / filename
        try:
            relative = filename.relative_to(project)
        except ValueError:
            relative = None
        if relative is not None and relative.suffix == ".py" and (
            relative.as_posix() == "run.py" or relative.parts[0] == "src"
        ):
            details["source_module"] = ".".join(relative.with_suffix("").parts)
            details["source_line"] = trace.tb_lineno
        trace = trace.tb_next
    return details


def safe_error_summary(error):
    return "; ".join(f"{key}={value}" for key, value in safe_error_details(error).items())


class _DiagnosticLogFilter(logging.Filter):
    def filter(self, record):
        message = redact_diagnostic_text(record.getMessage())
        if record.exc_info and record.exc_info[1] is not None:
            message += "\n" + safe_error_summary(record.exc_info[1])
        record.msg = message
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True

class _DiagnosticStream:
    """CLI Python text output only; hold incomplete writes until a line boundary."""
    def __init__(self, stream):
        from threading import Lock
        self._stream = stream
        self._pending = ""
        self._lock = Lock()

    def write(self, text):
        if not isinstance(text, str):
            raise TypeError("diagnostic text stream requires str")
        with self._lock:
            self._pending += text
            while True:
                boundary = re.search(r"[\r\n]", self._pending)
                if boundary is None:
                    break
                line = self._pending[:boundary.start()]
                ending = self._pending[boundary.start():boundary.end()]
                self._pending = self._pending[boundary.end():]
                self._stream.write(redact_diagnostic_text(line) + ending)
        return len(text)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        # A flush between fragmented writes must not expose a partial secret.
        with self._lock:
            self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def install_diagnostic_streams():
    """Wrap only normal Python CLI stdout/stderr; file JSON is unaffected."""
    import sys
    if not isinstance(sys.stdout, _DiagnosticStream):
        sys.stdout = _DiagnosticStream(sys.stdout)
    if not isinstance(sys.stderr, _DiagnosticStream):
        sys.stderr = _DiagnosticStream(sys.stderr)


logging.getLogger("src.ZeroMem").addFilter(_DiagnosticLogFilter())


def _metadata_path(value, project=None):
    """Export a location identifier, never a machine-specific absolute path."""
    project = Path(project or Path(__file__).resolve().parent.parent).resolve()
    text = os.fspath(value)
    if re.search(r"[A-Za-z][A-Za-z0-9+.-]*://", text):
        return "url-sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    foreign = PureWindowsPath(text)
    if foreign.is_absolute() and os.name != "nt":
        canonical = foreign.as_posix()
    else:
        try:
            target = Path(text).expanduser().resolve()
        except (OSError, ValueError):
            # Malformed diagnostic metadata must not prevent a valid answer.
            canonical = text
        else:
            try:
                return redact_diagnostic_text(target.relative_to(project).as_posix())
            except ValueError:
                canonical = str(target)
    return "external-sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _metadata_value(value, project, field=""):
    """Sanitize exported metadata only; runtime arguments and identity stay raw."""
    if isinstance(value, dict):
        sensitive = r"api[_-]?key|authorization|proxy[_-]?authorization|password|passwd|secret|access[_-]?token|refresh[_-]?token|credential|question|qid|prompt|messages|body|email|phone|mobile|user[_-]?(?:id|name)"
        return {
            (_metadata_value(key, project) if isinstance(key, str) else key):
            ("[REDACTED]" if re.fullmatch(sensitive, str(key), re.I)
             else _metadata_value(item, project, str(key)))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item, project, field) for item in value]
    if isinstance(value, str) and value:
        if re.search(r"[A-Za-z][A-Za-z0-9+.-]*://", value):
            return "url-sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
        path_field = field in {"expected_source_manifest", "project_root"} or field.endswith(("_path", "_dir", "_file"))
        explicit_path = (Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
                         or value.startswith(("/", "\\", "~/", "~\\", "./", "../", ".\\", "..\\")))
        if path_field or explicit_path:
            return _metadata_path(value, project)
        return redact_diagnostic_text(value)
    return value


def _session_keys(conv):
    keys = []
    for key in conv.keys():
        m = key if isinstance(key, str) else str(key)
        if m.startswith("session_") and not m.endswith("_date_time"):
            suffix = m.split("_", 1)[1]
            if suffix.isdigit():
                keys.append((int(suffix), m))
    keys.sort()
    return [k for _, k in keys]


def _format_session_passage(sample_id, conv, session_key):
    speaker_a = str(conv.get("speaker_a", "")).strip()
    speaker_b = str(conv.get("speaker_b", "")).strip()
    session_time = str(conv.get(f"{session_key}_date_time", "")).strip()
    turns = conv.get(session_key, []) or []
    session_num = session_key.split("_", 1)[1]
    lines = [f"[SampleID] {sample_id}", f"[Session] S{session_num}"]
    if session_time:
        lines.append(f"[Time] {session_time}")
    if speaker_a or speaker_b:
        lines.append(f"[Speakers] {speaker_a} | {speaker_b}")
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        dia_id = str(turn.get("dia_id", "")).strip()
        speaker = str(turn.get("speaker", "")).strip()
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        prefix = dia_id if dia_id else speaker
        if speaker and speaker not in prefix:
            prefix = f"{prefix} {speaker}".strip()
        lines.append(f"{prefix}: {text}")
    return "\n".join(lines)


BOUNDARY_FIELDS = (
    "sample_id",
    "conversation_id",
    "conv_id",
    "document_id",
    "doc_id",
    "story_id",
    "article_id",
    "context_id",
    "source_id",
)


TEXT_FIELDS = (
    "text",
    "chunk",
    "content",
    "passage",
    "context",
    "document",
    "article",
    "story",
    "input",
)


def _stringify_field_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [_stringify_field_value(v) for v in value]
        return "\n".join([p for p in parts if p])
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _mapping_boundary_items(item):
    if not isinstance(item, dict):
        return []
    out = []
    seen = set()
    candidates = [item]
    meta = item.get("metadata")
    if isinstance(meta, dict):
        candidates.append(meta)
    for source in candidates:
        for key in BOUNDARY_FIELDS:
            value = _stringify_field_value(source.get(key))
            if not value:
                continue
            sig = (key, value)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(sig)
    return out


def _primary_boundary_value(item):
    for _, value in _mapping_boundary_items(item):
        if value:
            return value
    return ""


def _extract_chunk_text(item):
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return _stringify_field_value(item)
    for key in TEXT_FIELDS:
        text = _stringify_field_value(item.get(key))
        if text:
            return text
    meta = item.get("metadata")
    if isinstance(meta, dict):
        for key in TEXT_FIELDS:
            text = _stringify_field_value(meta.get(key))
            if text:
                return text
    return json.dumps(item, ensure_ascii=False)


def _format_chunk_passage(idx, chunk):
    text = _extract_chunk_text(chunk)
    if isinstance(chunk, dict):
        boundary_tokens = [f"[{key}={value}]" for key, value in _mapping_boundary_items(chunk)]
        if boundary_tokens:
            text = " ".join(boundary_tokens) + "\n" + text
    return f"{idx}:{text}"


def _format_document_passages(chunks):
    """Use explicit document boundaries without consulting question annotations."""
    marker = re.compile(r"(?m)^Document[ \t]+(\d+):[ \t]*\r?(?:\n|$)")
    metadata = re.compile(r"(?m)^\[(?:SampleID|Session|Source)\][^\n]*(?:\n|$)")
    passages = []
    for idx, chunk in enumerate(chunks):
        text = _extract_chunk_text(chunk)
        matches = list(marker.finditer(text))
        prefix = text[:matches[0].start()] if matches else text
        # Ordinary prose and conversations keep their original representation.
        if not matches or metadata.sub("", prefix).strip():
            passages.append(_format_chunk_passage(len(passages), chunk))
            continue
        mapped = " ".join(f"[{key}={value}]" for key, value in _mapping_boundary_items(chunk))
        headers = re.findall(r"(?m)^\[(?:SampleID|Source)\][^\n]*", prefix)
        for position, match in enumerate(matches):
            stop = matches[position + 1].start() if position + 1 < len(matches) else len(text)
            content = text[match.end():stop].strip()
            title, _, body = content.partition("\n")
            lines = ([mapped] if mapped else []) + headers + [
                f"[Document] {match.group(1)}", f"[Title] {title.strip()}",
                f"[SourceChunks] {idx}", body.strip(),
            ]
            passages.append(f"{len(passages)}:" + "\n".join(line for line in lines if line))
    return passages


def _semantic_task_type(value):
    """Normalize optional semantic labels; numeric benchmark categories are ignored."""
    label = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    aliases = {
        "bridge": "multi_evidence",
        "comparison": "multi_evidence",
        "multi_evidence": "multi_evidence",
        "multi_hop": "multi_evidence",
        "multihop": "multi_evidence",
        "compositional": "multi_evidence",
        "aggregation": "multi_evidence",
        "temporal": "temporal",
        "time": "temporal",
        "inferential": "inferential",
        "open_domain": "inferential",
        "open_ended": "inferential",
        "inference": "inferential",
        "reasoning": "inferential",
        "factoid": "factoid",
        "single_hop": "factoid",
        "singlehop": "factoid",
        "direct": "factoid",
    }
    return aliases.get(label, "")


def _normalize_question_item(item, idx):
    row = dict(item) if isinstance(item, dict) else {"question": str(item)}
    row.setdefault("orig_idx", idx)
    boundary = _primary_boundary_value(row)
    if boundary and not _stringify_field_value(row.get("sample_id")):
        row["sample_id"] = boundary
    # Category is evaluation metadata. Runtime strategy may only consume an
    # explicitly supplied semantic task type or infer one from the question.
    task_type = _semantic_task_type(row.get("task_type"))
    if task_type:
        row["task_type"] = task_type
    else:
        row.pop("task_type", None)
    return row


def load_dataset(dataset_name):
    questions_path = Path(f"dataset/{dataset_name}/questions.json")
    chunks_path = Path(f"dataset/{dataset_name}/chunks.json")
    if questions_path.exists() and chunks_path.exists():
        with open(questions_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        normalized_questions = []
        for idx, item in enumerate(questions):
            normalized_questions.append(_normalize_question_item(item, idx))
        passages = _format_document_passages(chunks)
        return normalized_questions, passages

    flat_dataset_path = Path(f"{dataset_name}.json")
    if not flat_dataset_path.exists():
        raise FileNotFoundError("Could not find dataset files: expected dataset/<dataset>/questions.json and chunks.json, or <dataset>.json. Check --dataset_name and the working directory.")

    with open(flat_dataset_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    questions = []
    passages = []
    orig_idx = 0
    for sample in conversations:
        sample_id = sample.get("sample_id", "")
        conv = sample.get("conversation", {}) or {}
        if isinstance(conv, dict):
            for session_key in _session_keys(conv):
                passage = _format_session_passage(sample_id, conv, session_key)
                if passage.strip():
                    passages.append(f"{len(passages)}:{passage}")
        for qa in sample.get("qa", []):
            category = qa.get("category")
            task_type = _semantic_task_type(qa.get("task_type"))
            questions.append({
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "gold_answer": qa.get("answer", ""),
                "evidence": qa.get("evidence", []),
                "category": category,
                "task_type": task_type,
                "sample_id": sample_id,
                "orig_idx": orig_idx,
            })
            orig_idx += 1
    return questions, passages


def parse_skip_categories(value):
    raw = str(value or "").strip()
    if not raw:
        return set()
    out = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            continue
    return out


def _category_is_skipped(value, skip_categories):
    try:
        return int(value) in skip_categories
    except (TypeError, ValueError):
        return False


def make_qid(question_item, idx):
    stable_idx = question_item.get("orig_idx", idx) if isinstance(question_item, dict) else idx
    sample_id = question_item.get("sample_id", "") if isinstance(question_item, dict) else ""
    question = question_item.get("question", "") if isinstance(question_item, dict) else str(question_item)
    return f"{stable_idx}::{sample_id}::{question}"


def load_done_qids(jsonl_path):
    done = set()
    p = Path(jsonl_path)
    if not p.exists():
        return done
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            qid = obj.get("qid")
            if qid:
                done.add(qid)
    return done


def append_jsonl(jsonl_path, obj):
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def merge_jsonl_to_json(jsonl_path, json_path):
    items = []
    p = Path(jsonl_path)
    if not p.exists():
        return items
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    items.sort(key=lambda x: x.get("orig_idx", 0))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return items


def resolve_output_dir(args):
    if args.output_dir and str(args.output_dir).strip():
        return args.output_dir
    return os.path.join("results", args.dataset_name, "default_run")


def prepare_output_dir(out_dir, resume=False, reset_output_dir=False):
    out_path = Path(out_dir)
    if resume:
        out_path.mkdir(parents=True, exist_ok=True)
        return str(out_path)

    if out_path.exists() and reset_output_dir:
        shutil.rmtree(out_path)

    out_path.mkdir(parents=True, exist_ok=True)
    return str(out_path)


def _dataset_run_fingerprint(dataset_name, questions, passages):
    import hashlib

    sha = hashlib.sha256()
    sha.update(str(dataset_name).encode("utf-8"))
    sha.update(str(len(questions or [])).encode("utf-8"))
    sha.update(str(len(passages or [])).encode("utf-8"))
    for idx, q in enumerate(questions or []):
        sha.update(make_qid(q, idx).encode("utf-8"))
        if isinstance(q, dict):
            sha.update(str(q.get("answer", q.get("gold_answer", ""))).encode("utf-8"))
    for passage in passages or []:
        sha.update(hashlib.md5(str(passage).encode("utf-8")).hexdigest().encode("utf-8"))
    return sha.hexdigest()


def _output_has_prediction_files(out_dir):
    path = Path(out_dir)
    if not path.exists():
        return False
    return any(path.glob("predictions_*.json")) or any(path.glob("predictions_*.jsonl"))


def ensure_output_dir_isolated(out_dir, args, questions, passages):
    fingerprint = _dataset_run_fingerprint(args.dataset_name, questions, passages)
    config_path = Path(out_dir) / "run_config.json"
    has_predictions = _output_has_prediction_files(out_dir)
    if has_predictions and not args.resume and not args.reset_output_dir:
        raise RuntimeError(
            f"Output directory already contains prediction files: {_metadata_path(out_dir)}. "
            "Use --resume to continue the same run or --reset_output_dir to start clean."
        )
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                previous = json.load(f)
        except Exception:
            previous = {}
        previous_fp = previous.get("dataset_fingerprint")
        previous_dataset = previous.get("dataset_name")
        if previous_fp and previous_fp != fingerprint:
            raise RuntimeError(
                f"Output directory belongs to a different dataset/run: {_metadata_path(out_dir)}. "
                f"previous dataset={_metadata_value(previous_dataset, Path(__file__).resolve().parent.parent)}, "
                f"current dataset={_metadata_value(args.dataset_name, Path(__file__).resolve().parent.parent)}. "
                "Use a new --output_dir or --reset_output_dir."
            )
        if has_predictions and not previous_fp:
            raise RuntimeError(
                f"Output directory has legacy predictions without a dataset fingerprint: {_metadata_path(out_dir)}. "
                "Use a new --output_dir or --reset_output_dir to avoid mixed results."
            )
    elif has_predictions:
        raise RuntimeError(
            f"Output directory has predictions but no run_config.json: {_metadata_path(out_dir)}. "
            "Use a new --output_dir or --reset_output_dir to avoid mixed results."
        )
    return fingerprint


def snapshot_source_files(project):
    project = Path(project).resolve()
    files = sorted(project.glob("run.py")) + sorted((project / "src").rglob("*.py"))
    return {p.relative_to(project).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}


def loaded_source_modules(project):
    project = Path(project).resolve()
    modules = {}
    for name, module in list(sys.modules.items()):
        if name != "__main__" and name != "src" and not name.startswith("src."):
            continue
        file = getattr(module, "__file__", None)
        if file:
            if re.search(r"[A-Za-z][A-Za-z0-9+.-]*://", str(file)):
                modules[name] = {"path": _metadata_path(file, project),
                                 "within_project": False, "sha256": None}
                continue
            path = Path(file).resolve()
            try:
                file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                file_hash = None
            modules[name] = {"path": _metadata_path(path, project),
                             "within_project": project in path.parents, "sha256": file_hash}
    return modules


def save_run_metadata(out_dir, args, enable_dual_index, dataset_fingerprint=None,
                      llm_settings=None, api_endpoint="", started_at=None,
                      project_root=None, source_hashes=None):
    project = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
    if source_hashes is None:
        source_hashes = snapshot_source_files(project)
    arguments = vars(args).copy()
    settings = {k: v for k, v in arguments.items()
                if k not in {"resume", "reset_output_dir", "output_dir", "skip_internal_evaluation"}}
    settings.update(llm_settings=llm_settings, enable_dual_index=enable_dual_index)
    digest = lambda value: hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    identity = {
        "source_sha256": digest(source_hashes),
        "config_sha256": digest(settings),
        "dataset_fingerprint": dataset_fingerprint,
        "api_endpoint_sha256": hashlib.sha256(str(api_endpoint).encode("utf-8")).hexdigest(),
        "requested_model": _metadata_value(args.llm_model, project, "llm_model"),
    }
    path = Path(out_dir) / "run_config.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() and args.resume else {}
    previous_info = previous.get("run_info", {})
    same_identity = all(
        (_metadata_value(previous_info.get(k), project, "llm_model") if k == "requested_model"
         else previous_info.get(k)) == v for k, v in identity.items())
    if args.resume and _output_has_prediction_files(out_dir):
        if not previous_info.get("run_id") or not same_identity:
            raise RuntimeError("Resume source/config identity is missing or changed; use a new --output_dir.")
    run_info = dict(identity, source_file_count=len(source_hashes),
                    run_id=(previous_info.get("run_id") if same_identity else None) or str(uuid4()))
    meta = dict(_metadata_value(arguments, project), resolved_output_dir=_metadata_path(out_dir, project),
                project_root=".",
                enable_dual_index_resolved=enable_dual_index,
                dataset_fingerprint=dataset_fingerprint, source_files_sha256=source_hashes,
                llm_request_settings=_metadata_value(llm_settings, project), run_info=run_info,
                invocation_id=str(uuid4()),
                saved_at=previous.get("saved_at") or datetime.now().isoformat(),
                last_started_at=started_at or datetime.now().isoformat())
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def summarize_reader_calls(records):
    project = Path(__file__).resolve().parent.parent
    usage_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    totals = {k: 0 for k in usage_fields}
    coverage = {k: 0 for k in usage_fields}
    models = {}
    requested_models = {}
    successful_models = {}
    consistency_counts = {}
    for call in records:
        model = _metadata_value(call.get("model") or "unreported", project, "llm_model")
        models[model] = models.get(model, 0) + 1
        if call.get("status") == "success":
            requested = _metadata_value(call.get("requested_model") or "unreported", project, "llm_model")
            requested_models[requested] = requested_models.get(requested, 0) + 1
            successful_models[model] = successful_models.get(model, 0) + 1
            consistency = call.get("model_consistency", "unreported")
            consistency_counts[consistency] = consistency_counts.get(consistency, 0) + 1
        usage = call.get("usage") or {}
        for key in usage_fields:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                totals[key] += value
                coverage[key] += 1
    if not consistency_counts:
        model_status = "not_checked"
    elif (consistency_counts.get("mismatch", 0) or len(successful_models.keys() - {"unreported"}) > 1
          or len(requested_models.keys() - {"unreported"}) > 1):
        model_status = "inconsistent"
    elif ("unreported" in requested_models or
          sum(consistency_counts.get(k, 0) for k in ("exact", "alias_snapshot")) != sum(consistency_counts.values())):
        model_status = "unverified"
    else:
        model_status = "reported_consistent"
    return {
        "logical_calls": len(records),
        "successful_calls": sum(call.get("status") == "success" for call in records),
        "failed_calls": sum(call.get("status") == "failed" for call in records),
        "response_models": models,
        "model_consistency": {"status": model_status, "counts": consistency_counts,
                              "requested_models": requested_models,
                              "response_models": successful_models},
        "reported_token_totals": {k: totals[k] if not records or coverage[k] else None for k in usage_fields},
        "token_usage_coverage": coverage,
        "logical_call_usage_complete": all(coverage[k] == len(records) for k in usage_fields),
        "calls_without_usage": sum(call.get("usage") is None for call in records),
        "http_attempts_observed": sum(call.get("http_attempts", 0) for call in records),
        "retries_observed": sum(call.get("retries_observed", 0) for call in records),
        "api_worker_seconds_sum": sum(call.get("api_seconds", 0.0) for call in records),
    }


def _safe_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _wrap_text_evidence(texts, unit_type="passage"):
    return [{"unit_type": unit_type, "source_id": "", "session_time": "", "text": str(t)} for t in _safe_list(texts)]


def _pair_unit_evidence(source_ids, texts, session_times=None, unit_type="turn"):
    ids = _safe_list(source_ids)
    texts = _safe_list(texts)
    times = _safe_list(session_times)
    out = []
    for idx, text in enumerate(texts):
        source_id = ids[idx] if idx < len(ids) else ""
        session_time = times[idx] if idx < len(times) else ""
        out.append({"unit_type": unit_type, "source_id": str(source_id), "session_time": str(session_time), "text": str(text)})
    return out


def _compact_reader_evidence(item):
    compact = {
        "unit_type": str(item.get("unit_type", "evidence")),
        "source_id": str(item.get("source_id", "")),
        "session_time": str(item.get("session_time", "")),
        "text": str(item.get("text", "")),
    }
    if "focus_score" in item:
        compact["focus_score"] = float(item.get("focus_score", 0.0))
    return compact


def _merge_evidence_lists(*lists_):
    merged = []
    seen = set()
    for lst in lists_:
        for item in _safe_list(lst):
            key = (str(item.get("unit_type", "")), str(item.get("source_id", "")), str(item.get("session_time", "")), str(item.get("text", "")))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _hierarchy_evidence(branch):
    return _merge_evidence_lists(*(
        _pair_unit_evidence(
            branch.get(f"selected_{unit_type}_source_ids", []),
            branch.get(f"selected_{unit_type}_texts", []),
            branch.get(f"selected_{unit_type}_session_times", []),
            unit_type=unit_type,
        )
        for unit_type in ("trigram", "turn")
    ))


def extract_candidate_evidence(pred, mode):
    mode = (mode or "").lower()
    if mode == "graph":
        return _wrap_text_evidence(pred.get("graph_retrieval", {}).get("sorted_passage", []), unit_type="passage")
    if mode == "hierarchy":
        return _hierarchy_evidence(pred.get("hierarchical_retrieval", {}))

    branch = pred.get("fusion", {})
    if branch.get("route") in {"hierarchy", "hybrid"}:
        support_items = _wrap_text_evidence(branch.get("support_passages", []), unit_type="passage")
        return _merge_evidence_lists(_hierarchy_evidence(branch), support_items)
    return _wrap_text_evidence(branch.get("sorted_passage", []), unit_type="passage")


def extract_used_evidence(pred, mode):
    mode = (mode or "").lower()
    route = mode if mode != "fusion" else pred.get("fusion", {}).get("route", "graph")
    meta = _safe_list(pred.get("reader_context_meta", []))
    if meta:
        return [_compact_reader_evidence(item) for item in meta]

    if route in {"hierarchy", "hybrid"}:
        branch = pred.get("hierarchical_retrieval", {}) if mode == "hierarchy" else pred.get("fusion", {})
        merged = _hierarchy_evidence(branch)
        if merged:
            return merged
        used = _safe_list(pred.get("reader_context", []))
    else:
        used = (_safe_list(pred.get("reader_context", []))
                or _safe_list(pred.get("selected_sorted_passage", []))
                or _safe_list(pred.get("sorted_passage", [])))
    return _wrap_text_evidence(used, unit_type="passage")

def build_compact_result(pred, question_item, mode, qid, orig_idx):
    mode = (mode or "").lower()
    route = mode
    if mode == "fusion":
        route = pred.get("fusion", {}).get("route", "graph")

    question = pred.get("question")
    if not question and isinstance(question_item, dict):
        question = question_item.get("question", "")

    gold_answer = pred.get("gold_answer")
    if (gold_answer is None or gold_answer == "") and isinstance(question_item, dict):
        gold_answer = question_item.get("answer", question_item.get("gold_answer", ""))

    query_profile = pred.get("query_profile", {}) if isinstance(pred.get("query_profile"), dict) else {}
    compact = {
        "qid": qid,
        "orig_idx": orig_idx,
        "sample_id": question_item.get("sample_id", "") if isinstance(question_item, dict) else "",
        "category": question_item.get("category", None) if isinstance(question_item, dict) else None,
        "task_type": query_profile.get(
            "task_type",
            question_item.get("task_type", "") if isinstance(question_item, dict) else "",
        ),
        "adaptive_task_type": query_profile.get("adaptive_task_type", ""),
        "adaptive_task_confidence": query_profile.get("adaptive_task_confidence", 0.0),
        "task_type_source": query_profile.get("task_type_source", ""),
        "answer_cardinality": query_profile.get("answer_cardinality", "single"),
        "requires_answer_aggregation": bool(query_profile.get("requires_answer_aggregation", False)),
        "requires_evidence_closure": bool(query_profile.get("requires_evidence_closure", False)),
        "need_scores": query_profile.get("need_scores", {}),
        "calibrated_need_scores": query_profile.get("calibrated_need_scores", {}),
        "evidence_need_signals": query_profile.get("evidence_need_signals", {}),
        "eval_mode": mode,
        "route": route,
        "question": question or "",
        "gold_answer": gold_answer or "",
        "pred_answer": pred.get("pred_answer", "") or "",
        "reader_output": pred.get("reader_output"),
        "reader_answer": pred.get("reader_answer"),
        "dynamic_answer_target": pred.get("dynamic_answer_target", "") or "",
        "dynamic_answer_target_applied": bool(pred.get("dynamic_answer_target_applied", False)),
        "gold_evidence": _safe_list(question_item.get("evidence", [])) if isinstance(question_item, dict) else [],
        "candidate_evidence": extract_candidate_evidence(pred, mode),
        "used_evidence": extract_used_evidence(pred, mode),
    }
    for trace_key in ("reader_isolation_trace", "reader_final_isolation_trace", "reader_prompt_budget_trace", "inferential_reader_spotlight_trace", "question_echo_surface_trace", "postprocessor_expansion_guard_trace", "multi_evidence_graph_sample_trace", "multi_evidence_support_completion_trace"):
        if trace_key in pred:
            compact[trace_key] = pred.get(trace_key)
    if mode == "fusion":
        branch = pred.get("fusion", {})
        for key in ("explicit_document_units", "document_reference_trace", "document_hierarchy_support_trace"):
            if key in branch:
                compact[key] = branch[key]
    for key in ("run_info", "invocation_id", "performance"):
        if key in pred:
            compact[key] = pred[key]
    return compact
