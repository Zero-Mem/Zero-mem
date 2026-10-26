"""Final-QA client, model-label diagnostics, logging and embedding utilities."""

from hashlib import md5
import httpx
from openai import OpenAI
import re
import logging
import numpy as np
import os
from contextvars import ContextVar
from datetime import date
from time import perf_counter
from threading import Lock

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

logger = logging.getLogger("src.ZeroMem")
_reader_http_attempts = ContextVar("zeromem_reader_http_attempts", default=None)


def _record_reader_http_attempt(request):
    attempts = _reader_http_attempts.get()
    if attempts is not None:
        attempts.append(None)

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()


def classify_response_model(requested, reported):
    if not isinstance(reported, str) or not reported.strip():
        return "unreported"
    if requested == reported:
        return "exact"
    snapshot = re.fullmatch(re.escape(requested) + r"-(\d{4}-\d{2}-\d{2})", reported)
    if snapshot and not re.search(r"-\d{4}-\d{2}-\d{2}$", requested):
        try:
            date.fromisoformat(snapshot.group(1))
            return "alias_snapshot"
        except ValueError:
            pass
    return "mismatch"


from .data_io import (
    redact_diagnostic_text, safe_error_details, safe_error_summary,
    _DiagnosticLogFilter, _DiagnosticStream, install_diagnostic_streams, _metadata_value,
)


class ZeroMemReaderClient:
    """Final-QA API client with SDK retries and per-call usage diagnostics."""

    def __init__(self, llm_model):
        self._call_records = []
        self._call_records_lock = Lock()
        http_client = httpx.Client(
            timeout=300.0, trust_env=False,
            event_hooks={"request": [_record_reader_http_attempt]},
        )
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            http_client=http_client,
            timeout=300.0,
            # Keep the existing SDK retry policy; the reader is invoked once.
            max_retries=3,
        )
        self.llm_config = {
            "model": llm_model,
            "max_tokens": 512,
            "temperature": 0,
        }
        seed = str(os.getenv("LLM_SEED", "")).strip()
        if seed:
            self.llm_config["seed"] = int(seed)
    def infer(self, messages, return_metadata=False):
        """Send the supplied messages unchanged and optionally return call metadata."""
        attempts = []
        state = _reader_http_attempts.set(attempts)
        started = perf_counter()
        metadata = {
            "usage": None, "requested_model": self.llm_config["model"],
            "model": None, "model_consistency": "not_returned",
            "request_id": None, "response_id": None, "system_fingerprint": None,
            "http_attempts": 0, "retries_observed": 0, "api_seconds": 0.0,
            "status": "failed",
        }
        try:
            response = self.openai_client.chat.completions.create(
                **self.llm_config,
                messages=messages,
                timeout=300.0,
            )
            metadata.update(
                model=getattr(response, "model", None),
                request_id=getattr(response, "_request_id", None),
                response_id=getattr(response, "id", None),
                system_fingerprint=getattr(response, "system_fingerprint", None),
            )
            metadata["model_consistency"] = classify_response_model(
                metadata["requested_model"], metadata["model"])
            try:
                usage = getattr(response, "usage", None)
                if usage is not None:
                    metadata["usage"] = usage.model_dump()
            except Exception:
                # Recording must not prevent the original answer from returning.
                pass
            content = response.choices[0].message.content
            metadata["status"] = "success"
        except Exception as error:
            details = safe_error_details(error)
            metadata["error"] = details
            failure = RuntimeError(safe_error_summary(error))
            failure.zeromem_error_details = details
            failure.reader_call_metadata = metadata
            raise failure from None
        finally:
            metadata.update(
                api_seconds=perf_counter() - started,
                http_attempts=len(attempts),
                retries_observed=max(0, len(attempts) - 1),
            )
            metadata.update(_metadata_value(metadata, None))
            _reader_http_attempts.reset(state)
            with self._call_records_lock:
                self._call_records.append(metadata.copy())
        return (content, metadata) if return_metadata else content

    def snapshot_call_records(self):
        """Copy current call records without modifying the running counters."""
        with self._call_records_lock:
            return [record.copy() for record in self._call_records]



def setup_logging(log_file):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]  
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    for handler in handlers:
        handler.addFilter(_DiagnosticLogFilter())
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True
    )
    # Suppress noisy HTTP request logs (e.g., 401 Unauthorized) from httpx/openai
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val
