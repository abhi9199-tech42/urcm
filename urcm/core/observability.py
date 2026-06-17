"""
Observability module for URCM.

Provides structured event logging with privacy-aware redaction,
log rotation, and configurable levels.
"""

import os
import json
import time
import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()

_forbidden_key_substrings = {"vector", "embedding", "latent", "phoneme", "text_raw", "resonance"}
_max_string_length = 512
_max_list_length = 100
_max_dict_keys = 50


def _is_primitive(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool)) or x is None


def _redact_value(x: Any) -> Any:
    try:
        import numpy as np
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            return "[REDACTED_ARRAY]"
    except ImportError:
        pass
    if isinstance(x, (bytes, bytearray)):
        return "[REDACTED_BYTES]"
    if isinstance(x, (set,)):
        x = list(x)
    if isinstance(x, str):
        if len(x) > _max_string_length:
            return x[:_max_string_length] + "...(truncated)"
        return x
    if isinstance(x, list):
        if len(x) > _max_list_length:
            return [_sanitize_value(v) for v in x[:_max_list_length]] + ["...(truncated)"]
        return [_sanitize_value(v) for v in x]
    if isinstance(x, dict):
        keys = list(x.keys())[:_max_dict_keys]
        sanitized = {}
        for k in keys:
            sanitized[k] = _sanitize_value(x[k])
        if len(x.keys()) > _max_dict_keys:
            sanitized["_extra_keys_truncated"] = True
        return sanitized
    if _is_primitive(x):
        return x
    try:
        return str(x)[:_max_string_length]
    except Exception:
        return "[UNSERIALIZABLE]"


def _sanitize_value(x: Any) -> Any:
    if _is_primitive(x):
        if isinstance(x, str) and len(x) > _max_string_length:
            return x[:_max_string_length] + "...(truncated)"
        return x
    return _redact_value(x)


def _sanitize_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in fields.items():
        if any(s in k.lower() for s in _forbidden_key_substrings):
            out[k] = "[REDACTED]"
            continue
        out[k] = _sanitize_value(v)
    return out


def _rotate_log(path: str, max_bytes: int, backups: int) -> None:
    """Rotate log file if it exceeds max_bytes."""
    try:
        if os.path.exists(path) and os.path.getsize(path) >= max_bytes:
            for i in range(backups - 1, 0, -1):
                src = f"{path}.{i}"
                dst = f"{path}.{i+1}"
                if os.path.exists(src):
                    os.replace(src, dst)
            os.replace(path, f"{path}.1")
    except OSError as e:
        logger.warning(f"Log rotation failed: {e}")


def _get_config(key: str, default: Any) -> Any:
    """Get integer config from env var with fallback."""
    try:
        val = os.environ.get(key)
        if val is not None:
            return int(val)
    except (ValueError, TypeError):
        logger.warning(f"Invalid env var {key}={val}, using default {default}")
    return default


def record_event(event_type: str, fields: dict) -> None:
    """
    Record a structured event to the metrics log.
    Privacy-sensitive fields are automatically redacted.
    Failures are logged and do not crash the caller.
    """
    try:
        log_dir = os.path.join(os.getcwd(), "logs")
        metrics_file = os.path.join(log_dir, "metrics.jsonl")

        os.makedirs(log_dir, exist_ok=True)

        max_bytes = _get_config("URCM_LOG_MAX_BYTES", 5_000_000)
        backups = _get_config("URCM_LOG_BACKUPS", 5)

        _rotate_log(metrics_file, max_bytes, backups)

        env_name = os.environ.get("URCM_ENV")
        run_id = os.environ.get("URCM_RUN_ID")

        entry = {
            "ts": time.time(),
            "event": event_type,
            **_sanitize_fields(fields or {}),
        }
        if "env" not in entry and env_name:
            entry["env"] = env_name
        if "run_id" not in entry and run_id:
            entry["run_id"] = run_id

        with _lock:
            with open(metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    except OSError as e:
        logger.error(f"Failed to record event '{event_type}': {e}")
    except Exception as e:
        logger.exception(f"Unexpected error recording event '{event_type}': {e}")
