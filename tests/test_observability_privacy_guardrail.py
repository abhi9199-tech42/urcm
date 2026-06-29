import os
import json
import numpy as np
from urcm.core.observability import record_event

def _read_last_log_line(log_dir):
    p = os.path.join(log_dir, "logs", "metrics.jsonl")
    with open(p, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    assert lines, f"No log lines found in {p}"
    return json.loads(lines[-1])

def test_redacts_forbidden_keys_and_large_payloads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    record_event("privacy_test", {"resonance_vector": np.ones(8), "embedding": np.arange(10), "ok": "x"})
    entry = _read_last_log_line(str(tmp_path))
    assert entry["event"] == "privacy_test"
    assert entry["resonance_vector"] == "[REDACTED]"
    assert entry["embedding"] == "[REDACTED]"
    assert entry["ok"] == "x"

def test_truncates_long_list_and_string(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    long_list = list(range(200))
    long_str = "a" * 600
    record_event("truncate_test", {"long_list": long_list, "long_str": long_str})
    entry = _read_last_log_line(str(tmp_path))
    assert isinstance(entry["long_list"], list)
    assert entry["long_list"][-1] == "...(truncated)"
    assert isinstance(entry["long_str"], str)
    assert entry["long_str"].endswith("...(truncated)")
