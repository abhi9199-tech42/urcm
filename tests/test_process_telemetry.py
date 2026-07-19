import json
import os

from urcm.core.system import URCMSystem


def _read_events(tmp_dir):
    p = os.path.join(tmp_dir, "logs", "metrics.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f.read().splitlines() if line.strip()]

def test_process_query_emits_telemetry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sys = URCMSystem()
    sys.process_query("hello world")
    evs = _read_events(str(tmp_path))
    kinds = {e.get("event") for e in evs}
    assert "process_start" in kinds
    assert "process_end" in kinds
