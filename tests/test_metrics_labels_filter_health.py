import os
import json
from urcm.ops.metrics_exporter import render_metrics, compute_health

def write(fp, ev):
    with open(fp, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")

def test_prometheus_labels_and_filtering(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = os.path.join(tmp_path, "logs")
    os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, "metrics.jsonl")
    # dev env
    write(fp, {"event": "mu_path_complete", "converged": True, "reason": "Convergence", "env": "dev"})
    write(fp, {"event": "mu_path_complete", "converged": False, "reason": "Max Steps Reached", "env": "dev"})
    write(fp, {"event": "process_end", "final_mu": 0.5, "env": "dev"})
    # test env
    write(fp, {"event": "mu_path_complete", "converged": False, "reason": "Dead End", "env": "test"})
    text = render_metrics(str(tmp_path), env="dev")
    # HELP/TYPE lines present
    assert "# HELP urcm_process_total" in text
    assert "# TYPE urcm_process_total counter" in text
    # Labels present and filtered by env
    assert 'urcm_mu_path_complete_total{reason="converged",env="dev"}' in text
    assert 'urcm_mu_path_complete_total{reason="dead_end",env="dev"} 0' in text
    assert 'urcm_process_total{env="dev"}' in text

def test_health_endpoint_logic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = os.path.join(tmp_path, "logs")
    os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, "metrics.jsonl")
    # Compose 4 completions: 1 max_steps and 3 converged
    write(fp, {"event": "mu_path_complete", "converged": True, "reason": "Convergence", "env": "dev"})
    write(fp, {"event": "mu_path_complete", "converged": True, "reason": "Convergence", "env": "dev"})
    write(fp, {"event": "mu_path_complete", "converged": True, "reason": "Convergence", "env": "dev"})
    write(fp, {"event": "mu_path_complete", "converged": False, "reason": "Max Steps Reached", "env": "dev"})
    write(fp, {"event": "process_end", "final_mu": 0.55, "env": "dev"})
    os.environ["URCM_SLO_MIN_FINAL_MU"] = "0.5"
    os.environ["URCM_SLO_MAX_STEPS_RATE"] = "0.5"
    health = compute_health(str(tmp_path), env="dev")
    assert health["ok"] is True
    assert health["totals"]["completions"] == 4
