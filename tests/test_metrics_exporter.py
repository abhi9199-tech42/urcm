import json
import os

from urcm.ops.metrics_exporter import render_metrics


def write_event(fp, ev):
    with open(fp, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")

def test_exporter_counts_and_gauges(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log_dir = os.path.join(tmp_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    fp = os.path.join(log_dir, "metrics.jsonl")
    write_event(fp, {"event": "mu_loop_start", "mu": 0.1, "rho": 0.2, "chi": 0.1})
    write_event(fp, {"event": "mu_step", "step": 1, "mu": 0.2, "delta_mu": 0.1})
    write_event(fp, {"event": "mu_oscillation_detected"})
    write_event(fp, {"event": "mu_path_complete", "converged": True, "final_mu": 0.6, "reason": "Convergence"})
    write_event(fp, {"event": "process_end", "final_mu": 0.6, "paths": 1})
    text = render_metrics(str(tmp_path))
    metrics = {}
    for l in text.strip().splitlines():
        if not l or l.startswith("#"):
            continue
        parts = l.split()
        if len(parts) != 2:
            continue
        metrics[parts[0]] = float(parts[1])
    assert metrics['urcm_mu_step_total{env=""}'] == 1
    assert metrics['urcm_oscillation_total{env=""}'] == 1
    assert metrics['urcm_mu_path_complete_total{reason="converged",env=""}'] == 1
    assert metrics['urcm_process_total{env=""}'] == 1
    assert metrics['urcm_last_final_mu{env=""}'] == 0.6
    assert metrics['urcm_last_delta_mu{env=""}'] == 0.1
