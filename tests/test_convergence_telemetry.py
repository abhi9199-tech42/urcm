import os
import json
import numpy as np
from urcm.core.convergence_engine import MuConvergenceEngine
from urcm.core.data_models import ResonanceState

def _read_events(tmp_dir):
    p = os.path.join(tmp_dir, "logs", "metrics.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f.read().splitlines() if line.strip()]

def test_engine_emits_telemetry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    eng = MuConvergenceEngine(convergence_epsilon=1e-3, max_steps=5, competition_beam_width=1)
    v0 = np.ones(8)
    init = ResonanceState(
        resonance_vector=v0,
        mu_value=0.0,
        rho_density=0.0,
        chi_cost=0.0,
        stability_score=0.0,
        oscillation_phase=0.0,
        timestamp=0.0
    )
    def generator(state):
        return [ResonanceState(
            resonance_vector=state.resonance_vector,
            mu_value=0.0,
            rho_density=0.0,
            chi_cost=0.0,
            stability_score=0.0,
            oscillation_phase=0.0,
            timestamp=0.0
        )]
    eng.run_reasoning_loop(init, generator)
    evs = _read_events(str(tmp_path))
    kinds = {e.get("event") for e in evs}
    assert "mu_loop_start" in kinds
    assert "mu_step" in kinds
    assert "mu_path_complete" in kinds
