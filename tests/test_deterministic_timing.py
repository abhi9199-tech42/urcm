import numpy as np
from urcm.core.executive import ExecutiveController

def test_deterministic_timing_and_seed():
    e = ExecutiveController()
    e.current_state = np.zeros(e.engine.l2_dim)
    e.max_wall_ms = 10.0
    e.deterministic_seed = 123
    traj1 = e.run_loop(max_steps=100)
    m1 = e.last_loop_metrics
    e2 = ExecutiveController()
    e2.current_state = np.zeros(e2.engine.l2_dim)
    e2.max_wall_ms = 10.0
    e2.deterministic_seed = 123
    traj2 = e2.run_loop(max_steps=100)
    m2 = e2.last_loop_metrics
    assert m1.get("deterministic") and m2.get("deterministic")
    assert m1.get("elapsed_ms") is not None and m2.get("elapsed_ms") is not None
