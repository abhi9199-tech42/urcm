from urcm.core.executive import ExecutiveController

def test_run_loop_metrics_and_refusal_threshold():
    execu = ExecutiveController()
    execu.set_initial_state("truth")
    execu.add_goal("Reach unreachable", target_concept="nonexistent_goal", priority=1.0)
    execu.explain_refusal_threshold = 0.5
    traj = execu.run_loop(max_steps=5)
    assert isinstance(traj, list)
    assert "Low explain quality" in execu.last_refused_reason
    metrics = execu.last_loop_metrics
    assert metrics.get("steps", 0) >= 1
    assert metrics.get("avg_step_ms", 0.0) >= 0.0
    assert metrics.get("frustrations", 0) >= 0
