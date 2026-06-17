import time
from urcm.core.ingest import KnowledgeIngestion
from urcm.core.executive import ExecutiveController

def test_hard_envelopes_performance_and_feasibility():
    bp = "urcm_stress_hard.pkl"
    ing = KnowledgeIngestion(brain_path=bp, l2_dim=256)
    ing.ingest_text("all s are t.")
    cs = []
    # 20 variables with box constraints
    for i in range(1, 21):
        cs.append(({f"x{i}":1.0}, ">=", 0.0))
        cs.append(({f"x{i}":1.0}, "<=", 5.0))
    # Sum cap across a subset to induce coupling
    cs.append(({"x1":1.0,"x2":1.0,"x3":1.0,"x4":1.0,"x5":1.0}, "<=", 10.0))
    ing.brain_data["numeric_constraints"] = cs
    # Multiple nonlinear relations
    ing.brain_data["bilinear_pairs"] = [("w1","x1","x2"), ("w2","x3","x4"), ("w3","x5","x6"), ("w4","x7","x8"), ("w5","x9","x10")]
    ing.brain_data["square_pairs"] = [("z1","x11"), ("z2","x12"), ("z3","x13")]
    # Chain/log/sigmoid to increase hardness
    ing.brain_data["chain_pairs"] = [("c1","x14","x15"), ("c2","x16","x17")]
    ing.brain_data["log_pairs"] = [("u_log","x18")]
    ing.brain_data["sigmoid_pairs"] = [("u_sig","x19")]
    # Some integers
    ing.brain_data["int_vars"] = ["n_a", "n_b"]
    ing.save()
    execu = ExecutiveController(brain_path=bp)
    t0 = time.perf_counter()
    path = execu.plan_a_star("s","t")
    dt = (time.perf_counter() - t0)
    assert isinstance(path, list)
    assert dt < 5.0

def test_hard_integer_projection_violation_integrity():
    bp = "urcm_stress_hard_int.pkl"
    ing = KnowledgeIngestion(brain_path=bp, l2_dim=256)
    ing.ingest_text("all a are b. all b are d.")
    # Tight constraints causing projection violation with an integer variable
    ing.brain_data["numeric_constraints"] = [
        ({"n_k":1.0}, ">=", 0.0),
        ({"n_k":1.0}, "<=", 4.0),
        ({"y":1.0}, ">=", 0.0),
        ({"y":1.0}, "<=", 1.5),
        ({"n_k":1.0, "y":1.0}, "<=", 1.0),
    ]
    ing.brain_data["int_vars"] = ["n_k"]
    ing.brain_data["bilinear_pairs"] = [("w","y","y")]
    ing.save()
    execu = ExecutiveController(brain_path=bp)
    t0 = time.perf_counter()
    plan = execu.plan_a_star("a","d")
    dt = (time.perf_counter() - t0)
    assert plan in ([], ["a","b","d"])
    last = execu.last_loop_metrics.get("last_counterexample", {})
    if last:
        nk = last.get("assignment", {}).get("n_k", None)
        assert nk is not None
        assert abs(round(nk) - nk) < 1e-9
    assert dt < 5.0
