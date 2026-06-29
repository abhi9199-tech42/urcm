import time
from urcm.core.ingest import KnowledgeIngestion
from urcm.core.executive import ExecutiveController

def test_projection_rounding_integer_preserved_in_counterexample():
    bp = "urcm_stress_int_proj.pkl"
    ing = KnowledgeIngestion(brain_path=bp, l2_dim=128)
    ing.ingest_text("all a are b. all b are d.")
    ing.brain_data["numeric_constraints"] = [
        ({"n_k":1.0}, ">=", 0.0),
        ({"n_k":1.0}, "<=", 3.0),
        ({"y":1.0}, ">=", 0.0),
        ({"y":1.0}, "<=", 2.0),
        ({"n_k":1.0, "y":1.0}, "<=", 1.0),
    ]
    ing.brain_data["int_vars"] = ["n_k"]
    ing.save()
    execu = ExecutiveController(brain_path=bp)
    t0 = time.perf_counter()
    plan = execu.plan_a_star("a","d")
    dt = (time.perf_counter() - t0)
    assert plan in ([], ["a","b","d"])
    last = execu.last_loop_metrics.get("last_counterexample", {})
    # On projection violation branch, ensure integer rounding respected
    if last:
        assign = last.get("assignment", {})
        nk = assign.get("n_k", None)
        assert nk is not None
        # Check integrality after rounding/clamping
        assert abs(round(nk) - nk) < 1e-9
    # Ensure stress path returns quickly
    assert dt < 2.0

def test_stress_mixed_constraints_small_batch():
    bp = "urcm_stress_mixed.pkl"
    ing = KnowledgeIngestion(brain_path=bp, l2_dim=128)
    ing.ingest_text("all s are t.")
    cs = []
    for i in range(1, 11):
        cs.append(({f"x{i}":1.0}, ">=", 0.0))
        cs.append(({f"x{i}":1.0}, "<=", 5.0))
    cs.append(({"w":1.0}, "<=", 6.0))
    ing.brain_data["numeric_constraints"] = cs
    ing.brain_data["bilinear_pairs"] = [("w","x1","x2")]
    ing.brain_data["square_pairs"] = [("z","x3")]
    ing.brain_data["int_vars"] = ["n_m", "n_n"]
    ing.save()
    execu = ExecutiveController(brain_path=bp)
    t0 = time.perf_counter()
    path = execu.plan_a_star("s","t")
    dt = (time.perf_counter() - t0)
    assert isinstance(path, list)
    assert dt < 2.0
