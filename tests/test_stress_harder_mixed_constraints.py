import time
from urcm.core.ingest import KnowledgeIngestion
from urcm.core.executive import ExecutiveController

def test_extreme_mixed_envelopes_latency_and_result():
    bp = "urcm_stress_extreme.pkl"
    ing = KnowledgeIngestion(brain_path=bp, l2_dim=256)
    ing.ingest_text("all s are t.")
    cs = []
    for i in range(1, 31):
        cs.append(({f"x{i}":1.0}, ">=", 0.0))
        cs.append(({f"x{i}":1.0}, "<=", 5.0))
    cs.append(({"x1":1.0,"x2":1.0,"x3":1.0,"x4":1.0,"x5":1.0,"x6":1.0}, "<=", 12.0))
    ing.brain_data["numeric_constraints"] = cs
    ing.brain_data["bilinear_pairs"] = [("w1","x1","x2"), ("w2","x3","x4"), ("w3","x5","x6"), ("w4","x7","x8"), ("w5","x9","x10"), ("w6","x11","x12"), ("w7","x13","x14")]
    ing.brain_data["square_pairs"] = [("z1","x15"), ("z2","x16"), ("z3","x17"), ("z4","x18")]
    ing.brain_data["chain_pairs"] = [("c1","x19","x20"), ("c2","x21","x22"), ("c3","x23","x24")]
    ing.brain_data["log_pairs"] = [("u_log1","x25"), ("u_log2","x26")]
    ing.brain_data["sigmoid_pairs"] = [("u_sig1","x27"), ("u_sig2","x28")]
    ing.brain_data["int_vars"] = ["n_a", "n_b", "n_c"]
    ing.save()
    execu = ExecutiveController(brain_path=bp)
    t0 = time.perf_counter()
    path = execu.plan_a_star("s","t")
    dt = (time.perf_counter() - t0)
    assert isinstance(path, list)
    assert dt < 6.0

def test_extreme_infeasible_refusal_fast():
    bp = "urcm_stress_extreme_refusal.pkl"
    ing = KnowledgeIngestion(brain_path=bp, l2_dim=256)
    ing.ingest_text("all a are b.")
    ing.brain_data["numeric_constraints"] = [
        ({"x":1.0}, ">=", 0.0),
        ({"y":1.0}, ">=", 0.0),
        ({"x":1.0}, "<=", 1.0),
        ({"y":1.0}, "<=", 1.0),
        ({"w":1.0}, ">=", 100.0),
    ]
    ing.brain_data["bilinear_pairs"] = [("w","x","y")]
    ing.save()
    execu = ExecutiveController(brain_path=bp)
    t0 = time.perf_counter()
    plan = execu.plan_a_star("a","b")
    dt = (time.perf_counter() - t0)
    assert plan == []
    assert dt < 2.5
