import random
import time
from urcm.core.ingest import KnowledgeIngestion
from urcm.core.executive import ExecutiveController

def test_randomized_envelopes_latency_distribution():
    random.seed(42)
    runs = 20
    times = []
    successes = 0
    for i in range(runs):
        bp = f"urcm_rand_{i}.pkl"
        ing = KnowledgeIngestion(brain_path=bp, l2_dim=128)
        ing.ingest_text("all s are t.")
        cs = []
        n = 12
        for j in range(1, n+1):
            lo = 0.0
            hi = random.uniform(3.0, 6.0)
            cs.append(({f"x{j}":1.0}, ">=", lo))
            cs.append(({f"x{j}":1.0}, "<=", hi))
        pairs = []
        for k in range(3):
            a = f"x{random.randint(1, n)}"
            b = f"x{random.randint(1, n)}"
            w = f"w{k+1}"
            pairs.append((w, a, b))
        ing.brain_data["numeric_constraints"] = cs + [({pairs[0][0]:1.0}, "<=", 8.0)]
        ing.brain_data["bilinear_pairs"] = pairs
        ing.brain_data["square_pairs"] = [(f"z{k+1}", f"x{random.randint(1, n)}") for k in range(2)]
        ing.brain_data["int_vars"] = ["n_i"]
        ing.save()
        e = ExecutiveController(brain_path=bp)
        t0 = time.perf_counter()
        path = e.plan_a_star("s","t")
        dt = (time.perf_counter() - t0)
        times.append(dt)
        if isinstance(path, list):
            successes += 1
    times.sort()
    p95 = times[int(0.95 * runs) - 1]
    assert successes >= runs - 2
    assert p95 < 3.0
