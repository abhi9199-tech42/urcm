from urcm.core.ingest import KnowledgeIngestion
from urcm.core.executive import ExecutiveController

def test_planner_uses_smt_or_envelopes():
    bp = "urcm_planner_smt.pkl"
    ing = KnowledgeIngestion(brain_path=bp)
    ing.ingest_text("all a are b. x >= 1. y >= 1.")
    bd = ing.brain_data
    bd["numeric_constraints"] = [
        ({"x":1.0}, ">=", 1.0),
        ({"y":1.0}, ">=", 1.0),
        ({"x":1.0,"y":1.0}, "<=", 3.0),
    ]
    ing.brain_data = bd
    ing.save()
    e = ExecutiveController(brain_path=bp)
    path = e.plan_a_star("a","b")
    assert path == ["a","b"]
