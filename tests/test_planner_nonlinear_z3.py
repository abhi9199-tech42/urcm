from urcm.core.ingest import KnowledgeIngestion
from urcm.core.executive import ExecutiveController

def test_planner_nonlinear_smt_path():
    bp = "urcm_planner_nl.pkl"
    ing = KnowledgeIngestion(brain_path=bp)
    ing.ingest_text("all a are b.")
    bd = ing.brain_data
    bd["numeric_constraints"] = [
        ({"x":1.0}, ">=", 1.0),
        ({"y":1.0}, ">=", 1.0),
        ({"w":1.0}, "<=", 3.0),
    ]
    bd["bilinear_pairs"] = [("w","x","y")]
    bd["relations"] = [("all","a","b")]
    ing.brain_data = bd
    ing.save()
    e = ExecutiveController(brain_path=bp)
    path = e.plan_a_star("a","b")
    assert path == ["a","b"]
