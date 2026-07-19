from urcm.core.executive import ExecutiveController
from urcm.core.ingest import KnowledgeIngestion


def test_planning_bilinear_infeasible():
    bp = "urcm_bilinear2.pkl"
    ing = KnowledgeIngestion(brain_path=bp)
    ing.ingest_text("all a are b.")
    bd = ing.brain_data if hasattr(ing, "brain_data") else {}
    bd["numeric_constraints"] = [
        ({"x":1.0}, "<=", 2.0),
        ({"y":1.0}, "<=", 3.0),
        ({"x":1.0}, ">=", 0.0),
        ({"y":1.0}, ">=", 1.0),
        ({"w":1.0}, ">=", 7.0),
    ]
    bd["bilinear_pairs"] = [("w","x","y")]
    ing.brain_data = bd
    ing.save()
    e = ExecutiveController(brain_path=bp)
    path = e.plan_a_star("a","b")
    assert path == []
