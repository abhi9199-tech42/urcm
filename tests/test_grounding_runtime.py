from urcm.core.ingest import KnowledgeIngestion
from urcm.core.executive import ExecutiveController

def test_grounding_auto_populates_constraints_and_planner():
    bp = "urcm_runtime_grounding.pkl"
    ing = KnowledgeIngestion(brain_path=bp)
    text = "all cats are animals. w = x*y. x <= 2. y >= 1. z equals x squared."
    ing.ingest_text(text)
    ing.save()
    e = ExecutiveController(brain_path=bp)
    bd = e.engine.brain_data
    assert len(bd.get("numeric_constraints", [])) >= 2
    assert ("w","x","y") in bd.get("bilinear_pairs", [])
    assert ("z","x") in bd.get("square_pairs", [])
    # trivial planner path using relations
    path = e.plan_a_star("cats", "animals")
    assert path == ["cats","animals"]
