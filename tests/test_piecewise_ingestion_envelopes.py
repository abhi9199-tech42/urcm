from urcm.core.ingest import KnowledgeIngestion
from urcm.core.executive import ExecutiveController

def test_piecewise_ingestion_planning():
    bp = "urcm_piecewise.pkl"
    ing = KnowledgeIngestion(brain_path=bp)
    ing.ingest_text("all a are b. x >= 0. x <= 2. w piecewise x: ((0,0),(1,1),(2,4)).")
    bd = ing.brain_data
    bd["relations"] = [("all","a","b")]
    ing.brain_data = bd
    ing.save()
    e = ExecutiveController(brain_path=bp)
    path = e.plan_a_star("a", "b")
    assert path == ["a","b"]
