from urcm.core.ingest import KnowledgeIngestion
from urcm.core.executive import ExecutiveController

def test_type_guided_prefers_consistent_chain():
    bp = "urcm_type_guided.pkl"
    ing = KnowledgeIngestion(brain_path=bp, l2_dim=256)
    # Two chains to goal: strong universal and weaker mixed
    ing.ingest_text("all a are b. all b are d.")
    ing.ingest_text("some a are c. c implies d.")
    # Type annotations: keep a,b,d as same type; c different to trigger slight penalty
    ing.ingest_text("a is a entity. b is a entity. d is a entity. c is a tool.")
    ing.save()
    execu = ExecutiveController(brain_path=bp)
    path = execu.plan_a_star("a","d")
    assert path == ["a","b","d"]
