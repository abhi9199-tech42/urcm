import os
from urcm.core.ingest import KnowledgeIngestion
from urcm.core.executive import ExecutiveController

def test_explain_candidates_prioritization():
    brain_path = "urcm_prioritization.pkl"
    try:
        if os.path.exists(brain_path):
            os.remove(brain_path)
    except Exception:
        pass
    ing = KnowledgeIngestion(brain_path=brain_path, l2_dim=256)
    # Strong chain: a all b, b all d
    ing.ingest_text("all a are b. all b are d.")
    # Weaker chain: a some c, c implies d
    ing.ingest_text("some a are c. c implies d.")
    ing.save()

    execu = ExecutiveController(brain_path=brain_path)
    cands = execu.explain_candidates("a", "d", k=2)
    # Top candidate should be the all-all chain
    assert cands and cands[0]["chain"] == ["a","b","d"]
    # Weaker candidate also present
    assert any(c["chain"] == ["a","c","d"] for c in cands)
    try:
        os.remove(brain_path)
    except Exception:
        pass
