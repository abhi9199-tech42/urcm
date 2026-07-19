from urcm.core.executive import ExecutiveController
from urcm.core.ingest import KnowledgeIngestion


def test_assess_mastery_with_seed_relations():
    bp = "urcm_mastery.pkl"
    ing = KnowledgeIngestion(brain_path=bp, l2_dim=512)
    text = "a implies b. no c are d."
    ing.ingest_text(text)
    ing.save()
    execu = ExecutiveController(brain_path=bp)
    result = execu.assess_mastery()
    caps = result["capabilities"]
    assert caps["chain_discovery"]
    assert caps["contradiction_detection"]
    assert result["score"] >= 4
