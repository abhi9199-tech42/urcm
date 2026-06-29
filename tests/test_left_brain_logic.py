from urcm.core.ingest import KnowledgeIngestion
from urcm.core.executive import ExecutiveController

def seed_basic_logic(brain_path: str):
    ing = KnowledgeIngestion(brain_path=brain_path, l2_dim=512)
    text = (
        "rain implies wet. "
        "no rain are wet. "
        "all man are mortal. "
        "socrates is man. "
    )
    ing.ingest_text(text)
    ing.save()
    return brain_path

def test_modus_ponens_returns_consequent():
    bp = seed_basic_logic("urcm_lb_logic.pkl")
    execu = ExecutiveController(brain_path=bp)
    q = execu.modus_ponens("rain")
    assert q == "wet"

def test_modus_tollens_returns_not_premise():
    bp = seed_basic_logic("urcm_lb_logic.pkl")
    execu = ExecutiveController(brain_path=bp)
    res = execu.modus_tollens("rain", "wet")
    assert res == ("not", "rain")

def test_syllogism_chain_socrates_man_mortal():
    bp = seed_basic_logic("urcm_lb_logic.pkl")
    execu = ExecutiveController(brain_path=bp)
    path = execu.syllogism("socrates", "man", "mortal")
    assert path == ["socrates", "man", "mortal"]

def test_explain_quality_scores_chain_over_contradictions():
    bp = seed_basic_logic("urcm_lb_logic.pkl")
    execu = ExecutiveController(brain_path=bp)
    q = execu.explain_quality("socrates", "mortal")
    assert q["score"] >= 2.0
