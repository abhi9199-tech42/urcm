from urcm.core.executive import ExecutiveController
from urcm.core.ingest import KnowledgeIngestion


def seed_medical_knowledge(brain_path: str = "urcm_medical.pkl") -> str:
    """
    Seeds core medical triage rules and contraindications.
    Returns the brain path used for seeding.
    """
    ing = KnowledgeIngestion(brain_path=brain_path, l2_dim=512)
    corpus = (
        "chest_pain implies ecg. "
        "ecg implies troponin. "
        "shortness_of_breath implies chest_xray. "
        "fever implies infection. "
        "no allergy_penicillin are given_penicillin. "
        "all anticoagulant_patients are monitored except active_bleeding. "
        "hypertension implies bp_check. "
        "diabetes implies glucose_check. "
        "stroke implies ct_scan. "
    )
    ing.ingest_text(corpus)
    ing.save()
    return brain_path

def demo_medical_explain(start: str, goal: str, brain_path: str = "urcm_medical.pkl"):
    """
    Returns discovered chain and contradictions for a medical query.
    """
    execu = ExecutiveController(brain_path=brain_path)
    return execu.explain(start, goal)
