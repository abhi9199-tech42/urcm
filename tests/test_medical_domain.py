from urcm.domains.medical import seed_medical_knowledge, demo_medical_explain

def test_medical_chain_chest_pain_to_ecg():
    bp = seed_medical_knowledge()
    result = demo_medical_explain("chest_pain", "ecg", brain_path=bp)
    chain = result.get("discovered_chain", [])
    assert chain and chain[0] == "chest_pain" and chain[-1] == "ecg"

def test_medical_contradiction_penicillin_allergy():
    bp = seed_medical_knowledge()
    result = demo_medical_explain("allergy_penicillin", "given_penicillin", brain_path=bp)
    contradictions = result.get("contradictions", [])
    assert any(c[0] == "no" and c[1] == "allergy_penicillin" and c[2] == "given_penicillin" for c in contradictions)
