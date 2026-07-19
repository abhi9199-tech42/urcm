from urcm.core.executive import ExecutiveController


def test_explain_outputs_proof_core_and_refusal_flag():
    e = ExecutiveController()
    e.engine.brain_data = {"relations": [
        ("all","a","b"), ("no","a","b")
    ]}
    out = e.explain("a","b")
    assert isinstance(out.get("proof_core", []), list)
    assert out.get("refused", False) is True
    assert any(c for c in out.get("contradictions", []) if c[1]=="a" and c[2]=="b")
