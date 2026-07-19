from urcm.core.executive import ExecutiveController


def test_exec_records_proof_core_on_contradiction():
    e = ExecutiveController()
    e.engine.brain_data = {"relations": [
        ("all","a","b"), ("no","a","b")
    ]}
    res = e.explain("a","b")
    assert any(c for c in res.get("contradictions", []) if c[1]=="a" and c[2]=="b")
    assert isinstance(e.last_proof_core, list)
    assert e.last_refused_reason == "contradiction"
