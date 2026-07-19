from urcm.core.executive import ExecutiveController


def test_tms_supports_on_chain_and_retract():
    e = ExecutiveController()
    e.engine.brain_data = {"relations": [
        ("all","a","b"), ("all","b","c"), ("no","b","d")
    ]}
    res = e.explain("a", "c")
    ch = res.get("discovered_chain", [])
    assert ch == ["a","b","c"]
    assert e.tms.has(("all","a","b"))
    assert e.tms.has(("all","b","c"))
    res2 = e.explain("b","d")
    assert any(c for c in res2.get("contradictions", []) if c[1]=="b" and c[2]=="d")

def test_program_dsl_verification():
    e = ExecutiveController()
    spec = {
        "pre": [lambda ctx: True],
        "steps": ["s1","s2"],
        "post": [lambda r: "s1" in r["executed"] and "s2" in r["executed"]],
        "ctx": {}
    }
    out = e.verify_program(spec)
    assert out["pre_ok"] is True
    assert out["post_ok"] is True
    assert out["executed"] == ["s1","s2"]
