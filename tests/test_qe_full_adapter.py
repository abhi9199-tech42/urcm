from urcm.core.logic_gates import FormalLogic


def test_quantifier_eliminate_full_fallback():
    s = "forall x: p(x) implies q(x)"
    out = FormalLogic.quantifier_eliminate_full(s, ["a","b"])
    assert "p(a)" in out and "q(a)" in out
    assert "p(b)" in out and "q(b)" in out
