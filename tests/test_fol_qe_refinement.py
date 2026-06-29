from urcm.core.logic_gates import FormalLogic

def test_quantifier_eliminate_ground_forall_exists():
    q = "forall x: p(x) implies q(x)"
    out = FormalLogic.quantifier_eliminate_ground(q, ["a","b"])
    assert "p(a)" in out and "q(a)" in out and "p(b)" in out and "q(b)" in out
    e = "exists x: r(x)"
    out2 = FormalLogic.quantifier_eliminate_ground(e, ["c"])
    assert "r(c)" in out2

def test_fol_resolution_with_subsumption():
    clauses = [
        [("not", ("P","x")), ("Q","x")],
        [("not", ("Q","y")), ("R","y")],
        [("P","a")]
    ]
    assert FormalLogic.resolve_fol(clauses, ("R","a"))
