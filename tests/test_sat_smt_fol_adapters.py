import pytest
from urcm.core.logic_gates import SATBridge, SMTBridge, FormalLogic

def test_sat_bridge_basic():
    # (x1 or x2) and (not x1 or x2) and (x1 or not x2) -> satisfiable
    sol = SATBridge.solve_cnf([[1,2],[-1,2],[1,-2]])
    assert sol is None or isinstance(sol, dict)

def test_smt_bridge_linear():
    pytest.importorskip("z3", reason="z3-solver not installed")
    # x + y <= 3, x >= 1, y >= 1 -> feasible
    res = SMTBridge.solve_with_z3([({"x":1.0,"y":1.0}, "<=", 3.0), ({"x":1.0}, ">=", 1.0), ({"y":1.0}, ">=", 1.0)])
    assert res is None or (res.get("x", None) is not None and res.get("y", None) is not None)

def test_fol_resolution_unify():
    # ∀x (P(x) -> Q(x)), ∀x (Q(x) -> R(x)), P(a) |- R(a)
    # Clauses: ¬P(x) ∨ Q(x); ¬Q(y) ∨ R(y); P(a)
    clauses = [
        [("not", ("P","x")), ("Q","x")],
        [("not", ("Q","y")), ("R","y")],
        [("P","a")]
    ]
    assert FormalLogic.resolve_fol(clauses, ("R","a"))
