from urcm.core.logic_gates import SMTBridge

def test_smt_nonlinear_z3_basic_product():
    constraints = [
        ({"x":1.0}, ">=", 1.0),
        ({"y":1.0}, ">=", 1.0),
        ({"w":1.0}, "<=", 3.0),
    ]
    bilinear_pairs = [("w","x","y")]
    res = SMTBridge.solve_nonlinear_z3(constraints, bilinear_pairs=bilinear_pairs, square_pairs=[])
    assert res is not None
    assert res.get("x", 0.0) >= 1.0 and res.get("y", 0.0) >= 1.0
    assert res.get("w", 0.0) <= 3.0
