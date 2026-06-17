from urcm.core.logic_gates import ConstraintGraph

def test_square_envelope_bounds_feasible():
    vars = ["x","w"]
    constraints = [
        ({"x":1.0}, ">=", -2.0),
        ({"x":1.0}, "<=", 2.0),
        ({"w":1.0}, ">=", 0.0),
        ({"w":1.0}, "<=", 4.1),
    ]
    bounds = ConstraintGraph.solve_with_extended_envelopes(vars, constraints, square_pairs=[("w","x")], segments=4)
    assert bounds["x"][0] <= bounds["x"][1]
    assert bounds["w"][0] <= bounds["w"][1]

def test_chain_product_envelopes():
    vars = ["x","y","z","u","w"]
    constraints = [
        ({"x":1.0}, ">=", 0.0),
        ({"x":1.0}, "<=", 3.0),
        ({"y":1.0}, ">=", 0.5),
        ({"y":1.0}, "<=", 2.0),
        ({"z":1.0}, ">=", 0.5),
        ({"z":1.0}, "<=", 2.0),
        ({"w":1.0}, "<=", 12.0),
    ]
    bounds = ConstraintGraph.solve_with_extended_envelopes(vars, constraints, bilinear_pairs=[("u","x","y")], chain_pairs=[("w","u","z")], segments=3)
    assert bounds["w"][0] <= bounds["w"][1]
