from urcm.core.logic_gates import ConstraintGraph, Polytope


def test_polytope_bounds_and_contains():
    P = Polytope([
        ({"x":1.0}, ">=", 0.0),
        ({"x":1.0}, "<=", 2.0),
        ({"y":1.0}, ">=", 1.0),
        ({"y":1.0}, "<=", 3.0),
    ])
    Q = Polytope([
        ({"x":1.0, "y":1.0}, "<=", 4.0)
    ])
    R = P.intersection(Q)
    b = R.bounds(["x","y"])
    assert b["x"][0] <= b["x"][1] and b["y"][0] <= b["y"][1]
    assert R.contains({"x":1.0,"y":2.0})
    assert not R.contains({"x":3.0,"y":3.0})

def test_abs_linearization_constraints():
    cs = Polytope.linearize_abs("x","t")
    bounds = ConstraintGraph.solve_boxes(["x","t"], cs + [({"x":1.0}, ">=", -1.0), ({"x":1.0}, "<=", 1.0)])
    assert bounds["t"][0] >= 0.0
