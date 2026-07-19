from urcm.core.logic_gates import ConstraintGraph


def test_envelope_integration_bounds_w():
    vars = ["x","y","w"]
    constraints = [
        ({"x":1.0}, "<=", 2.0),
        ({"y":1.0}, "<=", 3.0),
        ({"x":1.0}, ">=", 0.0),
        ({"y":1.0}, ">=", 1.0),
    ]
    bilinear = [("w","x","y")]
    bounds = ConstraintGraph.solve_with_envelopes(vars, constraints, bilinear, iters=3)
    assert bounds["w"][1] <= 6.0 + 1e-6
    assert bounds["x"][0] >= 0.0 - 1e-6
    assert bounds["y"][0] >= 1.0 - 1e-6

def test_project_feasible_point():
    vars = ["x","y"]
    constraints = [
        ({"x":1.0}, "<=", 2.0),
        ({"y":1.0}, "<=", 3.0),
        ({"x":1.0,"y":1.0}, "<=", 4.0),
        ({"x":1.0}, ">=", 0.0),
        ({"y":1.0}, ">=", 0.0),
    ]
    b = {v: (-1e3, 1e3) for v in vars}
    b["x"] = (0.0, 2.0)
    b["y"] = (0.0, 3.0)
    x = ConstraintGraph.project_feasible_point(b, constraints, steps=200, alpha=0.2)
    s1 = x["x"]
    s2 = x["y"]
    assert 0.0 - 1e-6 <= s1 <= 2.0 + 1e-6
    assert 0.0 - 1e-6 <= s2 <= 3.0 + 1e-6
    assert s1 + s2 <= 4.0 + 1e-6
