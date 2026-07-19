from urcm.core.logic_gates import ConstraintGraph


def test_constraint_graph_three_vars():
    vars = ["x","y","z"]
    constraints = [
        ({"x":1.0}, "<=", 3.0),
        ({"y":1.0}, "<=", 2.0),
        ({"z":1.0}, ">=", 0.0),
        ({"x":1.0,"y":1.0,"z":1.0}, "<=", 5.0)
    ]
    bounds = ConstraintGraph.solve_boxes(vars, constraints)
    assert bounds["x"][0] <= 3.0 <= bounds["x"][1]
    assert bounds["y"][0] <= 2.0 <= bounds["y"][1]
    assert bounds["z"][0] <= 0.0 <= bounds["z"][1]

def test_fourier_motzkin_bounds_basic():
    vars = ["x","y"]
    constraints = [
        ({"x":1.0}, "<=", 4.0),
        ({"x":1.0,"y":1.0}, "<=", 6.0),
        ({"y":1.0}, "<=", 5.0)
    ]
    bounds = ConstraintGraph.fourier_motzkin_bounds(vars, constraints)
    assert bounds["x"][1] <= 4.0 + 1e-9
    assert bounds["y"][1] <= 5.0 + 1e-9
