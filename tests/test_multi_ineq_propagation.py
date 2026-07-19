from urcm.core.logic_gates import ConstraintPropagation


def test_two_variable_inequalities_box_intersection():
    res = ConstraintPropagation.solve_inequalities2([
        (1.0, 0.0, 3.0, "<="),
        (1.0, 0.0, 1.0, ">="),
        (0.0, 1.0, 3.0, "<="),
        (1.0, 1.0, 5.0, "<=")
    ])
    x = res["x"]
    y = res["y"]
    assert x[0] <= 1.0 <= x[1]
    assert x[0] <= 3.0 <= x[1]
    assert y[0] <= 3.0 <= y[1]
    assert res["infeasible"] is False

def test_infeasible_detection():
    res = ConstraintPropagation.solve_inequalities2([
        (1.0, 0.0, 1.0, "<="),
        (1.0, 0.0, 3.0, ">=")
    ])
    assert res["infeasible"] is True
