import numpy as np
from urcm.core.logic_gates import ConstraintGraph

def test_exp_envelope_bounds_feasible():
    vars = ["x","w"]
    constraints = [
        ({"x":1.0}, ">=", 0.0),
        ({"x":1.0}, "<=", 1.0),
        ({"w":1.0}, ">=", 1.0),
        ({"w":1.0}, "<=", float(np.exp(1.0)) + 0.1),
    ]
    bounds = ConstraintGraph.solve_with_extended_envelopes(vars, constraints, square_pairs=[], bilinear_pairs=[], chain_pairs=[], segments=3)
    assert bounds["x"][0] <= bounds["x"][1]
    assert bounds["w"][0] <= bounds["w"][1]
