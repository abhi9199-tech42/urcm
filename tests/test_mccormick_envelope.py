from urcm.core.logic_gates import ConstraintGraph


def test_mccormick_envelope_shapes():
    bounds = {"x": (0.0, 2.0), "y": (1.0, 3.0), "w": (-100.0, 100.0)}
    cs = ConstraintGraph.mccormick_envelope("x","y",bounds)
    assert isinstance(cs, list) and len(cs) == 4
    for coeffs, op, c in cs:
        assert op == "<="
        assert isinstance(coeffs, dict)
