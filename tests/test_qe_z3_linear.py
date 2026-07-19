import pytest

from urcm.core.logic_gates import FormalLogic

z3 = pytest.importorskip("z3", reason="z3-solver not installed")

def test_qe_z3_exists_interval_true():
    s = "exists x: x >= 1 and x <= 2"
    out = FormalLogic.quantifier_eliminate_z3_linear(s)
    assert out is not None
    assert "True" in out or "true" in out

def test_qe_z3_forall_false():
    s = "forall x: x >= 1"
    out = FormalLogic.quantifier_eliminate_z3_linear(s)
    assert out is not None
    assert "False" in out or "false" in out
