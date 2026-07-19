import pytest

from urcm.core.logic_gates import MathSymbolics


def test_rewrite_distribute_and_factor():
    d = MathSymbolics.rewrite_distribute("a*(b+c)")
    assert d == "a*b+a*c"
    f = MathSymbolics.rewrite_factor("a*b+a*c")
    assert f == "a*(b+c)"
    nf = MathSymbolics.rewrite_factor("a*b+x*c")
    assert nf is None

def test_solve_inequality_interval():
    lo, hi = MathSymbolics.solve_inequality_interval(2.0, 4.0, 10.0, "<=")
    assert hi == pytest.approx(3.0)
    assert lo < hi
    lo2, hi2 = MathSymbolics.solve_inequality_interval(-2.0, 4.0, 10.0, "<=")
    assert lo2 == pytest.approx(-3.0)
    assert hi2 > lo2
