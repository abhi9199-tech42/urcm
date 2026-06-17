from urcm.core.logic_gates import AlgebraEngine, SetTheory, CalculusEngine

def test_combine_like_terms():
    out = AlgebraEngine.combine_like_terms("2*x + 3*x")
    assert out == "5*x"

def test_derivative_poly():
    out = AlgebraEngine.derivative_poly("3*x^2+4*x", "x")
    assert out in {"6*x+4","6*x+4"}

def test_calculus_derivative_basic():
    out = CalculusEngine.derivative("2*x^3+3*x+5", "x")
    assert out == "6*x^2+3"

def test_calculus_integral_basic():
    out = CalculusEngine.integral_basic("6*x^2+3", "x")
    assert out in {"2*x^3+3*x","2*x^3+3*x"}

def test_set_theory_ops():
    a = {1,2,3}
    b = {3,4}
    assert SetTheory.union(a,b) == {1,2,3,4}
    assert SetTheory.intersection(a,b) == {3}
    assert SetTheory.difference(a,b) == {1,2}
    assert SetTheory.subset({1,2}, a) is True
