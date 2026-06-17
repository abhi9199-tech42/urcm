from urcm.core.logic_gates import CalculusEngine

def test_chain_rule_exp_linear():
    d = CalculusEngine.derivative("2*exp(3*x+1)", "x")
    assert d == "6*exp(3*x+1)"

def test_chain_rule_sin_linear():
    d = CalculusEngine.derivative("4*sin(2*x)", "x")
    assert d == "8*cos(2*x+0)"

def test_integral_exp_linear():
    i = CalculusEngine.integral_basic("6*exp(3*x+1)", "x")
    assert i == "2*exp(3*x+1)"
