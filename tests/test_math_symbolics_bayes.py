from urcm.core.logic_gates import MathSymbolics, BayesianNetwork

def test_fraction_simplify_and_ratio_equal():
    assert MathSymbolics.simplify_fraction("8","12") == (2,3)
    assert MathSymbolics.ratio_equal("2","3","4","6") is True
    assert MathSymbolics.solve_linear(2.0, 3.0, 9.0) == 3.0

def test_bayesian_network_enumeration():
    bn = BayesianNetwork()
    # A prior 0.3
    bn.add_node("A", [], lambda _: 0.3)
    # B depends on A: P(B|A)=0.8, P(B|¬A)=0.2
    bn.add_node("B", ["A"], lambda ps: 0.8 if ps[0] else 0.2)
    # C depends on B: P(C|B)=0.7, P(C|¬B)=0.1
    bn.add_node("C", ["B"], lambda ps: 0.7 if ps[0] else 0.1)
    p = bn.query("C", {})
    assert 0.1 < p < 0.7
    # Evidence B=True raises C's probability
    p_e = bn.query("C", {"B": True})
    assert p_e > p
