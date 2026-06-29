from urcm.core.logic_gates import BayesianNetwork

def test_bayes_cache_correctness():
    bn = BayesianNetwork()
    bn.add_node("A", [], lambda parents: 0.6)
    bn.add_node("B", ["A"], lambda parents: 0.9 if parents[0] else 0.1)
    p1 = bn.query("B", {"A": True})
    p2 = bn.query("B", {"A": True})
    assert abs(p1 - p2) < 1e-12
