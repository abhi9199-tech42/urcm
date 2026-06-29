from urcm.core.logic_gates import FormalLogic, ProbabilisticLogic

def test_cnf_and_resolution_simple():
    cnf = FormalLogic.to_cnf("a implies b")
    clauses = FormalLogic._collect_clauses(cnf)
    assert FormalLogic.resolution_proves(clauses, "b") is False
    # add 'a' as fact to clauses: (a) -> resolution proves b
    clauses2 = clauses + [{"a"}]
    assert FormalLogic.resolution_proves(clauses2, "b") is True

def test_probabilistic_combinators():
    p = ProbabilisticLogic.bayes(0.5, 0.8)
    assert 0.5 < p < 1.0
    o = ProbabilisticLogic.noisy_or([0.2, 0.3, 0.4])
    assert 0.5 <= o < 1.0
    a = ProbabilisticLogic.and_prod([0.9, 0.8])
    assert 0.7 < a < 1.0
