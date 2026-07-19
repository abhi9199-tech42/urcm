from urcm.core.logic_gates import InductionEngine


def test_hypothesis_induction_min_support():
    obs = [
        ("instance","dog","animal"),
        ("instance","dog","animal"),
        ("instance","cat","animal"),
    ]
    rules = InductionEngine.induce_rules(obs, min_support=2)
    assert ("all","dog","animal") in rules
    assert ("all","cat","animal") not in rules
