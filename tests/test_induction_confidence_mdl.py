from urcm.core.logic_gates import InductionEngine

def test_induction_confidence_and_mdl():
    obs = [
        ("instance","dog","animal"),
        ("instance","dog","animal"),
        ("instance","dog","animal"),
        ("no_instance","dog","plant"),
        ("instance","cat","animal"),
        ("not_instance","cat","plant"),
    ]
    rules = InductionEngine.induce_rules_extended(obs, min_support=2, mdl_lambda=0.05)
    rule_map = {(r[1], r[2]): r for r in rules}
    assert ("dog","animal") in rule_map
    meta = rule_map[("dog","animal")][3]
    assert meta["confidence"] >= 0.6
