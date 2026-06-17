from urcm.core.logic_gates import FormalLogic

def test_multi_arity_structured_forall_exists():
    text = "forall x exists y: (Rel(x,y) and P(x)) implies S(x,y). Rel(a,sk_a). P(a)."
    dom = ["a"]
    fol = FormalLogic.generate_fol_clauses_structured(text, domain_constants=dom, prune_cap=20)
    assert FormalLogic.resolve_fol(fol, ("s","a","sk_a")) is True

def test_zero_arity_predicate_resolution():
    text = "forall x: (P implies Q(x)). P. Q(a)."
    dom = ["a"]
    fol = FormalLogic.generate_fol_clauses_structured(text, domain_constants=dom, prune_cap=20)
    assert FormalLogic.resolve_fol(fol, ("q","a")) is True
