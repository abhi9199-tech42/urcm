from urcm.core.logic_gates import FormalLogic

def to_tuple_clause(cl):
    out = []
    for lit in cl:
        s = lit
        neg = False
        if isinstance(s, str) and s.startswith("¬"):
            neg = True
            s = s[1:]
        if "(" in s and ")" in s:
            pred = s.split("(")[0].lower()
            arg = s.split("(")[1].rstrip(")").lower()
            t = (pred, arg)
        elif "_" in s:
            pred, arg = s.split("_", 1)
            t = (pred.lower(), arg.lower())
        else:
            t = (s.lower(), "a")
        out.append(("not", t) if neg else t)
    return out

def test_forall_forall_exists_chain():
    text = "forall x forall y exists z: (P(x) and R(y) and T(z)) implies S(x,y). P(a). R(b). T(sk_a_b)."
    dom = ["a","b"]
    raw = FormalLogic.generate_fol_clauses(text, domain_constants=dom, prune_cap=50)
    fol = [to_tuple_clause(c) for c in raw]
    assert FormalLogic.resolve_fol(fol, ("s","a,b")) is True

def test_exists_exists_forall_chain():
    text = "exists x exists y forall z: (U(z) and V(x) and W(y)) implies H(z). V(sk). W(sk). U(c)."
    dom = ["c"]
    raw = FormalLogic.generate_fol_clauses(text, domain_constants=dom, prune_cap=50)
    fol = [to_tuple_clause(c) for c in raw]
    assert FormalLogic.resolve_fol(fol, ("h","c")) is True
