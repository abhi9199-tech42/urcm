from urcm.core.logic_gates import FormalLogic

def test_exists_forall_skolemization_and_resolution():
    text = "exists x forall y: (R(y) and P(x)) implies S(y). R(a). R(b). P(sk_x)."
    dom = ["a","b"]
    raw = FormalLogic.generate_fol_clauses(text, domain_constants=dom, prune_cap=50)
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
    fol = [to_tuple_clause(c) for c in raw]
    assert FormalLogic.resolve_fol(fol, ("s","a")) is True
    assert FormalLogic.resolve_fol(fol, ("s","b")) is True
