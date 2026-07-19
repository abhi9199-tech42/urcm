from urcm.core.logic_gates import FormalLogic


def test_forall_exists_skolem_grounding_and_resolution():
    text = "forall x exists y: (P(x) and R(y)) implies Q(x). P(a). R(sk_a)."
    dom = ["a"]
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
    ok = FormalLogic.resolve_fol(fol, ("q","a"))
    assert ok is True
