from urcm.core.logic_gates import FormalLogic

def test_generate_fol_clauses_and_resolve():
    text = "forall x: (P(x) and Q(x)) implies R(x). P(a). Q(a)."
    domain = ["a"]
    raw = FormalLogic.generate_fol_clauses(text, domain_constants=domain, prune_cap=50)
    # Convert string literals to FOL tuple form
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
                # fallback single symbol -> treat as predicate on constant c
                t = (s.lower(), "a")
            out.append(("not", t) if neg else t)
        return out
    fol = [to_tuple_clause(c) for c in raw]
    assert isinstance(fol, list) and len(fol) >= 3
    ok = FormalLogic.resolve_fol(fol, ("r","a"))
    assert ok is True
