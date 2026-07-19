from urcm.core.logic_gates import ClauseGenerator, FormalLogic


def test_forall_nested_implies_resolution():
    # Forall x: (P(x) and Q(x)) implies R(x)
    s = "forall x: (P(x) and Q(x)) implies R(x)"
    clauses = ClauseGenerator.parse_nested(s)
    assert clauses is not None and len(clauses) == 1
    # Convert to FOL tuple form understood by FormalLogic.resolve_fol
    def to_tuple_clause(cl):
        out = []
        for lit in cl:
            if lit.startswith("!"):
                body = lit[1:]
                pred, args = body.split("(")[0], body.split("(")[1].rstrip(")")
                out.append(("not", (pred.lower(), args.lower())))
            else:
                pred, args = lit.split("(")[0], lit.split("(")[1].rstrip(")")
                out.append((pred.lower(), args.lower()))
        return out
    fol_clauses = [to_tuple_clause(c) for c in clauses]
    # Ground facts: p(a), q(a)
    fol_clauses.append([("p","a")])
    fol_clauses.append([("q","a")])
    ok = FormalLogic.resolve_fol(fol_clauses, ("r","a"))
    assert ok is True
