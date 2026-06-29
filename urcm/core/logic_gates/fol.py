import re
from typing import Any, Dict, List, Optional, Tuple

_tp_cache: Dict[Tuple[Tuple[Tuple[str, ...], ...], str], Dict[str, Any]] = {}
_tp_order: List[Tuple[Tuple[Tuple[str, ...], ...], str]] = []
_tp_cap = 64

class FormalLogic:
    """
    Minimal propositional logic utilities:
    - parse simple expressions: variables (a..z), not, and, or, implies
    - transform to CNF using standard rewrites
    - resolution on clause sets (Horn-like small cases)
    """
    @staticmethod
    def _tokenize(expr: str):
        s = expr.lower().replace("(", " ( ").replace(")", " ) ")
        s = s.replace("and", " ∧ ").replace("or", " ∨ ").replace("not", " ¬ ").replace("implies", " → ")
        return s.split()

    @staticmethod
    def _to_ast(tokens):
        # Pratt parser with minimal precedence: ¬ > ∧ > ∨ > →
        def parse_expr(i=0, prec=0):
            def parse_primary(i):
                t = tokens[i]
                if t == "(":
                    node, j = parse_expr(i+1, 0)
                    if tokens[j] != ")":
                        raise ValueError("missing )")
                    return node, j+1
                elif t == "¬":
                    node, j = parse_primary(i+1)
                    return ("not", node), j
                else:
                    return ("var", t), i+1
            node, i = parse_primary(i)
            while i < len(tokens):
                op = tokens[i]
                if op == ")":
                    break
                # precedence
                if op in ("∧","∨","→"):
                    p = {"∧":3,"∨":2,"→":1}[op]
                    if p < prec:
                        break
                    rhs, j = parse_expr(i+1, p+1)
                    node = ({"∧":"and","∨":"or","→":"implies"}[op], node, rhs)
                    i = j
                else:
                    break
            return node, i
        node, j = parse_expr(0, 0)
        return node

    @staticmethod
    def _elim_implies(node):
        t = node[0]
        if t == "var":
            return node
        if t == "not":
            return ("not", FormalLogic._elim_implies(node[1]))
        a = FormalLogic._elim_implies(node[1])
        b = FormalLogic._elim_implies(node[2])
        if t == "implies":
            return ("or", ("not", a), b)
        return (t, a, b)

    @staticmethod
    def _push_not(node):
        t = node[0]
        if t == "var":
            return node
        if t == "not":
            inner = node[1]
            if inner[0] == "var":
                return ("not", inner)
            if inner[0] == "not":
                return FormalLogic._push_not(inner[1])
            if inner[0] == "and":
                return ("or", FormalLogic._push_not(("not", inner[1])), FormalLogic._push_not(("not", inner[2])))
            if inner[0] == "or":
                return ("and", FormalLogic._push_not(("not", inner[1])), FormalLogic._push_not(("not", inner[2])))
        return (t, FormalLogic._push_not(node[1]), FormalLogic._push_not(node[2]))

    @staticmethod
    def _to_cnf(node):
        # distribute or over and
        t = node[0]
        if t in ("var","not"):
            return node
        a = FormalLogic._to_cnf(node[1])
        b = FormalLogic._to_cnf(node[2])
        if t == "and":
            return ("and", a, b)
        if t == "or":
            if a[0] == "and":
                return ("and",
                        FormalLogic._to_cnf(("or", a[1], b)),
                        FormalLogic._to_cnf(("or", a[2], b)))
            if b[0] == "and":
                return ("and",
                        FormalLogic._to_cnf(("or", a, b[1])),
                        FormalLogic._to_cnf(("or", a, b[2])))
            return ("or", a, b)
        return ("and", a, b)

    @staticmethod
    def to_cnf(expr: str):
        tokens = FormalLogic._tokenize(expr)
        ast = FormalLogic._to_ast(tokens)
        ast2 = FormalLogic._elim_implies(ast)
        ast3 = FormalLogic._push_not(ast2)
        return FormalLogic._to_cnf(ast3)

    @staticmethod
    def _collect_clauses(node):
        def to_literals(n):
            if n[0] == "var":
                return {n[1]}
            if n[0] == "not" and n[1][0] == "var":
                return {"¬"+n[1][1]}
            if n[0] == "or":
                return to_literals(n[1]).union(to_literals(n[2]))
            return set()
        if node[0] == "and":
            left = FormalLogic._collect_clauses(node[1])
            right = FormalLogic._collect_clauses(node[2])
            return left + right
        return [to_literals(node)]

    @staticmethod
    def resolution_proves(clauses: list, target: str) -> bool:
        # clauses: list of sets of literals; target as literal "x" or "¬x"
        S = [set(c) for c in clauses]
        tgt_neg = "¬"+target if not target.startswith("¬") else target[1:]
        S.append({tgt_neg})
        new = set()
        def resolvent(Ci, Cj):
            res = []
            for l in list(Ci):
                if l.startswith("¬"):
                    comp = l[1:]
                else:
                    comp = "¬"+l
                if comp in Cj:
                    res_clause = (Ci - {l}) | (Cj - {comp})
                    res.append(frozenset(res_clause))
            return res
        while True:
            pairs = []
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    for r in resolvent(S[i], S[j]):
                        if not r:
                            return True
                        pairs.append(r)
            added = False
            for r in pairs:
                if r not in new and r not in map(frozenset, S):
                    new.add(r)
                    S.append(set(r))
                    added = True
            if not added:
                return False

    @staticmethod
    def unify(term1: tuple, term2: tuple, subst: Dict[str, str] = None) -> Optional[Dict[str, str]]:
        subst = dict(subst or {})
        def is_var(x):
            return isinstance(x, str) and len(x) == 1 and x.isalpha()
        if term1[0] != term2[0] or len(term1) != len(term2):
            return None
        for a, b in zip(term1[1:], term2[1:]):
            if is_var(a) and is_var(b):
                if a in subst and subst[a] != b:
                    return None
                subst[a] = b
            elif is_var(a):
                subst[a] = b
            elif is_var(b):
                subst[b] = a
            else:
                if a != b:
                    return None
        return subst

    @staticmethod
    def resolve_fol(clauses: List[List[tuple]], target: tuple, max_steps: int = 100) -> bool:
        S = [list(c) for c in clauses if not any(x[0] != "not" and ("not", x) in c for x in c)]
        tgt_neg = ("not", target)
        S.append([tgt_neg])
        def negate(t):
            return ("not", t) if t[0] != "not" else t[1]
        def complementary(a, b):
            if a[0] == "not" and b[0] != "not":
                return a[1][0] == b[0]
            if b[0] == "not" and a[0] != "not":
                return a[0] == b[1][0]
            return False
        def norm_clause(c):
            return tuple(sorted(c))
        def subsumes(c1, c2):
            return set(c1).issubset(set(c2))
        steps = 0
        max_steps = max_steps if isinstance(max_steps, int) and max_steps > 0 else 500
        while steps < max_steps:
            steps += 1
            new = []
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    Ci = S[i]
                    Cj = S[j]
                    for li in Ci:
                        for lj in Cj:
                            if complementary(li, lj):
                                pos = lj if lj[0] != "not" else li
                                neg = li if li[0] == "not" else lj
                                subst = FormalLogic.unify(pos if pos[0] != "not" else pos[1], neg[1] if neg[0] == "not" else neg, {})
                                if subst is None:
                                    continue
                                Ri = [x for x in Ci if x != li]
                                Rj = [x for x in Cj if x != lj]
                                res = []
                                for t in Ri + Rj:
                                    if t[0] == "not":
                                        sym = t[1][0]
                                        args = [subst.get(a, a) for a in t[1][1:]]
                                        res.append(("not", (sym, *args)))
                                    else:
                                        sym = t[0]
                                        args = [subst.get(a, a) for a in t[1:]]
                                        res.append((sym, *args))
                                if not res:
                                    return True
                                # remove tautologies
                                res_set = set(res)
                                taut = False
                                for t in list(res_set):
                                    if t[0] == "not":
                                        if (t[1][0], *t[1][1:]) in res_set:
                                            taut = True
                                            break
                                    else:
                                        if ("not", (t[0], *t[1:])) in res_set:
                                            taut = True
                                            break
                                if taut:
                                    continue
                                if res not in S and res not in new:
                                    keep = True
                                    for c in S:
                                        if subsumes(res, c):
                                            pass
                                        if subsumes(c, res):
                                            keep = False
                                            break
                                    if keep:
                                        new.append(res)
            if not new:
                return False
            S.extend(new)
        return False

    @staticmethod
    def to_cnf_quantified(expr: str, domain_constants: List[str], prune_cap: int = 200) -> List[set]:
        s = expr.strip()
        grounded = []
        m_prefix = re.match(r'^\s*((?:forall|exists)\s+[a-z](?:\s+(?:forall|exists)\s+[a-z])*)\s*:\s*(.+)$', s, flags=re.IGNORECASE)
        if m_prefix:
            qpart = m_prefix.group(1)
            body = m_prefix.group(2)
            toks = qpart.strip().split()
            chain = []
            i = 0
            while i < len(toks):
                qt = toks[i].lower()
                v = toks[i+1].lower()
                chain.append((qt, v))
                i += 2
            universals = [v for q, v in chain if q == "forall"]
            existentials = [v for q, v in chain if q == "exists"]
            if universals:
                from itertools import product
                dom = domain_constants[:max(1, min(len(domain_constants), prune_cap))]
                combos = list(product(dom, repeat=len(universals)))
                cap = max(1, min(len(combos), prune_cap))
                combos = combos[:cap]
                for combo in combos:
                    inst = body
                    for u, c in zip(universals, combo):
                        inst = re.sub(r'\b'+re.escape(u)+r'\b', c, inst)
                    if existentials:
                        key = "_".join(combo)
                        for e in existentials:
                            inst = re.sub(r'\b'+re.escape(e)+r'\b', "sk_"+key, inst)
                    grounded.append(inst)
            else:
                inst = body
                for e in existentials:
                    inst = re.sub(r'\b'+re.escape(e)+r'\b', "sk", inst)
                grounded.append(inst)
        else:
            m_forall = re.match(r'^\s*forall\s+([a-z])\s*:\s*(.+)$', s, flags=re.IGNORECASE)
            m_exists = re.match(r'^\s*exists\s+([a-z])\s*:\s*(.+)$', s, flags=re.IGNORECASE)
            m_alt = re.match(r'^\s*forall\s+([a-z])\s+exists\s+([a-z])\s*:\s*(.+)$', s, flags=re.IGNORECASE)
            m_alt2 = re.match(r'^\s*exists\s+([a-z])\s+forall\s+([a-z])\s*:\s*(.+)$', s, flags=re.IGNORECASE)
            if m_alt:
                x = m_alt.group(1)
                y = m_alt.group(2)
                body = m_alt.group(3)
                for c in domain_constants[:max(1, min(len(domain_constants), prune_cap))]:
                    inst = re.sub(r'\b'+re.escape(x)+r'\b', c, body)
                    inst = re.sub(r'\b'+re.escape(y)+r'\b', "sk_"+c, inst)
                    grounded.append(inst)
            elif m_alt2:
                e = m_alt2.group(1)
                u = m_alt2.group(2)
                body = m_alt2.group(3)
                sk = "sk_"+e
                for c in domain_constants[:max(1, min(len(domain_constants), prune_cap))]:
                    inst = re.sub(r'\b'+re.escape(u)+r'\b', c, body)
                    inst = re.sub(r'\b'+re.escape(e)+r'\b', sk, inst)
                    grounded.append(inst)
            elif m_forall:
                var = m_forall.group(1)
                body = m_forall.group(2)
                for c in domain_constants[:max(1, min(len(domain_constants), prune_cap))]:
                    inst = re.sub(r'\b'+re.escape(var)+r'\b', c, body)
                    grounded.append(inst)
            elif m_exists:
                var = m_exists.group(1)
                body = m_exists.group(2)
                sk = "sk_"+var
                inst = re.sub(r'\b'+re.escape(var)+r'\b', sk, body)
                grounded.append(inst)
            else:
                grounded.append(s)
        clauses = []
        for g in grounded:
            g_simple = re.sub(r'([a-zA-Z]+)\(([^)]+)\)', lambda m: f"{m.group(1)}_{m.group(2)}", g)
            cnf = FormalLogic.to_cnf(g_simple)
            clauses.extend(FormalLogic._collect_clauses(cnf))
        return clauses

    @staticmethod
    def generate_fol_clauses(text: str, domain_constants: List[str], prune_cap: int = 200) -> List[List[tuple]]:
        out = []
        parts = [p.strip() for p in re.split(r'[.!?]', text) if p.strip()]
        for p in parts:
            if re.search(r'\b(forall|exists)\b', p, flags=re.IGNORECASE):
                raw = FormalLogic.to_cnf_quantified(p, domain_constants, prune_cap)
                out.extend([list(c) for c in raw])
            else:
                cnf = FormalLogic.to_cnf(p.lower().replace("implies", " → "))
                out.extend([list(c) for c in FormalLogic._collect_clauses(cnf)])
        return out

    @staticmethod
    def _parse_literal(lit: str):
        neg = False
        s = lit
        if s.startswith("¬"):
            neg = True
            s = s[1:]
        if "(" in s and ")" in s:
            pred = s.split("(")[0].lower()
            args = s.split("(")[1].rstrip(")")
            args_list = [a.strip().lower() for a in args.split(",")]
            term = (pred, *args_list)
        elif "_" in s:
            pred, args = s.split("_", 1)
            args_list = [a.strip().lower() for a in args.split(",")]
            term = (pred.lower(), *args_list)
        else:
            term = (s.lower(),)
        return ("not", term) if neg else term

    @staticmethod
    def generate_fol_clauses_structured(text: str, domain_constants: List[str], prune_cap: int = 200) -> List[List[tuple]]:
        out = []
        parts = [p.strip() for p in re.split(r'[.!?]', text) if p.strip()]
        for p in parts:
            if re.search(r'\b(forall|exists)\b', p, flags=re.IGNORECASE):
                raw = FormalLogic.to_cnf_quantified(p, domain_constants, prune_cap)
                for c in raw:
                    out.append([FormalLogic._parse_literal(l) for l in c])
            else:
                psimple = re.sub(r'([a-zA-Z]+)\(([^)]+)\)', lambda m: f"{m.group(1)}_{m.group(2)}", p)
                cnf = FormalLogic.to_cnf(psimple.lower().replace("implies", " → "))
                for c in FormalLogic._collect_clauses(cnf):
                    out.append([FormalLogic._parse_literal(l) for l in c])
        return out

    @staticmethod
    def quantifier_eliminate(expr: str, logic: str = "QF_UFLIRA") -> Optional[str]:
        try:
            # This is a placeholder interface. Full QE support depends on underlying solver.
            # For now, we return the input serialized; advanced users can plug a solver with QE.
            return expr
        except Exception:
            return expr

    @staticmethod
    def quantifier_eliminate_ground(expr: str, domain_constants: List[str]) -> str:
        s = expr.strip()
        m_forall = re.match(r'^\s*forall\s+([a-z])\s*:\s*(.+)$', s, flags=re.IGNORECASE)
        m_exists = re.match(r'^\s*exists\s+([a-z])\s*:\s*(.+)$', s, flags=re.IGNORECASE)
        out = []
        if m_forall:
            var = m_forall.group(1)
            body = m_forall.group(2)
            for c in domain_constants:
                out.append(re.sub(rf'\b{var}\b', c, body))
            return " and ".join(out)
        if m_exists:
            var = m_exists.group(1)
            body = m_exists.group(2)
            c = domain_constants[0] if domain_constants else "sk"
            return re.sub(rf'\b{var}\b', c, body)
        return s

    @staticmethod
    def quantifier_eliminate_full(expr: str, domain_constants: List[str] = None) -> str:
        s = expr.strip()
        dom = domain_constants or ["a","b","c"]
        if re.search(r'\b(forall|exists)\b', s, flags=re.IGNORECASE):
            return FormalLogic.quantifier_eliminate_ground(s, dom)
        return s

    @staticmethod
    def quantifier_eliminate_z3_linear(expr: str) -> Optional[str]:
        try:
            import z3
        except Exception:
            return None
        s = expr.strip()
        m = re.match(r'^\s*(forall|exists)\s+([a-z])\s*:\s*(.+)$', s, flags=re.IGNORECASE)
        if not m:
            return None
        qtok = m.group(1).lower()
        var = m.group(2)
        body = m.group(3)
        x = z3.Real(var)
        def parse_atom(atom: str):
            atom = atom.strip()
            m1 = re.match(r'^'+re.escape(var)+r'\s*(<=|>=|=)\s*([+-]?\d+(?:\.\d+)?)$', atom)
            if m1:
                op = m1.group(1)
                val = float(m1.group(2))
                if op == "<=":
                    return x <= val
                if op == ">=":
                    return x >= val
                return x == val
            m2 = re.match(r'^([+-]?\d+(?:\.\d+)?)\s*\*\s*'+re.escape(var)+r'\s*(<=|>=|=)\s*([+-]?\d+(?:\.\d+)?)$', atom)
            if m2:
                a = float(m2.group(1)); op = m2.group(2); c = float(m2.group(3))
                expr = a * x
                if op == "<=":
                    return expr <= c
                if op == ">=":
                    return expr >= c
                return expr == c
            return None
        parts = [p.strip() for p in re.split(r'\band\b', body, flags=re.IGNORECASE)]
        zparts = []
        for p in parts:
            zc = parse_atom(p)
            if zc is None:
                return None
            zparts.append(zc)
        conj = z3.And(zparts) if zparts else z3.BoolVal(True)
        quant = z3.ForAll([x], conj) if qtok == "forall" else z3.Exists([x], conj)
        g = z3.Goal()
        g.add(quant)
        try:
            qe = z3.Tactic('qe')(g)
        except Exception:
            try:
                qe = z3.Tactic('qe2')(g)
            except Exception:
                return None
        if len(qe) == 0:
            return "True"
        # Convert to string
        out = str(qe[0].as_expr())
        return out

    @staticmethod
    def prune_clauses(clauses: List[set], cap: int = 200) -> List[set]:
        if len(clauses) <= cap:
            return clauses
        def score(c):
            lits = list(c)
            length_pen = len(lits)
            sym_pen = sum(len(l) for l in lits) * 0.05
            neg_pen = sum(1 for l in lits if l.startswith("¬")) * 0.2
            diversity_pen = len(set(l.replace("¬","") for l in lits))
            return length_pen + sym_pen + neg_pen + 0.1 * diversity_pen
        ranked = sorted(clauses, key=score)
        return ranked[:cap]

class TheoremProver:
    @staticmethod
    def prove(clauses: list, target: str) -> Dict[str, Any]:
        key = (tuple(sorted([tuple(sorted([str(x) for x in c])) for c in clauses])), target)
        if key in _tp_cache:
            return _tp_cache[key]
        S = [set(c) for c in clauses]
        tgt_neg = "¬"+target if not target.startswith("¬") else target[1:]
        S.append({tgt_neg})
        steps = []
        new = set()
        parents = {}
        index_map = {frozenset(c): i for i, c in enumerate(S)}
        def resolvent(Ci, Cj):
            res = []
            for l in list(Ci):
                comp = l[1:] if l.startswith("¬") else "¬"+l
                if comp in Cj:
                    res_clause = (Ci - {l}) | (Cj - {comp})
                    res.append(set(res_clause))
            return res
        def collect_core(idx, acc):
            if idx in parents:
                a, b = parents[idx]
                collect_core(a, acc)
                collect_core(b, acc)
            else:
                acc.add(idx)
        while True:
            pairs = []
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    for r in resolvent(S[i], S[j]):
                        if len(r) == 0:
                            core = set()
                            collect_core(i, core)
                            collect_core(j, core)
                            core_list = sorted(core)
                            steps.append({"resolve": (i, j), "result": []})
                            res = {"success": True, "target": target, "clauses": [list(c) for c in S], "steps": steps, "unsat_core": core_list}
                            _tp_cache[key] = res
                            _tp_order.append(key)
                            if len(_tp_order) > _tp_cap:
                                old = _tp_order.pop(0)
                                _tp_cache.pop(old, None)
                            return res
                        pairs.append((i, j, r))
            added = False
            for i, j, r in pairs:
                fr = frozenset(r)
                if fr not in new and fr not in index_map:
                    new.add(fr)
                    S.append(set(r))
                    idx = len(S) - 1
                    parents[idx] = (i, j)
                    index_map[fr] = idx
                    steps.append({"resolve": (i, j), "result": list(r)})
                    added = True
            if not added:
                res = {"success": False, "target": target, "clauses": [list(c) for c in S], "steps": steps}
                _tp_cache[key] = res
                _tp_order.append(key)
                if len(_tp_order) > _tp_cap:
                    old = _tp_order.pop(0)
                    _tp_cache.pop(old, None)
                return res

    @staticmethod
    def minimize_unsat_core(clauses: list, target: str, core_indices: List[int]) -> List[int]:
        if not core_indices:
            return core_indices
        core = list(core_indices)
        changed = True
        while changed:
            changed = False
            for i in list(core):
                trial = [clauses[j] for j in range(len(clauses)) if j in core and j != i]
                res = TheoremProver.prove(trial, target)
                if res.get("success"):
                    core.remove(i)
                    changed = True
        return sorted(core)

    @staticmethod
    def prove_pruned(clauses: list, target: str, max_pairs: int = 1000) -> Dict[str, Any]:
        key = (tuple(sorted([tuple(sorted([str(x) for x in c])) for c in clauses])), target)
        if key in _tp_cache:
            return _tp_cache[key]
        S = [set(c) for c in clauses]
        tgt_neg = "¬"+target if not target.startswith("¬") else target[1:]
        S.append({tgt_neg})
        steps = []
        new = set()
        parents = {}
        index_map = {frozenset(c): i for i, c in enumerate(S)}
        def weight(c):
            return len(c)
        while True:
            pairs = []
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    if len(pairs) >= max_pairs:
                        break
                    Ci, Cj = S[i], S[j]
                    if weight(Ci) + weight(Cj) > 8:
                        continue
                    for r in FormalLogic.resolution_proves.__defaults__ or []:
                        pass
                    for l in list(Ci):
                        comp = l[1:] if l.startswith("¬") else "¬"+l
                        if comp in Cj:
                            res_clause = (Ci - {l}) | (Cj - {comp})
                            if len(res_clause) == 0:
                                res = {"success": True, "target": target, "steps": steps}
                                _tp_cache[key] = res
                                _tp_order.append(key)
                                if len(_tp_order) > _tp_cap:
                                    old = _tp_order.pop(0)
                                    _tp_cache.pop(old, None)
                                return res
                            fr = frozenset(res_clause)
                            if fr not in new and fr not in index_map:
                                new.add(fr)
                                S.append(set(res_clause))
                                idx = len(S) - 1
                                parents[idx] = (i, j)
                                index_map[fr] = idx
                                steps.append({"resolve": (i, j), "result": list(res_clause)})
            if len(pairs) == 0:
                res = {"success": False, "target": target, "steps": steps}
                _tp_cache[key] = res
                _tp_order.append(key)
                if len(_tp_order) > _tp_cap:
                    old = _tp_order.pop(0)
                    _tp_cache.pop(old, None)
                return res

class SemanticParser:
    @staticmethod
    def parse_all_are(s: str) -> Optional[Tuple[str,str]]:
        t = s.strip().rstrip(".")
        m = re.match(r'^All\s+([a-zA-Z]+)\s+are\s+([a-zA-Z]+)$', t)
        if not m:
            return None
        return (m.group(1).lower(), m.group(2).lower())
    @staticmethod
    def to_edge(s: str) -> Optional[Tuple[str,str]]:
        p = SemanticParser.parse_all_are(s)
        if not p:
            return None
        return (p[0], p[1])

class ClauseGenerator:
    @staticmethod
    def implies(a: str, b: str) -> List[set]:
        return [set([f"!{a}", b])]
    @staticmethod
    def forall_implies(var: str, preds: List[Tuple[str,str]], consequent: Tuple[str,str]) -> List[set]:
        return [set([f"!{pred[0]}({var})" for pred in preds] + [f"{consequent[0]}({var})"])]
    @staticmethod
    def parse_nested(sentence: str) -> Optional[List[set]]:
        t = sentence.strip().rstrip(".")
        m = re.match(r'^forall\s+([a-zA-Z]+)\s*:\s*\((.+)\)\s*implies\s*([a-zA-Z_]+)\(([a-zA-Z_]+)\)$', t, flags=re.IGNORECASE)
        if not m:
            return None
        x = m.group(1)
        body = m.group(2)
        cons_pred = m.group(3).lower()
        cons_var = m.group(4).lower()
        parts = [p.strip() for p in re.split(r'\band\b', body, flags=re.IGNORECASE)]
        preds = []
        for p in parts:
            mm = re.match(r'^([a-zA-Z_]+)\(([a-zA-Z_]+)\)$', p)
            if not mm:
                return None
            if mm.group(2).lower() != x.lower():
                return None
            preds.append((mm.group(1).lower(), x.lower()))
        return ClauseGenerator.forall_implies(x.lower(), preds, (cons_pred, cons_var))

class ProverScheduler:
    @staticmethod
    def schedule(clauses: list, target: str, strategies: List[str], max_steps: int = 1000, max_seconds: float = 1.0) -> Dict[str, any]:
        import time as _t
        start = _t.perf_counter()
        for strat in strategies:
            proof = TheoremProver.prove(clauses, target)
            elapsed = _t.perf_counter() - start
            if proof.get("success"):
                proof["strategy"] = strat
                proof["elapsed"] = elapsed
                return proof
            if elapsed > max_seconds:
                return {"success": False, "strategy": strat, "elapsed": elapsed, "reason": "timeout"}
