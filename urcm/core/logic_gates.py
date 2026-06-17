import numpy as np
import re
from typing import List, Dict, Optional, Tuple, Callable, Any

_tc_cache: Dict[Tuple[Tuple[str, str], ...], List[Tuple[str, str]]] = {}
_tc_order: List[Tuple[Tuple[str, str], ...]] = []
_tc_cap = 256
_tp_cache: Dict[Tuple[Tuple[Tuple[str, ...], ...], str], Dict[str, Any]] = {}
_tp_order: List[Tuple[Tuple[Tuple[str, ...], ...], str]] = []
_tp_cap = 64

class GeometricLogic:
    """
    Implements Logic Gates using Energy Landscapes (Left Brain).
    
    Instead of binary True/False, we use 'Energy Modifiers':
    - AND(A, B): Creates a basin where BOTH A and B are active.
    - OR(A, B): Creates a double-well basin (bistable).
    - NOT(A): Creates a hill (repeller) at A.
    - IMPLIES(A, B): Creates a directional gradient flow from A to B.
    """
    
    def __init__(self, concept_map: Dict[str, np.ndarray]):
        self.concept_map = concept_map
        
    def get_vector(self, concept: str) -> Optional[np.ndarray]:
        return self.concept_map.get(concept)
        
    def apply_constraint(self, 
                       current_state: np.ndarray, 
                       logic_type: str, 
                       operands: List[str], 
                       weight: float = 1.0) -> np.ndarray:
        """
        Calculates the ENERGY GRADIENT for a logical constraint.
        Returns a vector (direction) to move the state towards satisfaction.
        """
        
        vectors = [self.get_vector(op) for op in operands]
        if any(v is None for v in vectors):
            return np.zeros_like(current_state) # Cannot apply if concepts unknown
            
        grad = np.zeros_like(current_state)
        
        if logic_type == "NOT":
            # NOT A: Avoid A.
            # Potential U = +1 * similarity(state, A)
            # Force F = -dU/dx = -1 * A (if cosine)
            # Simply: Push away from A.
            vec_a = vectors[0]
            # Direction: State -> Away from A = (State - A)
            # But simpler: Just subtract A vector scaled by weight
            # Repulsion force
            dist_sq = np.sum((current_state - vec_a)**2)
            if dist_sq < 0.1: # Only repel if close
                grad = (current_state - vec_a) * weight * 5.0 
            else:
                 grad = (current_state - vec_a) * weight
            
        elif logic_type == "AND":
            # A AND B: Be close to BOTH.
            # Target = Mean(A, B)
            # Force: Pull towards Mean
            vec_a, vec_b = vectors[0], vectors[1]
            target = (vec_a + vec_b) / 2.0
            target = target / np.linalg.norm(target) # Normalize
            grad = (target - current_state) * weight
            
        elif logic_type == "OR":
            # A OR B: Be close to A OR close to B.
            # Bistable potential.
            # Force: Pull towards whichever is closer.
            vec_a, vec_b = vectors[0], vectors[1]
            dist_a = np.linalg.norm(current_state - vec_a)
            dist_b = np.linalg.norm(current_state - vec_b)
            
            if dist_a < dist_b:
                grad = (vec_a - current_state) * weight
            else:
                grad = (vec_b - current_state) * weight
                
        elif logic_type == "IMPLIES":
            # IF A THEN B.
            # Logic: If close to A, MUST move to B.
            # If far from A, do nothing (vacuously true).
            vec_a, vec_b = vectors[0], vectors[1]
            
            # Check proximity to A (Antecedent)
            # Cosine sim or Euclidean? Euclidean for localized effects.
            dist_a = np.linalg.norm(current_state - vec_a)
            
            # Activation threshold (e.g. 0.8 similarity approx 0.6 dist)
            if dist_a < 1.0: 
                # Active! Pull towards B.
                grad = (vec_b - current_state) * weight * (1.0 - dist_a) # Stronger if closer to A
                
        return grad

class NumericLogic:
    """
    Numeric reasoning gates for simple arithmetic and comparisons.
    Operates on numeric literals extracted from concept names.
    """
    @staticmethod
    def _parse(value: str) -> Optional[float]:
        try:
            if isinstance(value, (int, float)):
                return float(value)
            s = str(value).strip().replace("_", "")
            # Accept forms like "three" only if mapped externally; here numeric only
            if re.match(r"^[+-]?(\d+)(\.\d+)?$", s):
                return float(s)
            return None
        except Exception:
            return None

    @staticmethod
    def add(a: str, b: str) -> Optional[float]:
        va, vb = NumericLogic._parse(a), NumericLogic._parse(b)
        if va is None or vb is None:
            return None
        return va + vb

    @staticmethod
    def sub(a: str, b: str) -> Optional[float]:
        va, vb = NumericLogic._parse(a), NumericLogic._parse(b)
        if va is None or vb is None:
            return None
        return va - vb

    @staticmethod
    def mul(a: str, b: str) -> Optional[float]:
        va, vb = NumericLogic._parse(a), NumericLogic._parse(b)
        if va is None or vb is None:
            return None
        return va * vb

    @staticmethod
    def eq(a: str, b: str) -> Optional[bool]:
        va, vb = NumericLogic._parse(a), NumericLogic._parse(b)
        if va is None or vb is None:
            return None
        return abs(va - vb) < 1e-9

    @staticmethod
    def gt(a: str, b: str) -> Optional[bool]:
        va, vb = NumericLogic._parse(a), NumericLogic._parse(b)
        if va is None or vb is None:
            return None
        return va > vb

    @staticmethod
    def lt(a: str, b: str) -> Optional[bool]:
        va, vb = NumericLogic._parse(a), NumericLogic._parse(b)
        if va is None or vb is None:
            return None
        return va < vb
    
    @staticmethod
    def div(a: str, b: str) -> Optional[float]:
        va, vb = NumericLogic._parse(a), NumericLogic._parse(b)
        if va is None or vb is None or abs(vb) < 1e-12:
            return None
        return va / vb
    
    @staticmethod
    def mod(a: str, b: str) -> Optional[float]:
        va, vb = NumericLogic._parse(a), NumericLogic._parse(b)
        if va is None or vb is None or abs(vb) < 1e-12:
            return None
        return va % vb

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
            from pysmt.shortcuts import Symbol, And, Exists, ForAll, is_sat, Not, get_env, serialize
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

class ProbabilisticLogic:
    """
    Minimal probabilistic combinators.
    """
    @staticmethod
    def bayes(prior: float, likelihood: float, evidence: float = None) -> float:
        prior = max(1e-9, min(1.0, prior))
        likelihood = max(1e-9, min(1.0, likelihood))
        if evidence is None:
            # Assume binary evidence space; normalize with law of total probability using 0.5 prior for not-hyp
            evidence = prior * likelihood + (1 - prior) * (1 - likelihood)
        return (prior * likelihood) / max(1e-9, evidence)

    @staticmethod
    def noisy_or(ps: List[float]) -> float:
        q = 1.0
        for p in ps:
            q *= (1 - max(0.0, min(1.0, p)))
        return 1 - q

    @staticmethod
    def and_prod(ps: List[float]) -> float:
        prod = 1.0
        for p in ps:
            prod *= max(0.0, min(1.0, p))
        return prod

class TemporalLogic:
    """
    Simple interval algebra: supports overlap, before, after.
    Intervals as tuples (start, end) with start < end.
    """
    @staticmethod
    def overlap(a: tuple, b: tuple) -> bool:
        return max(a[0], b[0]) < min(a[1], b[1])
    @staticmethod
    def before(a: tuple, b: tuple) -> bool:
        return a[1] <= b[0]
    @staticmethod
    def after(a: tuple, b: tuple) -> bool:
        return a[0] >= b[1]

class SpatialLogic:
    """
    Minimal geometry primitives.
    """
    @staticmethod
    def distance(p1: tuple, p2: tuple) -> float:
        return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))
    @staticmethod
    def triangle_inequality(p1: tuple, p2: tuple, p3: tuple) -> bool:
        d12 = SpatialLogic.distance(p1,p2)
        d23 = SpatialLogic.distance(p2,p3)
        d13 = SpatialLogic.distance(p1,p3)
        return d13 <= d12 + d23 + 1e-9
    
    @staticmethod
    def suvat_displacement(u: float, a: float, t: float) -> float:
        return float(u * t + 0.5 * a * t * t)
    
    @staticmethod
    def suvat_velocity(u: float, a: float, t: float) -> float:
        return float(u + a * t)

class MathSymbolics:
    @staticmethod
    def _to_int(val: str) -> Optional[int]:
        try:
            return int(str(val))
        except Exception:
            return None
    
    @staticmethod
    def simplify_fraction(num: str, den: str) -> Optional[tuple]:
        from math import gcd
        n = MathSymbolics._to_int(num)
        d = MathSymbolics._to_int(den)
        if n is None or d is None or d == 0:
            return None
        g = gcd(n, d)
        return (n // g, d // g)
    
    @staticmethod
    def ratio_equal(a_num: str, a_den: str, b_num: str, b_den: str) -> Optional[bool]:
        a = MathSymbolics.simplify_fraction(a_num, a_den)
        b = MathSymbolics.simplify_fraction(b_num, b_den)
        if a is None or b is None:
            return None
        return a[0] == b[0] and a[1] == b[1]
    
    @staticmethod
    def solve_linear(a: float, b: float, c: float) -> Optional[float]:
        if abs(a) < 1e-12:
            return None
        return (c - b) / a
    
    @staticmethod
    def rewrite_distribute(expr: str) -> Optional[str]:
        s = expr.replace(" ", "")
        m = re.match(r'^([a-zA-Z0-9]+)\*\(([a-zA-Z0-9]+)\+([a-zA-Z0-9]+)\)$', s)
        if not m:
            return None
        a, b, c = m.group(1), m.group(2), m.group(3)
        return f"{a}*{b}+{a}*{c}"
    
    @staticmethod
    def rewrite_factor(expr: str) -> Optional[str]:
        s = expr.replace(" ", "")
        m = re.match(r'^([a-zA-Z0-9]+)\*([a-zA-Z0-9]+)\+([a-zA-Z0-9]+)\*([a-zA-Z0-9]+)$', s)
        if not m:
            return None
        a, b, c, d = m.group(1), m.group(2), m.group(3), m.group(4)
        if a == c:
            return f"{a}*({b}+{d})"
        if b == d:
            return f"{b}*({a}+{c})"
        return None
    
    @staticmethod
    def solve_inequality_interval(a: float, b: float, c: float, op: str) -> Optional[Tuple[float,float]]:
        if abs(a) < 1e-12:
            return None
        x0 = (c - b) / a
        if op == "<=":
            return (-1e12, x0) if a > 0 else (x0, 1e12)
        if op == ">=":
            return (x0, 1e12) if a > 0 else (-1e12, x0)
        return None
    
class AlgebraEngine:
    @staticmethod
    def combine_like_terms(expr: str) -> Optional[str]:
        s = expr.replace(" ", "")
        m = re.match(r'^([+-]?\d+(?:\.\d+)?)\*([a-zA-Z]+)\+([+-]?\d+(?:\.\d+)?)\*\2$', s)
        if not m:
            return None
        a = float(m.group(1))
        b = float(m.group(3))
        var = m.group(2)
        c = a + b
        if abs(c - int(c)) < 1e-9:
            c = int(c)
        return f"{c}*{var}"
    
    @staticmethod
    def derivative_poly(expr: str, var: str = "x") -> Optional[str]:
        s = expr.replace(" ", "")
        parts = CalculusEngine._split_terms(s)
        terms = []
        for p in parts:
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)\*'+re.escape(var)+r'\^([0-9]+)$', p)
            if m:
                a = float(m.group(1))
                n = int(m.group(2))
                if n == 0:
                    continue
                coeff = a * n
                power = n - 1
                if abs(coeff - int(coeff)) < 1e-9:
                    coeff = int(coeff)
                if power == 0:
                    terms.append(str(coeff))
                elif power == 1:
                    terms.append(f"{coeff}*{var}")
                else:
                    terms.append(f"{coeff}*{var}^{power}")
            else:
                m2 = re.match(r'^([+-]?\d+(?:\.\d+)?)\*'+re.escape(var)+r'$', p)
                if m2:
                    a = float(m2.group(1))
                    if abs(a - int(a)) < 1e-9:
                        a = int(a)
                    terms.append(str(a))
        if not terms:
            return "0"
        return "+".join(terms)
    
class CalculusEngine:
    @staticmethod
    def _split_terms(expr: str) -> List[str]:
        s = expr.replace(" ", "")
        out = []
        buf = []
        depth = 0
        for ch in s:
            if ch == "(":
                depth += 1
                buf.append(ch)
            elif ch == ")":
                depth = max(0, depth - 1)
                buf.append(ch)
            elif ch == "+" and depth == 0:
                out.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        if buf:
            out.append("".join(buf))
        return out
    @staticmethod
    def derivative(expr: str, var: str = "x") -> Optional[str]:
        s = expr.replace(" ", "")
        parts = CalculusEngine._split_terms(s)
        terms = []
        for p in parts:
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)\*'+re.escape(var)+r'\^([0-9]+)$', p)
            if m:
                a = float(m.group(1)); n = int(m.group(2))
                if n == 0: continue
                coeff = a * n; power = n - 1
                if abs(coeff - int(coeff)) < 1e-9: coeff = int(coeff)
                if power == 0:
                    terms.append(str(coeff))
                elif power == 1:
                    terms.append(f"{coeff}*{var}")
                else:
                    terms.append(f"{coeff}*{var}^{power}")
                continue
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)\*'+re.escape(var)+r'$', p)
            if m:
                a = float(m.group(1))
                if abs(a - int(a)) < 1e-9: a = int(a)
                terms.append(str(a))
                continue
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)\*sin\('+re.escape(var)+r'\)$', p)
            if m:
                a = float(m.group(1))
                if abs(a - int(a)) < 1e-9: a = int(a)
                terms.append(f"{a}*cos({var})")
                continue
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)\*cos\('+re.escape(var)+r'\)$', p)
            if m:
                a = float(m.group(1))
                if abs(a - int(a)) < 1e-9: a = int(a)
                terms.append(f"{-a}*sin({var})")
                continue
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)\*exp\('+re.escape(var)+r'\)$', p)
            if m:
                a = float(m.group(1))
                if abs(a - int(a)) < 1e-9: a = int(a)
                terms.append(f"{a}*exp({var})")
                continue
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)\*ln\('+re.escape(var)+r'\)$', p)
            if m:
                a = float(m.group(1))
                if abs(a - int(a)) < 1e-9: a = int(a)
                terms.append(f"{a}*(1/{var})")
                continue
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)\*sin\(([+-]?\d+(?:\.\d+)?)\*'+re.escape(var)+r'(?:\+([+-]?\d+(?:\.\d+)?))?\)$', p)
            if m:
                c = float(m.group(1)); a = float(m.group(2)); b = m.group(3)
                k = c * a
                if b is None: b = "0"
                if abs(k - int(k)) < 1e-9: k = int(k)
                a_s = str(int(a)) if abs(a - int(a)) < 1e-9 else str(a)
                terms.append(f"{k}*cos({a_s}*{var}+{b})")
                continue
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)\*cos\(([+-]?\d+(?:\.\d+)?)\*'+re.escape(var)+r'(?:\+([+-]?\d+(?:\.\d+)?))?\)$', p)
            if m:
                c = float(m.group(1)); a = float(m.group(2)); b = m.group(3)
                k = -c * a
                if b is None: b = "0"
                if abs(k - int(k)) < 1e-9: k = int(k)
                a_s = str(int(a)) if abs(a - int(a)) < 1e-9 else str(a)
                terms.append(f"{k}*sin({a_s}*{var}+{b})")
                continue
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)\*exp\(([+-]?\d+(?:\.\d+)?)\*'+re.escape(var)+r'(?:\+([+-]?\d+(?:\.\d+)?))?\)$', p)
            if m:
                c = float(m.group(1)); a = float(m.group(2)); b = m.group(3)
                k = c * a
                if b is None: b = "0"
                if abs(k - int(k)) < 1e-9: k = int(k)
                a_s = str(int(a)) if abs(a - int(a)) < 1e-9 else str(a)
                terms.append(f"{k}*exp({a_s}*{var}+{b})")
                continue
        if not terms:
            return "0"
        return "+".join(terms)
    
    @staticmethod
    def integral_basic(expr: str, var: str = "x") -> Optional[str]:
        s = expr.replace(" ", "")
        parts = CalculusEngine._split_terms(s)
        terms = []
        for p in parts:
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)\*'+re.escape(var)+r'\^([0-9]+)$', p)
            if m:
                a = float(m.group(1)); n = int(m.group(2))
                k = a / (n + 1); power = n + 1
                if abs(k - int(k)) < 1e-9: k = int(k)
                if power == 1:
                    terms.append(f"{k}*{var}")
                else:
                    terms.append(f"{k}*{var}^{power}")
                continue
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)$', p)
            if m:
                a = float(m.group(1))
                if abs(a - int(a)) < 1e-9: a = int(a)
                terms.append(f"{a}*{var}")
                continue
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)\*sin\('+re.escape(var)+r'\)$', p)
            if m:
                a = float(m.group(1))
                k = -a
                if abs(k - int(k)) < 1e-9: k = int(k)
                terms.append(f"{k}*cos({var})")
                continue
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)\*cos\('+re.escape(var)+r'\)$', p)
            if m:
                a = float(m.group(1))
                if abs(a - int(a)) < 1e-9: a = int(a)
                terms.append(f"{a}*sin({var})")
                continue
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)\*exp\('+re.escape(var)+r'\)$', p)
            if m:
                a = float(m.group(1))
                if abs(a - int(a)) < 1e-9: a = int(a)
                terms.append(f"{a}*exp({var})")
                continue
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)\*sin\(([+-]?\d+(?:\.\d+)?)\*'+re.escape(var)+r'(?:\+([+-]?\d+(?:\.\d+)?))?\)$', p)
            if m:
                c = float(m.group(1)); a = float(m.group(2)); b = m.group(3)
                k = -c / max(1e-12, a)
                if b is None: b = "0"
                if abs(k - int(k)) < 1e-9: k = int(k)
                a_s = str(int(a)) if abs(a - int(a)) < 1e-9 else str(a)
                terms.append(f"{k}*cos({a_s}*{var}+{b})")
                continue
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)\*cos\(([+-]?\d+(?:\.\d+)?)\*'+re.escape(var)+r'(?:\+([+-]?\d+(?:\.\d+)?))?\)$', p)
            if m:
                c = float(m.group(1)); a = float(m.group(2)); b = m.group(3)
                k = c / max(1e-12, a)
                if b is None: b = "0"
                if abs(k - int(k)) < 1e-9: k = int(k)
                a_s = str(int(a)) if abs(a - int(a)) < 1e-9 else str(a)
                terms.append(f"{k}*sin({a_s}*{var}+{b})")
                continue
            m = re.match(r'^([+-]?\d+(?:\.\d+)?)\*exp\(([+-]?\d+(?:\.\d+)?)\*'+re.escape(var)+r'(?:\+([+-]?\d+(?:\.\d+)?))?\)$', p)
            if m:
                c = float(m.group(1)); a = float(m.group(2)); b = m.group(3)
                k = c / max(1e-12, a)
                if b is None: b = "0"
                if abs(k - int(k)) < 1e-9: k = int(k)
                a_s = str(int(a)) if abs(a - int(a)) < 1e-9 else str(a)
                terms.append(f"{k}*exp({a_s}*{var}+{b})")
                continue
        if not terms:
            return "0"
        return "+".join(terms)

class LinearSystem:
    @staticmethod
    def solve(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
        try:
            M = np.array(A, dtype=np.float64)
            y = np.array(b, dtype=np.float64)
            n, m = M.shape
            if n == m:
                x = np.linalg.solve(M, y)
            else:
                x, _, _, _ = np.linalg.lstsq(M, y, rcond=None)
            return [float(v) for v in x]
        except Exception:
            return None

class QuadraticOpt:
    @staticmethod
    def minimize_diag(Q: List[float], q: List[float]) -> Optional[List[float]]:
        try:
            Qd = np.array(Q, dtype=np.float64)
            qv = np.array(q, dtype=np.float64)
            inv = np.where(np.abs(Qd) < 1e-12, 0.0, 1.0 / Qd)
            x = -inv * qv
            return [float(v) for v in x]
        except Exception:
            return None
class SetTheory:
    @staticmethod
    def union(a: set, b: set) -> set:
        return set(a).union(set(b))
    @staticmethod
    def intersection(a: set, b: set) -> set:
        return set(a).intersection(set(b))
    @staticmethod
    def difference(a: set, b: set) -> set:
        return set(a).difference(set(b))
    @staticmethod
    def subset(a: set, b: set) -> bool:
        return set(a).issubset(set(b))

class BayesianNetwork:
    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self._cache: Dict[Tuple[str, bool, Tuple[Tuple[str, bool], ...]], float] = {}
    
    def add_node(self, name: str, parents: List[str], cpt: Callable[[List[bool]], float]) -> None:
        self.nodes[name] = {"parents": parents, "cpt": cpt}
    
    def query(self, target: str, evidence: Dict[str, bool]) -> float:
        names = list(self.nodes.keys())
        unknowns = [n for n in names if n not in evidence]
        key_true: Tuple[str, bool, Tuple[Tuple[str, bool], ...]] = (target, True, tuple(sorted(evidence.items())))
        key_false: Tuple[str, bool, Tuple[Tuple[str, bool], ...]] = (target, False, tuple(sorted(evidence.items())))
        if key_true in self._cache and key_false in self._cache:
            pt = self._cache[key_true]
            pf = self._cache[key_false]
            return pt / max(1e-12, (pt + pf))
        def prob_assignment(assign: Dict[str, bool]) -> float:
            p = 1.0
            for n in names:
                node = self.nodes[n]
                parent_vals: List[bool] = [assign[pn] for pn in node["parents"]]
                p_true: float = node["cpt"](parent_vals)
                p *= p_true if assign[n] else (1 - p_true)
            return p
        def enumerate_sum(target_val: bool) -> float:
            total = 0.0
            import itertools
            if len(unknowns) == 0:
                assign = dict(evidence)
                assign[target] = target_val
                return prob_assignment(assign)
            for bits in itertools.product([False, True], repeat=len(unknowns)):
                assign = dict(evidence)
                for i, n in enumerate(unknowns):
                    assign[n] = bits[i]
                assign[target] = target_val
                total += prob_assignment(assign)
            return total
        p_true = enumerate_sum(True)
        p_false = enumerate_sum(False)
        self._cache[key_true] = p_true
        self._cache[key_false] = p_false
        return p_true / max(1e-12, (p_true + p_false))
    
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

class DeterministicUtils:
    @staticmethod
    def set_seed(seed: int = 0):
        import random
        random.seed(seed)
        np.random.seed(seed)
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
        return {"success": False, "strategy": strategies[-1] if strategies else "", "elapsed": _t.perf_counter() - start}
    
class ConstraintPropagation:
    @staticmethod
    def solve_inequalities2(constraints: List[Tuple[float,float,float,str]]) -> Dict[str, Tuple[float,float]]:
        xmin, xmax = -1e3, 1e3
        ymin, ymax = -1e3, 1e3
        for _ in range(10):
            changed = False
            for a, b, c, op in constraints:
                if abs(a) < 1e-12 and abs(b) < 1e-12:
                    continue
                if abs(b) < 1e-12:
                    x0 = (c) / max(1e-12, a)
                    if op == "<=":
                        if a > 0:
                            new_xmax = min(xmax, x0)
                            changed |= new_xmax < xmax
                            xmax = new_xmax
                        else:
                            new_xmin = max(xmin, x0)
                            changed |= new_xmin > xmin
                            xmin = new_xmin
                    elif op == ">=":
                        if a > 0:
                            new_xmin = max(xmin, x0)
                            changed |= new_xmin > xmin
                            xmin = new_xmin
                        else:
                            new_xmax = min(xmax, x0)
                            changed |= new_xmax < xmax
                            xmax = new_xmax
                elif abs(a) < 1e-12:
                    y0 = (c) / max(1e-12, b)
                    if op == "<=":
                        if b > 0:
                            new_ymax = min(ymax, y0)
                            changed |= new_ymax < ymax
                            ymax = new_ymax
                        else:
                            new_ymin = max(ymin, y0)
                            changed |= new_ymin > ymin
                            ymin = new_ymin
                    elif op == ">=":
                        if b > 0:
                            new_ymin = max(ymin, y0)
                            changed |= new_ymin > ymin
                            ymin = new_ymin
                        else:
                            new_ymax = min(ymax, y0)
                            changed |= new_ymax < ymax
                            ymax = new_ymax
                else:
                    if op == "<=":
                        if b > 0:
                            yb = (c - a * xmin) / b
                            new_ymax = min(ymax, yb)
                            changed |= new_ymax < ymax
                            ymax = new_ymax
                        else:
                            yb = (c - a * xmin) / b
                            new_ymin = max(ymin, yb)
                            changed |= new_ymin > ymin
                            ymin = new_ymin
                    elif op == ">=":
                        if b > 0:
                            yb = (c - a * xmax) / b
                            new_ymin = max(ymin, yb)
                            changed |= new_ymin > ymin
                            ymin = new_ymin
                        else:
                            yb = (c - a * xmax) / b
                            new_ymax = min(ymax, yb)
                            changed |= new_ymax < ymax
                            ymax = new_ymax
            if not changed:
                break
        infeasible = xmin > xmax or ymin > ymax
        return {"x": (xmin, xmax), "y": (ymin, ymax), "infeasible": infeasible}
    
class ConstraintGraph:
    @staticmethod
    def initial_bounds_from_constraints(vars: List[str], constraints: List[Tuple[Dict[str,float], str, float]]) -> Dict[str, Tuple[float,float]]:
        bounds = {v: (-1e3, 1e3) for v in vars}
        for v in vars:
            lb = -1e3
            ub = 1e3
            for coeffs, op, c in constraints:
                if v not in coeffs:
                    continue
                a = coeffs[v]
                if op == "<=":
                    if a > 0:
                        ub = min(ub, c / max(a, 1e-12))
                    elif a < 0:
                        lb = max(lb, c / a)
                elif op == ">=":
                    if a > 0:
                        lb = max(lb, c / max(a, 1e-12))
                    elif a < 0:
                        ub = min(ub, c / a)
            if lb > ub:
                mid = 0.0
                lb, ub = min(mid, lb), max(mid, ub)
            bounds[v] = (lb, ub)
        return bounds
    @staticmethod
    def solve_boxes(vars: List[str], constraints: List[Tuple[Dict[str,float], str, float]], seed_bounds: Optional[Dict[str, Tuple[float,float]]] = None) -> Dict[str, Tuple[float,float]]:
        bounds = ConstraintGraph.initial_bounds_from_constraints(vars, constraints)
        if seed_bounds:
            for v in vars:
                if v in seed_bounds:
                    sl, su = seed_bounds[v]
                    bl, bu = bounds.get(v, (-1e3, 1e3))
                    nl = max(bl, sl)
                    nu = min(bu, su)
                    if nl > nu:
                        bounds[v] = (sl, su)
                    else:
                        bounds[v] = (nl, nu)
        for _ in range(10):
            changed = False
            for coeffs, op, c in constraints:
                for v in vars:
                    if v not in coeffs:
                        continue
                    a = coeffs[v]
                    others_min = sum(
                        coeffs[u] * (bounds[u][0] if coeffs[u] > 0 else bounds[u][1])
                        for u in vars if u in coeffs and u != v
                    )
                    others_max = sum(
                        coeffs[u] * (bounds[u][1] if coeffs[u] > 0 else bounds[u][0])
                        for u in vars if u in coeffs and u != v
                    )
                    if op == "<=":
                        if a > 0:
                            ub = (c - others_min) / a
                            old = bounds[v][1]
                            new = min(old, ub)
                            if new < old:
                                bounds[v] = (bounds[v][0], new)
                                changed = True
                        elif a < 0:
                            lb = (c - others_max) / a
                            old = bounds[v][0]
                            new = max(old, lb)
                            if new > old:
                                bounds[v] = (new, bounds[v][1])
                                changed = True
                    elif op == ">=":
                        if a > 0:
                            lb = (c - others_max) / a
                            old = bounds[v][0]
                            new = max(old, lb)
                            if new > old:
                                bounds[v] = (new, bounds[v][1])
                                changed = True
                        elif a < 0:
                            ub = (c - others_min) / a
                            old = bounds[v][1]
                            new = min(old, ub)
                            if new < old:
                                bounds[v] = (bounds[v][0], new)
                                changed = True
            if not changed:
                break
        return bounds
    
    @staticmethod
    def milp_optimize_bounds(vars: List[str], constraints: List[Tuple[Dict[str,float], str, float]], int_vars: Optional[List[str]] = None) -> Optional[Dict[str, Tuple[float,float]]]:
        try:
            import pulp
        except Exception:
            try:
                from ortools.linear_solver import pywraplp
            except Exception:
                return None
        int_set = set(int_vars or [])
        bounds = {}
        def solve_obj(is_max, var_name):
            try:
                prob = pulp.LpProblem("bounds", pulp.LpMaximize if is_max else pulp.LpMinimize)
                lp_vars = {}
                for v in vars:
                    if v in int_set:
                        lp_vars[v] = pulp.LpVariable(v, lowBound=None, upBound=None, cat=pulp.LpInteger)
                    else:
                        lp_vars[v] = pulp.LpVariable(v, lowBound=None, upBound=None, cat=pulp.LpContinuous)
                for coeffs, op, c in constraints:
                    expr = pulp.lpSum([a * lp_vars[v] for v, a in coeffs.items()])
                    if op == "<=":
                        prob += expr <= c
                    elif op == ">=":
                        prob += expr >= c
                prob += lp_vars[var_name]
                res = prob.solve(pulp.PULP_CBC_CMD(msg=False))
                if res != pulp.LpStatusOptimal:
                    return None
                return pulp.value(lp_vars[var_name])
            except Exception:
                return None
        for v in vars:
            lo = solve_obj(False, v)
            hi = solve_obj(True, v)
            if lo is None or hi is None:
                return None
            if lo > hi:
                lo, hi = hi, lo
            bounds[v] = (float(lo), float(hi))
        return bounds

    @staticmethod
    def fourier_motzkin_bounds(vars: List[str], constraints: List[Tuple[Dict[str,float], str, float]], cap: int = 200) -> Dict[str, Tuple[float,float]]:
        def normalize(cs):
            norm = []
            for coeffs, op, c in cs:
                if op == ">=":
                    coeffs = {k: -v for k, v in coeffs.items()}
                    c = -c
                norm.append((coeffs, "<=", c))
            return norm
        def prune(cs):
            if len(cs) <= cap:
                return cs
            cs = sorted(cs, key=lambda x: sum(abs(v) for v in x[0].values()))
            return cs[:cap]
        def eliminate_var(cs, v):
            P, N, Z = [], [], []
            for coeffs, _, c in cs:
                a = coeffs.get(v, 0.0)
                if a > 0: P.append((coeffs, c, a))
                elif a < 0: N.append((coeffs, c, a))
                else: Z.append(({k: val for k, val in coeffs.items() if k != v}, "<=", c))
            combined = []
            for pc, pc_c, pa in P:
                for nc, nc_c, na in N:
                    new_coeffs = {}
                    for w in pc.keys():
                        if w == v: continue
                        new_coeffs[w] = new_coeffs.get(w, 0.0) + pc[w] / pa
                    for w in nc.keys():
                        if w == v: continue
                        new_coeffs[w] = new_coeffs.get(w, 0.0) - nc[w] / (-na)
                    new_c = pc_c / pa - nc_c / (-na)
                    combined.append((new_coeffs, "<=", new_c))
            return prune(Z + combined)
        def bounds_for_var(v, cs):
            cs_norm = normalize(cs)
            order = [w for w in vars if w != v]
            for w in order:
                cs_norm = eliminate_var(cs_norm, w)
            ub = 1e12
            lb = -1e12
            for coeffs, _, c in cs_norm:
                a = coeffs.get(v, 0.0)
                if abs(a) < 1e-12:
                    continue
                if a > 0:
                    ub = min(ub, c / a)
                else:
                    lb = max(lb, c / a)
            return (lb, ub)
        final_bounds = {}
        for v in vars:
            final_bounds[v] = bounds_for_var(v, constraints)
        return final_bounds
    @staticmethod
    def z3_optimize_bounds(vars: List[str], constraints: List[Tuple[Dict[str,float], str, float]], int_vars: Optional[List[str]] = None) -> Optional[Dict[str, Tuple[float,float]]]:
        try:
            import z3
        except Exception:
            return None
        int_set = set(int_vars or [])
        zvars = {}
        def zv(name):
            if name not in zvars:
                zvars[name] = z3.Int(name) if name in int_set else z3.Real(name)
            return zvars[name]
        # Build common constraints
        base = []
        for coeffs, op, c in constraints:
            expr = 0
            for v, a in coeffs.items():
                expr = expr + a * zv(v)
            if op == "<=":
                base.append(expr <= c)
            elif op == ">=":
                base.append(expr >= c)
        bounds = {}
        for v in vars:
            # Minimize v
            opt_min = z3.Optimize()
            for con in base:
                opt_min.add(con)
            opt_min.minimize(zv(v))
            if opt_min.check() == z3.sat:
                mmin = opt_min.model()
                sval = mmin.eval(zv(v))
                sdec = str(sval.as_decimal(20)) if hasattr(sval, "as_decimal") else str(sval)
                if ("oo" in sdec) or ("inf" in sdec.lower()) or ("nan" in sdec.lower()):
                    return None
                try:
                    lo = float(sdec.replace("?", "0"))
                except Exception:
                    return None
            else:
                return None
            # Maximize v
            opt_max = z3.Optimize()
            for con in base:
                opt_max.add(con)
            opt_max.maximize(zv(v))
            if opt_max.check() == z3.sat:
                mmax = opt_max.model()
                sval = mmax.eval(zv(v))
                sdec = str(sval.as_decimal(20)) if hasattr(sval, "as_decimal") else str(sval)
                if ("oo" in sdec) or ("inf" in sdec.lower()) or ("nan" in sdec.lower()):
                    return None
                try:
                    hi = float(sdec.replace("?", "0"))
                except Exception:
                    return None
            else:
                return None
            if lo > hi:
                lo, hi = hi, lo
            bounds[v] = (lo, hi)
        return bounds
    
    @staticmethod
    def mccormick_envelope(x: str, y: str, bounds: Dict[str, Tuple[float,float]]) -> List[Tuple[Dict[str,float], str, float]]:
        lx, ux = bounds[x]
        ly, uy = bounds[y]
        # Lower bounds: z >= lx*y + x*ly - lx*ly and z >= ux*y + x*uy - ux*uy
        c1 = ({x: ly, y: lx, "w": -1.0}, "<=", -(lx * ly))             # -w + ly*x + lx*y <= -lx*ly
        c2 = ({x: uy, y: ux, "w": -1.0}, "<=", -(ux * uy))             # -w + uy*x + ux*y <= -ux*uy
        # Upper bounds: z <= ux*y + x*ly - ux*ly and z <= lx*y + x*uy - lx*uy
        c3 = ({x: -ly, y: -ux, "w": 1.0}, "<=", -(ux * ly))            # w - ly*x - ux*y <= -ux*ly
        c4 = ({x: -uy, y: -lx, "w": 1.0}, "<=", -(lx * uy))            # w - uy*x - lx*y <= -lx*uy
        return [c1, c2, c3, c4]
    
    @staticmethod
    def envelope_square(x: str, w: str, bounds: Dict[str, Tuple[float,float]], segments: int = 4) -> List[Tuple[Dict[str,float], str, float]]:
        lx, ux = bounds[x]
        lines = []
        # adaptive segmentation based on interval width
        width = abs(ux - lx)
        if width > 1e-9:
            seg = int(min(8, max(2, round(width * 2))))
            segments = max(segments, seg)
        if segments < 1:
            segments = 1
        # Upper bounds via secants over segments
        pts = [lx + (ux - lx) * i / segments for i in range(segments + 1)]
        for i in range(segments):
            x1, x2 = pts[i], pts[i+1]
            if abs(x2 - x1) < 1e-9:
                continue
            y1, y2 = x1*x1, x2*x2
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            # w <= m*x + b  ->  w - m*x <= b
            lines.append(({x: -m, w: 1.0}, "<=", b))
        # Lower bounds via tangents at endpoints
        for t in [lx, ux]:
            m = 2.0 * t
            b = t*t - m * t
            # w >= m*x + b  ->  -w + m*x <= -b
            lines.append(({x: m, w: -1.0}, "<=", -b))
        return lines
    
    @staticmethod
    def envelope_exp(x: str, w: str, bounds: Dict[str, Tuple[float,float]], segments: int = 4) -> List[Tuple[Dict[str,float], str, float]]:
        lx, ux = bounds[x]
        lines = []
        width = abs(ux - lx)
        if width > 1e-9:
            seg = int(min(8, max(2, round(width * 2))))
            segments = max(segments, seg)
        if segments < 1:
            segments = 1
        pts = [lx + (ux - lx) * i / segments for i in range(segments + 1)]
        for i in range(segments):
            x1, x2 = pts[i], pts[i+1]
            if abs(x2 - x1) < 1e-9:
                continue
            y1, y2 = np.exp(x1), np.exp(x2)
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            lines.append(({x: -m, w: 1.0}, "<=", b))
        for t in [lx, ux]:
            m = np.exp(t)
            b = np.exp(t) - m * t
            lines.append(({x: m, w: -1.0}, "<=", -b))
        return lines
    
    @staticmethod
    def envelope_log(x: str, w: str, bounds: Dict[str, Tuple[float,float]], segments: int = 4) -> List[Tuple[Dict[str,float], str, float]]:
        lx, ux = bounds[x]
        lx = max(lx, 1e-6)
        lines = []
        width = abs(ux - lx)
        if width > 1e-9:
            seg = int(min(8, max(2, round(width * 2))))
            segments = max(segments, seg)
        if segments < 1:
            segments = 1
        pts = [lx + (ux - lx) * i / segments for i in range(segments + 1)]
        for i in range(segments):
            x1, x2 = pts[i], pts[i+1]
            if abs(x2 - x1) < 1e-9:
                continue
            y1, y2 = np.log(x1), np.log(x2)
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            lines.append(({x: -m, w: 1.0}, "<=", b))
        for t in [lx, ux]:
            m = 1.0 / max(1e-6, t)
            b = np.log(t) - m * t
            lines.append(({x: m, w: -1.0}, "<=", -b))
        return lines
    
    @staticmethod
    def envelope_sigmoid(x: str, w: str, bounds: Dict[str, Tuple[float,float]], segments: int = 4) -> List[Tuple[Dict[str,float], str, float]]:
        lx, ux = bounds[x]
        lines = []
        width = abs(ux - lx)
        if width > 1e-9:
            seg = int(min(8, max(2, round(width * 2))))
            segments = max(segments, seg)
        if segments < 1:
            segments = 1
        def sig(t):
            return 1.0 / (1.0 + np.exp(-t))
        def dsig(t):
            s = sig(t)
            return s * (1.0 - s)
        pts = [lx + (ux - lx) * i / segments for i in range(segments + 1)]
        for i in range(segments):
            x1, x2 = pts[i], pts[i+1]
            if abs(x2 - x1) < 1e-9:
                continue
            y1, y2 = sig(x1), sig(x2)
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            lines.append(({x: -m, w: 1.0}, "<=", b))
        for t in [lx, ux]:
            m = dsig(t)
            b = sig(t) - m * t
            lines.append(({x: m, w: -1.0}, "<=", -b))
        return lines
    
    @staticmethod
    def envelope_piecewise(x: str, w: str, samples: List[Tuple[float,float]]) -> List[Tuple[Dict[str,float], str, float]]:
        if not samples or len(samples) < 2:
            return []
        pts = sorted(samples, key=lambda p: p[0])
        # Adaptive densification based on slope change
        def densify(pts, max_segments=8, tol=0.05):
            changed = True
            while changed and len(pts) < max_segments + 1:
                changed = False
                i = 0
                new_pts = []
                while i < len(pts) - 1:
                    x1, y1 = pts[i]
                    x2, y2 = pts[i+1]
                    new_pts.append((x1, y1))
                    if abs(x2 - x1) > 1e-9:
                        m = (y2 - y1) / (x2 - x1)
                        xm = 0.5 * (x1 + x2)
                        # linear prediction
                        yp = y1 + m * (xm - x1)
                        # heuristic curvature: assume convex/concave; if deviation large, add midpoint
                        ym = 0.5 * (y1 + y2)
                        if abs(ym - yp) > tol * (abs(y1) + abs(y2) + 1e-6):
                            changed = True
                            new_pts.append((xm, ym))
                    i += 1
                new_pts.append(pts[-1])
                pts = sorted(new_pts, key=lambda p: p[0])
            return pts
        pts = densify(pts)
        lines = []
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i+1]
            if abs(x2 - x1) < 1e-9:
                continue
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            lines.append(({x: -m, w: 1.0}, "<=", b))
        def tangent(i):
            if i == 0:
                x1, y1 = pts[0]
                x2, y2 = pts[1]
            elif i == len(pts) - 1:
                x1, y1 = pts[-2]
                x2, y2 = pts[-1]
            else:
                x1, y1 = pts[i-1]
                x2, y2 = pts[i+1]
            if abs(x2 - x1) < 1e-9:
                return None
            m = (y2 - y1) / (x2 - x1)
            xc, yc = pts[i]
            b = yc - m * xc
            return m, b
        for i in range(len(pts)):
            tb = tangent(i)
            if tb is None:
                continue
            m, b = tb
            lines.append(({x: m, w: -1.0}, "<=", -b))
        return lines
    
    @staticmethod
    def chain_product_envelopes(chain: List[Tuple[str,str,str]], bounds: Dict[str, Tuple[float,float]]) -> List[Tuple[Dict[str,float], str, float]]:
        envs = []
        for w, a, b in chain:
            envs.extend(ConstraintGraph.mccormick_envelope(a, b, bounds))
        return envs
    
    @staticmethod
    def solve_with_envelopes(vars: List[str], constraints: List[Tuple[Dict[str,float], str, float]], bilinear_pairs: List[Tuple[str,str,str]], iters: int = 3) -> Dict[str, Tuple[float,float]]:
        base = ConstraintGraph.solve_boxes(vars, constraints)
        aug = list(constraints)
        for w, x, y in bilinear_pairs:
            if w not in base:
                lx, ux = base.get(x, (-1e3, 1e3))
                ly, uy = base.get(y, (-1e3, 1e3))
                lo = min(lx*ly, lx*uy, ux*ly, ux*uy)
                hi = max(lx*ly, lx*uy, ux*ly, ux*uy)
                base[w] = (lo, hi)
            env = ConstraintGraph.mccormick_envelope(x, y, base)
            aug.extend(env)
        final_bounds = ConstraintGraph.solve_boxes(vars, aug, seed_bounds=base)
        milp = ConstraintGraph.milp_optimize_bounds(vars, aug)
        if milp:
            for v in vars:
                if v in milp:
                    lo = max(final_bounds.get(v, (-1e9, 1e9))[0], milp[v][0])
                    hi = min(final_bounds.get(v, (-1e9, 1e9))[1], milp[v][1])
                    if lo <= hi:
                        final_bounds[v] = (lo, hi)
        z3b = ConstraintGraph.z3_optimize_bounds(vars, aug)
        if z3b:
            for v in vars:
                if v in z3b:
                    lo = max(final_bounds.get(v, (-1e9, 1e9))[0], z3b[v][0])
                    hi = min(final_bounds.get(v, (-1e9, 1e9))[1], z3b[v][1])
                    if lo <= hi:
                        final_bounds[v] = (lo, hi)
        return final_bounds
    
    @staticmethod
    def solve_with_extended_envelopes(vars: List[str], constraints: List[Tuple[Dict[str,float], str, float]], bilinear_pairs: List[Tuple[str,str,str]] = None, square_pairs: List[Tuple[str,str]] = None, chain_pairs: List[Tuple[str,str,str]] = None, log_pairs: List[Tuple[str,str]] = None, sigmoid_pairs: List[Tuple[str,str]] = None, piecewise_pairs: List[Tuple[str,str,List[Tuple[float,float]]]] = None, segments: int = 4, int_vars: Optional[List[str]] = None) -> Dict[str, Tuple[float,float]]:
        bilinear_pairs = bilinear_pairs or []
        square_pairs = square_pairs or []
        chain_pairs = chain_pairs or []
        log_pairs = log_pairs or []
        sigmoid_pairs = sigmoid_pairs or []
        piecewise_pairs = piecewise_pairs or []
        base = ConstraintGraph.solve_boxes(vars, constraints)
        aug = list(constraints)
        for w, x, y in bilinear_pairs:
            if w not in base:
                lx, ux = base.get(x, (-1e3, 1e3))
                ly, uy = base.get(y, (-1e3, 1e3))
                lo = min(lx*ly, lx*uy, ux*ly, ux*uy)
                hi = max(lx*ly, lx*uy, ux*ly, ux*uy)
                base[w] = (lo, hi)
            aug.extend(ConstraintGraph.mccormick_envelope(x, y, base))
        if bilinear_pairs:
            split = max(2, min(4, segments))
            for w, x, y in bilinear_pairs:
                lx, ux = base.get(x, (-1e3, 1e3))
                ly, uy = base.get(y, (-1e3, 1e3))
                xs = [lx + (ux - lx) * i / split for i in range(split + 1)]
                ys = [ly + (uy - ly) * j / split for j in range(split + 1)]
                for i in range(split):
                    for j in range(split):
                        local = dict(base)
                        local[x] = (xs[i], xs[i+1])
                        local[y] = (ys[j], ys[j+1])
                        env_local = ConstraintGraph.mccormick_envelope(x, y, local)
                        aug.extend(env_local)
        for w, x in square_pairs:
            if w not in base:
                lx, ux = base.get(x, (-1e3, 1e3))
                lo = min(lx*lx, ux*ux, 0.0)
                hi = max(lx*lx, ux*ux)
                base[w] = (lo, hi if hi > lo else lo+1e-6)
            aug.extend(ConstraintGraph.envelope_square(x, w, base, segments=segments))
        for w, x in log_pairs:
            if w not in base:
                lx, ux = base.get(x, (1e-6, 1e3))
                lo = min(np.log(max(1e-6, lx)), np.log(max(1e-6, ux)))
                hi = max(np.log(max(1e-6, lx)), np.log(max(1e-6, ux)))
                base[w] = (lo, hi if hi > lo else lo+1e-6)
            aug.extend(ConstraintGraph.envelope_log(x, w, base, segments=segments))
        for w, x in sigmoid_pairs:
            if w not in base:
                lx, ux = base.get(x, (-10.0, 10.0))
                lo = min(1.0/(1.0+np.exp(-lx)), 1.0/(1.0+np.exp(-ux)))
                hi = max(1.0/(1.0+np.exp(-lx)), 1.0/(1.0+np.exp(-ux)))
                base[w] = (lo, hi if hi > lo else lo+1e-6)
            aug.extend(ConstraintGraph.envelope_sigmoid(x, w, base, segments=segments))
        for w, x, samples in piecewise_pairs:
            xs = [sx for sx, _ in samples]
            ys = [sy for _, sy in samples]
            if w not in base:
                lo = min(ys)
                hi = max(ys)
                base[w] = (lo, hi if hi > lo else lo+1e-6)
            aug.extend(ConstraintGraph.envelope_piecewise(x, w, samples))
        if chain_pairs:
            aug.extend(ConstraintGraph.chain_product_envelopes(chain_pairs, base))
        final_bounds = ConstraintGraph.solve_boxes(vars, aug, seed_bounds=base)
        milp = ConstraintGraph.milp_optimize_bounds(vars, aug, int_vars=int_vars)
        if milp:
            for v in vars:
                if v in milp:
                    lo = max(final_bounds.get(v, (-1e9, 1e9))[0], milp[v][0])
                    hi = min(final_bounds.get(v, (-1e9, 1e9))[1], milp[v][1])
                    if lo <= hi:
                        final_bounds[v] = (lo, hi)
        # Tighten with Z3 optimize; respect int_vars if provided
        z3b = ConstraintGraph.z3_optimize_bounds(vars, aug, int_vars=int_vars)
        if z3b:
            for v in vars:
                if v in z3b:
                    lo = max(final_bounds.get(v, (-1e9, 1e9))[0], z3b[v][0])
                    hi = min(final_bounds.get(v, (-1e9, 1e9))[1], z3b[v][1])
                    if lo <= hi:
                        final_bounds[v] = (lo, hi)
        # Preserve input variable bounds for stability
        inputs_preserve = set()
        inputs_preserve.update([x for _, x in square_pairs])
        for _, a, b in chain_pairs:
            inputs_preserve.add(a)
            inputs_preserve.add(b)
        for v in inputs_preserve:
            if v in base:
                final_bounds[v] = base[v]
        # Clamp inverted outputs to base when possible
        for w, _ in square_pairs:
            lo, hi = final_bounds.get(w, (-1e3, 1e3))
            if lo > hi:
                final_bounds[w] = base.get(w, (0.0, 1e3))
        for w, _, _ in chain_pairs:
            lo, hi = final_bounds.get(w, (-1e3, 1e3))
            if lo > hi:
                final_bounds[w] = base.get(w, (0.0, 1e3))
        return final_bounds
    
    @staticmethod
    def project_feasible_point(bounds: Dict[str, Tuple[float,float]], constraints: List[Tuple[Dict[str,float], str, float]], steps: int = 50, alpha: float = 0.1, bilinear_pairs: Optional[List[Tuple[str,str,str]]] = None) -> Dict[str, float]:
        x = {v: 0.5 * (b[0] + b[1]) for v, b in bounds.items()}
        def clamp(v, a, b):
            return max(a, min(b, v))
        for _ in range(steps):
            for coeffs, op, c in constraints:
                s = 0.0
                for v, a in coeffs.items():
                    s += a * x.get(v, 0.0)
                if op == "<=":
                    viol = s - c
                    if viol > 0:
                        norm = sum(abs(a) for a in coeffs.values()) + 1e-9
                        for v, a in coeffs.items():
                            x[v] = clamp(x[v] - alpha * viol * (a / norm), bounds[v][0], bounds[v][1])
                elif op == ">=":
                    viol = c - s
                    if viol > 0:
                        norm = sum(abs(a) for a in coeffs.values()) + 1e-9
                        for v, a in coeffs.items():
                            x[v] = clamp(x[v] + alpha * viol * (a / norm), bounds[v][0], bounds[v][1])
        if bilinear_pairs:
            for w, xv, yv in bilinear_pairs:
                if w in bounds and xv in x and yv in x:
                    prod = x[xv] * x[yv]
                    x[w] = clamp(prod, bounds[w][0], bounds[w][1])
        # Assign squares if specified via convention: store square pairs in bounds with tag
        if "square_pairs" in bounds:
            for w, xv in bounds["square_pairs"]:
                if w in bounds and xv in x:
                    sq = x[xv] * x[xv]
                    x[w] = clamp(sq, bounds[w][0], bounds[w][1])
        return x
    
    @staticmethod
    def quick_bilinear_feasible(vars: List[str], constraints: List[Tuple[Dict[str,float], str, float]], bilinear_pairs: List[Tuple[str,str,str]]) -> bool:
        if not bilinear_pairs:
            return False
        base = ConstraintGraph.initial_bounds_from_constraints(vars, constraints)
        for w, x, y in bilinear_pairs:
            lx, ux = base.get(x, (-1e3, 1e3))
            ly, uy = base.get(y, (-1e3, 1e3))
            corners = [lx*ly, lx*uy, ux*ly, ux*uy]
            p_lo = min(corners)
            p_hi = max(corners)
            for coeffs, op, c in constraints:
                if w in coeffs and len(coeffs) == 1:
                    a = coeffs[w]
                    if op == "<=":
                        # a*w <= c  => w <= c/a (if a>0), use lower product bound
                        if a > 0 and p_lo > c + 1e-9:
                            return False
                        if a < 0 and p_hi < c - 1e-9:
                            return False
                    elif op == ">=":
                        if a > 0 and p_hi < c - 1e-9:
                            return False
                        if a < 0 and p_lo > c + 1e-9:
                            return False
        return True
    
class Polytope:
    def __init__(self, constraints: List[Tuple[Dict[str,float], str, float]]):
        self.constraints = constraints
    
    def intersection(self, other: 'Polytope') -> 'Polytope':
        return Polytope(self.constraints + other.constraints)
    
    def contains(self, point: Dict[str,float]) -> bool:
        for coeffs, op, c in self.constraints:
            s = sum(a * point.get(v, 0.0) for v, a in coeffs.items())
            if op == "<=" and s > c + 1e-9:
                return False
            if op == ">=" and s < c - 1e-9:
                return False
        return True
    
    def bounds(self, vars: List[str]) -> Dict[str, Tuple[float,float]]:
        return ConstraintGraph.solve_boxes(vars, self.constraints)
    
    @staticmethod
    def linearize_abs(var: str, aux: str) -> List[Tuple[Dict[str,float], str, float]]:
        return [
            ({var: 1.0, aux: -1.0}, "<=", 0.0),
            ({var: -1.0, aux: -1.0}, "<=", 0.0),
            ({aux: 1.0}, ">=", 0.0),
        ]
    
    @staticmethod
    def parse_numeric_constraints(text: str) -> List[Tuple[Dict[str,float], str, float]]:
        cs = []
        for m in re.finditer(r'([a-zA-Z]+)\s*(<=|>=|=)\s*([+-]?\d+(?:\.\d+)?)', text):
            v = m.group(1)
            op = m.group(2)
            val = float(m.group(3))
            if op == "=":
                cs.append(({v: 1.0}, "<=", val))
                cs.append(({v: 1.0}, ">=", val))
            else:
                cs.append(({v: 1.0}, op, val))
        for m in re.finditer(r'([a-zA-Z]+)\s+(at\s+least|at\s+most)\s+([+-]?\d+(?:\.\d+)?)', text, flags=re.IGNORECASE):
            v = m.group(1)
            phrase = m.group(2).lower()
            val = float(m.group(3))
            if "least" in phrase:
                cs.append(({v: 1.0}, ">=", val))
            else:
                cs.append(({v: 1.0}, "<=", val))
        for m in re.finditer(r'([a-zA-Z]+)\s+(less\s+than\s+or\s+equal\s+to|greater\s+than\s+or\s+equal\s+to)\s+([+-]?\d+(?:\.\d+)?)', text, flags=re.IGNORECASE):
            v = m.group(1)
            phrase = m.group(2).lower()
            val = float(m.group(3))
            if "greater" in phrase:
                cs.append(({v: 1.0}, ">=", val))
            else:
                cs.append(({v: 1.0}, "<=", val))
        return cs
    
    @staticmethod
    def parse_relations(text: str) -> List[tuple]:
        rels = []
        for m in re.finditer(r'\ball\s+([a-z]+)\s+are\s+([a-z]+)\b', text, flags=re.IGNORECASE):
            rels.append(("all", m.group(1), m.group(2)))
        for m in re.finditer(r'\bno\s+([a-z]+)\s+are\s+([a-z]+)\b', text, flags=re.IGNORECASE):
            rels.append(("no", m.group(1), m.group(2)))
        return rels
    
class InductionEngine:
    @staticmethod
    def induce_rules(observations: List[tuple], min_support: int = 2) -> List[tuple]:
        counts = {}
        for typ, a, b in observations:
            if typ != "instance":
                continue
            key = (a, b)
            counts[key] = counts.get(key, 0) + 1
        rules = []
        for (a, b), cnt in counts.items():
            if cnt >= min_support:
                rules.append(("all", a, b))
        return rules
    
    @staticmethod
    def induce_rules_extended(observations: List[tuple], min_support: int = 2, neg_types: List[str] = None, mdl_lambda: float = 0.1) -> List[tuple]:
        neg_types = neg_types or ["no_instance", "not_instance"]
        pos_counts = {}
        neg_counts = {}
        for typ, a, b in observations:
            key = (a, b)
            if typ == "instance":
                pos_counts[key] = pos_counts.get(key, 0) + 1
            elif typ in neg_types or typ == "no":
                neg_counts[key] = neg_counts.get(key, 0) + 1
        rules = []
        scored = []
        for key, pos in pos_counts.items():
            neg = neg_counts.get(key, 0)
            total = pos + neg
            if pos >= min_support:
                conf = pos / max(1, total)
                a, b = key
                desc_len = len(a) + len(b)
                score = pos * conf - mdl_lambda * desc_len
                scored.append((score, ("all", a, b), conf, pos, neg))
        scored.sort(key=lambda x: (-x[0], -x[3], x[2]))
        for _, rule, conf, pos, neg in scored:
            rules.append((rule[0], rule[1], rule[2], {"confidence": conf, "pos": pos, "neg": neg}))
        return rules
    
    @staticmethod
    def transitive_closure(edges: List[Tuple[str,str]]) -> List[Tuple[str,str]]:
        key = tuple(sorted(edges))
        if key in _tc_cache:
            return _tc_cache[key]
        g = {}
        for a, b in edges:
            g.setdefault(a, set()).add(b)
        changed = True
        while changed:
            changed = False
            for a in list(g.keys()):
                add = set()
                for b in list(g[a]):
                    for c in list(g.get(b, set())):
                        if c not in g[a]:
                            add.add(c)
                if add:
                    g[a].update(add)
                    changed = True
        out = []
        for a, bs in g.items():
            for b in bs:
                out.append((a, b))
        out_sorted = sorted(out)
        _tc_cache[key] = out_sorted
        _tc_order.append(key)
        if len(_tc_order) > _tc_cap:
            old = _tc_order.pop(0)
            _tc_cache.pop(old, None)
        return out_sorted
    
class ProgramSynth:
    @staticmethod
    def synthesize(spec: Dict[str, any]) -> str:
        goal = spec.get("goal", "")
        if "sum" in goal:
            return "def compute(a,b):\n    return a+b\n"
        if "product" in goal:
            return "def compute(a,b):\n    return a*b\n"
        return "def compute(a,b):\n    return a\n"
    @staticmethod
    def verify(code: str, tests: List[Tuple[tuple, any]]) -> bool:
        ns = {}
        try:
            exec(code, ns)
            fn = ns.get("compute")
            if not callable(fn):
                return False
            for args, expect in tests:
                out = fn(*args)
                if out != expect:
                    return False
            return True
        except Exception:
            return False
    
    class Contract:
        def __init__(self, arg_types: List[str], precond: Callable[[tuple], bool], postcond: Callable[[tuple, any], bool]):
            self.arg_types = arg_types
            self.precond = precond
            self.postcond = postcond
    
    @staticmethod
    def verify_with_contracts(code: str, contracts: List['ProgramSynth.Contract'], samples: int = 50) -> Dict[str, any]:
        ns = {}
        try:
            exec(code, ns)
            fn = ns.get("compute")
            if not callable(fn):
                return {"ok": False, "reason": "no compute"}
            import random
            random.seed(0)
            def gen_val(t):
                if t == "int":
                    return random.randint(-10, 10)
                if t == "float":
                    return random.uniform(-10.0, 10.0)
                return 0
            counterexamples = []
            for c in contracts:
                for _ in range(samples):
                    args = tuple(gen_val(t) for t in c.arg_types)
                    if not c.precond(args):
                        continue
                    out = fn(*args)
                    if not c.postcond(args, out):
                        counterexamples.append({"args": args, "out": out})
                        break
            return {"ok": len(counterexamples) == 0, "counterexamples": counterexamples}
        except Exception as e:
            return {"ok": False, "reason": str(e)}
    
class SATBridge:
    @staticmethod
    def solve_cnf(clauses: List[List[int]]) -> Optional[Dict[int, bool]]:
        try:
            from pysat.solvers import Glucose3
        except Exception:
            return None
        s = Glucose3()
        for cl in clauses:
            s.add_clause(cl)
        sat = s.solve()
        if not sat:
            return None
        model = s.get_model()
        assign = {}
        for lit in model:
            v = abs(lit)
            assign[v] = lit > 0
        return assign
    
    @staticmethod
    def solve_maxsat(soft_clauses: List[List[int]], hard_clauses: List[List[int]] = None) -> Optional[Dict[int, bool]]:
        try:
            from pysat.examples.rc2 import RC2
            from pysat.formula import WCNF
        except Exception:
            return None
        wcnf = WCNF()
        for cl in hard_clauses or []:
            wcnf.append(cl)
        for cl in soft_clauses:
            wcnf.append(cl, weight=1)
        rc2 = RC2(wcnf)
        model = rc2.compute()
        if model is None:
            return None
        assign = {}
        for lit in model:
            v = abs(lit)
            assign[v] = lit > 0
        return assign
    
class SMTBridge:
    @staticmethod
    def solve_with_z3(assertions: List[tuple]) -> Optional[Dict[str, float]]:
        try:
            import z3
        except Exception:
            return None
        s = z3.Solver()
        vars = {}
        def get_var(name):
            if name not in vars:
                vars[name] = z3.Real(name)
            return vars[name]
        for coeffs, op, c in assertions:
            expr = 0
            for v, a in coeffs.items():
                expr = expr + a * get_var(v)
            if op == "<=":
                s.add(expr <= c)
            elif op == ">=":
                s.add(expr >= c)
            else:
                s.add(expr == c)
        if s.check() != z3.sat:
            return None
        m = s.model()
        out = {}
        for name, zvar in vars.items():
            val = m[zvar]
            if val is None:
                continue
            try:
                out[name] = float(val.as_decimal(20).replace("?", "0"))
            except Exception:
                sval = str(val)
                try:
                    if "/" in sval:
                        num, den = sval.split("/", 1)
                        out[name] = float(num) / float(den)
                    else:
                        out[name] = float(sval)
                except Exception:
                    continue
        return out
    
    @staticmethod
    def solve_nonlinear_z3(linear_assertions: List[tuple], bilinear_pairs: List[Tuple[str,str,str]] = None, square_pairs: List[Tuple[str,str]] = None) -> Optional[Dict[str, float]]:
        try:
            import z3
        except Exception:
            return None
        bilinear_pairs = bilinear_pairs or []
        square_pairs = square_pairs or []
        s = z3.Solver()
        vars = {}
        def get_var(name):
            if name not in vars:
                vars[name] = z3.Real(name)
            return vars[name]
        for coeffs, op, c in linear_assertions:
            expr = 0
            for v, a in coeffs.items():
                expr = expr + a * get_var(v)
            if op == "<=":
                s.add(expr <= c)
            elif op == ">=":
                s.add(expr >= c)
            else:
                s.add(expr == c)
        for w, x, y in bilinear_pairs:
            s.add(get_var(w) == get_var(x) * get_var(y))
        for w, x in square_pairs:
            s.add(get_var(w) == get_var(x) * get_var(x))
        if s.check() != z3.sat:
            return None
        m = s.model()
        out = {}
        for name, zvar in vars.items():
            val = m[zvar]
            if val is None:
                continue
            try:
                out[name] = float(val.as_decimal(20).replace("?", "0"))
            except Exception:
                sval = str(val)
                try:
                    if "/" in sval:
                        num, den = sval.split("/", 1)
                        out[name] = float(num) / float(den)
                    else:
                        out[name] = float(sval)
                except Exception:
                    continue
        return out
    
    @staticmethod
    def solve_mixed_integer_z3(linear_assertions: List[tuple], int_vars: List[str] = None) -> Optional[Dict[str, float]]:
        try:
            import z3
        except Exception:
            return None
        s = z3.Solver()
        int_vars = int_vars or []
        vars = {}
        def get_var(name):
            if name not in vars:
                if name in int_vars or name.startswith("n_"):
                    vars[name] = z3.Int(name)
                else:
                    vars[name] = z3.Real(name)
            return vars[name]
        for coeffs, op, c in linear_assertions:
            expr = 0
            for v, a in coeffs.items():
                zv = get_var(v)
                if isinstance(zv, z3.z3.IntNumRef) or isinstance(zv, z3.IntNumRef):
                    expr = expr + a * z3.ToReal(zv)
                else:
                    expr = expr + a * zv
            if op == "<=":
                s.add(expr <= c)
            elif op == ">=":
                s.add(expr >= c)
            else:
                s.add(expr == c)
        if s.check() != z3.sat:
            return None
        m = s.model()
        out = {}
        for name, zvar in vars.items():
            val = m[zvar]
            if val is None:
                continue
            try:
                out[name] = float(val.as_decimal(20).replace("?", "0"))
            except Exception:
                sval = str(val)
                try:
                    if "/" in sval:
                        num, den = sval.split("/", 1)
                        out[name] = float(num) / float(den)
                    else:
                        out[name] = float(sval)
                except Exception:
                    try:
                        out[name] = float(val.as_long())
                    except Exception:
                        continue
        return out
