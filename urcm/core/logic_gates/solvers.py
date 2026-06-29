from typing import Dict, List, Optional, Tuple


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
    def solve_maxsat(soft_clauses: List[List[int]], hard_clauses: Optional[List[List[int]]] = None) -> Optional[Dict[int, bool]]:
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
    def solve_nonlinear_z3(linear_assertions: List[tuple], bilinear_pairs: Optional[List[Tuple[str,str,str]]] = None, square_pairs: Optional[List[Tuple[str,str]]] = None) -> Optional[Dict[str, float]]:
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
    def solve_mixed_integer_z3(linear_assertions: List[tuple], int_vars: Optional[List[str]] = None) -> Optional[Dict[str, float]]:
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
