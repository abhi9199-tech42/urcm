import ast
from typing import Callable, Dict, List, Tuple

import numpy as np


def _safe_exec(code: str, ns: dict) -> bool:
    """Execute code only if it contains no dangerous constructs."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Delete,
                            ast.Global, ast.Nonlocal, ast.Raise,
                            ast.Assert, ast.Yield, ast.YieldFrom,
                            ast.Await, ast.AsyncFor, ast.AsyncWith)):
            return False
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in (
                'exec', 'eval', 'compile', '__import__', 'open',
                'getattr', 'setattr', 'delattr', 'globals', 'locals',
                'vars', 'dir', 'breakpoint', 'exit', 'quit'
            ):
                return False
    exec(code, ns)
    return True

_tc_cache: Dict[Tuple[Tuple[str, str], ...], List[Tuple[str, str]]] = {}
_tc_order: List[Tuple[Tuple[str, str], ...]] = []
_tc_cap = 256

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

class DeterministicUtils:
    @staticmethod
    def set_seed(seed: int = 0):
        import random
        random.seed(seed)
        np.random.seed(seed)

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
            if not _safe_exec(code, ns):
                return False
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
            if not _safe_exec(code, ns):
                return {"ok": False, "reason": "unsafe code"}
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
