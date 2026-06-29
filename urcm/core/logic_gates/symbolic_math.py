import re
from typing import List, Optional, Tuple

import numpy as np


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
