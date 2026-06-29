import re
from typing import Dict, List, Optional, Tuple

import numpy as np


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
                pass
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
