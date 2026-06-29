
import numpy as np


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
