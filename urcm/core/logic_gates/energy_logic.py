import re
from typing import Dict, List, Optional

import numpy as np


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
            norm = np.linalg.norm(target)
            if norm < 1e-9:
                target = vec_a  # Fallback to first vector if they cancel
            else:
                target = target / norm  # Normalize
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
