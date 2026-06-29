"""
Truth Maintenance System (TMS) for URCM.

Tracks facts and their supports; retracts invalidated facts.
Used by ExecutiveController for logical reasoning consistency.
"""

import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)


class TruthMaintenanceSystem:
    """
    Simple TMS: track facts and their supports; retract invalidated facts.
    """
    def __init__(self):
        self.facts: Set[tuple] = set()
        self.supports: Dict[tuple, set] = {}  # fact -> set of support keys

    def assert_fact(self, fact: tuple, support_key: str = "axiom") -> None:
        """Add a fact with a support justification."""
        self.facts.add(fact)
        s = self.supports.setdefault(fact, set())
        s.add(support_key)

    def retract_support(self, support_key: str) -> None:
        """Remove all facts justified by a given support key."""
        to_remove = []
        for fact, sup in self.supports.items():
            if support_key in sup:
                sup.remove(support_key)
                if not sup:
                    to_remove.append(fact)
        for f in to_remove:
            self.facts.discard(f)
            self.supports.pop(f, None)

    def has(self, fact: tuple) -> bool:
        """Check if a fact is currently believed."""
        return fact in self.facts
