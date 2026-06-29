"""
Data-driven heuristic engine for URCM.

Loads contextual heuristics from JSON and applies them as
constraint vectors during the cognitive cycle.
"""

import json
import logging
import os
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)

_HEURISTICS_PATH = os.path.join(os.path.dirname(__file__), "heuristics.json")


def _load_heuristics(path: str = _HEURISTICS_PATH) -> Dict[str, Any]:
    """Load heuristics from JSON file with fallback to empty defaults."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load heuristics from {path}: {e}")
        return {
            "antonyms": {}, "sequences": {}, "synonyms": {},
            "positive_anchors": [], "negative_anchors": [],
            "context_heuristics": [], "word_rules": {},
        }


class HeuristicEngine:
    """
    Applies data-driven heuristics to constraint vectors.

    Heuristics are loaded from a JSON file at init time.
    Each heuristic specifies trigger conditions and constraint outputs
    (attract/repulse word vectors with weights).
    """

    def __init__(self, path: str = _HEURISTICS_PATH):
        data = _load_heuristics(path)
        self.antonyms: Dict[str, str] = data.get("antonyms", {})
        self.sequences: Dict[str, str] = data.get("sequences", {})
        self.synonyms: Dict[str, str] = data.get("synonyms", {})
        self.positive_anchors: List[str] = data.get("positive_anchors", [])
        self.negative_anchors: List[str] = data.get("negative_anchors", [])
        self.context_heuristics: List[Dict] = data.get("context_heuristics", [])
        self.word_rules: Dict[str, Dict] = data.get("word_rules", {})

    def apply_antonym_constraints(
        self, current_word: str, concept_map: Dict, constraint_matrix: Any
    ) -> None:
        """Apply repulsion constraint for antonym pairs."""
        if current_word in self.antonyms:
            antonym = self.antonyms[current_word]
            if antonym in concept_map:
                idx_a = concept_map[current_word]
                idx_b = concept_map[antonym]
                constraint_matrix[idx_a, idx_b] = 6.0

    def apply_sequence_constraints(
        self, current_word: str, concept_map: Dict, constraint_matrix: Any
    ) -> None:
        """Apply attraction constraint for sequential word pairs."""
        if current_word in self.sequences:
            next_word = self.sequences[current_word]
            if next_word in concept_map:
                idx_a = concept_map[current_word]
                idx_b = concept_map[next_word]
                constraint_matrix[idx_a, idx_b] = -4.0

    def apply_sentiment_constraints(
        self, current_word: str, concept_map: Dict, constraint_matrix: Any
    ) -> None:
        """Apply attraction constraints for sentiment anchors."""
        if current_word in self.positive_anchors or current_word == "good":
            for anchor in self.positive_anchors:
                if anchor in concept_map:
                    idx_a = concept_map[current_word]
                    idx_b = concept_map[anchor]
                    constraint_matrix[idx_a, idx_b] = -3.0

        if current_word in self.negative_anchors or current_word == "bad":
            for anchor in self.negative_anchors:
                if anchor in concept_map:
                    idx_a = concept_map[current_word]
                    idx_b = concept_map[anchor]
                    constraint_matrix[idx_a, idx_b] = -3.0

    def apply_synonym_constraints(
        self, current_word: str, concept_map: Dict, constraint_matrix: Any
    ) -> None:
        """Apply attraction constraints for synonym pairs."""
        if current_word in self.synonyms:
            synonym = self.synonyms[current_word]
            if synonym in concept_map:
                idx_a = concept_map[current_word]
                idx_b = concept_map[synonym]
                constraint_matrix[idx_a, idx_b] = -12.0

    def apply_word_rules(
        self, current_word: str, concept_map: Dict, constraint_matrix: Any
    ) -> None:
        """Apply word-specific attract/repulse rules."""
        if current_word in self.word_rules:
            rule = self.word_rules[current_word]
            for item in rule.get("attract", []):
                word, weight = item["word"], item["weight"]
                if word in concept_map:
                    idx_a = concept_map[current_word]
                    idx_b = concept_map[word]
                    constraint_matrix[idx_a, idx_b] = weight
            for item in rule.get("repulse", []):
                word, weight = item["word"], item["weight"]
                if word in concept_map:
                    idx_a = concept_map[current_word]
                    idx_b = concept_map[word]
                    constraint_matrix[idx_a, idx_b] = weight

    def apply_context_heuristics(
        self,
        current_word: str,
        ctx: Set[str],
        concept_map: Dict,
        constraint_matrix: Any,
        max_tier: int = 2,
    ) -> None:
        """
        Apply all context-triggered heuristics.

        Each heuristic specifies:
        - trigger: conditions (current word, context words, negations)
        - attract: list of {word, weight} for attraction constraints
        - repulse: list of {word, weight} for repulsion constraints
        """
        for h in self.context_heuristics:
            if h.get("tier", 2) > max_tier:
                continue

            trigger = h.get("trigger", {})

            # Check current word constraint
            if "current" in trigger and current_word != trigger["current"]:
                continue

            # Check required context words
            if "words" in trigger:
                if not all(w in ctx for w in trigger["words"]):
                    continue

            # Check negation words (must NOT be in context)
            if "negations" in trigger:
                if any(n in ctx for n in trigger["negations"]):
                    continue

            # Check any_in constraint (at least one must be present)
            if "any_in" in trigger:
                if not any(w in ctx for w in trigger["any_in"]):
                    continue

            # Check current_not_in_ctx constraint
            if trigger.get("current_not_in_ctx") and current_word in ctx:
                continue

            # Check current_in constraint
            if "current_in" in trigger and current_word not in trigger["current_in"]:
                continue

            # Apply attract constraints
            for item in h.get("attract", []):
                word, weight = item["word"], item["weight"]
                if word in concept_map:
                    idx_a = concept_map[current_word]
                    idx_b = concept_map[word]
                    constraint_matrix[idx_a, idx_b] = weight

            # Apply repulse constraints
            for item in h.get("repulse", []):
                word, weight = item["word"], item["weight"]
                if word in concept_map:
                    idx_a = concept_map[current_word]
                    idx_b = concept_map[word]
                    constraint_matrix[idx_a, idx_b] = weight

    def apply_all(
        self,
        current_word: str,
        ctx: Set[str],
        concept_map: Dict,
        constraint_matrix: Any,
        max_tier: int = 2,
    ) -> None:
        """Apply all heuristic types in sequence."""
        self.apply_antonym_constraints(current_word, concept_map, constraint_matrix)
        self.apply_sequence_constraints(current_word, concept_map, constraint_matrix)
        self.apply_sentiment_constraints(current_word, concept_map, constraint_matrix)
        self.apply_synonym_constraints(current_word, concept_map, constraint_matrix)
        self.apply_word_rules(current_word, concept_map, constraint_matrix)
        self.apply_context_heuristics(current_word, ctx, concept_map, constraint_matrix, max_tier)
