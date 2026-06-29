"""
Unified μ-Resonance Cognitive Mesh (URCM) - A frequency-based reasoning system.

This package implements a frequency-based artificial reasoning system that replaces
discrete token-based processing with continuous frequency-based representations.
"""

__version__ = "0.1.0"
__author__ = "URCM Development Team"

from . import api, core
from .core.data_models import AttractorState, FrequencyPath, MeshSignal, PhonemeSequence, ReasoningPath, ResonanceState

__all__ = [
    "PhonemeSequence",
    "FrequencyPath",
    "ResonanceState",
    "AttractorState",
    "ReasoningPath",
    "MeshSignal"
]
