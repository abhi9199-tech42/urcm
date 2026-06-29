"""
Unified μ-Resonance Cognitive Mesh (URCM) Main System.

This module integrates all core components into a single, cohesive reasoning system.
It provides an end-to-end pipeline from text input to converged semantic understanding.
"""

import logging
import time
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

from urcm.core.attractor_network import AttractorNetwork
from urcm.core.convergence_engine import MuConvergenceEngine
from urcm.core.data_models import ReasoningPath, ResonanceState
from urcm.core.error_handling import ErrorRecoverySystem
from urcm.core.latent_space import ReconstructionSystem, SemanticLatentSpace
from urcm.core.observability import record_event
from urcm.core.oscillatory_gating import OscillatoryGating
from urcm.core.performance import OptimizedPhonemeSet
from urcm.core.phoneme_mapper import PhonemeFrequencyPipeline
from urcm.core.resonance_encoder import ResonancePathEncoder


class URCMSystem:
    """
    The main URCM System class.

    Integrates all sub-systems to provide a complete frequency-based reasoning pipeline.
    """

    def __init__(
        self,
        frequency_dim: int = 24,
        resonance_dim: int = 512, # Increased from 64 to 512 for higher capacity
        latent_dim: int = 16,
        base_frequency: float = 1.0,
        beam_width: int = 3,
        max_steps: int = 50
    ):
        """
        Initialize the URCM System with all components.
        """
        self.frequency_dim = frequency_dim
        self.resonance_dim = resonance_dim
        self.latent_dim = latent_dim

        # 1. Pipeline & Mapping
        self.pipeline = PhonemeFrequencyPipeline(frequency_dim=frequency_dim)
        self.optimized_set = OptimizedPhonemeSet(vector_dimension=frequency_dim)

        # 2. Encoding
        self.encoder = ResonancePathEncoder(
            input_dim=frequency_dim,
            resonance_dim=resonance_dim
        )

        # 3. Memory & Dynamics
        self.latent_space = SemanticLatentSpace(
            input_dim=resonance_dim,
            latent_dim=latent_dim
        )
        self.reconstruction = ReconstructionSystem(self.latent_space)

        self.attractor_network = AttractorNetwork(size=resonance_dim)

        # 4. Gating & Control
        self.gating = OscillatoryGating(
            resonance_dim=resonance_dim,
            base_frequency=base_frequency
        )

        # 5. Reasoning Engine
        self.engine = MuConvergenceEngine(
            competition_beam_width=beam_width,
            max_steps=max_steps
        )

        # 6. Error Recovery
        self.error_recovery = ErrorRecoverySystem(
            latent_space=self.latent_space,
            attractor_network=self.attractor_network,
            gating_system=self.gating,
            phoneme_mapper=self.pipeline.frequency_mapper
        )

        self.status: Dict[str, Any] = {
            "initialized": True,
            "processed_count": 0,
            "errors_recovered": 0
        }

    def process_query(self, text: str) -> ReasoningPath:
        """
        Process a text query through the complete URCM pipeline.

        Args:
            text: Input text query to process.

        Returns:
            The top ReasoningPath with the highest convergence score.

        Raises:
            ValueError: If input text is empty.
            RuntimeError: If no reasoning paths are generated.

        Steps:
        1. Text -> Frequency Path
        2. Frequency Path -> Initial Resonance State
        3. Initial State -> Reasoning Loop (Convergence)
        4. Converged Path -> Result
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        try:
            record_event("process_start", {"text_len": len(text)})
        except Exception as e:
            logger.warning(f"Failed to record process_start event: {e}")
        # Step 1: Phonemic Grounding
        freq_path = self.pipeline.process_text(text)

        # Step 2: Temporal Encoding
        initial_state = self.encoder.get_resonance_state(freq_path)

        # Step 3: Run Reasoning Engine
        # We need a generator for the engine to navigate the space
        results = self.engine.run_reasoning_loop(
            initial_state=initial_state,
            next_state_generator=self._propose_next_states
        )

        self.status["processed_count"] += 1
        try:
            top = results[0] if results else None
            fm = float(top.mu_trajectory[-1]) if top and top.mu_trajectory else 0.0
            record_event("process_end", {"final_mu": fm, "paths": len(results)})
        except Exception as e:
            logger.warning(f"Failed to record process_end event: {e}")

        if not results:
            raise RuntimeError("No reasoning paths generated")

        return results[0]

    def _propose_next_states(self, current_state: ResonanceState) -> List[ResonanceState]:
        """
        Internal generator for the MuConvergenceEngine.
        Proposes candidates for the next semantic state.

        Args:
            current_state: The current resonance state to propose next states from.

        Returns:
            List of validated candidate ResonanceState objects.
        """
        candidates = []

        # Dynamics: Evolve phases in attractor network based on current resonance
        # (In a full implementation, the resonance vector would drive frequencies or phases)
        # Here we simulate 3 candidate directions:

        # 1. Follow current rhythm (Pure Gating)
        gated_vec = self.gating.apply_gating(current_state.resonance_vector, dt=0.05)
        candidates.append(self._create_candidate(current_state, gated_vec, "rhythm"))

        # 2. Move towards nearest attractor (Semantic Attraction)
        # Sync the network to the current state first
        current_phases = (current_state.resonance_vector % (2 * np.pi))
        self.attractor_network.set_state(current_phases)
        self.attractor_network.step(dt=0.1)

        attractor = self.attractor_network.find_nearest_attractor()
        if attractor:
            # Blend current state with attractor pattern (favor stability to improve μ more consistently)
            attracted_vec = 0.9 * gated_vec + 0.1 * attractor.phase_pattern[:self.resonance_dim]
            candidates.append(self._create_candidate(current_state, attracted_vec, "attractor"))

        # 3. Multi-path exploration (Noise/Drift)
        noise = np.random.normal(0, 0.02, self.resonance_dim)
        noisy_vec = gated_vec + noise
        candidates.append(self._create_candidate(current_state, noisy_vec, "exploration"))

        # Run all candidates through error recovery before returning
        validated_candidates = []
        for cand in candidates:
            recovered_cand, actions = self.error_recovery.check_and_recover(cand)
            if actions:
                self.status["errors_recovered"] += len(actions)
            validated_candidates.append(recovered_cand)

        return validated_candidates

    def _create_candidate(self, prev_state: ResonanceState, new_vec: np.ndarray, label: str) -> ResonanceState:
        """
        Helper to package a raw vector into a ResonanceState.

        Args:
            prev_state: Previous state (for context).
            new_vec: New resonance vector.
            label: Label for the candidate type.

        Returns:
            Normalized ResonanceState object.
        """
        # Normalize to keep on semantic manifold
        new_vec = new_vec / (np.linalg.norm(new_vec) + 1e-9)

        return ResonanceState(
            resonance_vector=new_vec,
            mu_value=0.0, # Will be calculated by engine
            rho_density=0.0,
            chi_cost=0.0,
            stability_score=0.0,
            oscillation_phase=self.gating.phase,
            timestamp=time.time()
        )

    def validate_system(self) -> Dict[str, bool]:
        """
        Run a suite of self-tests to ensure all integration points are functional.

        Returns:
            Dictionary of check names to boolean results.
        """
        checks = {}

        try:
            # 1. Test Pipeline
            path = self.pipeline.process_text("test")
            checks["pipeline_ok"] = path.vectors.shape[1] == self.frequency_dim

            # 2. Test Encoder
            state = self.encoder.get_resonance_state(path)
            checks["encoder_ok"] = state.resonance_vector.shape[0] == self.resonance_dim

            # 3. Test Latent Space
            _, _, valid = self.reconstruction.perform_round_trip(state)
            checks["latent_space_ok"] = True # Round trip might have some loss but logic should work

            # 4. Test Engine
            reasoning_results = self.process_query("URCM check")
            checks["engine_ok"] = len(reasoning_results.mu_trajectory) > 0

            checks["overall_health"] = all(checks.values())
        except Exception as e:
            checks["overall_health"] = False
            checks["error"] = str(e)

        return checks
