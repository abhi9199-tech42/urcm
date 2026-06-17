
import numpy as np
from typing import List, Optional, Callable, Dict, Tuple
from urcm.core.data_models import ResonanceState, ReasoningPath
from urcm.core.theory import URCMTheory
from urcm.core.observability import record_event

class MuConvergenceEngine:
    """
    Core reasoning engine that drives semantic convergence based on μ-stability.
    
    This engine manages multiple competing reasoning paths, selecting those with
    the highest Resonance (μ) and pruning others. It rigorously dictates when
    reasoning terminates based on the stability of Δμ.
    """
    
    def __init__(
        self,
        rho_threshold: float = 0.5,
        convergence_epsilon: float = 1e-3,
        max_steps: int = 50,
        competition_beam_width: int = 3,
        oscillation_window: int = 4,
        oscillation_std_threshold: float = 0.02
    ):
        """
        Initialize the convergence engine.
        
        Args:
            rho_threshold: Minimum semantic density required to be considered a valid path.
            convergence_epsilon: Threshold for Δμ below which the system is considered converged.
            max_steps: Maximum number of reasoning steps (infinite loop prevention).
            competition_beam_width: Number of parallel paths to maintain (Beam Search width).
        """
        self.rho_threshold = rho_threshold
        self.convergence_epsilon = convergence_epsilon
        self.max_steps = max_steps
        self.beam_width = competition_beam_width
        self.osc_window = max(oscillation_window, 3)
        self.osc_std_thresh = max(oscillation_std_threshold, 0.0)
        
    def calculate_state_metrics(self, state: ResonanceState) -> ResonanceState:
        """
        Ensures a state has valid μ, ρ, and χ metrics computed.
        Re-calculates if necessary using URCMTheory.
        """
        # If metrics are placeholders (e.g. from a raw generator), calculate them
        # Note: We assume resonance_vector is populated.
        
        if state.rho_density == 0.0 and state.chi_cost == 0.0:
            rho = URCMTheory.calculate_rho(state.resonance_vector)
            # Define base transformation cost relative to neutral origin to avoid zero chi
            chi = float(np.linalg.norm(state.resonance_vector))
            mu = URCMTheory.compute_mu(rho, chi)
            
            # stability = mu
            stability = mu 
            
            # Return new state with computed metrics
            return ResonanceState(
                resonance_vector=state.resonance_vector,
                mu_value=mu,
                rho_density=rho,
                chi_cost=chi,
                stability_score=stability,
                oscillation_phase=state.oscillation_phase,
                timestamp=state.timestamp
            )
        return state
    
    def _maybe_emit_oscillation(self, mu_traj: List[float], step: int):
        """
        Detects oscillation in recent μ deltas and emits a telemetry event.
        Heuristic: sign-alternating Δμ in the last window and non-trivial std.
        """
        try:
            if len(mu_traj) < self.osc_window + 1:
                return
            window = mu_traj[-(self.osc_window+1):]
            deltas = [window[i+1] - window[i] for i in range(len(window)-1)]
            # Require alternating signs across the window
            signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in deltas]
            # Ignore zeros for alternation check
            filtered = [s for s in signs if s != 0]
            if len(filtered) < self.osc_window:
                return
            alternates = all(filtered[i] != filtered[i-1] for i in range(1, len(filtered)))
            std = float(np.std(window))
            if alternates and std >= self.osc_std_thresh:
                record_event("mu_oscillation_detected", {
                    "step": int(step),
                    "std": std,
                    "window": [float(x) for x in window]
                })
        except Exception:
            pass

    def evaluate_paths(self, active_paths: List[ReasoningPath]) -> List[ReasoningPath]:
        """
        Sort and prune paths based on their current tip's μ value.
        """
        # Sort by current μ (descending)
        # We look at the last value in mu_trajectory
        sorted_paths = sorted(
            active_paths, 
            key=lambda p: p.mu_trajectory[-1] if p.mu_trajectory else 0.0, 
            reverse=True
        )
        
        # Keep top N (Beam Width)
        return sorted_paths[:self.beam_width]

    def check_convergence(self, path: ReasoningPath) -> bool:
        """
        Determines if a specific path has converged based on Δμ < ε.
        """
        if len(path.mu_trajectory) < 2:
            return False
            
        # Calculate recent delta
        current_mu = path.mu_trajectory[-1]
        prev_mu = path.mu_trajectory[-2]
        delta_mu = abs(current_mu - prev_mu)
        
        # Convergence condition: Change is minimal AND state is stable (positive mu)
        if delta_mu < self.convergence_epsilon and current_mu > 0:
            return True
            
        return False

    def run_reasoning_loop(
        self, 
        initial_state: ResonanceState,
        next_state_generator: Callable[[ResonanceState], List[ResonanceState]]
    ) -> List[ReasoningPath]:
        """
        Executes the main resonance loop.
        
        Args:
            initial_state: The starting resonance state (e.g. from encoded query).
            next_state_generator: Function that proposes candidate next states.
            
        Returns:
            List of converged ReasoningPath objects (best ones first).
        """
        # Bootstrap initial path
        initial_state = self.calculate_state_metrics(initial_state)
        try:
            record_event("mu_loop_start", {"mu": float(initial_state.mu_value), "rho": float(initial_state.rho_density), "chi": float(initial_state.chi_cost)})
        except Exception:
            pass
        
        root_path = ReasoningPath(
            initial_state=initial_state,
            intermediate_states=[],
            final_state=initial_state, 
            mu_trajectory=[initial_state.mu_value, initial_state.mu_value], # Duplicate for initial T=0 stutter
            convergence_achieved=False,
            termination_reason="Running"
        )
        
        active_paths = [root_path]
        completed_paths = []
        
        step_count = 0
        
        while active_paths and step_count < self.max_steps:
            step_count += 1
            new_candidates = []
            
            for path in active_paths:
                current_tip = path.final_state
                
                # Check if already converged (shouldn't happen if we manage lists right, but safety check)
                if path.convergence_achieved:
                    completed_paths.append(path)
                    continue
                
                # Generate potential next moves
                proposals = next_state_generator(current_tip)
                
                if not proposals:
                    # Dead end
                    path.termination_reason = "Dead End (No further states)"
                    completed_paths.append(path)
                    continue
                    
                for proposal in proposals:
                    # Calculate proper metrics relative to history
                    # Chi is cost of transition from current_tip to proposal
                    chi = URCMTheory.calculate_chi(proposal.resonance_vector, current_tip.resonance_vector)
                    rho = URCMTheory.calculate_rho(proposal.resonance_vector)
                    mu = URCMTheory.compute_mu(rho, chi)
                    
                    # Create resolved state
                    next_state = ResonanceState(
                        resonance_vector=proposal.resonance_vector,
                        mu_value=mu,
                        rho_density=rho,
                        chi_cost=chi,
                        stability_score=mu, # Simplified
                        oscillation_phase=proposal.oscillation_phase,
                        timestamp=proposal.timestamp
                    )
                    
                    # Fork the path
                    new_trajectory = path.mu_trajectory + [mu]
                    # We archive the OLD tip into intermediates
                    new_intermediates = path.intermediate_states + [current_tip]
                    
                    new_path = ReasoningPath(
                        initial_state=path.initial_state,
                        intermediate_states=new_intermediates,
                        final_state=next_state,
                        mu_trajectory=new_trajectory,
                        convergence_achieved=False,
                        termination_reason="Running"
                    )
                    
                    try:
                        if len(new_trajectory) >= 2:
                            dm = float(abs(new_trajectory[-1] - new_trajectory[-2]))
                        else:
                            dm = 0.0
                        record_event("mu_step", {"step": step_count, "mu": float(mu), "delta_mu": dm})
                    except Exception:
                        pass
                    
                    # Check convergence immediately for this new step
                    if self.check_convergence(new_path):
                        new_path.convergence_achieved = True
                        new_path.termination_reason = "Convergence (Δμ < ε)"
                        completed_paths.append(new_path)
                    else:
                        # Oscillation detector on the extended μ trajectory
                        self._maybe_emit_oscillation(new_path.mu_trajectory, step_count)
                        new_candidates.append(new_path)
            
            # Competition: Select best active candidates to continue
            active_paths = self.evaluate_paths(new_candidates)
            
        # Handle max steps
        for path in active_paths:
            path.termination_reason = "Max Steps Reached"
            completed_paths.append(path)
            
        # Return all completed paths, sorted by final mu stability
        results = sorted(
            completed_paths, 
            key=lambda p: p.mu_trajectory[-1] if p.mu_trajectory else 0.0, 
            reverse=True
        )
        try:
            for p in results:
                m = float(p.mu_trajectory[-1]) if p.mu_trajectory else 0.0
                record_event("mu_path_complete", {"final_mu": m, "converged": bool(p.convergence_achieved), "reason": p.termination_reason})
        except Exception:
            pass
        return results
