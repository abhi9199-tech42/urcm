import numpy as np
from typing import Tuple, List, Optional
from .resonance_encoder import ResonancePathEncoder

class GeometricMemory:
    """
    Implements Bounded Memory Deposition (One-Shot Learning) for URCM.
    
    Instead of iterative Gradient Descent (Backprop), this module directly
    'deposits' attractor basins into the resonance weight matrix (W_res).
    
    Theory:
    W_new = W_old - (W_old * u - v) * u.T / |u|^2
    Where u = input state, v = target next state.
    This is a rank-1 update that enforces W * u = v immediately.
    """
    
    def __init__(self, resonance_dim: int, capacity_factor: float = 0.5):
        self.resonance_dim = resonance_dim
        # Capacity limit based on matrix rank and spectral properties
        # Ideally N * 0.14 for Hopfield, but higher for Resonance due to non-linearity
        self.capacity_limit = int(resonance_dim * capacity_factor)
        self.deposited_count = 0
        
    def deposit_attractor(self, 
                         W_res: np.ndarray, 
                         state_vector: np.ndarray, 
                         next_state_vector: np.ndarray = None) -> np.ndarray:
        """
        Deposits a single attractor into the weight matrix in ONE SHOT.
        
        If next_state_vector is None, it creates a fixed point (state -> state).
        If provided, it creates a transition (state -> next_state).
        
        Args:
            W_res: Current resonance matrix (N, N)
            state_vector: The state to stabilize (N,)
            next_state_vector: The target state (N,)
            
        Returns:
            W_updated: The updated weight matrix.
        """
        if next_state_vector is None:
            next_state_vector = state_vector
            
        # Ensure vectors are normalized for numerical stability of the update direction
        # But we need the ACTUAL state magnitude for the projection.
        u = state_vector
        norm_u_sq = np.dot(u, u)
        
        if norm_u_sq < 1e-6:
            return W_res # Too small to deposit
            
        # 1. Calculate the required linear projection
        # We need: tanh(u @ W) = v
        # So: u @ W = arctanh(v)
        
        # Clip target to safe range to prevent arctanh explosion
        # Keeping it linear-ish (< 0.9) helps stability
        safe_target = np.clip(next_state_vector, -0.95, 0.95)
        linear_target = np.arctanh(safe_target)
        
        # 2. Calculate current linear projection
        current_projection = np.dot(u, W_res)
        
        # 3. Calculate Error in LINEAR space
        error = linear_target - current_projection
        
        # 4. Rank-1 Update
        # W_new = W_old + outer(u, error) * (1 / |u|^2)
        # This forces u @ W_new = u @ W_old + u @ u.T * error / |u|^2 = linear_target
        
        update = np.outer(u, error) / norm_u_sq
        W_updated = W_res + update
        
        self.deposited_count += 1
        return W_updated

    def deposit_sequence(self, 
                        W_res: np.ndarray, 
                        trajectory: List[np.ndarray],
                        broaden: bool = True) -> np.ndarray:
        """
        Deposits a sequence of states as a flow channel.
        s1 -> s2 -> s3 -> ... -> s_final -> s_final
        
        Args:
            broaden: If True, injects noise around the path to create 
                     a "Funnel" (Attractor Basin), improving stability.
        """
        W_curr = W_res.copy()
        
        # Basin Broadening Parameters
        noise_samples = 4
        noise_scale = 0.10
        
        for i in range(len(trajectory) - 1):
            curr = trajectory[i]
            nxt = trajectory[i+1]
            
            # 1. Deposit Core Path
            W_curr = self.deposit_attractor(W_curr, curr, nxt)
            
            # 2. Deposit Funnel (Basin)
            if broaden:
                for _ in range(noise_samples):
                    # Create noisy version of 'curr' that also maps to 'nxt'
                    noise = np.random.normal(0, noise_scale, curr.shape)
                    noisy_curr = curr + noise
                    # Normalize to keep on hypersphere
                    noisy_curr = noisy_curr / np.linalg.norm(noisy_curr) * np.linalg.norm(curr)
                    
                    W_curr = self.deposit_attractor(W_curr, noisy_curr, nxt)
            
        # Make the last state a fixed point attractor
        last = trajectory[-1]
        W_curr = self.deposit_attractor(W_curr, last, last)
        
        # Broaden the fixed point too
        if broaden:
             for _ in range(noise_samples):
                noise = np.random.normal(0, noise_scale, last.shape)
                noisy_last = last + noise
                noisy_last = noisy_last / np.linalg.norm(noisy_last) * np.linalg.norm(last)
                W_curr = self.deposit_attractor(W_curr, noisy_last, last)
        
        return W_curr

    def check_capacity(self) -> float:
        """Returns percentage of capacity used."""
        return self.deposited_count / self.capacity_limit
