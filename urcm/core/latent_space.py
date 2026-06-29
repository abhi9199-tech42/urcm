
from typing import Tuple

import numpy as np

from urcm.core.data_models import ResonanceState


class SemanticLatentSpace:
    """
    Manages the projection of resonance states into a compressed latent space
    and their reconstruction, ensuring minimal semantic drift.
    """

    def __init__(self, input_dim: int = 64, latent_dim: int = 16, mu_threshold_drift: float = 0.8):
        """
        Initialize the Latent Space manager.

        Args:
            input_dim: Dimension of the Resonance Vector space.
            latent_dim: Dimension of the compressed Latent space.
            mu_threshold_drift: Minimum retained μ-ratio to consider projection valid (drift check).
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.mu_threshold = mu_threshold_drift

        # Initialize projection matrices (Random Orthogonal for now, effectively PCA-like placeholder)
        # E: Encoder (Input -> Latent)
        # D: Decoder (Latent -> Input)
        rng = np.random.RandomState(42)

        # Random matrix
        M = rng.randn(input_dim, latent_dim)
        # QR decomposition to get orthogonal basis for more stable projection
        Q, _ = np.linalg.qr(M)

        self.E = Q.T # (Latent, Input)
        self.D = Q   # (Input, Latent)

    def project(self, state: ResonanceState) -> np.ndarray:
        """
        Project a resonance state into the latent space.
        z = E * r
        """
        if state.resonance_vector.shape[0] != self.input_dim:
            raise ValueError(f"State dimension {state.resonance_vector.shape[0]} mismatch expected {self.input_dim}")

        latent_vector = np.dot(self.E, state.resonance_vector)
        return latent_vector

    def reconstruct(self, latent_vector: np.ndarray) -> np.ndarray:
        """
        Reconstruct a resonance vector from the latent space.
        r_hat = D * z
        """
        if latent_vector.shape[0] != self.latent_dim:
             raise ValueError(f"Latent dimension {latent_vector.shape[0]} mismatch expected {self.latent_dim}")

        reconstructed_vector = np.dot(self.D, latent_vector)
        return reconstructed_vector

    def validate_reconstruction(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray
    ) -> Tuple[float, bool]:
        """
        Validates the reconstruction quality.
        Returns (Loss, IsValid).
        Loss is L1 norm ||Original - Reconstructed||_1
        """
        # L1 Loss (Manhattan distance)
        loss = float(np.sum(np.abs(original - reconstructed)))

        # Validation logic:
        # A simple threshold on loss relative to signal magnitude
        signal_energy = np.sum(np.abs(original)) + 1e-9
        relative_loss = loss / signal_energy

        # If we lost > (1 - threshold) of information, it's drifting
        is_valid = bool((1.0 - relative_loss) >= self.mu_threshold)

        return loss, is_valid

class ReconstructionSystem:
    """
    High-level system for validating Full Round-Trip fidelity.
    Resonance -> Latent -> Resonance
    """

    def __init__(self, latent_space: SemanticLatentSpace):
        self.space = latent_space

    def perform_round_trip(self, state: ResonanceState) -> Tuple[np.ndarray, float, bool]:
        """
        Executes a full round trip projection and reconstruction.

        Returns:
            - Reconstructed Vector (ndarray)
            - Reconstruction Loss (float)
            - Validation Status (bool)
        """
        # 1. Project
        z = self.space.project(state)

        # 2. Reconstruct
        r_hat = self.space.reconstruct(z)

        # 3. Validate
        loss, valid = self.space.validate_reconstruction(state.resonance_vector, r_hat)

        return r_hat, loss, valid
