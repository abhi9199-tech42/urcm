"""Integration test fixtures and mocks for URCM."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def mock_frequency_dim():
    """Small frequency dimension for fast tests."""
    return 16


@pytest.fixture
def mock_resonance_dim():
    """Small resonance dimension for fast tests."""
    return 16


@pytest.fixture
def mock_latent_dim():
    """Small latent dimension for fast tests."""
    return 4


@pytest.fixture
def mock_phoneme_vectors(mock_frequency_dim):
    """Generate mock phoneme vectors for testing."""
    np.random.seed(42)
    phonemes = "aāiīuūeoṃḥkgñcjhṭḍhtdṇptnbmyrlvśṣsh"
    vectors = {}
    for p in phonemes:
        v = np.random.randn(mock_frequency_dim).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        vectors[p] = v
    return vectors


@pytest.fixture
def mock_urcm_system(mock_frequency_dim, mock_resonance_dim, mock_latent_dim):
    """Create a minimal URCMSystem for fast tests."""
    from urcm.core.system import URCMSystem
    return URCMSystem(
        frequency_dim=mock_frequency_dim,
        resonance_dim=mock_resonance_dim,
        latent_dim=mock_latent_dim,
        beam_width=2,
        max_steps=10,
    )


@pytest.fixture
def mock_reasoning_path():
    """Create a minimal ReasoningPath for testing downstream consumers."""
    import time

    from urcm.core.data_models import ReasoningPath, ResonanceState
    state = ResonanceState(
        resonance_vector=np.array([0.5, -0.3, 0.8, -0.1]),
        mu_value=0.75,
        rho_density=0.6,
        chi_cost=0.2,
        stability_score=0.9,
        oscillation_phase=0.5,
        timestamp=time.time(),
    )
    return ReasoningPath(
        states=[state],
        mu_trajectory=[0.75],
        converged=True,
        final_reason="converged",
        steps_taken=1,
    )


@pytest.fixture(autouse=True)
def temp_log_dir(tmp_path, monkeypatch):
    """Redirect logs to temp directory during tests."""
    monkeypatch.setenv("URCM_LOG_MAX_BYTES", "10000")
    monkeypatch.setenv("URCM_LOG_BACKUPS", "2")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def test_client():
    """FastAPI test client for integration tests."""
    try:
        from httpx import ASGITransport, AsyncClient

        from urcm.api import app
        transport = ASGITransport(app=app)
        client = AsyncClient(transport=transport, base_url="http://test")
        yield client
        # Note: client.close() should be called in async context
    except ImportError:
        pytest.skip("httpx not installed")


@pytest.fixture
def sample_queries():
    """Sample queries for integration testing."""
    return [
        "What is the meaning of life?",
        "Explain quantum entanglement.",
        "Describe the water cycle.",
        "What is artificial intelligence?",
        "How does gravity work?",
    ]
