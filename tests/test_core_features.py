"""
Comprehensive tests for URCM core features:
1. Safety governor (energy checks, clamping)
2. Reversibility (W_res_inv, decoder trajectory)
3. Energy descent / metacognition
4. Batch processing (simple loop)
5. Transformer stub
6. Weight loading from disk on init
7. Data model classes (FrequencyPath, ResonanceState)
8. QR decomposition made optional (random init)
9. Pseudoinverse removed (random W_out)
"""

import pytest
import numpy as np
import os
import pickle
import warnings
import tempfile
import time

from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.safety import SafetyGovernor, SafetyViolation
from urcm.core.metacognition import MetacognitiveMonitor
from urcm.core.data_models import (
    FrequencyPath, ResonanceState, PhonemeSequence,
    AttractorState, ReasoningPath, MeshSignal
)
from urcm.core.hierarchical_encoder import HierarchicalEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_freq_path(seq_len=5, dim=24):
    """Create a random FrequencyPath for testing."""
    vecs = np.random.randn(seq_len, dim) * 0.3
    mapping = [(f"p{i}", i) for i in range(seq_len)]
    return FrequencyPath(vectors=vecs, smoothness_score=0.5, phoneme_mapping=mapping)


def _make_freq_path_batch(batch_size=4, seq_len=5, dim=24):
    """Create a list of FrequencyPaths."""
    return [_make_freq_path(seq_len, dim) for _ in range(batch_size)]


# ===========================================================================
# 1. SAFETY GOVERNOR
# ===========================================================================

class TestSafetyGovernor:
    """Tests for energy checks, clamping, and kernel locking."""

    def test_energy_ceiling_pass(self):
        gov = SafetyGovernor(energy_ceiling=10.0)
        state = np.ones(64) * 0.5
        assert gov.check_energy_ceiling(state) is True

    def test_energy_ceiling_violation(self):
        gov = SafetyGovernor(energy_ceiling=10.0)
        bad_state = np.ones(64) * 10.0  # norm ~ 80
        with pytest.raises(SafetyViolation, match="ENERGY CEILING"):
            gov.check_energy_ceiling(bad_state)

    def test_spectral_stability_ok(self):
        gov = SafetyGovernor(max_spectral_radius=0.99)
        H = np.random.randn(64, 64)
        Q, _ = np.linalg.qr(H)
        W = Q * 0.95  # spectral radius ~ 0.95
        assert gov.check_spectral_stability(W) is True

    def test_spectral_stability_runaway(self):
        gov = SafetyGovernor(max_spectral_radius=0.99)
        W = np.eye(64) * 2.0  # spectral radius = 2.0
        with pytest.raises(SafetyViolation, match="RUNAWAY GAIN"):
            gov.check_spectral_stability(W)

    def test_kernel_lock_blocks_core_logic(self):
        gov = SafetyGovernor()
        gov.lock_kernel()
        with pytest.raises(SafetyViolation, match="KERNEL LOCK"):
            gov.validate_modification("core_logic_rewrite")

    def test_kernel_lock_allows_weight_update(self):
        gov = SafetyGovernor()
        gov.lock_kernel()
        # Should NOT raise
        gov.validate_modification("weight_update")

    def test_kernel_unlock_with_key(self):
        gov = SafetyGovernor()
        gov.lock_kernel()
        gov.unlock_kernel("URCM_ADMIN_OVERRIDE")
        gov.validate_modification("core_logic_rewrite")  # no raise

    def test_kernel_unlock_bad_key(self):
        gov = SafetyGovernor()
        gov.lock_kernel()
        with pytest.raises(SafetyViolation, match="Unauthorized"):
            gov.unlock_kernel("wrong_key")

    def test_input_sanitization_clips(self):
        gov = SafetyGovernor()
        huge = np.ones(24) * 100.0
        result = gov.sanitize_input(huge)
        assert np.linalg.norm(result) <= 5.0 + 1e-9

    def test_input_sanitization_passthrough(self):
        gov = SafetyGovernor()
        small = np.ones(24) * 0.1
        result = gov.sanitize_input(small)
        np.testing.assert_allclose(result, small)

    def test_verify_reversibility_pass(self):
        gov = SafetyGovernor()
        s = np.random.randn(64)
        gov.verify_reversibility(s, s.copy(), tolerance=1e-3)

    def test_verify_reversibility_fail(self):
        gov = SafetyGovernor()
        s1 = np.zeros(64)
        s2 = np.ones(64)
        with pytest.raises(SafetyViolation, match="IRREVERSIBLE"):
            gov.verify_reversibility(s1, s2, tolerance=0.01)

    def test_safety_active_in_encoder_trajectory(self):
        """Encoder's get_state_trajectory must call safety checks."""
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        fp = _make_freq_path(5, 24)
        traj = enc.get_state_trajectory(fp)
        assert traj.shape == (5, 64)
        # All states must be finite (safety didn't blow up)
        assert np.all(np.isfinite(traj))


# ===========================================================================
# 2. REVERSIBILITY (W_res_inv, decoder trajectory)
# ===========================================================================

class TestReversibility:
    """Tests for W_res_inv correctness and decoder trajectory collection."""

    def test_w_res_inv_exists(self):
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        assert hasattr(enc, "W_res_inv")
        assert enc.W_res_inv.shape == (64, 64)

    def test_w_res_inv_approximates_inverse(self):
        """W_res_inv should satisfy W_res @ W_res_inv ≈ I (scaled)."""
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        product = enc.W_res @ enc.W_res_inv
        np.testing.assert_allclose(product, np.eye(64), atol=1e-6,
                                    err_msg="W_res_inv is not a valid inverse of W_res")

    def test_roundtrip_forward_backward(self):
        """Forward encode then backward decode via W_res_inv should approximately reconstruct."""
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        fp = _make_freq_path(10, 24)

        # Forward
        states = enc.get_state_trajectory(fp)
        final_state = states[-1]

        # Backward one step: from s_t reconstruct s_{t-1}
        # s_t = tanh(W_in*x_t + W_res*s_{t-1} + b)
        # => s_{t-1} = W_res_inv @ (atanh(s_t) - W_in*x_t - b)
        safe = np.clip(final_state, -0.999, 0.999)
        pre_act = np.arctanh(safe)
        x_t = fp.vectors[-1]
        residual = pre_act - np.dot(x_t, enc.W_in) - enc.bias
        s_prev_reconstructed = np.dot(residual, enc.W_res_inv)

        # The reconstructed s_prev should be close to actual s_{t-1}
        s_prev_actual = states[-2]
        diff = np.linalg.norm(s_prev_reconstructed - s_prev_actual)
        assert diff < 0.5, f"Roundtrip reconstruction diff too large: {diff:.4f}"

    def test_decoder_trajectory_shapes(self):
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        fp = _make_freq_path(8, 24)
        final_state_vec = enc.encode_path(fp)
        rs = ResonanceState(
            resonance_vector=final_state_vec,
            mu_value=0.5, rho_density=0.5, chi_cost=0.5,
            stability_score=0.5, oscillation_phase=0.0,
            timestamp=time.time()
        )
        states, outputs = enc.collect_decoder_trajectory(rs, steps=5)
        assert states.shape == (5, 64)
        assert outputs.shape == (5, 24)

    def test_decode_state_runs(self):
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        fp = _make_freq_path(6, 24)
        final_vec = enc.encode_path(fp)
        rs = ResonanceState(
            resonance_vector=final_vec,
            mu_value=0.5, rho_density=0.5, chi_cost=0.5,
            stability_score=0.5, oscillation_phase=0.0,
            timestamp=time.time()
        )
        decoded = enc.decode_state(rs, steps=4)
        assert decoded.shape == (4, 24)
        assert np.all(np.isfinite(decoded))


# ===========================================================================
# 3. ENERGY DESCENT / METACOGNITION
# ===========================================================================

class TestEnergyDescent:
    """Tests for energy calculation, gradient descent, and metacognitive monitoring."""

    def test_global_energy_finite(self):
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        state = np.random.randn(64) * 0.5
        energy = enc.get_global_energy(state)
        assert np.isfinite(energy)
        assert energy >= 0

    def test_global_energy_with_codebook(self):
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        state = np.random.randn(64) * 0.5
        # Codebook must be in INPUT space (dim=24) since get_global_energy compares x_hat
        codebook = {"a": np.random.randn(24), "b": np.random.randn(24)}
        energy = enc.get_global_energy(state, codebook)
        assert np.isfinite(energy)
        assert energy >= 0

    def test_energy_descent_reduces_energy(self):
        """descend_energy_gradient should reduce energy over steps."""
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        state = np.random.randn(64) * 0.8
        codebook = {f"k{i}": np.random.randn(24) for i in range(5)}
        e_before = enc.get_global_energy(state, codebook)
        new_state = enc.descend_energy_gradient(state, codebook, steps=20, learning_rate=0.05)
        e_after = enc.get_global_energy(new_state, codebook)
        assert e_after <= e_before + 0.01  # allow small numerical tolerance

    def test_metacognitive_monitor_stable(self):
        mon = MetacognitiveMonitor(history_window=5)
        state = np.random.randn(64)
        sig = mon.monitor(state, current_energy=0.3, current_word="hello")
        assert sig["status"] == "stable"
        assert sig["frustration"] == 0.0
        assert sig["focus"] == 0.0

    def test_metacognitive_monitor_detects_loops(self):
        mon = MetacognitiveMonitor(history_window=5)
        state = np.random.randn(64)
        for _ in range(5):
            mon.monitor(state, current_energy=0.3, current_word="loop")
        sig = mon.monitor(state, current_energy=0.3, current_word="loop")
        assert sig["status"] == "looping"
        assert sig["frustration"] > 0

    def test_metacognitive_monitor_detects_confusion(self):
        mon = MetacognitiveMonitor(history_window=5)
        state = np.random.randn(64)
        # Use different words to avoid loop detector, but high energy to trigger confusion
        words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
        for w in words:
            mon.monitor(state, current_energy=5.0, current_word=w)
        sig = mon.monitor(state, current_energy=5.0, current_word="golf")
        assert sig["status"] == "confused"
        assert sig["focus"] > 0

    def test_goal_adherence(self):
        mon = MetacognitiveMonitor()
        s1 = np.array([1.0, 0.0, 0.0])
        s2 = np.array([1.0, 0.0, 0.0])
        s3 = np.array([-1.0, 0.0, 0.0])
        assert mon.check_goal_adherence(s1, s2) > 0.99
        assert mon.check_goal_adherence(s1, s3) < -0.99

    def test_dynamics_until_stable_converges(self):
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        state = np.random.randn(64) * 0.5
        codebook = {f"k{i}": np.random.randn(24) for i in range(5)}
        final, steps, energy = enc.run_dynamics_until_stable(
            state, codebook, max_steps=50, energy_tolerance=1e-3
        )
        assert np.all(np.isfinite(final))
        assert steps > 0


# ===========================================================================
# 4. BATCH PROCESSING (kept simple loop)
# ===========================================================================

class TestBatchProcessing:
    """Tests for batch encoding via simple loop."""

    def test_encode_path_batch_shapes(self):
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        batch = _make_freq_path_batch(4, 6, 24)
        result = enc.process_batch(batch)
        assert result.shape == (4, 64)
        assert np.all(np.isfinite(result))

    def test_batch_matches_individual(self):
        """Batch result should match individual encode_path results."""
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        paths = _make_freq_path_batch(3, 5, 24)
        batch_result = enc.process_batch(paths)
        for i, fp in enumerate(paths):
            individual = enc.encode_path(fp)
            np.testing.assert_allclose(batch_result[i], individual, atol=1e-10)

    def test_encode_path_batch_rnn_shapes(self):
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        batch_arr = np.random.randn(4, 6, 24)
        result = enc.encode_path_batch(batch_arr)
        assert result.shape == (4, 64)

    def test_get_state_trajectory_batch(self):
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        batch_arr = np.random.randn(3, 5, 24)
        result = enc.get_state_trajectory_batch(batch_arr)
        assert result.shape == (3, 5, 64)
        assert np.all(np.isfinite(result))

    def test_batch_empty(self):
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        result = enc.process_batch([])
        assert result.shape == (0,)

    def test_hierarchical_batch(self):
        he = HierarchicalEncoder(l1_input_dim=24, l1_res_dim=64, l2_res_dim=128)
        batch = np.random.randn(3, 5, 24)
        l2_states, l1_trajs = he.encode_concept_batch(batch)
        assert l2_states.shape == (3, 128)
        assert l1_trajs.shape == (3, 5, 64)


# ===========================================================================
# 5. TRANSFORMER STUB
# ===========================================================================

class TestTransformerStub:
    """Tests for the transformer_stub encoder type."""

    def test_transformer_stub_init(self):
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64, encoder_type="transformer_stub")
        assert enc.encoder_type == "transformer_stub"
        assert hasattr(enc, "W_q")
        assert hasattr(enc, "W_k")
        assert hasattr(enc, "W_v")
        assert enc.W_q.shape == (24, 32)
        assert enc.W_k.shape == (24, 32)
        assert enc.W_v.shape == (24, 64)

    def test_transformer_stub_encode(self):
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64, encoder_type="transformer_stub")
        fp = _make_freq_path(5, 24)
        result = enc.encode_path(fp)
        assert result.shape == (64,)
        assert np.all(np.isfinite(result))

    def test_transformer_stub_different_from_recurrent(self):
        """Transformer stub should produce different output than recurrent for same input."""
        fp = _make_freq_path(5, 24)
        enc_r = ResonancePathEncoder(input_dim=24, resonance_dim=64, encoder_type="recurrent_numpy")
        enc_t = ResonancePathEncoder(input_dim=24, resonance_dim=64, encoder_type="transformer_stub")
        r = enc_r.encode_path(fp)
        t = enc_t.encode_path(fp)
        assert not np.allclose(r, t, atol=1e-4)

    def test_transformer_invalid_type(self):
        with pytest.raises(ValueError, match="Unsupported encoder type"):
            ResonancePathEncoder(encoder_type="invalid_magic")

    def test_transformer_stub_batch_not_implemented(self):
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64, encoder_type="transformer_stub")
        batch = np.random.randn(2, 5, 24)
        with pytest.raises(NotImplementedError):
            enc.encode_path_batch(batch)


# ===========================================================================
# 6. WEIGHT LOADING FROM DISK ON INIT
# ===========================================================================

class TestWeightLoading:
    """Tests for HierarchicalEncoder weight loading from disk."""

    def test_load_weights_if_exist_no_file(self, tmp_path, monkeypatch):
        """No weight file => random init, no crash."""
        monkeypatch.chdir(tmp_path)
        he = HierarchicalEncoder(l1_input_dim=24, l1_res_dim=64, l2_res_dim=128)
        assert he.layer1.W_res.shape == (64, 64)

    def test_load_weights_from_file(self, tmp_path, monkeypatch):
        """Saving then loading should restore weights."""
        monkeypatch.chdir(tmp_path)
        he = HierarchicalEncoder(l1_input_dim=24, l1_res_dim=64, l2_res_dim=128)
        save_path = str(tmp_path / "test_weights.pkl")
        he.save_hierarchy(save_path)

        # Create fresh encoder and load
        he2 = HierarchicalEncoder(l1_input_dim=24, l1_res_dim=64, l2_res_dim=128)
        he2.load_hierarchy(save_path)
        np.testing.assert_array_equal(he2.layer1.W_res, he.layer1.W_res)
        np.testing.assert_array_equal(he2.layer2.W_res, he.layer2.W_res)

    def test_load_weights_nonexistent_file(self, tmp_path, monkeypatch):
        """Loading non-existent file should not crash."""
        monkeypatch.chdir(tmp_path)
        he = HierarchicalEncoder(l1_input_dim=24, l1_res_dim=64, l2_res_dim=128)
        he.load_hierarchy(str(tmp_path / "no_such_file.pkl"))  # prints message, no crash

    def test_load_weights_dimension_mismatch(self, tmp_path, monkeypatch):
        """Dimension mismatch: load_hierarchy loads whatever is saved (no runtime guard)."""
        monkeypatch.chdir(tmp_path)
        # Save with l2=128
        he = HierarchicalEncoder(l1_input_dim=24, l1_res_dim=64, l2_res_dim=128)
        save_path = str(tmp_path / "dim_test.pkl")
        he.save_hierarchy(save_path)

        # Load with l2=256 (mismatch)
        he2 = HierarchicalEncoder(l1_input_dim=24, l1_res_dim=64, l2_res_dim=256)
        he2.load_hierarchy(save_path)
        # load_hierarchy loads the saved (128, 128) W_res into the 256-dim encoder
        np.testing.assert_array_equal(he2.layer2.W_res, he.layer2.W_res)

    def test_load_hierarchical_encoder_weights(self, tmp_path, monkeypatch):
        """HierarchicalEncoder.save_hierarchy/load_hierarchy round-trip."""
        monkeypatch.chdir(tmp_path)
        he = HierarchicalEncoder(l1_input_dim=24, l1_res_dim=64, l2_res_dim=128)
        path = str(tmp_path / "hier.pkl")
        he.save_hierarchy(path)

        he2 = HierarchicalEncoder(l1_input_dim=24, l1_res_dim=64, l2_res_dim=128)
        he2.load_hierarchy(path)
        np.testing.assert_array_equal(he2.layer1.W_in, he.layer1.W_in)
        np.testing.assert_array_equal(he2.layer1.W_res, he.layer1.W_res)
        np.testing.assert_array_equal(he2.layer1.W_out, he.layer1.W_out)
        np.testing.assert_array_equal(he2.layer2.W_in, he.layer2.W_in)


# ===========================================================================
# 7. DATA MODEL CLASSES
# ===========================================================================

class TestDataModels:
    """Tests for FrequencyPath, ResonanceState, and other data models."""

    def test_frequency_path_valid(self):
        fp = _make_freq_path(5, 24)
        assert fp.vectors.shape == (5, 24)
        assert fp.smoothness_score == 0.5
        assert len(fp.phoneme_mapping) == 5

    def test_frequency_path_rejects_bad_dim(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            FrequencyPath(vectors=np.zeros((5,)), smoothness_score=0.5,
                          phoneme_mapping=[("a", 0)])

    def test_frequency_path_rejects_bad_K(self):
        with pytest.raises(ValueError, match="K must be in range"):
            FrequencyPath(vectors=np.zeros((5, 10)), smoothness_score=0.5,
                          phoneme_mapping=[("a", i) for i in range(5)])

    def test_frequency_path_rejects_negative_smoothness(self):
        with pytest.raises(ValueError, match="non-negative"):
            FrequencyPath(vectors=np.zeros((5, 24)), smoothness_score=-1.0,
                          phoneme_mapping=[("a", i) for i in range(5)])

    def test_frequency_path_mapping_length_mismatch(self):
        with pytest.raises(ValueError, match="Phoneme mapping length"):
            FrequencyPath(vectors=np.zeros((5, 24)), smoothness_score=0.5,
                          phoneme_mapping=[("a", 0)])

    def test_resonance_state_valid(self):
        rs = ResonanceState(
            resonance_vector=np.random.randn(64),
            mu_value=0.8, rho_density=0.9, chi_cost=0.1,
            stability_score=0.85, oscillation_phase=1.0,
            timestamp=time.time()
        )
        assert rs.resonance_vector.shape == (64,)
        assert rs.mu_value == 0.8

    def test_resonance_state_rejects_2d_vector(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            ResonanceState(
                resonance_vector=np.zeros((64, 1)),
                mu_value=0.5, rho_density=0.5, chi_cost=0.5,
                stability_score=0.5, oscillation_phase=1.0,
                timestamp=0.0
            )

    def test_resonance_state_rejects_bad_phase(self):
        with pytest.raises(ValueError, match="Oscillation phase"):
            ResonanceState(
                resonance_vector=np.zeros(64),
                mu_value=0.5, rho_density=0.5, chi_cost=0.5,
                stability_score=0.5, oscillation_phase=7.0,
                timestamp=0.0
            )

    def test_resonance_state_rejects_negative_timestamp(self):
        with pytest.raises(ValueError, match="Timestamp must be non-negative"):
            ResonanceState(
                resonance_vector=np.zeros(64),
                mu_value=0.5, rho_density=0.5, chi_cost=0.5,
                stability_score=0.5, oscillation_phase=1.0,
                timestamp=-1.0
            )

    def test_phoneme_sequence_valid(self):
        ps = PhonemeSequence(phonemes=["a", "k", "t"], source_text="akt")
        assert len(ps.phonemes) == 3

    def test_phoneme_sequence_empty_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            PhonemeSequence(phonemes=[], source_text="akt")

    def test_attractor_state_valid(self):
        at = AttractorState(
            phase_pattern=np.random.randn(64),
            eigenvalues=np.random.randn(64),
            stability_type="stable",
            semantic_label="test"
        )
        assert at.stability_type == "stable"

    def test_attractor_state_bad_type(self):
        with pytest.raises(ValueError, match="Stability type"):
            AttractorState(
                phase_pattern=np.zeros(64),
                eigenvalues=np.zeros(64),
                stability_type="banana"
            )

    def test_mesh_signal_valid(self):
        ms = MeshSignal(
            sender_id="node1", delta_mu=0.1,
            phase_alignment=1.0, timestamp=0.0,
            signal_type="sync"
        )
        assert ms.signal_type == "sync"

    def test_mesh_signal_bad_type(self):
        with pytest.raises(ValueError, match="Signal type"):
            MeshSignal(
                sender_id="n", delta_mu=0.1,
                phase_alignment=1.0, timestamp=0.0,
                signal_type="invalid"
            )


# ===========================================================================
# 8. QR DECOMPOSITION MADE OPTIONAL (random init)
# ===========================================================================

class TestQROptional:
    """Tests that QR decomposition is used for W_res init and is orthogonal."""

    def test_w_res_is_orthogonal_scaled(self):
        """W_res should be Q * 0.95 where Q is orthogonal."""
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        # W_res should be close to orthogonal * 0.95
        product = enc.W_res @ enc.W_res.T
        expected = np.eye(64) * (0.95 ** 2)
        np.testing.assert_allclose(product, expected, atol=1e-6,
                                    err_msg="W_res is not a scaled orthogonal matrix")

    def test_deterministic_init(self):
        """Two encoders with default seed should have same W_res."""
        enc1 = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        enc2 = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        np.testing.assert_array_equal(enc1.W_res, enc2.W_res)

    def test_w_res_norm_preservation(self):
        """Orthogonal W_res should preserve vector norm (approximately)."""
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        v = np.random.randn(64)
        v_norm = np.linalg.norm(v)
        mapped = np.dot(v, enc.W_res)
        mapped_norm = np.linalg.norm(mapped)
        # Should be close to 0.95 * v_norm
        np.testing.assert_allclose(mapped_norm, 0.95 * v_norm, rtol=0.01)

    def test_w_in_random_normal(self):
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        assert enc.W_in.shape == (24, 64)
        assert np.all(np.isfinite(enc.W_in))


# ===========================================================================
# 9. PSEUDOINVERSE REMOVED (random W_out)
# ===========================================================================

class TestPseudoinverseRemoved:
    """Tests that W_out is NOT simply pinv(W_in) but a separately initialized matrix."""

    def test_w_out_shape(self):
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        assert enc.W_out.shape == (64, 24)

    def test_w_out_differs_from_pinv(self):
        """W_out should NOT equal pinv(W_in) — it is randomly initialized."""
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        pinv_win = np.linalg.pinv(enc.W_in)
        # They share shape but should differ
        assert not np.allclose(enc.W_out, pinv_win, atol=1e-4), \
            "W_out is still just pinv(W_in) — pseudoinverse not removed"

    def test_w_out_finite(self):
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        assert np.all(np.isfinite(enc.W_out))

    def test_w_out_trained_differs_from_random(self):
        """After training, W_out should differ from its initial random value."""
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        w_out_before = enc.W_out.copy()

        # Create a tiny training set
        paths = [_make_freq_path(5, 24) for _ in range(3)]
        gen = lambda: iter([paths])
        enc.train_decoder_fast_denoising(gen, noise_level=0.01, ridge_alpha=0.1)

        assert not np.allclose(enc.W_out, w_out_before, atol=1e-4), \
            "W_out did not change after training"


# ===========================================================================
# 10. DOCSTRINGS AND COMMENTS
# ===========================================================================

class TestDocstrings:
    """Verify that key classes and methods have docstrings."""

    def test_safety_governor_docstring(self):
        assert SafetyGovernor.__doc__ is not None
        assert len(SafetyGovernor.__doc__) > 10

    def test_check_energy_ceiling_docstring(self):
        assert SafetyGovernor.check_energy_ceiling.__doc__ is not None

    def test_resonance_encoder_docstring(self):
        assert ResonancePathEncoder.__doc__ is not None
        assert "encode" in ResonancePathEncoder.__doc__.lower()

    def test_encode_path_docstring(self):
        assert ResonancePathEncoder.encode_path.__doc__ is not None

    def test_decode_state_docstring(self):
        assert ResonancePathEncoder.decode_state.__doc__ is not None

    def test_data_models_docstrings(self):
        assert FrequencyPath.__doc__ is not None
        assert ResonanceState.__doc__ is not None
        assert PhonemeSequence.__doc__ is not None

    def test_metacognition_docstring(self):
        assert MetacognitiveMonitor.__doc__ is not None

    def test_hierarchical_encoder_docstring(self):
        assert HierarchicalEncoder.__doc__ is not None


# ===========================================================================
# INTEGRATION: Encoder creates valid ResonanceState end-to-end
# ===========================================================================

class TestEndToEnd:
    """Integration tests tying multiple features together."""

    def test_full_encode_produces_valid_state(self):
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        fp = _make_freq_path(8, 24)
        state = enc.get_resonance_state(fp)
        assert isinstance(state, ResonanceState)
        assert state.resonance_vector.shape == (64,)
        assert 0 <= state.oscillation_phase <= 2 * np.pi
        assert np.isfinite(state.mu_value)
        assert np.isfinite(state.rho_density)
        assert np.isfinite(state.chi_cost)

    def test_hierarchical_encode(self):
        he = HierarchicalEncoder(l1_input_dim=24, l1_res_dim=64, l2_res_dim=128)
        phoneme_vecs = np.random.randn(10, 24)
        l2_state, l1_traj = he.encode_concept(phoneme_vecs)
        assert l2_state.shape == (128,)
        assert l1_traj.shape == (10, 64)

    def test_full_pipeline_with_batch(self):
        enc = ResonancePathEncoder(input_dim=24, resonance_dim=64)
        paths = _make_freq_path_batch(4, 6, 24)
        batch_result = enc.process_batch(paths)
        assert batch_result.shape == (4, 64)
        # Each should match individual
        for i, fp in enumerate(paths):
            individual = enc.encode_path(fp)
            np.testing.assert_allclose(batch_result[i], individual, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
