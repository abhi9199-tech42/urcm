import numpy as np
import sys
import os

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from urcm.core.system import URCMSystem
from urcm.core.phoneme_mapper import PhonemeFrequencyMapper
from urcm.core.data_models import ResonanceState

def test_phoneme_reversibility():
    """
    Test if Phoneme -> Vector -> Phoneme is reversible via Nearest Neighbor.
    """
    print("\n--- Test 1: Phoneme <-> Vector Reversibility ---")
    mapper = PhonemeFrequencyMapper(frequency_dim=24)
    
    test_phonemes = ['a', 'k', 't', 'p', 'y', 's']
    success_count = 0
    
    for p in test_phonemes:
        # 1. Map to Vector
        vec = mapper.map_phoneme(p)
        
        # 2. Reverse Map (Nearest Neighbor)
        best_match = None
        best_dist = float('inf')
        
        for cand_p, cand_vec in mapper.phoneme_vectors.items():
            dist = np.linalg.norm(vec - cand_vec)
            if dist < best_dist:
                best_dist = dist
                best_match = cand_p
        
        if best_match == p:
            print(f"✅ {p} -> Vector -> {best_match} (Dist: {best_dist:.6f})")
            success_count += 1
        else:
            print(f"❌ {p} -> Vector -> {best_match} (Dist: {best_dist:.6f})")
            
    print(f"Result: {success_count}/{len(test_phonemes)} Phonemes Reversible.")

def test_latent_reversibility():
    """
    Test if Resonance -> Latent -> Resonance is reversible (Round Trip).
    """
    print("\n--- Test 2: Resonance -> Latent -> Resonance (Compression) ---")
    system = URCMSystem()
    
    # Create a dummy resonance state
    # In a real scenario, this comes from the encoder
    original_vec = np.random.normal(0, 1, system.resonance_dim)
    original_vec = original_vec / np.linalg.norm(original_vec)
    
    state = ResonanceState(
        resonance_vector=original_vec,
        mu_value=0.9,
        rho_density=0.8,
        chi_cost=0.1,
        stability_score=0.95,
        oscillation_phase=0.0,
        timestamp=0.0
    )
    
    # Perform Round Trip
    reconstructed_vec, loss, valid = system.reconstruction.perform_round_trip(state)
    
    print(f"Original Vector Norm: {np.linalg.norm(original_vec):.4f}")
    print(f"Reconstructed Norm: {np.linalg.norm(reconstructed_vec):.4f}")
    print(f"Reconstruction Loss (L1): {loss:.4f}")
    print(f"Valid (Drift < Threshold): {valid}")
    
    if valid:
        print("✅ Latent Compression is Stable.")
    else:
        print("⚠️ Latent Compression shows significant drift.")

def test_semantic_reversibility_gap():
    """
    Demonstrate the missing link: Resonance -> Sequence.
    """
    print("\n--- Test 3: The 'Holy Grail' Gap (Resonance -> Sequence) ---")
    system = URCMSystem()
    text = "quantum physics"
    
    print(f"Input: '{text}'")
    
    # 1. Forward Pass
    freq_path = system.pipeline.process_text(text)
    resonance_state = system.encoder.get_resonance_state(freq_path)
    
    print("Forward Mapping: Text -> FrequencyPath -> ResonanceState ✅")
    print(f"Resonance Vector Shape: {resonance_state.resonance_vector.shape}")
    
    # 2. Backward Pass (Hypothetical)
    print("Backward Mapping: ResonanceState -> FrequencyPath -> Text ❓")
    
    if hasattr(system.encoder, 'decode_state'):
        print("✅ Encoder has 'decode_state' method.")
        
        # Try to run it
        print("   Attempting to decode resonance state...")
        try:
            # "quantum physics" is ~14 phonemes. Let's try to decode 15 steps.
            decoded_vectors = system.encoder.decode_state(resonance_state, steps=15)
            print(f"   Decoded shape: {decoded_vectors.shape}")
            
            # Verify signal strength
            signal_norm = np.linalg.norm(decoded_vectors)
            print(f"   Decoded Signal Energy: {signal_norm:.4f}")
            
            if signal_norm > 0.1:
                 print("✅ Decoder produced non-zero signal (Generative Output Active).")
                 
                 # Try to map back to phonemes
                 print("   Mapping decoded vectors back to Phonemes...")
                 mapper = PhonemeFrequencyMapper(frequency_dim=24)
                 decoded_phonemes = []
                 for vec in decoded_vectors:
                     # Find nearest phoneme
                     best_match = None
                     best_dist = float('inf')
                     for cand_p, cand_vec in mapper.phoneme_vectors.items():
                         dist = np.linalg.norm(vec - cand_vec)
                         if dist < best_dist:
                             best_dist = dist
                             best_match = cand_p
                     decoded_phonemes.append(best_match)
                 
                 print(f"   Reconstructed Sequence: {'-'.join(decoded_phonemes)}")
                 print("   (Note: This is a 'Dream' sequence from the resonance state)")
                 
            else:
                 print("⚠️ Decoder output is silent/zero.")
        except Exception as e:
            print(f"❌ Error during decoding: {e}")
             
    else:
        print("❌ Encoder is MISSING 'decode_state' method.")
        print("   -> Current architecture is 'Many-to-One' (Lossy Compression).")
        print("   -> To achieve the Holy Grail, we need a Generative Decoder (e.g., RNN/Transformer Decoder)")
        print("      that takes the Resonance Vector and unrolls it back to Phonemes.")

if __name__ == "__main__":
    print("🧪 URCM Reversibility Verification")
    test_phoneme_reversibility()
    test_latent_reversibility()
    test_semantic_reversibility_gap()
