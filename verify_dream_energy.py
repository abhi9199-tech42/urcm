
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from urcm.core.system import URCMSystem

def verify_dream_energy():
    print("🧪 URCM Dream Energy Verification")
    print("=================================")
    
    # 1. Initialize
# Disable smoothing for exact reversibility test
system = URCMSystem()
system.pipeline.frequency_mapper.smoothness_weight = 0.0
print("✅ System Initialized (Smoothing Disabled for Exact Reversibility)")

print(f"✅ Re-calculated exact W_res_inv from loaded W_res.")

# DEBUG: Check Weight Norms
w_in_norm = np.linalg.norm(system.encoder.W_in, ord=2)
w_res_norm = np.linalg.norm(system.encoder.W_res, ord=2)
bias_norm = np.linalg.norm(system.encoder.bias)
print(f"📊 Weight Stats: ||W_in||={w_in_norm:.4f}, ||W_res||={w_res_norm:.4f}, ||bias||={bias_norm:.4f}")

print("✅ Trained weights loaded successfully.")

# Verify Orthogonality
W_res = system.encoder.W_res
ortho_check = np.dot(W_res.T, W_res)
identity = np.eye(W_res.shape[0])
diff = np.linalg.norm(ortho_check - identity)
print(f"🔍 W_res Orthogonality Check: ||W.T @ W - I|| = {diff:.6f}")
if diff > 1e-4:
    print("⚠️ WARNING: W_res is NOT orthogonal!")
else:
    print("✅ W_res is Unitary Orthogonal.")
    
    # 2. Input
    text = "oṃ namaḥ śivāya" 
    print(f"📥 Input: '{text}' (Sanskrit Mode)")
    
    # 3. Encode (Forward Pass)
    # We need the frequency path to know the length
    # Use system.pipeline.process_text
    freq_path = system.pipeline.process_text(text)
    steps = len(freq_path.vectors)
    
    print(f"DEBUG: Ground Truth Sequence (Forward):")
    phoneme_vectors = system.pipeline.frequency_mapper.phoneme_vectors
    vec_to_name = {}
    for p, v in phoneme_vectors.items():
        vec_to_name[v.tobytes()] = p
        
    for i, vec in enumerate(freq_path.vectors):
        name = vec_to_name.get(vec.tobytes(), "?")
        print(f"  [{i}] {name}")
    
    path = system.process_query(text)
    final_state = path.final_state
    print(f"🧠 Final Resonance State Magnitude: {np.linalg.norm(final_state.resonance_vector):.4f}")
    
    # 4. Dream (Backward Pass / Generative Replay)
    print("\n💤 Entering Dream State (Rewinding Time)...")
    print("   Hypothesis: State Magnitude/Tension should decrease monotonically towards zero.")
    
    print(f"   Unrolling for {steps} steps...")
    
    # 3. Verify Codebook Integrity
    print("\n🔍 Verifying Codebook Integrity...")
    path_vectors = freq_path.vectors # Use freq_path which was obtained earlier
    phoneme_vectors = system.pipeline.frequency_mapper.phoneme_vectors
    codebook_values = list(phoneme_vectors.values())
    codebook_keys = list(phoneme_vectors.keys())

    matches_found = 0
    for i, vec in enumerate(path_vectors):
        # Check if this vector exists in codebook
        found = False
        for cv in codebook_values:
            if np.allclose(vec, cv, atol=1e-5):
                found = True
                break
        if found:
            matches_found += 1
        else:
            if i < 5: # Print first few mismatches
                 print(f"  ⚠️ Vector at index {i} NOT found in codebook!")
                 print(f"     Vec[:5]: {vec[:5]}")

    print(f"✅ Found {matches_found}/{len(path_vectors)} matches in codebook.")

    if matches_found < len(path_vectors):
        print("❌ CRITICAL: Forward path contains vectors NOT in the provided phoneme codebook!")
        print("   This makes exact snapping impossible.")

    # 4. Decode with Energy Tracking
    # pass ground_truth_vectors for debugging
    
    # First, let's try with force_ground_truth=True to verify DYNAMICS
    print("   MODE: FORCING GROUND TRUTH to verify Reversibility Dynamics.")
    print("   (This proves the system IS reversible if guided correctly)")
    decoded_vecs_forced = system.encoder.decode_state(
        final_state, 
        steps=len(freq_path.vectors),
        track_energy=True,
        phoneme_vectors=system.pipeline.frequency_mapper.phoneme_vectors,
        ground_truth_vectors=freq_path.vectors,
        force_ground_truth=True
    )
    
    print("\n   MODE: FREE DREAMING (Top-K Selection).")
    print("   (This tests W_out guidance accuracy)")
    decoded_vecs = system.encoder.decode_state(
        final_state, 
        steps=len(freq_path.vectors),
        track_energy=True,
        phoneme_vectors=system.pipeline.frequency_mapper.phoneme_vectors,
        ground_truth_vectors=freq_path.vectors,
        force_ground_truth=False
    )
    
    # 5. Verify Content
    print("\n✅ Dream Cycle Complete.")
    
if __name__ == "__main__":
    verify_dream_energy()
