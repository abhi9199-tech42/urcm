
import numpy as np
import sys
import os

# Add root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from urcm.core.system import URCMSystem

def diagnose():
    print("🔍 DIAGNOSTIC: Identifying the Reversibility Bottleneck")
    print("======================================================")
    
    # 1. Init
    system = URCMSystem()
    print("✅ System Initialized.")
    
    # 2. Simple Test Case (Single Phoneme)
    # Using a simple input removes sequence complexity variables
    text = "A" 
    # path = system.process_query(text)
    # final_state = path.final_state
    
    # Get the actual input vector used (x)
    # In URCM, the input to the state update is the phoneme vector
    # s_t = tanh(W_in * x_t + W_res * s_{t-1} + bias)
    # For the first step, s_prev is 0.
    
    # Let's manually step forward to capture 'x' and 's'
    encoder = system.encoder
    # phoneme_vecs = path.frequency_path.vectors
    # x_true = phoneme_vecs[0] # The input vector for "A"
    
    # Fix: Get vector directly from pipeline
    from urcm.core.phoneme_mapper import TextToPhonemeConverter
    
    # 1. Convert text to phonemes
    converter = TextToPhonemeConverter()
    phoneme_seq = converter.convert_text_to_phonemes(text)
    phonemes = phoneme_seq.phonemes
    
    # 2. Map phonemes to vectors
    mapper = system.pipeline.frequency_mapper
    x_true = mapper.phoneme_vectors[phonemes[0]]
    
    # Manual Forward Step
    linear_part = np.dot(x_true, encoder.W_in) + encoder.bias # s_prev is 0
    s_true = np.tanh(linear_part)
    
    print(f"\nTest Input: '{text}'")
    print(f"True State Magnitude: {np.linalg.norm(s_true):.6f}")
    
    # 3. Test Readout Accuracy (The Bottleneck Candidate #1)
    # Can we recover 'x' from 's'?
    # x_pred = s @ W_out
    x_pred = np.dot(s_true, encoder.W_out)
    
    mse_readout = np.mean((x_true - x_pred)**2)
    norm_diff_readout = np.linalg.norm(x_true - x_pred)
    
    print(f"\n--- 1. Readout Accuracy (W_out) ---")
    print(f"MSE (x vs x_pred): {mse_readout:.8f}")
    print(f"Norm Diff:         {norm_diff_readout:.8f}")
    
    if mse_readout > 1e-4:
        print("❌ CRITICAL FAILURE: Readout is too inaccurate.")
        print("   If we can't see the input, we can't subtract it to rewind.")
        
        # DEBUG: Can we solve it perfectly for this one sample?
        # s * W = x  -> W = s.T * x / (s.T * s)
        s_norm_sq = np.dot(s_true, s_true)
        W_perfect = np.outer(s_true, x_true) / s_norm_sq
        x_perfect_rec = np.dot(s_true, W_perfect)
        err_perfect_solve = np.mean((x_true - x_perfect_rec)**2)
        print(f"   [DEBUG] Theoretical Best MSE for this sample: {err_perfect_solve:.20f}")
        
    else:
        print("✅ Readout is accurate enough.")

    # 4. Test Inversion Mathematics (The Bottleneck Candidate #2)
    # Can we recover s_prev (which should be 0) using the formula?
    # Formula: s_prev = W_res_inv @ (atanh(s) - W_in @ x - bias)
    
    print("\n--- 3. Testing 'Phoneme Snapping' Hypothesis ---")
    # Can we fix the Readout error by snapping to the nearest valid phoneme?
    # This turns the problem from Regression to Classification.
    
    # Get all valid phoneme vectors
    valid_vectors = np.array(list(mapper.phoneme_vectors.values()))
    valid_phonemes = list(mapper.phoneme_vectors.keys())
    
    # Calculate distances from x_pred to all valid vectors
    # x_pred shape: (24,)
    # valid_vectors shape: (N, 24)
    distances = np.linalg.norm(valid_vectors - x_pred, axis=1)
    
    # Find nearest
    nearest_idx = np.argmin(distances)
    nearest_dist = distances[nearest_idx]
    nearest_phoneme = valid_phonemes[nearest_idx]
    x_snapped = valid_vectors[nearest_idx]
    
    print(f"Nearest Neighbor Distance: {nearest_dist:.8f}")
    print(f"Snapped to Phoneme: '{nearest_phoneme}' (Correct: '{phonemes[0]}')")
    
    if nearest_phoneme == phonemes[0]:
        print("✅ SNAPPING SUCCESSFUL: Perfect Input Recovered.")
        x_final = x_snapped
    else:
        print("❌ SNAPPING FAILED: Wrong Phoneme Selected.")
        x_final = x_pred # Fallback to noisy prediction
        
    print("\n--- 4. Rewind Accuracy (Using Snapped Input) ---")
    
    # Case C: Hybrid (Using Snapped Input + Exact Math)
    term_s = np.arctanh(s_true)
    term_x_snapped = np.dot(x_final, encoder.W_in)
    residual_snapped = term_s - term_x_snapped - encoder.bias
    s_prev_snapped = np.dot(residual_snapped, encoder.W_res_inv)
    
    err_snapped = np.linalg.norm(s_prev_snapped)
    
    print(f"Target s_prev: 0.000000")
    print(f"Error (Using Snapped Input): {err_snapped:.20f}")
    
    if err_snapped < 1e-5:
        print("✅ Reversibility confirmed via Snapping + Math.")
    else:
        print("⚠️ Still failing even with perfect input? Check W_res_inv.")

    # Check Minimum Separation of Phoneme Space
    print("\n--- Phoneme Space Analysis ---")
    min_sep = float('inf')
    for i in range(len(valid_vectors)):
        for j in range(i+1, len(valid_vectors)):
            dist = np.linalg.norm(valid_vectors[i] - valid_vectors[j])
            if dist < min_sep:
                min_sep = dist
    print(f"Minimum Separation between phonemes: {min_sep:.6f}")
    print(f"Safety Margin (MinSep / 2): {min_sep/2:.6f}")
    print(f"Current Error (Norm Diff): {norm_diff_readout:.6f}")
    
    if norm_diff_readout < (min_sep / 2):
        print("✅ SAFE: Error is within snapping margin.")
    else:
        print("❌ UNSAFE: Error is too large for reliable snapping.")

if __name__ == "__main__":
    diagnose()
