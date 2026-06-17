import numpy as np
import time
from urcm.core.system import URCMSystem

def verify_full_system():
    print("🧪 RIGOROUS SYSTEM VERIFICATION")
    print("==============================")
    
    # 1. Load System
    try:
        system = URCMSystem()
        print("✅ System Loaded.")
    except Exception as e:
        print(f"❌ System Init Failed: {e}")
        return

    codebook = system.pipeline.frequency_mapper.phoneme_vectors
    keys = list(codebook.keys())
    
    # TEST 1: TOPOLOGY (Geometric Separation)
    print("\n[TEST 1] Semantic Topology Check")
    min_dist = float('inf')
    avg_dist = 0
    count = 0
    
    # Sample pairs to avoid O(N^2) spam if N is large, but 51 is fine.
    # We check states, not just inputs.
    states = {}
    for k in keys:
        h = np.dot(codebook[k], system.encoder.W_in)
        states[k] = np.tanh(h)
        
    for i, k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            d = np.linalg.norm(states[k1] - states[k2])
            if d < min_dist:
                min_dist = d
            avg_dist += d
            count += 1
            
    avg_dist /= count
    print(f"   Min Separation: {min_dist:.4f} (Should be > 1.0)")
    print(f"   Avg Separation: {avg_dist:.4f}")
    
    if min_dist > 1.0:
        print("✅ PASS: Concepts are distinct.")
    else:
        print("❌ FAIL: Concepts are too close (Risk of Confusion).")

    # TEST 2: REVERSIBILITY (Clean Input)
    print("\n[TEST 2] Reversibility (Encoding -> Decoding)")
    total_mse = 0
    for k in keys:
        # Encode
        s = states[k]
        # Decode
        safe_s = np.clip(s, -0.999, 0.999)
        h = np.arctanh(safe_s)
        x_hat = np.dot(h, system.encoder.W_out)
        
        mse = np.mean((x_hat - codebook[k])**2)
        total_mse += mse
        
    avg_mse = total_mse / len(keys)
    print(f"   Avg Reconstruction MSE: {avg_mse:.8f}")
    
    if avg_mse < 1e-5:
        print("✅ PASS: Perfect Reversibility.")
    else:
        print("❌ FAIL: Lossy Compression.")

    # TEST 3 & 4: DYNAMICS & HALTING (Stress Test)
    print("\n[TEST 3 & 4] Dynamics & Halting (Stress Test)")
    print("   Injecting Noise (0.15) into ALL 51 concepts...")
    
    success_count = 0
    total_steps = 0
    halted_count = 0
    
    np.random.seed(42)
    
    for k in keys:
        target_vec = codebook[k]
        ideal_s = states[k]
        
        # Add Noise
        noise = np.random.normal(0, 0.15, ideal_s.shape)
        noisy_s = np.tanh(np.arctanh(ideal_s * 0.9) + noise)
        
        # THINK (Run Dynamics)
        final_s, steps, final_energy = system.encoder.run_dynamics_until_stable(
            noisy_s, codebook, max_steps=100, energy_tolerance=1e-4
        )
        
        total_steps += steps
        if steps < 100:
            halted_count += 1
        else:
            print(f"      ⚠️ Thought loop detected for '{k}' (Energy: {final_energy:.4f})")
            
        # Check Result
        # Decode final thought
        safe_final = np.clip(final_s, -0.999, 0.999)
        h_final = np.arctanh(safe_final)
        x_final = np.dot(h_final, system.encoder.W_out)
        
        # Find nearest
        nearest_dist = float('inf')
        nearest_k = None
        for ck, cv in codebook.items():
            d = np.linalg.norm(x_final - cv)
            if d < nearest_dist:
                nearest_dist = d
                nearest_k = ck
                
        if nearest_k == k:
            success_count += 1
        else:
            print(f"   ❌ Failed on '{k}' -> Hallucinated '{nearest_k}' (Dist: {nearest_dist:.4f})")
            
    accuracy = success_count / len(keys) * 100
    avg_steps = total_steps / len(keys)
    
    print(f"   Accuracy: {accuracy:.1f}% ({success_count}/{len(keys)})")
    print(f"   Avg Thinking Steps: {avg_steps:.1f}")
    print(f"   Auto-Halted: {halted_count}/{len(keys)}")
    
    if accuracy > 95:
        print("✅ PASS: Robust Intelligence.")
    else:
        print("❌ FAIL: System is easily confused.")
        
    if halted_count == len(keys):
        print("✅ PASS: Automatic Halting works 100%.")
    else:
        print(f"⚠️ WARNING: Some thoughts didn't halt ({len(keys)-halted_count}). Limit Cycles too wide?")

    print("\nSUMMARY")
    if min_dist > 1.0 and avg_mse < 1e-5 and accuracy > 95:
        print("🚀 SYSTEM IS READY FOR UI / DEPLOYMENT.")
    else:
        print("🛑 SYSTEM NEEDS REFINEMENT.")

if __name__ == "__main__":
    verify_full_system()
