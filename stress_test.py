import numpy as np
import time
from urcm.core.system import URCMSystem

def verify_robustness_hard():
    print("🔥 EXTREME ROBUSTNESS & STRESS TEST")
    print("===================================")
    
    # 1. Load System
    try:
        system = URCMSystem()
        print("✅ System Loaded.")
    except Exception as e:
        print(f"❌ System Init Failed: {e}")
        return

    codebook = system.pipeline.frequency_mapper.phoneme_vectors
    keys = list(codebook.keys())
    
    # Pre-calculate ideal states
    states = {}
    for k in keys:
        h = np.dot(codebook[k], system.encoder.W_in)
        states[k] = np.tanh(h)

    # ---------------------------------------------------------
    # TEST 1: HIGH NOISE TOLERANCE (The "Storm" Test)
    # ---------------------------------------------------------
    print("\n[TEST 1] High Noise Tolerance (Sigma=0.35)")
    print("   Note: 0.35 is very high for tanh saturation regions.")
    
    success_count = 0
    np.random.seed(42)
    
    for k in keys:
        ideal_s = states[k]
        
        # Add Heavy Noise
        noise = np.random.normal(0, 0.35, ideal_s.shape)
        # We add noise inside the pre-tanh space to respect bounds, or post-tanh?
        # Let's add it to the state directly and clip, simulating "synaptic noise".
        noisy_s = np.clip(ideal_s + noise, -1, 1)
        
        # Run Dynamics (with slight thermal noise to escape local minima)
        final_s, steps, _ = system.encoder.run_dynamics_until_stable(
            noisy_s, codebook, max_steps=100, energy_tolerance=1e-4, noise_injection=0.02
        )
        
        # Decode and Check
        safe_final = np.clip(final_s, -0.999, 0.999)
        h_final = np.arctanh(safe_final)
        x_final = np.dot(h_final, system.encoder.W_out)
        
        # Nearest Neighbor
        nearest_dist = float('inf')
        nearest_k = None
        for ck, cv in codebook.items():
            d = np.linalg.norm(x_final - cv)
            if d < nearest_dist:
                nearest_dist = d
                nearest_k = ck
                
        if nearest_k == k:
            success_count += 1
            
    accuracy = success_count / len(keys) * 100
    print(f"   Accuracy @ 0.35 Noise: {accuracy:.1f}%")
    if accuracy > 80:
        print("✅ PASS: Highly Robust.")
    else:
        print("⚠️ WARNING: Fragile under heavy noise.")

    # ---------------------------------------------------------
    # TEST 2: NEURON DROPOUT (The "Brain Damage" Test)
    # ---------------------------------------------------------
    print("\n[TEST 2] Neuron Dropout (20% Zeroed)")
    print("   Simulating loss of 20% of resonance nodes.")
    
    success_count = 0
    dropout_rate = 0.2
    
    for k in keys:
        ideal_s = states[k].copy()
        
        # Create mask
        mask = np.random.choice([0, 1], size=ideal_s.shape, p=[dropout_rate, 1-dropout_rate])
        damaged_s = ideal_s * mask
        
        # Run Dynamics
        final_s, steps, _ = system.encoder.run_dynamics_until_stable(
            damaged_s, codebook, max_steps=100
        )
        
        # Decode
        safe_final = np.clip(final_s, -0.999, 0.999)
        h_final = np.arctanh(safe_final)
        x_final = np.dot(h_final, system.encoder.W_out)
        
        # Check
        nearest_dist = float('inf')
        nearest_k = None
        for ck, cv in codebook.items():
            d = np.linalg.norm(x_final - cv)
            if d < nearest_dist:
                nearest_dist = d
                nearest_k = ck
                
        if nearest_k == k:
            success_count += 1

    accuracy = success_count / len(keys) * 100
    print(f"   Accuracy @ 20% Dropout: {accuracy:.1f}%")
    if accuracy > 90:
        print("✅ PASS: Holographic Redundancy Confirmed.")
    else:
        print("❌ FAIL: Localized representations (Not holographic).")

    # ---------------------------------------------------------
    # TEST 3: ADVERSARIAL SUPERPOSITION (The "Bistability" Test)
    # ---------------------------------------------------------
    print("\n[TEST 3] Adversarial Superposition (A + B)")
    print("   Input = 0.5 * State(A) + 0.5 * State(B)")
    print("   System MUST converge to A or B, not stay in between.")
    
    pairs = [
        ('a', 'i'),
        ('k', 'g'), # Similar sounds
        ('p', 'b'),
        ('t', 'd'),
        ('m', 'n')
    ]
    
    resolved_count = 0
    
    for p1, p2 in pairs:
        if p1 not in keys or p2 not in keys:
            continue
            
        s1 = states[p1]
        s2 = states[p2]
        
        # Mix
        mixed_s = 0.5 * s1 + 0.5 * s2
        
        # Run Dynamics (Needs strong noise to break 50/50 symmetry)
        # Using Metacognitive Frustration (Shock) to break out of confusion
        final_s, steps, _ = system.encoder.run_dynamics_until_stable(
            mixed_s, codebook, max_steps=100, noise_injection=0.05, temperature=1.0, max_shocks=5
        )
        
        # Decode
        safe_final = np.clip(final_s, -0.999, 0.999)
        h_final = np.arctanh(safe_final)
        x_final = np.dot(h_final, system.encoder.W_out)
        
        # Check distances
        d1 = np.linalg.norm(x_final - codebook[p1])
        d2 = np.linalg.norm(x_final - codebook[p2])
        
        print(f"   Mixture '{p1}'+'{p2}' -> Dist({p1})={d1:.2f}, Dist({p2})={d2:.2f}")
        
        # Success if it picked a side (one dist is low < 1.0, other is high)
        if min(d1, d2) < 0.8: # Relaxed threshold for stress test
            resolved_count += 1
            winner = p1 if d1 < d2 else p2
            print(f"      ✅ Resolved to '{winner}'")
        else:
            print(f"      ❌ STUCK in confusion.")

    if resolved_count == len(pairs):
        print("✅ PASS: Strong Decision Boundaries (Winner-Take-All).")
    else:
        print("⚠️ WARNING: Weak attractors for mixed states.")

if __name__ == "__main__":
    verify_robustness_hard()
