import numpy as np
import matplotlib.pyplot as plt
from urcm.core.system import URCMSystem

def thinking_test():
    print("🧠 Starting Autonomous Thinking Test...")
    print("   Scenario: System is given a noisy, confused thought.")
    print("   Goal: System must autonomously resolve it to a clear concept.")
    
    # 1. Load System
    system = URCMSystem()
    print("✅ System Loaded.")
    
    # 2. Pick a Concept and Corrupt it
    target_phoneme = 'a'
    target_vec = system.pipeline.frequency_mapper.phoneme_vectors[target_phoneme]
    
    # Generate Ideal State
    h_ideal = np.dot(target_vec, system.encoder.W_in)
    s_ideal = np.tanh(h_ideal)
    
    # Add Noise (Confusion)
    # High noise to test robustness
    np.random.seed(101)
    noise = np.random.normal(0, 0.3, s_ideal.shape)
    s_current = np.tanh(np.arctanh(s_ideal * 0.9) + noise)
    
    initial_energy = system.encoder.get_global_energy(s_current, system.pipeline.frequency_mapper.phoneme_vectors)
    print(f"\n   Target: '{target_phoneme}'")
    print(f"   Initial Energy: {initial_energy:.6f} (High Confusion)")
    
    # 3. Autonomous Thinking Loop (Dynamics)
    max_steps = 50
    energy_threshold = 1e-4
    
    energies = []
    
    print("\n   Thinking Process:")
    print("   Step | Energy   | Delta    | Drift | Action")
    print("   ---------------------------------------------")
    
    prev_energy = initial_energy
    
    for t in range(max_steps):
        # A. Measure Energy
        energy = system.encoder.get_global_energy(s_current, system.pipeline.frequency_mapper.phoneme_vectors)
        energies.append(energy)
        
        delta = prev_energy - energy
        
        # B. Check Halting
        # If energy is low enough OR change is negligible
        action = "Thinking..."
        if abs(delta) < energy_threshold and t > 5:
            action = "HALT (Stable)"
        
        # C. Run Dynamics (s_t+1 = tanh(W_res * s_t))
        # No input (Autonomous)
        s_next = np.tanh(np.dot(s_current, system.encoder.W_res))
        
        drift = np.linalg.norm(s_next - s_current)
        
        print(f"   {t:<4} | {energy:.6f} | {delta:+.6f} | {drift:.4f} | {action}")
        
        if action.startswith("HALT"):
            break
            
        s_current = s_next
        prev_energy = energy
        
    # 4. Final Result
    print("\n   Final Thought Analysis:")
    final_energy = energies[-1]
    
    # Decode
    safe_state = np.clip(s_current, -0.999, 0.999)
    h_final = np.arctanh(safe_state)
    x_final = np.dot(h_final, system.encoder.W_out)
    
    # Find nearest concept
    min_dist = float('inf')
    found_phoneme = None
    
    for p, vec in system.pipeline.frequency_mapper.phoneme_vectors.items():
        d = np.linalg.norm(x_final - vec)
        if d < min_dist:
            min_dist = d
            found_phoneme = p
            
    print(f"   Resolved Concept: '{found_phoneme}'")
    print(f"   Distance to Concept: {min_dist:.6f}")
    
    if found_phoneme == target_phoneme:
        print("✅ SUCCESS: Correctly identified original thought despite noise.")
    else:
        print(f"❌ FAILURE: Hallucinated '{found_phoneme}'.")

if __name__ == "__main__":
    thinking_test()
