import numpy as np
import matplotlib.pyplot as plt
from urcm.core.system import URCMSystem
import time

def verify_canonical_energy():
    print("🚀 Verifying Invariant Core: Canonical Energy Function")
    
    try:
        system = URCMSystem()
        print("✅ System Initialized.")
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return

    # 1. Create a "Confused" State
    # We take a valid state and add noise
    target_phoneme = "m"
    target_vec = system.pipeline.frequency_mapper.phoneme_vectors[target_phoneme]
    
    # Generate ideal state for 'm'
    # s = tanh(x @ W_in)
    # W_in is (Input, Resonance), x is (Input,)
    h = np.dot(target_vec, system.encoder.W_in)
    ideal_state = np.tanh(h)
    
    # Add heavy noise
    np.random.seed(42)
    # Optimized Topology should handle 0.2 easily.
    noise = np.random.normal(0, 0.2, size=ideal_state.shape)
    confused_state = np.tanh(np.arctanh(ideal_state * 0.9) + noise) # Messy state
    
    print(f"\n🧠 Starting with Confused State (Target: '{target_phoneme}')")
    
    # Get codebook for the system to use
    codebook = system.pipeline.frequency_mapper.phoneme_vectors
    
    # 2. Measure Initial Energy
    initial_energy = system.encoder.get_global_energy(confused_state, codebook)
    print(f"  Initial Energy: {initial_energy:.6f}")
    
    # 3. THINKING PROCESS (Gradient Descent)
    print("\n🤔 Thinking (Minimizing Energy)...")
    
    current_state = confused_state.copy()
    energies = [initial_energy]
    
    steps = 20
    learning_rate = 0.05
    
    print(f"{'Step':<5} | {'Energy':<15} | {'Delta':<15}")
    print("-" * 45)
    
    for t in range(steps):
        # Perform one step of "Thinking"
        # The system internally calculates the gradient and slides down
        new_state = system.encoder.descend_energy_gradient(
            current_state, 
            codebook, 
            steps=1, 
            learning_rate=learning_rate
        )
        
        # Measure
        energy = system.encoder.get_global_energy(new_state, codebook)
        delta = energies[-1] - energy
        
        energies.append(energy)
        current_state = new_state
        
        print(f"{t+1:<5} | {energy:<15.6f} | {delta:<15.6f}")
        
    # 4. Verify Monotonic Descent
    print("\n📊 Results:")
    if energies[-1] < energies[0]:
        print(f"✅ Energy Reduced: {energies[0]:.4f} -> {energies[-1]:.4f}")
    else:
        print("❌ Energy Failed to Reduce")
        
    # Check if it's strictly monotonic (Invariant Requirement)
    # We allow small numerical jitter, but generally > 0
    decreases = [energies[i] - energies[i+1] for i in range(len(energies)-1)]
    if all(d > -1e-5 for d in decreases):
        print("✅ Monotonic Descent Verified (Invariant Holds)")
    else:
        print("⚠️ Descent not strictly monotonic (Check step size)")

    # 5. Check Final Thought
    # What does the system think it is now?
    # Predict
    safe_state = np.clip(current_state, -1.0+1e-9, 1.0-1e-9)
    x_hat = np.dot(np.arctanh(safe_state), system.encoder.W_out)
    
    min_dist = float('inf')
    pred = "?"
    for p, vec in codebook.items():
        d = np.linalg.norm(x_hat - vec)
        if d < min_dist:
            min_dist = d
            pred = p
            
    print(f"\n🗣️ Final Thought: '{pred}' (Distance: {min_dist:.6f})")
    if pred == target_phoneme:
        print("✅ Concept Recovered Successfully.")
    else:
        print(f"❌ Hallucinated '{pred}' instead of '{target_phoneme}'.")

if __name__ == "__main__":
    verify_canonical_energy()
