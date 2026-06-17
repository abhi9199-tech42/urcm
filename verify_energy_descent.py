import numpy as np
import matplotlib.pyplot as plt
from urcm.core.system import URCMSystem
import time
import os

def calculate_pseudo_energy(state, codebook):
    """
    Calculates a pseudo-energy metric based on distance to the nearest valid phoneme.
    E(s) = min_k ||s - codebook_k||^2
    
    In a true attractor system, this should decrease monotonically as the system 
    'snaps' to a valid thought.
    """
    distances = []
    for vec in codebook:
        d = np.linalg.norm(state - vec)
        distances.append(d)
    return min(distances)

def calculate_velocity(state_t, state_prev):
    """
    Calculates the 'velocity' of the thought process.
    V = ||s_t - s_{t-1}||
    
    A stable thought (Fixed Point) implies V -> 0.
    """
    if state_prev is None:
        return 0.0
    return np.linalg.norm(state_t - state_prev)

def verify_descent():
    print("Loading URCM System...")
    try:
        # URCMSystem initializes ResonancePathEncoder, which auto-loads weights from project root
        system = URCMSystem()
        print("✅ System Initialized.")
        
        # Verify weights loaded
        # We can check if W_out is just the pinv of W_in (untrained) or something else
        # Or just trust the logs which will print "Loading trained weights..."
    except Exception as e:
        print(f"❌ Could not initialize system: {e}")
        return

    # Test Input: A fragment of a mantra
    # We feed it, then let it 'dream' (free run) to see if it settles.
    input_sequence = "om namah"
    print(f"\n🧪 Injecting Seed Thought: '{input_sequence}'")
    
    # 1. Encode initial sequence to get momentum
    # Use the system's pipeline to get vectors
    freq_path = system.pipeline.process_text(input_sequence)
    
    current_state = np.zeros(system.encoder.resonance_dim)
    
    # Run standard forward pass for seed
    for x in freq_path.vectors:
        # s_t = tanh(x @ W_in + s_{t-1} @ W_res + b)
        # Inputs are row vectors
        h = np.dot(x, system.encoder.W_in) + np.dot(current_state, system.encoder.W_res) + system.encoder.bias
        # Safe tanh
        current_state = np.tanh(h)

    print("🌱 Seed Processed. Entering Free Resonance (Dreaming)...")
    
    # 2. Free Resonance (Dreaming)
    # We remove input (x=0) or use the 'predicted' x (closed loop).
    # Closed Loop: s_t -> predict x -> feed back x -> s_{t+1}
    # This is "Thinking".
    
    steps = 50
    energies = []
    velocities = []
    
    # Get codebook for energy calc
    codebook = list(system.pipeline.frequency_mapper.phoneme_vectors.values())
    
    prev_state = current_state.copy()
    
    print(f"\n{'Step':<5} | {'Phoneme':<10} | {'Energy (Dist)':<15} | {'Velocity':<15}")
    print("-" * 55)

    for t in range(steps):
        # A. Decode current thought (Readout)
        # In a full system, this would use the robust "Snapping"
        # For raw physics tracking, we look at the raw state
        
        # Predict x from s using the encoder's predict method (if available) or manual W_out
        # ResonancePathEncoder doesn't have predict_phoneme publicly exposed easily like this, 
        # but we can implement it here.
        
        # x_pred = arctanh(s) @ W_out
        safe_state = np.clip(current_state, -1.0 + 1e-9, 1.0 - 1e-9)
        h_state = np.arctanh(safe_state)
        x_raw = np.dot(h_state, system.encoder.W_out)
        
        # Snap to nearest phoneme
        min_dist = float('inf')
        predicted_phoneme = "?"
        target_vec = None
        
        for p, vec in system.pipeline.frequency_mapper.phoneme_vectors.items():
            dist = np.linalg.norm(x_raw - vec)
            if dist < min_dist:
                min_dist = dist
                predicted_phoneme = p
                target_vec = vec
        
        # Energy = Distance to snapped phoneme (how clear is the thought?)
        energy = min_dist
        energies.append(energy)
        
        # C. Velocity Calculation
        velocity = calculate_velocity(current_state, prev_state)
        velocities.append(velocity)
        
        print(f"{t:<5} | {predicted_phoneme:<10} | {energy:<15.6f} | {velocity:<15.6f}")
        
        # D. Closed Loop Dynamics (The "Next Thought")
        # The system "hears" its own thought and updates state
        # x_input = target_vec (The Snapped/Clean concept)
        
        if target_vec is not None:
            # Update State: s_{t+1} = tanh(x_snapped @ W_in + s_t @ W_res)
            # This is the "Cycle Consistency" loop in action
            h_next = np.dot(target_vec, system.encoder.W_in) + np.dot(current_state, system.encoder.W_res) + system.encoder.bias
            
            # Update prev
            prev_state = current_state.copy()
            current_state = np.tanh(h_next)
        else:
            # If we can't snap, the dream collapses or drifts
            print("   (Dream drift - no clear phoneme)")
            break
            
    # Summary
    print("\n📊 Analysis:")
    # Check trend
    start_energy = np.mean(energies[:5])
    end_energy = np.mean(energies[-5:])
    
    if end_energy < start_energy:
        print(f"✅ Energy Decreased ({start_energy:.4f} -> {end_energy:.4f}) (Optimization occurred)")
    else:
        print(f"⚠️ Energy Fluctuated ({start_energy:.4f} -> {end_energy:.4f})")
        
    if velocities[-1] < 0.1:
        print(f"✅ Velocity Low ({velocities[-1]:.4f}) (Converging)")
    else:
        print(f"⚠️ System kept moving (Velocity {velocities[-1]:.4f}) - Cyclic or Chaotic")

if __name__ == "__main__":
    verify_descent()
