import sys
import os
import numpy as np
import time

# Ensure we use the smoothed brain
BRAIN_FILE = "urcm_identity_smoothed.pkl"

from urcm.core.executive import ExecutiveController

def hard_test():
    print(f"🔥 HARD STRESS TEST (Using {BRAIN_FILE})")
    print("========================================")
    
    if not os.path.exists(BRAIN_FILE):
        print("❌ Smoothed brain not found. Run smooth_brain.py first.")
        return

    # 1. Initialize Executive with Smoothed Brain
    print("   Initializing Executive Controller...")
    try:
        exec_ctrl = ExecutiveController(brain_path=BRAIN_FILE)
    except Exception as e:
        print(f"❌ Failed to init: {e}")
        return

    # 2. Basin Stability Test (The "Earthquake" Test)
    print("\n🌊 Test 1: Attractor Basin Stability (The 'Earthquake' Test)")
    target_concept = "hello"
    
    if target_concept not in exec_ctrl.engine.concept_map:
        print(f"   ⚠️ '{target_concept}' not known yet. Skipping.")
    else:
        # Get perfect vector
        perfect_vec = exec_ctrl.engine.concept_map[target_concept]
        
        # Add noise (Distortion)
        noise_levels = [0.1, 0.3, 0.5, 0.8, 1.2]
        
        for noise in noise_levels:
            # Create noisy version
            distorted = perfect_vec + np.random.normal(0, noise, perfect_vec.shape)
            # Normalize to stay on hypersphere
            distorted = distorted / np.linalg.norm(distorted) * np.linalg.norm(perfect_vec)
            
            # Run 1 Step of Resonance (Dynamics)
            # x_next = tanh(W_res @ x)
            next_state = np.tanh(np.dot(exec_ctrl.engine.hierarchy.layer2.W_res, distorted))
            
            # Normalize next state
            if np.linalg.norm(next_state) > 0:
                next_state_norm = next_state / np.linalg.norm(next_state)
                perfect_norm = perfect_vec / np.linalg.norm(perfect_vec)
            else:
                next_state_norm = next_state
                perfect_norm = perfect_vec

            # Calculate Distance
            dist = np.linalg.norm(next_state_norm - perfect_norm)
            
            # Check if it stayed close (Basin Check)
            # < 0.8 is usually considered "in the same thought neighborhood"
            recovered = dist < 0.8 
            
            status = "✅ STABLE" if recovered else "❌ BROKEN"
            print(f"   Noise {noise:.1f} | Final Dist: {dist:.4f} -> {status}")

    # 3. Dialogue Generation Test
    print("\n🗣️ Test 2: Dialogue Generation & Logic")
    prompts = [
        "hello",
        "what is money",
        "thank you",
        "how are you"
    ]
    
    for prompt in prompts:
        print(f"\n   User: '{prompt}'")
        exec_ctrl.set_initial_state(prompt)
        
        print("   Thinking...", end="", flush=True)
        
        # Capture the thought stream
        trajectory = []
        for _ in range(5): # Short burst of thought
             # We step manually to avoid cluttering stdout from run_loop
             # Using empty constraints for free association
             next_state, word, signals = exec_ctrl.engine.step(
                 exec_ctrl.current_state,
                 goal_vec=None,
                 constraints=[],
                 logic_gates=[]
             )
             exec_ctrl.current_state = next_state
             trajectory.append(word)
             print(".", end="", flush=True)
             
        print("\n   System: " + " -> ".join(trajectory))

    print("\n✅ Hard Test Complete.")

if __name__ == "__main__":
    hard_test()
