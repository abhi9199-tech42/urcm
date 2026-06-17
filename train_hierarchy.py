import numpy as np
from urcm.core.hierarchical_encoder import HierarchicalEncoder
import pickle

def train_hierarchy():
    print("🧠 Phase 3: Training Hierarchical Concepts (Words)...")
    
    # 1. Initialize
    encoder = HierarchicalEncoder(l1_input_dim=24, l1_res_dim=64, l2_res_dim=128)
    
    # 2. Define Vocabulary (Words as sequences of phoneme vectors)
    # We simulate phoneme vectors with random stable patterns
    np.random.seed(42)
    vocab_size = 5
    seq_length = 4
    
    # Generate unique phoneme prototypes
    phonemes = {
        'a': np.random.normal(0, 1, 24),
        'i': np.random.normal(0, 1, 24),
        'u': np.random.normal(0, 1, 24),
        'k': np.random.normal(0, 1, 24),
        'm': np.random.normal(0, 1, 24)
    }
    
    words = {
        "ama": [phonemes['a'], phonemes['m'], phonemes['a']],
        "ami": [phonemes['a'], phonemes['m'], phonemes['i']],
        "kuku": [phonemes['k'], phonemes['u'], phonemes['k'], phonemes['u']],
        "miku": [phonemes['m'], phonemes['i'], phonemes['k'], phonemes['u']],
        "kai": [phonemes['k'], phonemes['a'], phonemes['i']]
    }
    
    print(f"📚 Vocabulary: {list(words.keys())}")
    
    # 3. Generate L1 Trajectories (Inputs for L2)
    l2_training_data = []
    l2_targets = {} # We want the final state to be the attractor
    
    print("   Generating Layer 1 Trajectories...")
    for word, seq in words.items():
        seq_array = np.array(seq)
        _, l1_traj = encoder.encode_concept(seq_array)
        l2_training_data.append(l1_traj)
        
        # Target: We want the system to settle into a stable state for this word.
        # Ideally, we let the reservoir decide the state, then reinforce it.
        # Or we force it to a specific code.
        # Let's use the "Self-Organization" approach:
        # Run the reservoir, take the final state, and make it an attractor.
        final_state_raw = encoder.layer2.encode_path(l1_traj)
        l2_targets[word] = final_state_raw
        
    # 4. Train Layer 2 Attractors (W_res)
    # Update W_res so that: tanh(target @ W_res) ~= target
    # This makes the target a Fixed Point.
    
    print("   Training Layer 2 Attractors (W_res)...")
    lr = 0.01
    epochs = 1000
    
    # We only train W_res of layer 2
    W_res = encoder.layer2.W_res
    
    for epoch in range(epochs):
        total_error = 0
        for word, target_state in l2_targets.items():
            # Current mapping: s_next = tanh(s @ W_res)
            # We want s_next == s (Fixed Point)
            
            # Forward
            arg = np.dot(target_state, W_res)
            pred_next = np.tanh(arg)
            
            # Error (We want pred_next to be target_state)
            error = target_state - pred_next
            total_error += np.mean(error**2)
            
            # Backprop (Approximate for W_res)
            # dE/dW = error * d(tanh)/dx * x
            dtanh = 1 - pred_next**2
            grad = np.outer(target_state, error * dtanh)
            
            W_res += lr * grad
            
            # Enforce constraints (Spectral Radius / Orthogonality if needed)
            # For now, just basic Hebbian-like update
            
        if epoch % 100 == 0:
            print(f"      Epoch {epoch}: Mean MSE = {total_error/len(words):.6f}")
            
    encoder.layer2.W_res = W_res
    print("✅ Training Complete.")
    
    # 5. Verify Stability
    print("\n🔍 Verifying Word Stability in Layer 2...")
    for word, target in l2_targets.items():
        # Run one step of dynamics
        arg = np.dot(target, encoder.layer2.W_res)
        next_s = np.tanh(arg)
        dist = np.linalg.norm(next_s - target)
        print(f"   Word '{word}': Drift = {dist:.6f} {'✅ Stable' if dist < 0.1 else '⚠️ Drifting'}")
        
    # Save
    encoder.save_hierarchy("urcm_hierarchy_trained.pkl")

if __name__ == "__main__":
    train_hierarchy()
