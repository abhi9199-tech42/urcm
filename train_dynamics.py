import numpy as np
import pickle
import os
from urcm.core.system import URCMSystem

def train_dynamics():
    print("🧠 Phase 2: Training Attractor Dynamics (W_res)...")
    print("   Goal: Make phonemes stable 'Fixed Points' in the recurrent flow.")
    
    # 1. Load System
    try:
        system = URCMSystem()
        print("✅ System Loaded.")
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return

    # 2. Prepare Data (Attractors)
    codebook = system.pipeline.frequency_mapper.phoneme_vectors
    keys = list(codebook.keys())
    # X_in shape: (N, Input_Dim)
    X_in = np.array([codebook[k] for k in keys])
    
    # Calculate Ideal States (The Fixed Points we want)
    # s = tanh(x @ W_in)
    H_ideal = np.dot(X_in, system.encoder.W_in)
    S_ideal = np.tanh(H_ideal)
    
    # 3. Create Training Dataset for Dynamics
    # We want: s_{t+1} = tanh(s_t @ W_res) -> s_{ideal}
    # So we want: s_t @ W_res = arctanh(s_{ideal})
    # To ensure attraction, we train: Noisy_State -> Ideal_State
    
    print(f"   Generating attraction basins for {len(keys)} concepts...")
    
    noise_levels = [0.0, 0.05, 0.1, 0.15] # Curriculum of noise
    X_train_list = []
    Y_train_list = []
    
    # Pre-calculate target pre-activations
    # Clip to avoid infinity in arctanh
    safe_S_ideal = np.clip(S_ideal, -0.999, 0.999)
    H_target = np.arctanh(safe_S_ideal)
    
    # 🔍 Hard Negative Mining (Fixing Weak Concepts)
    # Based on verification failure: 'ā', 'ḷ', 'ai', 'kh'
    weak_concepts = ['ā', 'ḷ', 'ai', 'kh']
    weak_indices = [keys.index(k) for k in weak_concepts if k in keys]
    print(f"   Boosting robustness for weak concepts: {weak_concepts}")
    
    # Standard Training Only (Adversarial Mixing Removed to preserve basin depth)
    # Deep basins naturally compete; mixing creates shallow spurious valleys.

    for noise in noise_levels + [0.20]: # Conservative noise training
        # Standard Training
        noise_matrix = np.random.normal(0, noise, S_ideal.shape)
        S_noisy = np.tanh(np.arctanh(safe_S_ideal) + noise_matrix)
        
        X_train_list.append(S_noisy)
        Y_train_list.append(H_target) 

        # Boost Weak Concepts (Oversample 5x with Extra Noise)
        if noise > 0:
            for idx in weak_indices:
                # 5 extra copies with higher noise range
                for _ in range(5):
                    # Adaptive noise: 0.2 to 0.3 for robustness
                    extra_noise = np.random.normal(0, noise * 1.5, S_ideal[idx].shape)
                    s_boost = np.tanh(np.arctanh(safe_S_ideal[idx]) + extra_noise)
                    
                    X_train_list.append(s_boost.reshape(1, -1))
                    Y_train_list.append(H_target[idx].reshape(1, -1))
                    
    X_train = np.vstack(X_train_list)
    Y_train = np.vstack(Y_train_list)
    
    print(f"   Training Data: {X_train.shape} samples.")
    
    # 4. Solve Ridge Regression for W_res
    # X @ W_res = Y
    # W_res = (X.T @ X + lambda*I)^-1 @ X.T @ Y
    
    # Stronger regularization to smooth the attractor basins
    lambda_reg = 0.05 
    n_res = system.encoder.resonance_dim
    
    print("   Solving dynamics equation...")
    A = np.dot(X_train.T, X_train) + lambda_reg * np.eye(n_res)
    B = np.dot(X_train.T, Y_train)
    
    W_res_new = np.linalg.solve(A, B)
    
    # 5. Stability Analysis
    # Check Spectral Radius (Eigenvalues)
    # If > 1, system will explode. If < 1, it will die.
    # We want it around 1.0 (Criticality) or just stable attractors.
    eigvals = np.linalg.eigvals(W_res_new)
    rho = np.max(np.abs(eigvals))
    print(f"   Spectral Radius (rho): {rho:.4f}")
    
    if rho > 1.08: # Strict stability
        print("⚠️ Warning: Dynamics unstable. Scaling down to 1.08...")
        W_res_new = W_res_new / rho * 1.08
        print(f"   New rho: {np.max(np.abs(np.linalg.eigvals(W_res_new))):.4f}")
    else:
        print("   Spectral Radius accepted.")
    
    # 6. Verify Attraction
    print("\n🔍 Verifying Attractor Dynamics...")
    test_idx = 0 # Test with first phoneme
    target_state = S_ideal[test_idx]
    
    # Start with noise
    noisy_start = np.tanh(np.arctanh(target_state * 0.9) + np.random.normal(0, 0.1, target_state.shape))
    curr = noisy_start
    
    dist_start = np.linalg.norm(curr - target_state)
    print(f"   Start Dist: {dist_start:.4f}")
    
    for t in range(10):
        # s_{t+1} = tanh(s_t @ W_res)
        # Note: No input here! Autonomous dynamics.
        curr = np.tanh(np.dot(curr, W_res_new))
        dist = np.linalg.norm(curr - target_state)
        print(f"   Step {t+1}: Dist {dist:.4f}")
        
    # 7. Save
    root_dir = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(root_dir, "urcm_weights.pkl")
    
    with open(weight_path, "rb") as f:
        weights = pickle.load(f)
        
    weights["W_res"] = W_res_new
    # Update Inverse too
    try:
        weights["W_res_inv"] = np.linalg.inv(W_res_new)
    except:
        weights["W_res_inv"] = np.linalg.pinv(W_res_new)
        
    with open(weight_path, "wb") as f:
        pickle.dump(weights, f)
        
    print(f"\n💾 Updated W_res saved to {weight_path}")

if __name__ == "__main__":
    train_dynamics()
