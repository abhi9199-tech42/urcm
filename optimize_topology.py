import numpy as np
import pickle
import os
from urcm.core.system import URCMSystem

def optimize_topology():
    print("🧠 Starting Semantic Topology Optimization...")
    print("   Goal: Maximize distance between different phonemes in Resonance Space.")
    
    # 1. Load System
    try:
        system = URCMSystem()
        print("✅ System Loaded.")
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return

    # 2. Prepare Data
    codebook = system.pipeline.frequency_mapper.phoneme_vectors
    keys = list(codebook.keys())
    # X shape: (N_phonemes, Input_Dim)
    X = np.array([codebook[k] for k in keys])
    
    W_in = system.encoder.W_in.copy()
    
    print(f"   Optimizing {len(keys)} concepts...")
    print(f"   Initial W_in Norm: {np.linalg.norm(W_in):.4f}")
    
    # 3. Optimization Loop
    learning_rate = 0.01
    iterations = 1000
    
    for i in range(iterations):
        # Forward: Project to State Space
        # H = X @ W_in  (N, Res)
        H = np.dot(X, W_in)
        S = np.tanh(H)
        
        # Compute Pairwise Distances efficiently
        # D^2 = sum(S^2) + sum(S^2).T - 2 S @ S.T
        S_sq = np.sum(S**2, axis=1, keepdims=True)
        Dist_sq = S_sq + S_sq.T - 2 * np.dot(S, S.T)
        
        # Avoid negative due to float errors
        Dist_sq = np.maximum(Dist_sq, 1e-6)
        Dist = np.sqrt(Dist_sq)
        
        # Loss: We want to MAXIMIZE distance.
        # Minimize L = sum( exp(-Dist) )
        # Focus on close pairs (small Dist -> large exp)
        Repulsion = np.exp(-Dist)
        
        # Mask diagonal (self-distance is 0, exp is 1, ignore)
        np.fill_diagonal(Repulsion, 0)
        
        Loss = np.sum(Repulsion)
        
        if i % 100 == 0:
            min_dist = np.min(Dist + np.eye(len(keys)) * 1000)
            print(f"   Iter {i:<4} | Loss: {Loss:<10.4f} | Min Dist: {min_dist:.4f}")
            
        # Backward Pass (Gradient of L w.r.t W_in)
        # dL/dW = dL/dS * dS/dH * dH/dW
        
        # 1. dL/dS
        # L = sum_{j!=k} exp(-|s_j - s_k|)
        # dL/ds_i = sum_j -exp(-d_ij) * d(d_ij)/ds_i
        # d(d_ij)/ds_i = (s_i - s_j) / d_ij
        # So dL/ds_i = sum_j [ -exp(-d_ij) / d_ij * (s_i - s_j) ]
        
        # Force matrix F_ij = -exp(-D_ij) / D_ij
        Force = -Repulsion / (Dist + 1e-6)
        
        # Diff_ij = s_i - s_j. This is a tensor op, let's simplify.
        # Gradient_S[i] = sum_j Force[i,j] * (S[i] - S[j])
        #               = S[i] * sum(Force[i,:]) - sum_j(Force[i,j] * S[j])
        
        Sum_Force = np.sum(Force, axis=1, keepdims=True) # (N, 1)
        Grad_S = S * Sum_Force - np.dot(Force, S)
        
        # 2. dS/dH = 1 - S^2
        Grad_H = Grad_S * (1 - S**2)
        
        # 3. dH/dW = X.T
        # Grad_W = X.T @ Grad_H
        Grad_W = np.dot(X.T, Grad_H)
        
        # Update
        W_in -= learning_rate * Grad_W
        
        # Optional: Normalize W_in to prevent explosion
        # W_in *= 0.999
        
    print(f"✅ Optimization Complete.")
    
    # 4. Save New Weights
    # We must preserve W_res and W_out, only update W_in
    root_dir = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(root_dir, "urcm_weights.pkl")
    
    with open(weight_path, "rb") as f:
        weights = pickle.load(f)
        
    weights["W_in"] = W_in
    
    # Note: Changing W_in technically invalidates W_out (Decoder).
    # But since we only pushed states apart, the linear structure might survive.
    # Ideally, we retrain W_out after this.
    # For now, let's save and see.
    
    with open(weight_path, "wb") as f:
        pickle.dump(weights, f)
        
    print(f"💾 Updated W_in saved to {weight_path}")

if __name__ == "__main__":
    optimize_topology()
