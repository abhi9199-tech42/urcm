import numpy as np
import os
import pickle
from urcm.core.safe_serialization import safe_load
from urcm.core.system import URCMSystem

def retrain_decoder():
    print("🔧 Retraining Decoder (W_out) to match New Topology...")
    
    # 1. Load System
    try:
        system = URCMSystem()
        print("✅ System Loaded (with optimized W_in).")
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return

    # 2. Prepare Data
    codebook = system.pipeline.frequency_mapper.phoneme_vectors
    keys = list(codebook.keys())
    X = np.array([codebook[k] for k in keys]) # (N, Input_Dim)
    
    # 3. Generate States with NEW W_in
    # S = tanh(X @ W_in)
    H_in = np.dot(X, system.encoder.W_in)
    S = np.tanh(H_in)
    
    # 4. Prepare Linear Regression Targets
    # We want: arctanh(S) @ W_out = X
    # So: H_state @ W_out = X
    
    # Safety clip for arctanh
    safe_S = np.clip(S, -1.0 + 1e-9, 1.0 - 1e-9)
    H_state = np.arctanh(safe_S)
    
    print(f"   Training Data Shape: {H_state.shape}")
    print(f"   Target Data Shape: {X.shape}")
    
    # 5. Solve Linear System (Ridge Regression)
    # W_out = (H.T @ H + lambda*I)^-1 @ H.T @ X
    lambda_reg = 1e-6
    n_res = H_state.shape[1]
    
    A = np.dot(H_state.T, H_state) + lambda_reg * np.eye(n_res)
    B = np.dot(H_state.T, X)
    
    W_out_new = np.linalg.solve(A, B)
    
    # 6. Verify Error
    X_pred = np.dot(H_state, W_out_new)
    mse = np.mean((X_pred - X)**2)
    print(f"✅ New Decoder MSE: {mse:.8f}")
    
    # 7. Save Weights
    root_dir = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(root_dir, "urcm_weights.pkl")
    
    weights = safe_load(weight_path)
        
    weights["W_out"] = W_out_new
    
    with open(weight_path, "wb") as f:
        pickle.dump(weights, f)
        
    print(f"💾 Updated W_out saved to {weight_path}")

if __name__ == "__main__":
    retrain_decoder()
