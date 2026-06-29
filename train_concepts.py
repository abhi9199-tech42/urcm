import numpy as np
import os
from urcm.core.safe_serialization import safe_load
import sys

def train_concepts():
    print("🧠 Phase 3: Concept Dynamics Training (Layer 2)")
    print("=============================================")
    
    # 1. Load Brain (Concepts)
    brain_path = "urcm_identity.pkl"
    if not os.path.exists(brain_path):
        print(f"❌ Brain file '{brain_path}' not found.")
        return
        
    print(f"1. Loading Brain from {brain_path}...")
    brain = safe_load(brain_path)
    if brain is None:
        print(f"❌ Failed to load {brain_path}")
        return
        
    if "concept_map" not in brain:
        print("❌ No concept_map found in brain.")
        return
        
    concept_map = brain["concept_map"]
    concepts = list(concept_map.keys())
    print(f"   Found {len(concepts)} concepts.")
    
    # 2. Prepare Training Data
    # X = Concept Vectors
    # Y = Target Vectors (Self-Stability)
    
    print("2. Preparing Training Data...")
    X_list = []
    
    # Check dim
    first_vec = concept_map[concepts[0]]
    dim = first_vec.shape[0]
    print(f"   Vector Dimension: {dim}")
    
    # Collect all vectors
    for word, vec in concept_map.items():
        X_list.append(vec)
        
    X = np.array(X_list) # (N, Dim)
    
    # Target: Stability (Identity Mapping)
    # We want s_{t+1} = tanh(s_t @ W) approx s_t
    # So s_t @ W = arctanh(s_t)
    # But vectors might be > 1.0 or < -1.0? 
    # Let's check stats.
    print(f"   Stats: Min={X.min():.4f}, Max={X.max():.4f}, Mean={X.mean():.4f}")
    
    # If vectors are from L2, they are output of tanh, so they should be in (-1, 1).
    # If they are saturated (near 1/-1), arctanh will explode.
    # We clip them for the target calculation.
    
    safe_X = np.clip(X, -0.999, 0.999)
    Y_target = np.arctanh(safe_X)
    
    # 3. Ridge Regression
    # X @ W = Y
    # W = (X.T @ X + lambda*I)^-1 @ X.T @ Y
    
    print("3. Solving for W_res (Ridge Regression)...")
    lambda_reg = 0.1 # Regularization
    
    # Add noise for robustness?
    # X_train = X + noise
    # Let's duplicate data with noise to force wider basins
    
    noise_level = 0.05
    X_noisy = np.tanh(np.arctanh(safe_X) + np.random.normal(0, noise_level, X.shape))
    
    # Combine Clean + Noisy
    X_train = np.vstack([X, X_noisy])
    Y_train = np.vstack([Y_target, Y_target])
    
    print(f"   Training samples: {X_train.shape[0]}")
    
    A = np.dot(X_train.T, X_train) + lambda_reg * np.eye(dim)
    B = np.dot(X_train.T, Y_train)
    
    W_res = np.linalg.solve(A, B)
    
    # 4. Check Stability
    eigvals = np.linalg.eigvals(W_res)
    rho = np.max(np.abs(eigvals))
    print(f"   Spectral Radius: {rho:.4f}")
    
    if rho > 1.0:
        print("   ⚠️  System is expansive (rho > 1). Scaling to 0.99 for stability.")
        W_res = W_res / rho * 0.99
    
    # 5. Save to Weights File
    weight_path = "urcm_weights.pkl"
    weights = {}
    
    if os.path.exists(weight_path):
        with open(weight_path, "rb") as f:
            try:
                from urcm.core.safe_serialization import safe_load
                loaded = safe_load(weight_path)
                if loaded is not None:
                    weights = loaded
            except Exception:
                pass
                
    weights["l2_W_res"] = W_res
    print(f"4. Saving to {weight_path}...")
    
    with open(weight_path, "wb") as f:
        pickle.dump(weights, f)
        
    print("✅ Concept Dynamics Trained Successfully.")

if __name__ == "__main__":
    train_concepts()
