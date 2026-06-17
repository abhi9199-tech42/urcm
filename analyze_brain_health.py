import pickle
import numpy as np
import os

BRAIN_PATH = "urcm_identity.pkl"

def analyze():
    if not os.path.exists(BRAIN_PATH):
        print("❌ Brain file not found.")
        return

    print(f"📊 Analyzing {BRAIN_PATH}...")
    try:
        with open(BRAIN_PATH, "rb") as f:
            data = pickle.load(f)
            
        W = data["l2_W_res"]
        print(f"   Shape: {W.shape}")
        
        # 1. Weight Stats
        print(f"   Min/Max Weight: {np.min(W):.4f} / {np.max(W):.4f}")
        print(f"   Mean/Std: {np.mean(W):.4f} / {np.std(W):.4f}")
        
        # 2. Spectral Radius (Eigenvalues)
        # Computing full eigenvalues for 512x512 is fast.
        print("   Computing Eigenvalues...")
        evals = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(evals))
        print(f"   Spectral Radius (max |lambda|): {max_eig:.4f}")
        
        # 3. Singular Values (SVD) - better for capacity check
        print("   Computing Singular Values...")
        s = np.linalg.svd(W, compute_uv=False)
        print(f"   Max Singular Value: {np.max(s):.4f}")
        print(f"   Min Singular Value: {np.min(s):.4f}")
        
        # Effective Rank (number of singular values > threshold)
        threshold = 1e-5
        rank = np.sum(s > threshold)
        print(f"   Effective Rank (s > {threshold}): {rank}/{W.shape[0]}")
        
        # Energy Distribution
        energy_90 = np.searchsorted(np.cumsum(s) / np.sum(s), 0.90)
        print(f"   90% Energy in top {energy_90} components.")
        
        if max_eig > 1.5:
             print("\n⚠️ WARNING: High Spectral Radius (>1.5). System may be chaotic/unstable.")
        elif max_eig < 0.5:
             print("\n⚠️ WARNING: Low Spectral Radius (<0.5). System may be dissipative/fading.")
        else:
             print("\n✅ Spectral Radius is in healthy range (0.5 - 1.5).")

        if rank == W.shape[0]:
             print("⚠️ Full Rank. Matrix might be saturated with noise.")
        else:
             print(f"✅ Rank is not saturated ({rank}). Capacity available.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    analyze()
