import numpy as np
import os
from urcm.core.safe_serialization import safe_load
import shutil
import time

INPUT_PATH = "urcm_identity.pkl"
SNAPSHOT_PATH = "urcm_identity_snapshot.pkl"
OUTPUT_PATH = "urcm_identity_smoothed.pkl"
TARGET_RANK = 480 # Keep almost full rank (was 300)
TARGET_RADIUS = 0.99 # Force stability

def smooth_brain():
    print(f"🧠 Smoothing Brain Process Started...")
    
    max_retries = 5
    data = None
    
    # Retry loop for Copy AND Load
    for i in range(max_retries):
        try:
            # 1. Try to Copy
            shutil.copy2(INPUT_PATH, SNAPSHOT_PATH)
            
            # 2. Try to Load immediately to verify integrity
            print(f"   Attempt {i+1}: Loading snapshot...")
            data = safe_load(SNAPSHOT_PATH)
                
            print("   ✅ Snapshot verified.")
            break # Success
            
        except (EOFError, pickle.UnpicklingError, PermissionError) as e:
            print(f"   ⚠️ Attempt {i+1} failed (File contention?): {e}")
            time.sleep(2) # Wait longer for write to finish
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            time.sleep(1)

    if data is None:
        print("❌ Failed to acquire valid brain snapshot after retries. Aborting.")
        return

    try:
        W = data["l2_W_res"]
        print(f"   Original Shape: {W.shape}")
        
        # SVD
        print("   Running SVD Decomp...")
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        
        print(f"   Original Energy (Sum S): {np.sum(S):.2f}")
        
        # Truncate
        S_new = S.copy()
        S_new[TARGET_RANK:] = 0 # Zero out tail
        
        print(f"   Smoothed Energy (Sum S): {np.sum(S_new):.2f} ({(np.sum(S_new)/np.sum(S))*100:.1f}%)")
        
        # Reconstruct
        W_clean = U @ np.diag(S_new) @ Vt
        
        # Spectral Normalization
        print("   Checking Spectral Radius...")
        evals = np.linalg.eigvals(W_clean)
        max_eig = np.max(np.abs(evals))
        print(f"   Current Spectral Radius: {max_eig:.4f}")
        
        if max_eig > TARGET_RADIUS:
            print(f"   Scaling down to {TARGET_RADIUS}...")
            scale_factor = TARGET_RADIUS / max_eig
            W_clean *= scale_factor
        
        # Update Data
        data["l2_W_res"] = W_clean
        
        with open(OUTPUT_PATH, "wb") as f:
             pickle.dump(data, f)
            
        print("✅ Smoothed Brain Saved to 'urcm_identity_smoothed.pkl'.")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")

if __name__ == "__main__":
    smooth_brain()
