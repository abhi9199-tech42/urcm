
import pickle
import os
import numpy as np

def fix_weights():
    print("🔧 Inspecting and Fixing Weight Files...")
    
    weights_path = "urcm_weights.pkl"
    brain_path = "urcm_identity.pkl"
    
    weights = {}
    
    # 1. Load existing weights if available
    if os.path.exists(weights_path):
        try:
            with open(weights_path, "rb") as f:
                weights = pickle.load(f)
            print(f"📄 Current urcm_weights.pkl keys: {list(weights.keys())}")
        except Exception as e:
            print(f"❌ Error reading weights file: {e}")
            weights = {}
            
    # 2. Load brain to get authoritative weights
    if os.path.exists(brain_path):
        print(f"🧠 Loading {brain_path}...")
        try:
            with open(brain_path, "rb") as f:
                brain = pickle.load(f)
            print(f"   Brain keys: {list(brain.keys())}")
            
            # Extract L2 Weights (Critical for Concepts)
            if "l2_W_res" in brain:
                weights["l2_W_res"] = brain["l2_W_res"]
                print(f"   ✅ Extracted l2_W_res: {brain['l2_W_res'].shape}")
            
            if "l2_W_in" in brain:
                weights["l2_W_in"] = brain["l2_W_in"]
                print(f"   ✅ Extracted l2_W_in: {brain['l2_W_in'].shape}")
                
            # Extract L1 Weights (Phonemes)
            if "l1_W_res" in brain:
                weights["l1_W_res"] = brain["l1_W_res"]
                print(f"   ✅ Extracted l1_W_res: {brain['l1_W_res'].shape}")
            
            if "l1_W_in" in brain:
                weights["l1_W_in"] = brain["l1_W_in"]
                print(f"   ✅ Extracted l1_W_in: {brain['l1_W_in'].shape}")
                
            # If l1 weights are missing in weights file, try to find them or init random
            # (Brain usually only stores Concept Map and L2 weights, not L1 if not saved explicitly)
            
        except Exception as e:
            print(f"❌ Error reading brain file: {e}")
    else:
        print("⚠️ No brain file found (urcm_identity.pkl).")

    # 3. Ensure all keys exist (Initialize if missing)
    # L1 (Phoneme Level) - 24 -> 64
    if "l1_W_res" not in weights:
        print("   ⚠️ Missing l1_W_res, initializing orthogonal...")
        H = np.random.randn(64, 64)
        Q, R = np.linalg.qr(H)
        weights["l1_W_res"] = Q * 0.95
        
    if "l1_W_in" not in weights:
        print("   ⚠️ Missing l1_W_in, initializing random...")
        weights["l1_W_in"] = np.random.normal(0, 0.1, (24, 64))

    # L2 (Concept Level) - 64 -> 128 (or dynamic)
    # We must match the dimension of l2_W_res if it exists
    l2_dim = 128
    if "l2_W_res" in weights:
        l2_dim = weights["l2_W_res"].shape[0]
    else:
        print(f"   ⚠️ Missing l2_W_res, initializing orthogonal ({l2_dim}x{l2_dim})...")
        H = np.random.randn(l2_dim, l2_dim)
        Q, R = np.linalg.qr(H)
        weights["l2_W_res"] = Q * 0.95
        
    if "l2_W_in" not in weights:
        print(f"   ⚠️ Missing l2_W_in, initializing random (64x{l2_dim})...")
        weights["l2_W_in"] = np.random.normal(0, 0.1, (64, l2_dim))

    # 4. Save fixed weights
    try:
        with open(weights_path, "wb") as f:
            pickle.dump(weights, f)
        print(f"✅ Saved fixed weights to {weights_path}")
        print(f"   Keys: {list(weights.keys())}")
    except Exception as e:
        print(f"❌ Error saving weights: {e}")

if __name__ == "__main__":
    fix_weights()
