import numpy as np
from urcm.core.system import URCMSystem

def check_mapping():
    print("🔍 Checking 'm' -> 'ḍ' Confusion...")
    system = URCMSystem()
    
    vec_m = system.pipeline.frequency_mapper.phoneme_vectors['m']
    vec_d = system.pipeline.frequency_mapper.phoneme_vectors['ḍ']
    
    # 1. Forward Pass (Encode)
    # s = tanh(x @ W_in)
    h_m = np.dot(vec_m, system.encoder.W_in)
    s_m = np.tanh(h_m)
    
    # 2. Backward Pass (Decode)
    # x_hat = arctanh(s) @ W_out
    safe_s = np.clip(s_m, -1.0+1e-9, 1.0-1e-9)
    h_back = np.arctanh(safe_s)
    x_hat = np.dot(h_back, system.encoder.W_out)
    
    # 3. Check Distances
    dist_m = np.linalg.norm(x_hat - vec_m)
    dist_d = np.linalg.norm(x_hat - vec_d)
    
    print(f"Original 'm' -> System -> Output")
    print(f"  Distance to 'm': {dist_m:.6f}")
    print(f"  Distance to 'ḍ': {dist_d:.6f}")
    
    if dist_d < dist_m:
        print("❌ CRITICAL FAULT: The system inherently thinks 'm' is 'ḍ'.")
        print("   W_out is not the inverse of W_in for this vector.")
    else:
        print("✅ Mapping is correct (m is closer). The Gradient Descent might be taking a wrong turn.")

if __name__ == "__main__":
    check_mapping()
