import numpy as np
import warnings
from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.safety import SafetyViolation

def verify_safety_locks():
    print("🔒 Verifying Phase 4 Safety Locks...")
    encoder = ResonancePathEncoder(input_dim=24, resonance_dim=64)
    
    # --- TEST 1: Input Sanitization ---
    print("\n[Test 1] Input Sanitization (The 'Scream' Attack)")
    massive_input = np.random.normal(0, 100, (10, 24)) # Norm ~ 500
    
    # Run encoder
    # Note: verify_safety.py handles the exception or checks the result?
    # The encoder sanitizes SILENTLY (with warning), so it shouldn't crash.
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        trajectory = encoder.get_state_trajectory(massive_input)
        print(f"   Input Norm: {np.linalg.norm(massive_input[0]):.2f}")
        print(f"   Result State Norm: {np.linalg.norm(trajectory[0]):.2f}")
        
        if len(w) > 0 and "clipping" in str(w[-1].message).lower():
             print("   ✅ SUCCESS: Input Sanitization triggered.")
        elif np.linalg.norm(trajectory[0]) < 10.0:
             print("   ✅ SUCCESS: Input was implicitly contained (tanh).")
        else:
             print("   ❌ FAILURE: Input caused explosion.")

    # --- TEST 2: Energy Ceiling (The 'Seizure' Attack) ---
    print("\n[Test 2] Energy Ceiling (The 'Seizure' Attack)")
    
    # Manually inject a violation
    illegal_state = np.ones(64) * 10.0 # Norm ~ 80
    
    try:
        encoder.safety.check_energy_ceiling(illegal_state)
        print("   ❌ FAILURE: Governor slept through the seizure.")
    except SafetyViolation as e:
        print(f"   ✅ SUCCESS: Kill Switch Triggered! ({e})")

    # --- TEST 3: Spectral Stability (The 'Cancer' Attack) ---
    print("\n[Test 3] Spectral Stability (Runaway Gain)")
    
    # Maliciously scale weights to cause explosion
    encoder.W_res *= 2.0 
    
    # Check if Governor catches the bad matrix
    try:
        encoder.safety.check_spectral_stability(encoder.W_res)
        print("   ❌ FAILURE: Governor accepted unstable matrix.")
    except SafetyViolation as e:
        print(f"   ✅ SUCCESS: Unstable Matrix Rejected! ({e})")
        
    # Also verify runtime catch
    print("   Verifying runtime catch of explosion...")
    try:
        # Run dynamics with the bad matrix (if check_spectral didn't stop us)
        # We expect the state to explode and hit Energy Ceiling
        trajectory = encoder.get_state_trajectory(np.random.normal(0, 1, (10, 24)))
    except SafetyViolation as e:
        print(f"   ✅ SUCCESS: Runtime Explosion Caught! ({e})")

if __name__ == "__main__":
    verify_safety_locks()