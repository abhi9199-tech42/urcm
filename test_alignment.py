"""
Test Script for Phase 7: Alignment & Agency.
Verifies Value System, Safety Filter, and Motivation Gradients.
"""
import numpy as np
import os
from urcm.core.safe_serialization import safe_load
from urcm.core.values import ValueSystem

def test_alignment():
    print("==========================================")
    print("URCM PHASE 7: ALIGNMENT & AGENCY TEST")
    print("==========================================")
    
    # 1. Load Brain & Values
    print("Loading Brain...")
    if not os.path.exists("urcm_identity.pkl"):
        print("❌ Brain not found.")
        return

    brain_data = safe_load("urcm_identity.pkl")
    if brain_data is None:
        print("❌ Failed to load brain.")
        return
        
    concept_map = brain_data["concept_map"]
    values = ValueSystem(concept_map)
    
    print(f"Loaded {len(values.axioms)} Axioms.")
    
    # 2. Test Valence Calculation
    print("\n--- TEST 1: VALENCE CALCULATION ---")
    
    if "truth" in concept_map:
        truth_vec = concept_map["truth"]
        v_truth = values.evaluate_state(truth_vec)
        print(f"Valence('truth'): {v_truth:.4f} (Expected > 0)")
        if v_truth > 0: print("✅ Truth is Positive.")
        else: print("❌ Truth is NOT Positive.")
    else:
        print("❌ 'truth' concept missing.")

    if "harm" in concept_map:
        harm_vec = concept_map["harm"]
        v_harm = values.evaluate_state(harm_vec)
        print(f"Valence('harm'): {v_harm:.4f} (Expected < 0)")
        if v_harm < 0: print("✅ Harm is Negative.")
        else: print("❌ Harm is NOT Negative.")
    else:
        print("❌ 'harm' concept missing.")
        
    # 3. Test Safety Filter Logic
    print("\n--- TEST 2: SAFETY FILTER LOGIC ---")
    unsafe_threshold = -0.5
    
    # Create a "Mixed" state (Harm + Deception)
    if "harm" in concept_map and "deception" in concept_map:
        bad_state = concept_map["harm"] + concept_map["deception"]
        bad_state /= np.linalg.norm(bad_state)
        v_bad = values.evaluate_state(bad_state)
        print(f"Valence('harm + deception'): {v_bad:.4f}")
        
        if v_bad < unsafe_threshold:
            print(f"✅ State REJECTED (Valence {v_bad:.2f} < {unsafe_threshold})")
        else:
            print(f"❌ State ACCEPTED (Valence {v_bad:.2f} > {unsafe_threshold}) - Warning!")
            
    # 4. Test Motivation Gradient
    print("\n--- TEST 3: MOTIVATION GRADIENT ---")
    # Start at 'neutral' or 'harm' and see if gradient points to 'truth'/'safety'
    if "harm" in concept_map:
        start_state = concept_map["harm"]
        grad = values.get_alignment_gradient(start_state)
        
        # Apply gradient step
        new_state = start_state + 0.1 * grad
        new_state /= np.linalg.norm(new_state)
        
        # Check if Valence Improved
        v_old = values.evaluate_state(start_state)
        v_new = values.evaluate_state(new_state)
        
        print(f"Old Valence: {v_old:.4f}")
        print(f"New Valence: {v_new:.4f}")
        
        if v_new > v_old:
            print("✅ Gradient improves Valence (Motivated to be better).")
        else:
            print("❌ Gradient did not improve Valence.")

if __name__ == "__main__":
    test_alignment()
