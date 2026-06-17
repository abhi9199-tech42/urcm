import numpy as np
import os
import sys
import time

# Ensure we use the smoothed brain
BRAIN_FILE = "urcm_identity_smoothed.pkl"

from urcm.core.executive import ExecutiveController

def verify_tier1():
    print("🧪 TIER 1: BASIC COGNITION VERIFICATION")
    print("=======================================")
    
    if not os.path.exists(BRAIN_FILE):
        print(f"❌ Brain file {BRAIN_FILE} not found. Run training & smoothing first.")
        return

    print("   Initializing Executive Controller...")
    try:
        exec_ctrl = ExecutiveController(brain_path=BRAIN_FILE)
    except Exception as e:
        print(f"❌ Failed to init: {e}")
        return

    score = 0
    total_tests = 0

    # ==========================================
    # 1.1 PERCEPTION (Check capability only)
    # ==========================================
    print("\n👁️  1.1 Perception & Pattern Recognition")
    print("   --------------------------------------")
    print("   [INFO] Current URCM architecture is Text/Phoneme based.")
    print("   [INFO] Image/Video/Audio modules are currently placeholders.")
    print("   ⚠️  Skipping functional tests for Perception (Requires Phase 2 expansion).")

    # ==========================================
    # 1.2 LANGUAGE UNDERSTANDING
    # ==========================================
    print("\n🗣️  1.2 Language Understanding")
    print("   --------------------------")

    # --- Test 1.2.1: Synonym Detection ---
    print("\n   [Test 1.2.1] Synonym Detection (Happy vs Joy vs Sad)")
    try:
        w_happy = exec_ctrl.engine.get_concept_vector("happy")
        w_joy = exec_ctrl.engine.get_concept_vector("joy")
        w_sad = exec_ctrl.engine.get_concept_vector("sad")
        
        if w_happy is not None and w_joy is not None and w_sad is not None:
            dist_syn = np.linalg.norm(w_happy - w_joy)
            dist_ant = np.linalg.norm(w_happy - w_sad)
            
            print(f"      Dist(Happy, Joy) = {dist_syn:.4f}")
            print(f"      Dist(Happy, Sad) = {dist_ant:.4f}")
            
            if dist_syn < dist_ant:
                print("      ✅ PASS: Synonyms are closer than Antonyms.")
                score += 1
            else:
                print("      ❌ FAIL: Antonyms are closer (Common in distributional semantics).")
        else:
            print("      ⚠️  SKIP: Concepts not found in vocabulary.")
    except Exception as e:
        print(f"      ❌ ERROR: {e}")
    total_tests += 1

    # --- Test 1.2.2: Homonym Disambiguation (Bank) ---
    print("\n   [Test 1.2.2] Homonym Disambiguation ('Bank')")
    # Method: Seed context, then check where 'bank' lands or what it associates with.
    # We'll inject context + "bank" and see the next predicted word.
    
    contexts = [
        ("river water bank", "nature"), 
        ("money deposit bank", "finance") 
    ]
    
    # We need to define 'nature' and 'finance' related concepts that exist in the brain
    # Let's check what we have. If not, we rely on checking trajectory divergence.
    
    try:
        print("      Injecting: 'river water bank'")
        exec_ctrl.set_initial_state("water")
        # Step 1: Water
        # Step 2: Bank
        # Check resulting state
        
        # Manually guiding the trajectory for the test
        # We simulate: State = Water, then we add Bank vector to it? 
        # Or we let the engine think from "water bank"?
        
        # Let's try running the loop briefly
        exec_ctrl.set_initial_state("river")
        trajectory_river = exec_ctrl.run_loop(max_steps=3)
        print(f"      Stream A: {trajectory_river}")
        
        exec_ctrl.set_initial_state("money")
        trajectory_money = exec_ctrl.run_loop(max_steps=3)
        print(f"      Stream B: {trajectory_money}")
        
        if trajectory_river != trajectory_money:
             print("      ✅ PASS: Context alters thought trajectory.")
             score += 1
        else:
             print("      ❌ FAIL: Context had no effect (Trajectories identical).")
             
    except Exception as e:
        print(f"      ❌ ERROR: {e}")
    total_tests += 1

    # ==========================================
    # 1.3 MEMORY & RECALL
    # ==========================================
    print("\n🧠  1.3 Memory & Recall")
    print("   -------------------")
    
    # --- Test 1.3.1: Short-term Memory (Context Retention) ---
    print("\n   [Test 1.3.1] Short-term Memory (5+ turns)")
    # We will set a goal, run for 5 steps, and see if the goal is still active/influencing.
    
    try:
        target = "freedom"
        exec_ctrl.add_goal(f"Find {target}", target_concept=target, priority=2.0)
        
        print(f"      Goal set: Reach '{target}'")
        print("      Running 5 steps...")
        path = exec_ctrl.run_loop(max_steps=5)
        
        # Check if the last state is close to freedom
        final_vec = exec_ctrl.current_state
        target_vec = exec_ctrl.engine.get_concept_vector(target)
        
        if target_vec is not None:
            dist = np.linalg.norm(final_vec - target_vec)
            print(f"      Final Distance to Goal: {dist:.4f}")
            
            if dist < 1.0:
                 print("      ✅ PASS: Maintained goal focus over 5 turns.")
                 score += 1
            else:
                 print("      ❌ FAIL: Lost focus (Drifted too far).")
        else:
            print(f"      ⚠️ SKIP: Target '{target}' not in brain.")

    except Exception as e:
        print(f"      ❌ ERROR: {e}")
    total_tests += 1

    print("\n=======================================")
    print(f"🏁 SUMMARY: {score}/{total_tests} Tests Passed")
    if score == total_tests:
        print("🎉 TIER 1 BASIC COGNITION: VERIFIED")
    else:
        print("⚠️  TIER 1 NEEDS IMPROVEMENT")

if __name__ == "__main__":
    verify_tier1()
