import sys
from urcm.core.reasoning import ReasoningEngine

def main():
    print("==========================================")
    print("URCM PHASE 6: COGNITIVE REASONING TEST")
    print("==========================================")
    print("Goal: Demonstrate 'Adaptation' of strategy under constraints.")
    print("Scenario: WWII Midway (Ingested from ww2_scenarios.json)")
    
    try:
        engine = ReasoningEngine()
    except Exception as e:
        print(f"Failed to load engine: {e}")
        return

    # Debug: Print Vocabulary Size
    print(f"Vocabulary Size: {len(engine.concept_map)}")
    
    # ---------------------------------------------------------
    # TEST 0: Sandbox Verification (New Knowledge)
    # ---------------------------------------------------------
    print("\n--- Test 0: Sandbox Knowledge (DeepSeek/AlexNet) ---")
    sandbox_queries = ["deepseek", "alexnet", "reinforcement"]
    for q in sandbox_queries:
        if engine.get_concept_vector(q) is not None:
            print(f"Query: '{q}'")
            traj = engine.solve(q, [], steps=5)
            print(f"  -> Trajectory: {' -> '.join(traj)}")
        else:
            print(f"Query: '{q}' -> NOT FOUND (Ingestion needed)")

    if "midway" in engine.concept_map:
        print("Concept 'midway' exists.")
    else:
        print("Concept 'midway' MISSING.")

    # 1. BASELINE: What is the strategy for Pacific?
    # Expected: "1942", "Midway", etc.
    print("\n--- TEST 1: BASELINE RECALL (No Constraints) ---")
    query = "pacific"
    trajectory_baseline = engine.solve(query, constraints=[], steps=5)
    print(f"Result: {' -> '.join(trajectory_baseline)}")
    
    # 2. ADAPTATION: Force 'Midway' context
    # Constraint: SEEK "Midway"
    # Expected: Shift from "Avoid" path to "Midway" path (or mixed)
    print("\n--- TEST 2: ADAPTATION (Constraint: SEEK MIDWAY) ---")
    constraints = [("midway", -5.0)] # Negative = Seek (Strong)
    trajectory_adapted = engine.solve(query, constraints=constraints, steps=5)
    print(f"Result: {' -> '.join(trajectory_adapted)}")
    
    # 3. CONTRADICTION CHECK: Pacific + Rhine (Market Garden)
    # "Rhine" is in the brain (from previous runs).
    # Force it to think about Rhine while starting at Pacific.
    print("\n--- TEST 3: CONTRADICTION (Constraint: SEEK RHINE) ---")
    constraints_rhine = [("rhine", -5.0)] 
    trajectory_rhine = engine.solve(query, constraints=constraints_rhine, steps=5)
    print(f"Result: {' -> '.join(trajectory_rhine)}")

    # Analysis
    print("\n--- ANALYSIS ---")
    
    baseline_str = " -> ".join(trajectory_baseline)
    adapted_str = " -> ".join(trajectory_adapted)
    rhine_str = " -> ".join(trajectory_rhine)
    
    if trajectory_baseline != trajectory_adapted:
        print("✅ SUCCESS: Strategy adapted (Trajectory changed).")
        if "midway" in adapted_str:
             print("✅ SUCCESS: Successfully steered towards 'midway'.")
    else:
        print("❌ FAILURE: Strategy did not adapt.")
        
    if "rhine" in rhine_str:
        print("✅ SUCCESS: Successfully steered towards 'rhine' (Contradiction forced).")

if __name__ == "__main__":
    main()
