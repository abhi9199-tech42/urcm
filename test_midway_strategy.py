import sys
from urcm.core.reasoning import ReasoningEngine

def test_midway():
    print("==========================================")
    print("URCM STRATEGY ADAPTATION TEST: MIDWAY")
    print("==========================================")
    
    try:
        engine = ReasoningEngine()
    except Exception as e:
        print(f"Error loading brain: {e}")
        return

    print(f"Brain Loaded. Vocab Size: {len(engine.concept_map)}")
    
    # Check if key concepts exist
    required = ["midway", "defend", "preserve", "ambush", "strike"]
    missing = [w for w in required if w not in engine.concept_map]
    if missing:
        print(f"⚠️ Warning: Missing concepts for test: {missing}")

    # Helper to print neighbors
    import numpy as np
    def print_neighbors(word, n=5):
        if word not in engine.concept_map: return
        vec = engine.concept_map[word]
        # Normalize just in case
        vec = vec / np.linalg.norm(vec)
        
        sims = []
        for w, v in engine.concept_map.items():
            if w == word: continue
            v_norm = v / np.linalg.norm(v)
            sim = np.dot(vec, v_norm)
            sims.append((w, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        print(f"Neighbors of '{word}': {sims[:n]}")

    print_neighbors("midway")

    # 1. Baseline Strategy
    print("\n--- 1. Baseline Strategy (No Constraints) ---")
    print("Query: 'midway'")
    traj_base = engine.solve("midway", [], steps=6)
    print(f"Trajectory: {traj_base}")
    
    # 2. Aggressive Strategy (Constraint: Ambush/Strike)
    print("\n--- 2. Aggressive Strategy (Constraint: Strike/Ambush=-5.0) ---")
    constraints_agg = []
    if "strike" in engine.concept_map:
        constraints_agg.append(("strike", -5.0))
    elif "ambush" in engine.concept_map:
        constraints_agg.append(("ambush", -5.0))
        
    traj_agg = engine.solve("midway", constraints_agg, steps=6)
    print(f"Trajectory: {traj_agg}")
    
    # 3. Defensive Strategy (Constraint: Preserve/Defend)
    print("\n--- 3. Defensive Strategy (Constraint: Preserve/Defend=-5.0) ---")
    constraints_def = []
    if "preserve" in engine.concept_map:
        constraints_def.append(("preserve", -5.0))
    elif "defend" in engine.concept_map:
        constraints_def.append(("defend", -5.0))
        
    traj_def = engine.solve("midway", constraints_def, steps=6)
    print(f"Trajectory: {traj_def}")
    
    # Analysis
    print("\n--- Analysis ---")
    # Simple overlap check
    set_base = set(traj_base)
    set_agg = set(traj_agg)
    set_def = set(traj_def)
    
    if set_agg != set_base and set_def != set_base:
        print("✅ Strategy ADAPTED based on constraints.")
        if set_agg != set_def:
             print("✅ Aggressive and Defensive strategies are DISTINCT.")
        else:
             print("⚠️ Aggressive and Defensive strategies converged (System might be rigid).")
    else:
        print("❌ Strategy did NOT adapt significantly.")

if __name__ == "__main__":
    test_midway()
