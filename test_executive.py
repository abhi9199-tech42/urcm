from urcm.core.executive import ExecutiveController

def test_executive_memory():
    print("\n--- Testing Executive Controller (Left Brain) ---")
    
    try:
        exec_ctrl = ExecutiveController()
    except Exception as e:
        print(f"Skipping: {e}")
        return

    # 1. Initialize Thought
    exec_ctrl.set_initial_state("confusion")
    
    # 2. Add Goals (Stack: Bottom -> Top)
    # Goal A: Long term - Reach 'Truth'
    exec_ctrl.add_goal("Find Ultimate Truth", target_concept="truth [satya]", priority=1.0)
    
    # Goal B: Short term - Reach 'Knowledge' first
    exec_ctrl.add_goal("Acquire Knowledge", target_concept="knowledge", priority=2.0)
    
    print("\n[Test] Goals Added. Stack should be: Truth -> Knowledge (Top)")
    
    # 3. Run
    trajectory = exec_ctrl.run_loop(max_steps=15)
    
    print(f"\nFinal Trajectory: {trajectory}")
    
    # Validation
    # Did we hit Knowledge?
    if "knowledge" in trajectory:
        print("✅ Hit Sub-goal 'Knowledge'")
    else:
        print("❌ Missed Sub-goal 'Knowledge'")
        
    # Did we hit Truth?
    # Note: decode() might return 'truth' or 'truth [satya]' depending on the map key/synonyms.
    # We check for both or partial match.
    hit_truth = False
    for t in trajectory:
        if "truth" in t.lower():
            hit_truth = True
            break
            
    if hit_truth:
        print("✅ Hit Main Goal 'Truth'")
    else:
        print("❌ Missed Main Goal 'Truth'")

if __name__ == "__main__":
    test_executive_memory()
