from urcm.core.sanskrit_bridge import SanskritBridge
from urcm.core.reasoning import ReasoningEngine

def test_bridge():
    print("Testing Sanskrit Bridge...")
    bridge = SanskritBridge()
    
    # Test direct mapping
    terms = ["consciousness", "war", "energy", "unknown_term"]
    for t in terms:
        s = bridge.get_sanskrit(t)
        print(f"'{t}' -> {s}")
        
    # Test Reasoning Integration
    print("\nTesting Reasoning Integration...")
    try:
        engine = ReasoningEngine()
        # Mock a trajectory
        traj = ["war", "energy", "peace"]
        translated = engine.bridge.translate_trajectory(traj)
        print(f"Original: {traj}")
        print(f"Translated: {translated}")
        
        # Test Solve (if brain is loaded)
        if "war" in engine.concept_map:
            print("\nTesting Solve('war')...")
            real_traj = engine.solve("war", [], steps=2)
            print(f"Real Trajectory: {real_traj}")
        else:
            print("\n'war' not in brain, skipping solve test.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_bridge()
