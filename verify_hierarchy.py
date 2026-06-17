import numpy as np
from urcm.core.hierarchical_encoder import HierarchicalEncoder

def test_hierarchy():
    print("🏗️  Initializing Hierarchical Encoder...")
    encoder = HierarchicalEncoder()
    
    # 1. Create Synthetic Inputs
    # "ama" = [vec_a, vec_m, vec_a]
    # We'll just generate random stable vectors to simulate phonemes
    np.random.seed(42)
    vec_a = np.random.normal(0, 1, (24,))
    vec_m = np.random.normal(0, 1, (24,))
    vec_i = np.random.normal(0, 1, (24,))
    
    # Sequence 1: "ama"
    seq_ama = np.array([vec_a, vec_m, vec_a])
    
    # Sequence 2: "ami"
    seq_ami = np.array([vec_a, vec_m, vec_i])
    
    print("\n🧪 Test 1: Determinism")
    state1, _ = encoder.encode_concept(seq_ama)
    state2, _ = encoder.encode_concept(seq_ama)
    
    diff = np.linalg.norm(state1 - state2)
    print(f"   Diff between identical runs: {diff:.6f}")
    if diff < 1e-9:
        print("   ✅ Determinism PASS")
    else:
        print("   ❌ Determinism FAIL")
        
    print("\n🧪 Test 2: Differentiation")
    state_ama, traj_ama = encoder.encode_concept(seq_ama)
    state_ami, traj_ami = encoder.encode_concept(seq_ami)
    
    dist = np.linalg.norm(state_ama - state_ami)
    print(f"   L2 Distance between 'ama' and 'ami': {dist:.6f}")
    
    # Check L1 trajectory difference
    # Last step of L1 trajectory should be different because inputs were different at last step
    l1_diff = np.linalg.norm(traj_ama[-1] - traj_ami[-1])
    print(f"   L1 Final State Diff: {l1_diff:.6f}")
    
    if dist > 0.1:
        print("   ✅ Differentiation PASS (Concepts are distinct)")
    else:
        print("   ❌ Differentiation FAIL (Concepts are too similar)")
        
    print("\n📊 Dimensions Check:")
    print(f"   L1 Output (Trajectory): {traj_ama.shape} (Expected T x 64)")
    print(f"   L2 Output (Concept):    {state_ama.shape} (Expected 128)")

if __name__ == "__main__":
    test_hierarchy()
