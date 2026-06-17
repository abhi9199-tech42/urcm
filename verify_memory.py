import numpy as np
from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.memory import GeometricMemory

def verify_one_shot_learning():
    print("🧠 Verifying One-Shot Geometric Memory Deposition...")
    
    # 1. Initialize
    dim = 64
    encoder = ResonancePathEncoder(input_dim=24, resonance_dim=dim)
    memory = GeometricMemory(resonance_dim=dim)
    
    # 2. Generate a random "Concept" (State Vector)
    # This represents a new word or idea we want to teach it.
    concept_state = np.random.normal(0, 0.5, (dim,))
    concept_state = np.tanh(concept_state) # Normalize to valid range
    
    # 3. Test Before Learning
    print("   Testing before learning...")
    # Run one step of dynamics
    next_state_pre = np.tanh(np.dot(concept_state, encoder.W_res) + encoder.bias)
    drift_pre = np.linalg.norm(next_state_pre - concept_state)
    print(f"   Pre-Learning Drift: {drift_pre:.4f} (Should be high)")
    
    # 4. ONE-SHOT LEARN
    print("   ⚡ Depositing Memory (One-Shot)...")
    encoder.W_res = memory.deposit_attractor(encoder.W_res, concept_state)
    
    # 5. Test After Learning
    print("   Testing after learning...")
    next_state_post = np.tanh(np.dot(concept_state, encoder.W_res) + encoder.bias)
    drift_post = np.linalg.norm(next_state_post - concept_state)
    print(f"   Post-Learning Drift: {drift_post:.4f}")
    
    if drift_post < 0.1:
        print("   ✅ SUCCESS: Concept stabilized instantly.")
    else:
        print("   ❌ FAILURE: Concept still drifting.")

    # 6. Verify it didn't break the matrix (Spectral Check)
    # Approximate spectral radius via power iteration
    v = np.random.randn(dim)
    for _ in range(10):
        v = np.dot(v, encoder.W_res)
        v = v / np.linalg.norm(v)
    spectral_radius = np.linalg.norm(np.dot(v, encoder.W_res))
    print(f"   Spectral Radius: {spectral_radius:.4f}")

if __name__ == "__main__":
    verify_one_shot_learning()