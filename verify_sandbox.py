"""
URCM Verification: Sandbox Knowledge & Reasoning.
Compares URCM's newly acquired knowledge against a simulated 'Standard LLM' baseline.
"""

from urcm.core.reasoning import ReasoningEngine
import numpy as np
import time

def verify():
    print("==========================================")
    print("URCM SANDBOX VERIFICATION")
    print("Loading Brain from 'urcm_identity.pkl'...")
    print("==========================================")
    
    try:
        engine = ReasoningEngine()
    except FileNotFoundError:
        print("❌ Error: Brain file not found. Run 'explore.py' first.")
        return

    print(f"✅ Brain Loaded. Vocabulary Size: {len(engine.concept_map)}")
    
    # Test Cases: (Query, Constraint, Description)
    tests = [
        {
            "query": "DeepSeek",
            "constraint": [], 
            "desc": "Recall: Who/What is DeepSeek?",
            "llm_baseline": "DeepSeek is an AI company/model developed in China..."
        },
        {
            "query": "AlexNet",
            "constraint": [],
            "desc": "Recall: What is AlexNet?",
            "llm_baseline": "AlexNet is a convolutional neural network designed by Alex Krizhevsky..."
        },
        {
            "query": "Reinforcement",
            "constraint": [("reward", -0.5)], # Seek Reward
            "desc": "Inference: Reinforcement + Seek Reward",
            "llm_baseline": "Reinforcement learning involves agents taking actions to maximize cumulative reward..."
        }
    ]
    
    for test in tests:
        q = test["query"]
        print(f"\n🧪 TEST: {test['desc']}")
        print(f"   [Standard LLM]: {test['llm_baseline']}")
        
        # Check if concept exists
        if not engine.get_concept_vector(q):
            print(f"   [URCM]: ❌ Concept '{q}' NOT FOUND in memory.")
            continue
            
        # Run Reasoning
        start_time = time.time()
        trajectory = engine.solve(q, test["constraint"], steps=8)
        duration = time.time() - start_time
        
        # Format Trajectory
        traj_str = " -> ".join(trajectory)
        print(f"   [URCM]: {traj_str}")
        print(f"   (Time: {duration:.4f}s)")
        
        # Validation Logic
        if len(set(trajectory)) > 1:
            print("   ✅ Result: DYNAMIC (Trajectory evolved)")
        else:
            print("   ⚠️ Result: STATIC (Fixed Point / Memory)")

if __name__ == "__main__":
    verify()
