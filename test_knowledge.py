import numpy as np
import pickle
import sys
from urcm.core.hierarchical_encoder import HierarchicalEncoder

def main():
    print("TESTING KNOWLEDGE RECALL & GENERATION")
    
    # Load Brain
    with open("urcm_identity.pkl", "rb") as f:
        brain = pickle.load(f)
        
    concept_map = brain["concept_map"]
    # Reverse map for decoding
    inv_map = {tuple(v): k for k, v in concept_map.items()}
    
    # Helper to find nearest word
    def decode(vec):
        best_word = "?"
        best_dist = float('inf')
        for word, w_vec in concept_map.items():
            dist = np.linalg.norm(vec - w_vec)
            if dist < best_dist:
                best_dist = dist
                best_word = word
        return best_word

    # Detect Dimension from loaded weights
    l2_dim = brain["l2_W_res"].shape[0]
    hierarchy = HierarchicalEncoder(l2_res_dim=l2_dim)
    hierarchy.layer2.W_res = brain["l2_W_res"]
    
    # Test Cases
    prompts = [
        "sky", 
        "cats", 
        "urcm", 
        "neural", 
        "water",
        "midway",      # From WWII JSON
        "resonance",   # From White Paper
        "stability",   # From White Paper
        "arnhem"       # From WWII JSON
    ]
    
    for prompt in prompts:
        if prompt not in concept_map:
            print(f"Skipping {prompt} (unknown)")
            continue
            
        print(f"\nPrompt: '{prompt}'")
        state = concept_map[prompt]
        
        # Run Dynamics for 5 steps (simulate sentence generation)
        output = [prompt]
        for _ in range(5):
            # s_t+1 = tanh(s_t @ W)
            # No input, just internal flow
            state = np.tanh(np.dot(state, hierarchy.layer2.W_res))
            word = decode(state)
            output.append(word)
            
        print("Generated: " + " ".join(output))

if __name__ == "__main__":
    main()
