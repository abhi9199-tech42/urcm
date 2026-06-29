import numpy as np
from urcm.core.safe_serialization import safe_load

def inspect():
    brain = safe_load("urcm_identity.pkl")
    if brain is None:
        print("Failed to load brain file")
        return
        
    cmap = brain["concept_map"]
    print(f"Total Concepts: {len(cmap)}")
    
    keys = list(cmap.keys())
    vecs = []
    for k in keys[:5]:
        v = cmap[k]
        print(f"Key: {k}, Vec Mean: {v.mean():.6f}, Vec Std: {v.std():.6f}")
        vecs.append(v)
        
    # Check distinctness
    v1 = vecs[0]
    v2 = vecs[1]
    dist = np.linalg.norm(v1 - v2)
    print(f"Distance between {keys[0]} and {keys[1]}: {dist:.6f}")
    
    # Check if 'truthah' is in keys
    print(f"'truthah' in keys: {'truthah' in keys}")
    
    # Decode a vector
    # This simulates what ReasoningEngine.decode() does
    print("\nDecoding test:")
    test_vec = vecs[0]
    best_word = "?"
    best_dist = float('inf')
    for word, w_vec in cmap.items():
        dist = np.linalg.norm(test_vec - w_vec)
        if dist < best_dist:
            best_dist = dist
            best_word = word
            
    print(f"Vector for '{keys[0]}' decodes to: '{best_word}' (Dist: {best_dist})")

if __name__ == "__main__":
    inspect()
