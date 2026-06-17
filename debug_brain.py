import pickle
import os

def check_brain():
    brain_path = "urcm_identity.pkl"
    if not os.path.exists(brain_path):
        print("Brain not found!")
        return
        
    with open(brain_path, "rb") as f:
        data = pickle.load(f)
        
    vocab = list(data["concept_map"].keys())
    print(f"Vocab Size: {len(vocab)}")
    
    print("\nSearch for 'ambush':")
    found = False
    for w in vocab:
        if "ambush" in w:
            print(f"  Found: '{w}'")
            found = True
            
    if not found:
        print("  'ambush' NOT found in vocab.")
        
    print("\nFirst 20 words in vocab:")
    print(vocab[:20])

if __name__ == "__main__":
    check_brain()
