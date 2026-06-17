import pickle
import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from urcm.core.hierarchical_encoder import HierarchicalEncoder
from urcm.core.phoneme_mapper import TextToPhonemeConverter

def verify_interim_progress():
    # Default to smoothed if available, else identity
    BRAIN_PATH = "urcm_identity_smoothed.pkl" if os.path.exists("urcm_identity_smoothed.pkl") else "urcm_identity.pkl"

    if len(sys.argv) > 1:
        BRAIN_PATH = sys.argv[1]

    print(f"🕵️ INTERIM TRAINING VERIFICATION (Using: {BRAIN_PATH})")
    print("================================")
    
    # 1. Load Brain (with retry for file contention)
    brain_data = None
    max_retries = 5
    
    for i in range(max_retries):
        try:
            with open(BRAIN_PATH, "rb") as f:
                brain_data = pickle.load(f)
            print(f"✅ Successfully loaded '{BRAIN_PATH}'")
            break
        except PermissionError:
            print(f"⚠️ File locked (Training is writing?). Retrying {i+1}/{max_retries}...")
            time.sleep(2)
        except Exception as e:
            print(f"❌ Error loading brain: {e}")
            return

    if not brain_data:
        print("❌ Could not load brain data.")
        return

    # 2. Check Vocabulary Size
    concept_map = brain_data.get("concept_map", {})
    vocab_size = len(concept_map)
    print(f"\n📚 Current Vocabulary Size: {vocab_size} concepts")
    
    # 3. Check for New Concepts (from recent logs)
    target_words = ["hello", "money", "apartment", "gift", "color", "dress", "thanks"]
    print("\n🔍 Concept Check:")
    found_count = 0
    for word in target_words:
        if word in concept_map:
            print(f"   ✅ '{word}' exists.")
            found_count += 1
        else:
            print(f"   ❌ '{word}' NOT found.")
            
    print(f"   (Found {found_count}/{len(target_words)})")

    # 4. Functional Test: Associative Recall (Concept -> Concept)
    print("\n🧠 Associative Recall Test (L2 Resonance):")
    
    try:
        # Load L2 Weights
        l2_W_res = brain_data["l2_W_res"]
        
        def get_association(start_word):
            if start_word not in concept_map:
                return "Unknown", 0.0
                
            # 1. Start State
            state = concept_map[start_word]
            
            # 2. Run Dynamics (One Step)
            # x_next = tanh(W_res @ x)
            next_state = np.tanh(np.dot(l2_W_res, state))
            
            # 3. Decode
            best_word = "?"
            best_dist = float('inf')
            
            # Normalize
            if np.linalg.norm(next_state) > 0:
                next_state = next_state / np.linalg.norm(next_state)
            
            for word, vec in concept_map.items():
                if word == start_word: continue # Ignore self-loop for association
                
                dist = np.linalg.norm(next_state - vec)
                if dist < best_dist:
                    best_dist = dist
                    best_word = word
            
            return best_word, best_dist

        # Test Associations
        test_inputs = ["hello", "money", "thank", "what", "color"]
        
        for word in test_inputs:
            assoc, dist = get_association(word)
            print(f"   Context: '{word}' -> Associated: '{assoc}' (Dist: {dist:.4f})")
            
    except Exception as e:
        print(f"❌ Error during associative test: {e}")

if __name__ == "__main__":
    verify_interim_progress()
