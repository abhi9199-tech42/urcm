import numpy as np
import pickle
import os
from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.hierarchical_encoder import HierarchicalEncoder
from urcm.core.memory import GeometricMemory
from urcm.core.phoneme_mapper import TextToPhonemeConverter
from urcm.core.identity import BASIC_VOCABULARY, IDENTITY_CONCEPTS

def train_identity():
    print("🎓 URCM Phase 5: Identity Training (One-Shot)")
    print("=============================================")
    
    # 1. Initialize System
    print("1. Initializing Cognitive Core...")
    # We use the Hierarchical Encoder for deeper concepts
    # But for "talking", we need a shared resonance space.
    # Let's train the Layer 2 (Concept) space.
    
    hierarchy = HierarchicalEncoder()
    memory = GeometricMemory(resonance_dim=128) # Layer 2 is 128 dim
    converter = TextToPhonemeConverter()
    
    # 2. Generate/Deposit Concepts
    print(f"2. Learning {len(BASIC_VOCABULARY)} basic concepts...")
    
    # We need to map: Word -> Phonemes -> Layer 1 Trajectory -> Layer 2 State
    # Then Stabilize Layer 2 State.
    
    concept_map = {} # Word -> Resonance State (for reverse lookup later)
    
    for word in BASIC_VOCABULARY:
        # A. Convert to Phonemes
        # Note: Method name is convert_text_to_phonemes, and it returns a PhonemeSequence object
        phoneme_seq = converter.convert_text_to_phonemes(word)
        if not phoneme_seq or not phoneme_seq.phonemes:
            print(f"   ⚠️ Skipping '{word}': No phonemes found.")
            continue
            
        phonemes = phoneme_seq.phonemes

            
        # B. Get Phoneme Vectors (Input to Layer 1)
        # We need a dummy mapper or use the converter's internal logic if exposed.
        # The converter returns a list of phoneme strings.
        # We need to map strings to vectors.
        # Let's use the hierarchy's layer1.W_in logic implicitly? 
        # No, we need vectors.
        # Let's create a random vector for each unique phoneme for now if not defined.
        # Actually, let's use the system's standard pipeline if possible.
        
        # Hack: We'll generate consistent random vectors for phonemes based on their hash
        # to ensure "a" always equals "a".
        
        l1_inputs = []
        for p in phonemes:
            # Deterministic hash to vector
            seed = sum(ord(c) for c in p)
            np.random.seed(seed) 
            vec = np.random.normal(0, 1, (24,)) # Layer 1 input dim
            vec = vec / np.linalg.norm(vec)
            l1_inputs.append(vec)
        
        l1_inputs = np.array(l1_inputs)
        
        # C. Process through Hierarchy
        # Get L2 State
        l2_state, _ = hierarchy.encode_concept(l1_inputs)
        
        # D. DEPOSIT MEMORY (One-Shot)
        # We want this L2 state to be a fixed point attractor.
        # "When you think of 'URCM', stay thinking of 'URCM'."
        hierarchy.layer2.W_res = memory.deposit_attractor(hierarchy.layer2.W_res, l2_state)
        
        # Store for Dictionary (Reverse Lookup)
        concept_map[word] = l2_state
        print(f"   ✓ Learned: '{word}'")
        
    # 3. Deposit Identity Definitions (Associative Links)
    # "Who are you?" -> "URCM"
    print("3. Connecting Identity Associations...")
    for key, value in IDENTITY_CONCEPTS.items():
        if key in concept_map:
            key_state = concept_map[key]
            # Associate Key with Value? 
            # This requires sequence learning (Question -> Answer).
            # For now, we just ensure the KEY concepts are stable.
            # We already did that in step 2 (if they are in vocab).
            pass

    # 4. Save the Brain
    print("4. Saving Trained Weights and Concept Map...")
    
    # Save Weights
    weights = {
        "l1_W_res": hierarchy.layer1.W_res,
        "l1_W_in": hierarchy.layer1.W_in,
        "l1_W_out": hierarchy.layer1.W_out,
        "l2_W_res": hierarchy.layer2.W_res, # The important one
        "l2_W_in": hierarchy.layer2.W_in,
        "l2_W_out": hierarchy.layer2.W_out,
        "concept_map": concept_map
    }
    
    with open("urcm_identity.pkl", "wb") as f:
        pickle.dump(weights, f)
        
    print("✅ Identity Training Complete.")

if __name__ == "__main__":
    train_identity()