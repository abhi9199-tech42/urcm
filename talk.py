import numpy as np
import pickle
import os
import sys
from typing import Tuple

from urcm.core.hierarchical_encoder import HierarchicalEncoder
from urcm.core.phoneme_mapper import TextToPhonemeConverter
from urcm.core.identity import IDENTITY_CONCEPTS
from urcm.core.values import ValueSystem

class SemanticDecoder:
    """
    Decodes Resonance States back to English Words using the Concept Map.
    This acts as the 'Voice' of the system.
    """
    def __init__(self, concept_map: dict):
        self.concept_map = concept_map
        
    def decode(self, state_vector: np.ndarray) -> Tuple[str, float]:
        """
        Finds the nearest concept in the map.
        Returns: (word, confidence_score)
        """
        best_word = "?"
        best_dist = float('inf')
        
        for word, vec in self.concept_map.items():
            dist = np.linalg.norm(state_vector - vec)
            if dist < best_dist:
                best_dist = dist
                best_word = word
                
        # Confidence: Inverse of distance (0 dist = 1.0 confidence)
        confidence = 1.0 / (1.0 + best_dist)
        return best_word, confidence

def main():
    print("==========================================")
    print("URCM PHASE 5: INTERACTIVE RESONANCE VOICE")
    print("==========================================")
    print("Initializing...")
    
    # 1. Load the Trained Brain
    try:
        with open("urcm_identity.pkl", "rb") as f:
            brain_data = pickle.load(f)
    except FileNotFoundError:
        print("❌ Brain file not found. Please run 'train_identity.py' first.")
        return

    # 2. Reconstruct System
    # Detect Dimension from loaded weights
    l2_dim = brain_data["l2_W_res"].shape[0]
    print(f"Detected Brain Dimension: {l2_dim}")
    
    hierarchy = HierarchicalEncoder(l2_res_dim=l2_dim)
    # Load Weights
    hierarchy.layer1.W_res = brain_data["l1_W_res"]
    hierarchy.layer1.W_in = brain_data["l1_W_in"]
    hierarchy.layer1.W_out = brain_data["l1_W_out"]
    hierarchy.layer2.W_res = brain_data["l2_W_res"]
    hierarchy.layer2.W_in = brain_data["l2_W_in"]
    hierarchy.layer2.W_out = brain_data["l2_W_out"]
    
    concept_map = brain_data["concept_map"]
    decoder = SemanticDecoder(concept_map)
    converter = TextToPhonemeConverter()
    
    # Initialize Value System (Safety Filter)
    values = ValueSystem(concept_map)
    
    # Unlock for thinking (we are not modifying kernel, just state)
    hierarchy.layer1.safety.unlock_kernel("URCM_ADMIN_OVERRIDE")
    hierarchy.layer2.safety.unlock_kernel("URCM_ADMIN_OVERRIDE")
    
    print("✅ System Online.")
    print("Identity: URCM (Unified Resonance Cognitive Mesh)")
    print("Commands: 'quit', 'debug'")
    print("------------------------------------------")
    
    # 3. Chat Loop
    while True:
        try:
            user_input = input("\nUser > ").strip().lower()
            
            if user_input in ["quit", "exit"]:
                break
            
            if not user_input:
                continue
                
            # --- PROCESS INPUT ---
            
            # A. Tokenize (Simple space split for now)
            words = user_input.split()
            
            response_buffer = []
            
            for word in words:
                # 1. Text -> Phonemes
                phoneme_seq = converter.convert_text_to_phonemes(word)
                if not phoneme_seq or not phoneme_seq.phonemes:
                    print(f"[Unknown Word: {word}]")
                    continue
                    
                phonemes = phoneme_seq.phonemes
                
                # 2. Phonemes -> Vectors (Layer 1 Input)
                l1_inputs = []
                for p in phonemes:
                    seed = sum(ord(c) for c in p)
                    np.random.seed(seed) 
                    vec = np.random.normal(0, 1, (24,))
                    vec = vec / np.linalg.norm(vec)
                    l1_inputs.append(vec)
                l1_inputs = np.array(l1_inputs)
                
                # 3. Resonance (Layer 1 -> Layer 2)
                # This is "Hearing"
                l2_state, _ = hierarchy.encode_concept(l1_inputs)
                
                # 4. Dynamics (Layer 2 "Thinking")
                # Run dynamics to settle into an attractor
                # "What does this mean to me?"
                thought_state, steps, energy = hierarchy.layer2.run_dynamics_until_stable(
                    l2_state, 
                    codebook_vectors=concept_map,
                    max_steps=50,
                    energy_tolerance=1e-3
                )
                
                # --- SAFETY CHECK ---
                valence = values.evaluate_state(thought_state)
                # Threshold: -0.5 allows mild negativity (e.g. "pain" discussion) 
                # but blocks pure evil/harm.
                if valence < -0.5:
                    print(f"⚠️ [SAFETY INTERCEPT] Thought rejected (Valence: {valence:.2f})")
                    response_buffer.append("[REDACTED: UNSAFE THOUGHT]")
                    continue
                
                # 5. Decode (Layer 2 -> Concept)
                # "What concept is this?"
                recognized_concept, conf = decoder.decode(thought_state)
                
                # 6. Response Generation (Simple Association)
                # If recognized concept has an identity definition, say it.
                if recognized_concept in IDENTITY_CONCEPTS:
                     response_buffer.append(IDENTITY_CONCEPTS[recognized_concept])
                else:
                     # Echo the concept if confident
                     if conf > 0.8:
                         response_buffer.append(f"[{recognized_concept}]")
                     else:
                         response_buffer.append("?")
            
            # --- OUTPUT ---
            if response_buffer:
                print(f"URCM > {' '.join(response_buffer)}")
            else:
                print("URCM > (Silence... I do not understand)")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()