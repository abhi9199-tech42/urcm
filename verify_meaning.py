
import sys
import os
import numpy as np
from typing import List, Tuple

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from urcm.core.system import URCMSystem

def verify_semantic_preservation():
    print("Initializing URCM System for Semantic Verification...")
    system = URCMSystem()
    
    # Test cases with clear semantic meaning
    test_cases = [
        "The cat sat on the mat.",
        "Artificial intelligence is transforming the world.",
        "Gravity pulls objects towards the center of the earth.",
        "Red apples are sweet and delicious."
    ]
    
    print(f"\n{'Input Text':<50} | {'Reconstructed Phonemes (Approx)':<40} | {'Cosine Sim':<10}")
    print("-" * 110)
    
    for text in test_cases:
        # 1. Forward Pass: Text -> Resonance State
        result_path = system.process_query(text)
        final_state = result_path.final_state
        
        # 2. Backward Pass: Resonance State -> Reconstructed Signal
        # Note: URCM is a compression engine, so perfect text reconstruction isn't the primary goal,
        # but the resonance state should be close to the original input in latent space.
        
        # We can check this by performing a round-trip check using the ReconstructionSystem
        reconstructed_vec, loss, valid = system.reconstruction.perform_round_trip(final_state)
        
        # Calculate Cosine Similarity between Original and Reconstructed Latent Vector
        # (This measures how much "meaning" was preserved in the compression)
        
        # Get original latent representation
        original_latent = system.latent_space.project(final_state)
        
        # Re-encode the reconstruction (which is in resonance space) to latent space for comparison
        # Or compare in resonance space directly
        
        # Let's compare the original resonance vector vs the reconstructed resonance vector
        v1 = final_state.resonance_vector
        v2 = reconstructed_vec
        
        # Cosine Similarity
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        similarity = dot_product / (norm_v1 * norm_v2) if (norm_v1 > 0 and norm_v2 > 0) else 0.0
        
        # For display, we show the similarity score. 
        # High similarity (> 0.9) means the "meaning" (resonance state) is preserved.
        
        print(f"{text:<50} | {'[Latent Vector Reconstruction]':<40} | {similarity:.4f}")

if __name__ == "__main__":
    verify_semantic_preservation()
