import time
import numpy as np
import pandas as pd
import sys
import os
from typing import List, Dict, Any

# Add root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from urcm.core.system import URCMSystem
from tests.comprehensive_test_data import TEST_DATA_100

def levenshtein_ratio(s1, s2):
    """Calculates similarity ratio between two sequences."""
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    
    rows = len(s1) + 1
    cols = len(s2) + 1
    dist = [[0 for x in range(cols)] for x in range(rows)]

    for i in range(rows):
        dist[i][0] = i
    for j in range(cols):
        dist[0][j] = j

    for col in range(1, cols):
        for row in range(1, rows):
            if s1[row-1] == s2[col-1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row-1][col] + 1,      # deletion
                                 dist[row][col-1] + 1,      # insertion
                                 dist[row-1][col-1] + cost) # substitution
    
    max_len = max(len(s1), len(s2))
    return 1.0 - (dist[rows-1][cols-1] / max_len)

def generate_extra_inputs(base_data: Dict[str, List[str]], target_count: int = 200) -> List[str]:
    """Expands dataset to target count."""
    all_inputs = []
    for cat, items in base_data.items():
        all_inputs.extend(items)
    
    current_count = len(all_inputs)
    needed = target_count - current_count
    
    print(f"Base data: {current_count} inputs. Generating {needed} more...")
    
    extra_inputs = []
    # Strategy 1: Reverse inputs (Semantic stress test)
    for i in range(needed):
        source = all_inputs[i % current_count]
        # Reverse words
        words = source.split()
        new_text = " ".join(words[::-1])
        extra_inputs.append(new_text)
        
    return all_inputs + extra_inputs

def run_scale_test():
    print("🚀 Starting URCM Scale Test (100 Types / 200 Inputs)...")
    
    # 1. Setup Data
    inputs = generate_extra_inputs(TEST_DATA_100, 200)
    print(f"✅ Loaded {len(inputs)} inputs.")
    
    # 2. Initialize System
    system = URCMSystem()
    print("✅ System Initialized.")
    
    # 3. Run Loop
    results = []
    start_time = time.time()
    
    print("\nProcessing...")
    for i, text in enumerate(inputs):
        if not text or not text.strip():
            print(f"⚠️ Skipping empty input {i}")
            continue
            
        try:
            # Step A: Forward Pass
            path = system.process_query(text)
            final_state = path.final_state
            
            # Step B: Get Ground Truth Phonemes
            input_phonemes = system.pipeline.text_converter.convert_text_to_phonemes(text).phonemes
            
            # Step C: Backward Pass (The Holy Grail)
            # Unroll same length as input for fair comparison, or fixed length?
            # Let's unroll roughly the length of input phonemes
            decode_steps = len(input_phonemes)
            decoded_vectors = system.encoder.decode_state(final_state, steps=decode_steps)
            
            # Map back to phonemes
            mapper = system.pipeline.frequency_mapper
            decoded_phonemes = []
            for vec in decoded_vectors:
                best_match = None
                best_dist = float('inf')
                for cand_p, cand_vec in mapper.phoneme_vectors.items():
                    dist = np.linalg.norm(vec - cand_vec)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = cand_p
                decoded_phonemes.append(best_match)
            
            # Step D: Calculate Metrics
            # 1. Similarity
            match_score = levenshtein_ratio(input_phonemes, decoded_phonemes)
            
            # 2. Signal Strength
            signal_energy = np.mean([np.linalg.norm(v) for v in decoded_vectors])
            
            # 3. Diversity
            unique_ratio = len(set(decoded_phonemes)) / len(decoded_phonemes) if decoded_phonemes else 0
            
            results.append({
                "id": i,
                "input_len": len(text),
                "mu": final_state.mu_value,
                "rho": final_state.rho_density,
                "reconstruction_score": match_score,
                "signal_energy": signal_energy,
                "diversity": unique_ratio,
                "input_snippet": text[:30] + "...",
                "decoded_snippet": "-".join(decoded_phonemes[:5]) + "..."
            })
            
            if (i+1) % 20 == 0:
                print(f"  Progress: {i+1}/200...")
                
        except Exception as e:
            print(f"❌ Error on input {i}: {e}")
            
    total_time = time.time() - start_time
    print(f"\n✅ Processing Complete in {total_time:.2f}s")
    
    # 4. Analysis
    df = pd.DataFrame(results)
    
    print("\n--- 📊 Aggregate Metrics ---")
    print(f"Total Inputs: {len(df)}")
    print(f"Avg Stability (μ): {df['mu'].mean():.4f}")
    print(f"Avg Density (ρ):   {df['rho'].mean():.4f}")
    print(f"Avg Reconstruction Score: {df['reconstruction_score'].mean():.4f} (0-1)")
    print(f"Avg Signal Energy: {df['signal_energy'].mean():.4f}")
    print(f"Avg Output Diversity: {df['diversity'].mean():.4f}")
    
    # 5. Save Results
    df.to_csv("scale_test_results.csv", index=False)
    print("\n📄 Detailed results saved to 'scale_test_results.csv'")
    
    # 6. Show Best/Worst Reconstructions
    print("\n🏆 Best Reconstruction:")
    best = df.loc[df['reconstruction_score'].idxmax()]
    print(f"  Input: {best['input_snippet']}")
    print(f"  Score: {best['reconstruction_score']:.4f}")
    
    print("\n📉 Worst Reconstruction:")
    worst = df.loc[df['reconstruction_score'].idxmin()]
    print(f"  Input: {worst['input_snippet']}")
    print(f"  Score: {worst['reconstruction_score']:.4f}")

if __name__ == "__main__":
    run_scale_test()
