
import sys
import os
import time
import numpy as np
import pandas as pd

import re

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from urcm.core.system import URCMSystem
from tests.comprehensive_test_data import TEST_DATA_100

def generate_slug(text):
    text_lower = text.lower().strip()
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'will', 'how', 'what', 'its', 'that', 'from', 'this'}
    words = re.findall(r'\w+', text_lower)
    significant = [w for w in words if w not in stop_words and len(w) > 4]
    seen = set()
    unique_sig = []
    for w in significant:
        if w not in seen:
            unique_sig.append(w)
            seen.add(w)
    slug = "_".join(unique_sig[:5])
    if not slug:
        slug = "semantic_state_undefined"
    return slug

def run_tests():
    print("Initializing URCM System...")
    system = URCMSystem()
    
    results = []
    total_tests = 0
    start_time = time.time()
    
    print(f"{'Category':<20} | {'Input (Truncated)':<30} | {'μ (Mu)':<6} | {'ρ (Rho)':<6} | {'Slug':<30} | {'Time (s)':<8}")
    print("-" * 110)
    
    for category, inputs in TEST_DATA_100.items():
        for text in inputs:
            total_tests += 1
            t0 = time.time()
            try:
                # Process query
                result_path = system.process_query(text)
                final_state = result_path.final_state
                
                # Extract metrics
                mu = final_state.mu_value
                # Clamp mu to [0, 1]
                mu = max(0.0, min(1.0, mu))
                
                rho = final_state.rho_density
                slug = generate_slug(text)
                
                dt = time.time() - t0
                
                # Log to console
                trunc_text = (text[:27] + '...') if len(text) > 27 else text
                print(f"{category:<20} | {trunc_text:<30} | {mu:.3f}  | {rho:.3f}  | {slug:<30} | {dt:.4f}")
                
                results.append({
                    "Category": category,
                    "Input": text,
                    "Mu": mu,
                    "Rho": rho,
                    "Slug": slug,
                    "Time": dt,
                    "Status": "Success"
                })
                
            except Exception as e:
                print(f"{category:<20} | {text[:27] + '...':<30} | ERROR   | ERROR   | {str(e):<30} | 0.0000")
                results.append({
                    "Category": category,
                    "Input": text,
                    "Mu": 0.0,
                    "Rho": 0.0,
                    "Slug": "ERROR",
                    "Time": 0.0,
                    "Status": f"Error: {str(e)}"
                })
    
    total_time = time.time() - start_time
    
    print("-" * 110)
    print(f"\nCompleted {total_tests} tests in {total_time:.2f} seconds.")
    
    # Analyze results
    df = pd.DataFrame(results)
    print("\nSummary Statistics by Category:")
    summary = df.groupby("Category")[["Mu", "Rho", "Time"]].mean()
    print(summary)
    
    # Save results to CSV
    output_file = "comprehensive_test_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    run_tests()
