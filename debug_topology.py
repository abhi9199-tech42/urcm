import numpy as np
from urcm.core.system import URCMSystem

def analyze_topology():
    print("🗺️ Analyzing Semantic Topology...")
    
    try:
        system = URCMSystem()
        print("✅ System Initialized.")
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return

    codebook = system.pipeline.frequency_mapper.phoneme_vectors
    keys = list(codebook.keys())
    vectors = list(codebook.values())
    n = len(keys)
    
    print(f"📊 Analyzing {n} phonemes...")
    
    # Calculate Pairwise Distances
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(vectors[i] - vectors[j])
            distances.append((dist, keys[i], keys[j]))
            
    # Sort by distance (closest first)
    distances.sort(key=lambda x: x[0])
    
    print("\n⚠️ Dangerous Proximities (Risk of Hallucination):")
    print(f"{'Dist':<10} | {'Pair':<15}")
    print("-" * 30)
    
    for d, k1, k2 in distances[:10]:
        print(f"{d:<10.4f} | {k1} <-> {k2}")
        
    # Check specific pair from previous error
    # 'm' vs 'ḍ'
    vec_m = codebook.get('m')
    vec_d = codebook.get('ḍ')
    
    if vec_m is not None and vec_d is not None:
        dist_md = np.linalg.norm(vec_m - vec_d)
        print(f"\n🔍 Specific Check ('m' vs 'ḍ'): {dist_md:.4f}")
    else:
        print("\n🔍 Could not find 'm' or 'ḍ' in codebook.")
        
    # Check average distance
    avg_dist = np.mean([d for d, _, _ in distances])
    print(f"\n📏 Average Separation: {avg_dist:.4f}")
    
    # Recommendation
    if distances[0][0] < 0.1:
        print("❌ CRITICAL: Topology Collapse. Some phonemes are indistinguishable.")
    elif distances[0][0] < 1.0:
        print("⚠️ WARNING: Tight clusters. Requires high-precision W_out.")
    else:
        print("✅ Topology Healthy. Good separation.")

if __name__ == "__main__":
    analyze_topology()
