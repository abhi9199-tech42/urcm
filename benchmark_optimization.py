import numpy as np
import time
from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.hierarchical_encoder import HierarchicalEncoder

def benchmark_encoding():
    print("🚀 Benchmarking ResonancePathEncoder...")
    
    input_dim = 24
    res_dim = 64
    seq_len = 10
    batch_size = 1000
    
    encoder = ResonancePathEncoder(input_dim, res_dim)
    
    # Generate random data: List of (seq_len, input_dim) arrays
    data = [np.random.normal(0, 1, (seq_len, input_dim)) for _ in range(batch_size)]
    data_tensor = np.array(data)
    
    print(f"   Input: {batch_size} sequences of length {seq_len}, dim {input_dim}")
    
    # Measure Sequential Processing
    start_time = time.time()
    results_seq = []
    for seq in data:
        res = encoder.encode_path(seq)
        results_seq.append(res)
    end_time = time.time()
    
    duration_seq = end_time - start_time
    print(f"   Sequential Time: {duration_seq:.4f}s ({batch_size/duration_seq:.2f} seq/s)")
    
    # Measure Batched Processing
    start_time = time.time()
    results_batch = encoder.encode_path_batch(data_tensor)
    end_time = time.time()
    
    duration_batch = end_time - start_time
    print(f"   Batched Time:    {duration_batch:.4f}s ({batch_size/duration_batch:.2f} seq/s)")
    
    speedup = duration_seq / duration_batch
    print(f"   🚀 Speedup: {speedup:.2f}x")
    
    # Verify Correctness
    diff = np.linalg.norm(np.array(results_seq) - results_batch)
    if diff < 1e-9:
        print("   ✅ Batched results match Sequential results")
    else:
        print(f"   ❌ Mismatch! Diff: {diff}")
        
    print("\n🚀 Benchmarking HierarchicalEncoder (Deep Stack)...")
    h_encoder = HierarchicalEncoder()
    
    # Measure Sequential Hierarchy
    start_time = time.time()
    for seq in data:
        h_encoder.encode_concept(seq)
    end_time = time.time()
    duration_h_seq = end_time - start_time
    print(f"   Sequential Hierarchy: {duration_h_seq:.4f}s")
    
    # Measure Batched Hierarchy
    start_time = time.time()
    h_encoder.encode_concept_batch(data_tensor)
    end_time = time.time()
    duration_h_batch = end_time - start_time
    print(f"   Batched Hierarchy:    {duration_h_batch:.4f}s")
    
    print(f"   🚀 Hierarchy Speedup: {duration_h_seq / duration_h_batch:.2f}x")
    
    return duration_seq

if __name__ == "__main__":
    benchmark_encoding()
