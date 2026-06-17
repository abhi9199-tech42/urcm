import time
import numpy as np
import psutil
import os
from urcm.core.system import URCMSystem

def benchmark_performance():
    print("⚡ URCM PERFORMANCE BENCHMARK")
    print("============================")
    
    # 1. Load System
    start_load = time.time()
    system = URCMSystem()
    end_load = time.time()
    print(f"✅ System Loaded in {(end_load - start_load)*1000:.2f} ms")
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024
    print(f"   Memory Usage: {mem_before:.2f} MB")
    
    # Prepare Data
    codebook = system.pipeline.frequency_mapper.phoneme_vectors
    keys = list(codebook.keys())
    sample_key = keys[0]
    sample_vec = codebook[sample_key]
    
    # Pre-calculate state for dynamics test
    h_state = np.dot(sample_vec, system.encoder.W_in)
    sample_state = np.tanh(h_state)
    
    # BENCHMARK 1: ENCODING (Frequency -> Resonance)
    # Simulating a path of length 10 (typical phoneme duration)
    path_len = 10
    input_seq = np.tile(sample_vec, (path_len, 1))
    
    N_loops = 1000
    print(f"\n[TEST 1] Encoding Latency (Seq Len {path_len}) - {N_loops} iterations")
    
    start_time = time.time()
    for _ in range(N_loops):
        _ = system.encoder._encode_recurrent(input_seq)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / N_loops) * 1000
    throughput = N_loops / total_time
    
    print(f"   Avg Latency: {avg_latency:.4f} ms")
    print(f"   Throughput:  {throughput:.2f} inputs/sec")
    
    if avg_latency < 10:
        print("✅ PASS: Real-time capable (<10ms)")
    else:
        print("⚠️ WARN: Slow encoding")

    # BENCHMARK 2: THINKING (Dynamics Convergence)
    # Measuring "Thought Speed"
    print(f"\n[TEST 2] Thinking Speed (Dynamics until Stable) - {N_loops} iterations")
    
    # Inject noise to force actual computation
    np.random.seed(42)
    noisy_state = np.tanh(np.arctanh(sample_state * 0.9) + np.random.normal(0, 0.1, sample_state.shape))
    
    start_time = time.time()
    total_steps = 0
    for _ in range(N_loops):
        _, steps, _ = system.encoder.run_dynamics_until_stable(
            noisy_state, codebook, max_steps=20, energy_tolerance=1e-3
        )
        total_steps += steps
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / N_loops) * 1000
    avg_steps = total_steps / N_loops
    throughput = N_loops / total_time
    
    print(f"   Avg Thinking Time: {avg_latency:.4f} ms")
    print(f"   Avg Convergence Steps: {avg_steps:.1f}")
    print(f"   Throughput: {throughput:.2f} thoughts/sec")
    
    # Human reaction time is ~200ms. 
    # If thinking takes < 50ms, it feels "instant".
    if avg_latency < 50:
        print("✅ PASS: Super-human reaction speed (<50ms)")
    else:
        print("⚠️ WARN: Sluggish thoughts")

    # BENCHMARK 3: DECODING (Resonance -> Frequency)
    print(f"\n[TEST 3] Decoding Latency (Readout) - {N_loops} iterations")
    
    start_time = time.time()
    for _ in range(N_loops):
        # Linear projection + tanh
        h = np.arctanh(np.clip(sample_state, -0.999, 0.999))
        _ = np.dot(h, system.encoder.W_out)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / N_loops) * 1000
    throughput = N_loops / total_time
    
    print(f"   Avg Latency: {avg_latency:.4f} ms")
    print(f"   Throughput:  {throughput:.2f} outputs/sec")
    
    print("\nOVERALL STATUS")
    if throughput > 100:
        print("🚀 SYSTEM IS HIGH-PERFORMANCE.")
    else:
        print("🐢 SYSTEM NEEDS OPTIMIZATION.")

if __name__ == "__main__":
    benchmark_performance()
