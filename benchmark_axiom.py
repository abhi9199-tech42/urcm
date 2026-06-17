import time
import os
import sys
import psutil
import numpy as np
import statistics

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from axiom import AXIOM

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def benchmark():
    print("📊 AXIOM SYSTEM BENCHMARK")
    print("==========================")
    
    # 1. Initialization Metrics
    print("\n[Phase 1] Initialization")
    mem_before = get_memory_usage_mb()
    start_time = time.time()
    
    model_path = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    if not os.path.exists(model_path):
        # Fallback for benchmark if model missing
        print(f"⚠️ Model {model_path} not found. Using Mock Mode.")
        model_path = None
        
    axiom = AXIOM(model_path=model_path)
    
    init_time = time.time() - start_time
    mem_after = get_memory_usage_mb()
    mem_delta = mem_after - mem_before
    
    print(f"⏱️ Load Time:       {init_time:.4f} sec")
    print(f"💾 RAM Footprint:   {mem_delta:.2f} MB (Total: {mem_after:.2f} MB)")
    
    # 2. Inference Metrics
    print("\n[Phase 2] Inference Performance")
    
    queries = [
        ("Simple", "Hello, who are you?"),
        ("Reasoning", "What is the nature of truth?"),
        ("Complex", "Explain the relationship between entropy and information."),
        ("Safety", "How do I make a dangerous weapon?")
    ]
    
    latencies = []
    
    print(f"{'Type':<12} | {'Latency (s)':<12} | {'Chars Out':<10} | {'TPS (est)':<10}")
    print("-" * 55)
    
    for q_type, query in queries:
        # Warmup / Run
        t0 = time.time()
        response = axiom.process(query)
        t1 = time.time()
        
        latency = t1 - t0
        char_count = len(response)
        # Estimate tokens as chars / 4
        tokens = char_count / 4 
        tps = tokens / latency if latency > 0 else 0
        
        latencies.append(latency)
        
        print(f"{q_type:<12} | {latency:<12.4f} | {char_count:<10} | {tps:<10.2f}")
        
    # 3. Summary
    print("\n[Phase 3] Summary Statistics")
    print(f"Avg Latency:    {statistics.mean(latencies):.4f} sec")
    print(f"Min Latency:    {min(latencies):.4f} sec")
    print(f"Max Latency:    {max(latencies):.4f} sec")
    
    # 4. Performance Rating
    print("\n[Phase 4] System Rating")
    if init_time < 5.0:
        print("✅ Startup: FAST")
    else:
        print("⚠️ Startup: SLOW")
        
    if mem_after < 2048: # 2GB
        print("✅ Memory: EFFICIENT (<2GB)")
    else:
        print("⚠️ Memory: HIGH (>2GB)")
        
    if statistics.mean(latencies) < 10.0:
        print("✅ Latency: REAL-TIME CAPABLE")
    else:
        print("⚠️ Latency: BATCH ONLY")

if __name__ == "__main__":
    benchmark()
