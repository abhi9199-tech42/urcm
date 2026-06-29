"""Diagnose the batch throughput anomaly"""
import numpy as np
import torch
import time
import tracemalloc
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.phoneme_mapper import PhonemeFrequencyPipeline

# Optimized pipeline
@torch.jit.script
def resonance_step(state, x_t, W_in, W_res, bias):
    return torch.tanh(x_t @ W_in + state @ W_res + bias).clamp(-10.0, 10.0)

@torch.jit.script
def encode_batch(vectors_batch, W_in, W_res, bias):
    B, T, _ = vectors_batch.shape
    R = W_res.shape[0]
    states = torch.zeros(B, R, dtype=vectors_batch.dtype)
    for t in range(T):
        states = torch.tanh(vectors_batch[:, t, :] @ W_in + states @ W_res + bias).clamp(-10.0, 10.0)
    return states


class OptPipe:
    def __init__(self):
        np.random.seed(42)
        W_in = np.random.normal(0, 0.1, (24, 512)).astype(np.float32)
        H = np.random.randn(512, 512).astype(np.float32)
        Q, _ = np.linalg.qr(H)
        W_res = (Q * 0.95).astype(np.float32)
        bias = np.random.normal(0, 0.01, 512).astype(np.float32)
        self.W_in = torch.from_numpy(W_in)
        self.W_res = torch.from_numpy(W_res)
        self.bias = torch.from_numpy(bias)
        # warmup
        d = torch.zeros(2, 10, 24)
        encode_batch(d, self.W_in, self.W_res, self.bias)

    def encode(self, batch_np):
        t = torch.from_numpy(batch_np.astype(np.float32))
        return encode_batch(t, self.W_in, self.W_res, self.bias).numpy()


pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
encoder_orig = ResonancePathEncoder(input_dim=24, resonance_dim=512)
opt = OptPipe()
text = "All humans are mortal."

# Pre-build all frequency paths
print("Pre-building frequency paths...")
freq_paths = []
for i in range(1100):
    fp = pipeline.process_text(text + f" {i % 10}")
    freq_paths.append(fp)

print()
print("DIAGNOSTIC BENCHMARK (5 runs per size, reporting min/mean/max)")
print("=" * 80)

for batch_size in [1, 10, 50, 100, 200, 500, 1000]:
    vectors_batch = np.array([freq_paths[i].vectors for i in range(batch_size)])

    # --- Original ---
    orig_times = []
    for run in range(5):
        subset = freq_paths[:batch_size]
        t0 = time.perf_counter()
        for fp in subset:
            encoder_orig.encode_path(fp)
        t1 = time.perf_counter()
        orig_times.append((t1 - t0) * 1000 / batch_size)

    # --- Optimized ---
    opt_times = []
    for run in range(5):
        t0 = time.perf_counter()
        opt.encode(vectors_batch)
        t1 = time.perf_counter()
        opt_times.append((t1 - t0) * 1000 / batch_size)

    # --- Memory ---
    tracemalloc.start()
    try:
        opt.encode(vectors_batch)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    orig_min, orig_mean, orig_max = min(orig_times), np.mean(orig_times), max(orig_times)
    opt_min, opt_mean, opt_max = min(opt_times), np.mean(opt_times), max(opt_times)
    speedup = orig_mean / opt_mean if opt_mean > 0 else 0

    print(f"  N={batch_size:>5} | Orig: {orig_mean:>6.2f}ms (min={orig_min:.2f} max={orig_max:.2f}) | "
          f"Opt: {opt_mean:>6.3f}ms (min={opt_min:.3f} max={opt_max:.3f}) | "
          f"Speedup: {speedup:>5.1f}x | Peak: {peak/1024:.0f}KB")

print()
print("ANALYSIS:")
print("  If min/mean/max are close => stable measurement")
print("  If min << mean => GC or JIT cache misses in later runs")
print("  If optimal batch = sweet spot => memory bandwidth or cache limit")
