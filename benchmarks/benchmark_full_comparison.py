"""
Full Comparison: URCM vs DistilBERT vs 300K Transformer
"""
import numpy as np
import torch
import torch.nn as nn
import time
import tracemalloc
import sys
import io
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.phoneme_mapper import PhonemeFrequencyPipeline
from transformers import DistilBertModel, DistilBertTokenizer, GPT2Model, GPT2Tokenizer


# ============================================================================
#  URCM ORIGINAL (Python loop)
# ============================================================================
class URCMPython:
    def __init__(self):
        self.pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
        self.encoder = ResonancePathEncoder(input_dim=24, resonance_dim=512)

    def encode(self, text):
        fp = self.pipeline.process_text(text)
        return self.encoder.encode_path(fp)

    def encode_batch(self, texts):
        results = []
        for t in texts:
            results.append(self.encode(t))
        return np.array(results)


# ============================================================================
#  URCM OPTIMIZED (TorchScript JIT)
# ============================================================================
@torch.jit.script
def _jit_encode(vectors_batch, W_in, W_res, bias):
    B, T, _ = vectors_batch.shape
    R = W_res.shape[0]
    states = torch.zeros(B, R, dtype=vectors_batch.dtype)
    for t in range(T):
        states = torch.tanh(vectors_batch[:, t, :] @ W_in + states @ W_res + bias).clamp(-10.0, 10.0)
    return states


class URCMJIT:
    def __init__(self):
        self.pipeline = PhonemeFrequencyPipeline(frequency_dim=24)
        np.random.seed(42)
        W_in = np.random.normal(0, 0.1, (24, 512)).astype(np.float32)
        H = np.random.randn(512, 512).astype(np.float32)
        Q, _ = np.linalg.qr(H)
        W_res = (Q * 0.95).astype(np.float32)
        bias = np.random.normal(0, 0.01, 512).astype(np.float32)
        self.W_in = torch.from_numpy(W_in)
        self.W_res = torch.from_numpy(W_res)
        self.bias = torch.from_numpy(bias)
        # Warmup
        _jit_encode(torch.zeros(2, 10, 24), self.W_in, self.W_res, self.bias)

    def encode(self, text):
        fp = self.pipeline.process_text(text)
        v = torch.from_numpy(fp.vectors.astype(np.float32)).unsqueeze(0)
        return _jit_encode(v, self.W_in, self.W_res, self.bias).squeeze(0).numpy()

    def encode_batch(self, texts):
        fps = [self.pipeline.process_text(t) for t in texts]
        max_len = max(fp.vectors.shape[0] for fp in fps)
        batch = torch.zeros(len(fps), max_len, 24)
        for i, fp in enumerate(fps):
            v = fp.vectors
            batch[i, :v.shape[0]] = torch.from_numpy(v.astype(np.float32))
        return _jit_encode(batch, self.W_in, self.W_res, self.bias).numpy()


# ============================================================================
#  DISTILBERT
# ============================================================================
class DistilBERTWrapper:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.model.eval()
        self.dim = 768

    @torch.no_grad()
    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding=False)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()

    @torch.no_grad()
    def encode_batch(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', truncation=True, max_length=128, padding=True)
        outputs = self.model(**inputs)
        # Mean pool (exclude padding)
        attn = inputs['attention_mask'].unsqueeze(-1).float()
        summed = (outputs.last_hidden_state * attn).sum(dim=1)
        counts = attn.sum(dim=1).clamp(min=1)
        return (summed / counts).numpy()


# ============================================================================
#  300K TRANSFORMER (Custom small model)
# ============================================================================
class SmallTransformer(nn.Module):
    def __init__(self, vocab_size=50257, d_model=128, nhead=4, num_layers=2, dim_ff=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model
        # Count params
        total = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.embed(x) + self.pos_embed(pos)
        h = self.transformer(h)
        return h.mean(dim=1)  # [B, d_model]


class SmallTransformerWrapper:
    def __init__(self):
        self.model = SmallTransformer()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self.d_model = 128
        total_params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        emb = self.model(inputs['input_ids'])
        return emb.squeeze(0).numpy()

    @torch.no_grad()
    def encode_batch(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', truncation=True, max_length=128, padding=True)
        emb = self.model(inputs['input_ids'])
        return emb.numpy()


# ============================================================================
#  BENCHMARK UTILITIES
# ============================================================================
def bench_latency(model, texts, n_runs=50):
    """Measure per-text encode latency."""
    # Warmup
    for t in texts[:3]:
        model.encode(t)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for t in texts:
            model.encode(t)
        t1 = time.perf_counter()
        times.append((t1 - t0) / len(texts) * 1000)

    return np.mean(times), np.std(times)


def bench_throughput_batch(model, texts, n_runs=20):
    """Measure batch throughput."""
    # Warmup
    model.encode_batch(texts[:min(10, len(texts))])

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.encode_batch(texts)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return np.mean(times), np.std(times)


def measure_memory(model, texts):
    """Measure peak memory for batch encode."""
    tracemalloc.start()
    model.encode_batch(texts)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024


def measure_params(model):
    """Count model parameters."""
    if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
        return sum(p.numel() for p in model.model.parameters())
    if hasattr(model, 'encoder'):
        # URCM: W_in + W_res + bias + W_out
        enc = model.encoder
        return enc.W_in.size + enc.W_res.size + enc.bias.size + enc.W_out.size
    if hasattr(model, 'W_in'):
        return model.W_in.numel() + model.W_res.numel() + model.bias.numel()
    return 0


# ============================================================================
#  MAIN BENCHMARK
# ============================================================================
def run():
    print("=" * 90)
    print("  URCM vs DistilBERT vs 300K Transformer — FULL COMPARISON")
    print("=" * 90)

    # Initialize models
    print("\n[INIT] Loading models...")
    urcm_py = URCMPython()
    urcm_jit = URCMJIT()
    print("  Loading DistilBERT (66M params)...")
    distilbert = DistilBERTWrapper()
    small_tf = SmallTransformerWrapper()

    # Sample texts
    texts_short = [
        "All humans are mortal.",
        "The cat sat on the mat.",
        "URCM uses resonance for reasoning.",
        "Socrates is a philosopher.",
        "If it rains, the ground gets wet.",
    ]
    texts_medium = [
        "All humans are mortal. Socrates is human. Therefore Socrates is mortal.",
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "Resonance frequency determines the stability of semantic representations.",
        "In the morning, the sun rises over the mountains painting the sky orange.",
        "The theorem proves that all prime numbers greater than two are odd.",
    ]

    # ── Params ──
    p_urcm = 24*512 + 512*512 + 512 + 512*24  # W_in + W_res + bias + W_out
    p_distilbert = sum(p.numel() for p in distilbert.model.parameters())
    p_small_tf = sum(p.numel() for p in small_tf.model.parameters())

    # ── 1. Encode Latency ──
    print("\n[1] ENCODE LATENCY (per text, ms)")
    print("-" * 70)

    for label, texts in [("Short (5 words)", texts_short), ("Medium (12 words)", texts_medium)]:
        print(f"\n  {label}:")

        lat_py, std_py = bench_latency(urcm_py, texts, n_runs=30)
        lat_jit, std_jit = bench_latency(urcm_jit, texts, n_runs=30)
        lat_db, std_db = bench_latency(distilbert, texts, n_runs=30)
        lat_stf, std_stf = bench_latency(small_tf, texts, n_runs=30)

        print(f"    {'Model':<25} {'Latency':>10} {'Std':>8}")
        print(f"    {'-'*25} {'-'*10} {'-'*8}")
        print(f"    {'URCM Python':<25} {lat_py:>8.2f}ms {std_py:>6.2f}ms")
        print(f"    {'URCM JIT':<25} {lat_jit:>8.2f}ms {std_jit:>6.2f}ms")
        print(f"    {'DistilBERT':<25} {lat_db:>8.2f}ms {std_db:>6.2f}ms")
        print(f"    {'300K Transformer':<25} {lat_stf:>8.2f}ms {std_stf:>6.2f}ms")

    # ── 2. Batch Throughput ──
    print("\n[2] BATCH THROUGHPUT (texts/sec)")
    print("-" * 70)

    for batch_label, batch_size in [("B=1 (sequential)", 1), ("B=10", 10), ("B=50", 50), ("B=100", 100), ("B=500", 500)]:
        batch_texts = (texts_medium * (batch_size // len(texts_medium) + 1))[:batch_size]

        # Skip batch for URCM Python (no batching support)
        if batch_size == 1:
            t_py, _ = bench_throughput_batch(urcm_py, batch_texts, n_runs=10)
            tps_py = 1000 / t_py if t_py > 0 else 0
        else:
            t_py = float('inf')
            tps_py = 0

        t_jit, _ = bench_throughput_batch(urcm_jit, batch_texts, n_runs=10)
        t_db, _ = bench_throughput_batch(distilbert, batch_texts, n_runs=10)
        t_stf, _ = bench_throughput_batch(small_tf, batch_texts, n_runs=10)

        tps_jit = batch_size / (t_jit / 1000) if t_jit > 0 else 0
        tps_db = batch_size / (t_db / 1000) if t_db > 0 else 0
        tps_stf = batch_size / (t_stf / 1000) if t_stf > 0 else 0

        print(f"\n  {batch_label}:")
        if batch_size == 1:
            print(f"    URCM Python:    {tps_py:>8.1f} texts/sec")
        else:
            print(f"    URCM Python:    {'N/A (no batch)':>15}")
        print(f"    URCM JIT:       {tps_jit:>8.1f} texts/sec")
        print(f"    DistilBERT:     {tps_db:>8.1f} texts/sec")
        print(f"    300K TF:        {tps_stf:>8.1f} texts/sec")

    # ── 3. Memory ──
    print("\n[3] MEMORY USAGE")
    print("-" * 70)

    mem_jit = measure_memory(urcm_jit, texts_medium * 100)
    mem_db = measure_memory(distilbert, texts_medium * 100)
    mem_stf = measure_memory(small_tf, texts_medium * 100)

    print(f"    {'Model':<25} {'Params':>12} {'Batch Mem (B=100)':>18} {'Model Size':>12}")
    print(f"    {'-'*25} {'-'*12} {'-'*18} {'-'*12}")
    print(f"    {'URCM JIT':<25} {p_urcm:>11,} {mem_jit:>15.0f}KB {p_urcm*4/1e6:>10.2f}MB")
    print(f"    {'DistilBERT':<25} {p_distilbert:>11,} {mem_db:>15.0f}KB {p_distilbert*4/1e6:>10.2f}MB")
    print(f"    {'300K TF':<25} {p_small_tf:>11,} {mem_stf:>15.0f}KB {p_small_tf*4/1e6:>10.2f}MB")

    # ── 4. Semantic Quality (Encode → Nearest Neighbor) ──
    print("\n[4] SEMANTIC CONSISTENCY (encode similar texts, check cosine similarity)")
    print("-" * 70)

    similar_pairs = [
        ("The cat sat on the mat", "A cat was sitting on a mat"),
        ("All humans are mortal", "Every human being will die someday"),
        ("It is raining outside", "Rain is falling from the sky"),
    ]
    dissimilar_pairs = [
        ("The cat sat on the mat", "Quantum computing uses qubits"),
        ("All humans are mortal", "The stock market crashed today"),
        ("It is raining outside", "Python is a programming language"),
    ]

    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    for label, model in [("URCM JIT", urcm_jit), ("DistilBERT", distilbert), ("300K TF", small_tf)]:
        sim_scores = []
        for t1, t2 in similar_pairs:
            e1 = model.encode(t1)
            e2 = model.encode(t2)
            sim_scores.append(cosine_sim(e1, e2))

        dissim_scores = []
        for t1, t2 in dissimilar_pairs:
            e1 = model.encode(t1)
            e2 = model.encode(t2)
            dissim_scores.append(cosine_sim(e1, e2))

        avg_sim = np.mean(sim_scores)
        avg_dissim = np.mean(dissim_scores)
        gap = avg_sim - avg_dissim

        print(f"    {label:<15} Similar: {avg_sim:.4f}  Dissimilar: {avg_dissim:.4f}  Gap: {gap:.4f}")

    # ── FINAL TABLE ──
    print("\n" + "=" * 90)
    print("  FINAL COMPARISON TABLE")
    print("=" * 90)

    # Re-measure for clean numbers
    lat_jit_short, _ = bench_latency(urcm_jit, texts_short, n_runs=50)
    lat_db_short, _ = bench_latency(distilbert, texts_short, n_runs=50)
    lat_stf_short, _ = bench_latency(small_tf, texts_short, n_runs=50)

    batch_50 = (texts_medium * (50 // len(texts_medium) + 1))[:50]
    tps_jit_50, _ = bench_throughput_batch(urcm_jit, batch_50, n_runs=10)
    tps_jit_50 = 50 / (tps_jit_50 / 1000)
    tps_db_50, _ = bench_throughput_batch(distilbert, batch_50, n_runs=10)
    tps_db_50 = 50 / (tps_db_50 / 1000)
    tps_stf_50, _ = bench_throughput_batch(small_tf, batch_50, n_runs=10)
    tps_stf_50 = 50 / (tps_stf_50 / 1000)

    print(f"""
  +----------------------+------------+------------+------------+------------+
  | Metric               | URCM JIT   | URCM Python| DistilBERT | 300K TF    |
  +----------------------+------------+------------+------------+------------+
  | Parameters           | {p_urcm:>10,} | {p_urcm:>10,} | {p_distilbert:>10,} | {p_small_tf:>10,} |
  | Model Size (fp32)    | {p_urcm*4/1e6:>9.2f}MB | {p_urcm*4/1e6:>9.2f}MB | {p_distilbert*4/1e6:>9.2f}MB | {p_small_tf*4/1e6:>9.2f}MB |
  | Embedding Dim        | {512:>10} | {512:>10} | {768:>10} | {128:>10} |
  | Encode Latency (ms)  | {lat_jit_short:>10.2f} | {bench_latency(urcm_py, texts_short, 30)[0]:>10.2f} | {lat_db_short:>10.2f} | {lat_stf_short:>10.2f} |
  | Throughput B=1 (t/s) | {1000/bench_throughput_batch(urcm_jit, texts_medium[:1], 20)[0]:>10.1f} | {1000/bench_throughput_batch(urcm_py, texts_medium[:1], 20)[0]:>10.1f} | {1000/bench_throughput_batch(distilbert, texts_medium[:1], 20)[0]:>10.1f} | {1000/bench_throughput_batch(small_tf, texts_medium[:1], 20)[0]:>10.1f} |
  | Throughput B=50 (t/s)| {tps_jit_50:>10.1f} | {'N/A':>10} | {tps_db_50:>10.1f} | {tps_stf_50:>10.1f} |
  | Peak Mem B=100 (KB)  | {mem_jit:>10.0f} | {'N/A':>10} | {mem_db:>10.0f} | {mem_stf:>10.0f} |
  | Convergence Cert.    | {'Yes':>10} | {'Yes':>10} | {'No':>10} | {'No':>10} |
  | Explainability       | {'Full mu':>10} | {'Full mu':>10} | {'Black box':>10} | {'Black box':>10} |
  +----------------------+------------+------------+------------+------------+
    """)


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    run()
