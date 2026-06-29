"""Profile the real bottleneck: URCM with high data"""
import time, numpy as np, sys, io, tracemalloc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from urcm.core.resonance_encoder import ResonancePathEncoder
from urcm.core.phoneme_mapper import PhonemeFrequencyPipeline

encoder = ResonancePathEncoder(input_dim=24, resonance_dim=512)
pipeline = PhonemeFrequencyPipeline(frequency_dim=24)

text = "All humans are mortal. Socrates is human."

# Component timing
phonemes = pipeline.text_converter.convert_text_to_phonemes(text)
freq_path = pipeline.frequency_mapper.create_frequency_path(phonemes.phonemes)

N = 500

t0 = time.perf_counter()
for _ in range(N):
    pipeline.text_converter.convert_text_to_phonemes(text)
t1 = time.perf_counter()
t_text2phoneme = (t1 - t0) / N * 1000

t0 = time.perf_counter()
for _ in range(N):
    pipeline.frequency_mapper.create_frequency_path(phonemes.phonemes)
t1 = time.perf_counter()
t_phoneme2freq = (t1 - t0) / N * 1000

t0 = time.perf_counter()
for _ in range(N):
    encoder.encode_path(freq_path)
t1 = time.perf_counter()
t_encode = (t1 - t0) / N * 1000

total = t_text2phoneme + t_phoneme2freq + t_encode

print("COMPONENT BREAKDOWN (single text, {} iterations)".format(N))
print("=" * 60)
print("{:<35} {:>8.2f}ms {:>6.1f}%".format("Text -> Phonemes (Python loop)", t_text2phoneme, t_text2phoneme/total*100))
print("{:<35} {:>8.2f}ms {:>6.1f}%".format("Phonemes -> Freq Vectors", t_phoneme2freq, t_phoneme2freq/total*100))
print("{:<35} {:>8.2f}ms {:>6.1f}%".format("Freq -> Resonance (Python loop)", t_encode, t_encode/total*100))
print("-" * 60)
print("{:<35} {:>8.2f}ms {:>6.1f}%".format("TOTAL per text", total, 100))

# Batch scaling - the REAL problem
print()
print("BATCH THROUGHPUT (texts/sec)")
print("=" * 60)
print("{:>10} {:>12} {:>12} {:>12}".format("N", "Total(ms)", "PerText(ms)", "Texts/sec"))
print("{:>10} {:>12} {:>12} {:>12}".format("-"*10, "-"*12, "-"*12, "-"*12))

for n in [1, 10, 50, 100, 500]:
    tracemalloc.start()
    try:
        t0 = time.perf_counter()
        for i in range(n):
            fp = pipeline.process_text(text)
            encoder.encode_path(fp)
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    total_ms = (t1 - t0) * 1000
    per_ms = total_ms / n
    tps = n / (t1 - t0)
    print("{:>10} {:>10.1f}ms {:>10.3f}ms {:>10.1f}".format(n, total_ms, per_ms, tps))

print()
print("WHERE TIME GOES:")
print("=" * 60)
print("  1. Text->Phonemes: Character-by-character Python loop")
print("     - Greedy regex match per character")
print("     - Dict lookup + list append per char")
print()
print("  2. Phonemes->Freq: Per-phoneme dict lookup + normalize")
print("     - No batch/fancy-index support")
print()
print("  3. Freq->State: Per-timestep Python recurrent loop")
print("     - 2 matmuls + tanh + 2 safety calls PER STEP")
print("     - Numpy can't optimize across iterations")
print()
print("ALL THREE STAGES ARE SEQUENTIAL PYTHON LOOPS.")
print("THAT'S THE REAL PROBLEM.")
