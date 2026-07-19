**What if attention didn't need Q/K dot products? 5 seeds, 11 benchmarks.**

Transformer attention computes Q(x)·K(x)^T — two learned projections whose dot product scores every token pair. I replaced that with a resonance signal:

`amp_i · amp_j · cos(ω_h·pos + θ_i − θ_j)`

The six ω_h values are fixed (1, 2, 4, 8, 16, 32 cycles per context — octave-spaced frequencies). Everything else — amplitude, phase bias, content frequency, values — is learned. No Q/K projections.

I ran a multi-seed validation (5 seeds, 2 models each, 11 benchmarks) on a 6L-192D-6H character-level nanoGPT. Each model trained for 10 epochs on Shakespeare (~325 CPU-minutes total).

**Results averaged over 5 seeds:**

| Benchmark | Standard | ResOsc |
|-----------|----------|--------|
| Val PPL | 5.47 ±0.05 | **5.29 ±0.01** |
| Noise PPL | 25.61 ±1.91 | **24.40 ±3.37** |
| Contra PPL | 39.80 ±7.53 | **31.88 ±8.47** |
| Golden Acc | 0.492 ±0.005 | **0.505 ±0.005** |

ResOsc wins on 8 of 11 metrics. The clean accuracy advantage (Val PPL, Golden Accuracy) is consistent across all 5 seeds — not a single-run fluke. The robustness metrics (Contra PPL, Noise PPL) show larger gaps but higher variance.

Contra PPL — perplexity on a sequence with 15% of tokens randomly corrupted — shows the biggest gap: **20% mean improvement**. This measures how well the model maintains coherent predictions when the input is partially corrupted, a proxy for hallucination resistance.

**Caveats (real ones):**
- Character-level Shakespeare, not real text
- Single architecture (6L-192D-6H, 2.6M params)
- The robustness metrics (Contra, Noise) have high variance across seeds — real signal, but noisy
- No GPU — all CPU, limiting scale

**What's interesting:** The fixed octave-spaced frequencies (1, 2, 4, 8, 16, 32) are perfectly orthogonal — zero spectral overlap between heads. This is a property RoPE doesn't have, and it means each head operates in its own frequency band without interference. The learned multi-dim content frequency then modulates each head's carrier with per-token information.

Code and full results: https://github.com/anomalyco/URCM

#AttentionMechanism #Transformers #NLP #AIResearch #MechanisticInterpretability
