# URCM: Unified u-Resonance Cognitive Mesh
## Investment Pitch — GPU Training Budget

---

## What We Built

A **27.4M parameter cognitive mesh** that reasons through attractor dynamics, not autoregressive sampling:

| Component | Params | Role |
|-----------|--------|------|
| ARC (English encoder) | 15.0M | Encodes text into concept space |
| Brain (PTIL processor) | 8.3M | Processes symbolic reasoning codes |
| Bridge (cross-domain) | 4.0M | Maps between English and reasoning spaces |
| **Total** | **27.4M** | |

**Key insight**: LLMs generate text token-by-token (stochastic). URCM reasons through **energy minimization** in a learned attractor landscape (deterministic).

---

## Toy Benchmark Results (CPU, untrained Bridge)

```
Metric       |  Baseline |   URCM
-------------|-----------|--------
Consistency  |    0.663  |  1.000  (deterministic reasoning)
Formal align |    0.413  |  0.889  (persona resonance)
```

- **100% consistency** — same query always produces same reasoning trajectory
- **Persona-aware** — formal persona gets 2.15x alignment boost over baseline
- **33.8s for 72 comparisons** on CPU — GPU would be 10-50x faster

---

## What GPU Training Enables

### Phase 1: Bridge Training (~10 GPU-hours on P100)
- Train Bridge on 202 English-PTIL pairs
- Expected: persona resonance jumps from 0.3 to 0.8+ across all personas
- Loss: alignment (MSE) + prediction (cross-entropy)

### Phase 2: Full Reasoning Expansion (~50 GPU-hours)
- Expand Brain from 202 to 10,000+ reasoning pairs
- Train on causal, logical, moral, mathematical reasoning chains
- Build FAISS index on brain vectors for retrieval

### Phase 3: Scale Demo (~100 GPU-hours)
- Run 96-task consistency benchmark on P100
- Compare: GPT-4 + system prompt vs URCM attractor + resonance
- Target: show URCM achieves 2-3x consistency improvement

---

## Why This Matters

### Current AI Limitations
1. **Stochastic** — same query gives different answers each time
2. **Persona drift** — LLMs forget who they are mid-conversation
3. **No reasoning backbone** — just pattern matching, not structured thought

### URCM Solution
1. **Deterministic attractors** — energy descent always finds the same stable state
2. **Persona embedding** — identity is encoded in concept space, not just prompts
3. **Cross-domain bridge** — English reasoning maps to structured PTIL codes

### Market Opportunity
- **Enterprise AI**: Consistent brand persona across millions of customer interactions
- **AI Safety**: Predictable, auditable reasoning trajectories
- **Healthcare**: Empathetic counselors that never drift from protocol
- **Education**: Tutoring systems that maintain teaching style

---

## Budget Request

| Item | Cost | Duration |
|------|------|----------|
| P100 GPU hours (Kaggle) | Free tier | 30h/month |
| A100 GPU hours (cloud) | ~$3/hour | Need ~160 hours |
| **Total for full demo** | **~$480** | **2 months** |

### Milestones
1. **Month 1** ($240): Bridge training + 96-task benchmark on P100
2. **Month 2** ($240): Scale Brain + FAISS index + enterprise demo

---

## Files & Reproducibility

- `urcm_best.pt` — ARC encoder (15M params)
- `urcm_ptil.pt` — Brain processor (8.3M params)
- `bridge_enhanced.pt` — Trained Bridge (4M params)
- `vocab_best.json` — 22,929 token vocabulary
- `ptil_encoded_corpus.json` — 202 training pairs
- `toy_benchmark.py` — Reproducible toy test (runs on CPU in 34s)

All code is open. All models are on Kaggle datasets. The benchmark runs locally with zero dependencies beyond PyTorch.

---

## Contact

URCM Project — built on pre-tokenization intelligence research
GitHub: abhi9199-tech42/urcm
Kaggle: abhi12334455
