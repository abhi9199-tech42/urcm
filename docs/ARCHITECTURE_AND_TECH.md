# URCM Tech Stack and Architecture

Unified μ-Resonance Cognitive Mesh (URCM) — a value-grounded, resonance-based cognitive engine that generates and structures concept trajectories, with optional LLM fluency via AXIOM.

## Summary
- Core paradigm: continuous resonance in concept space with energy minimization and safety constraints
- Right-brain dynamics (resonance, metacognition, logic gating) + Left-brain structure (Paninian grammar)
- Knowledge accretion via geometric memory and deterministic concept vectors
- Optional LLM merger for fluent generation, guided by URCM’s structured insight and values

## Architecture Diagram (ASCII)
```
                 ┌────────────────────────────┐
 Input           │  Phoneme Frequency Pipeline│
 (Text) ───────▶ │  (Text→Phonemes→Frequencies) 
                 └───────────┬────────────────┘
                             │ FrequencyPath (L1)
                             ▼
                 ┌────────────────────────────┐
                 │  Resonance Path Encoder    │
                 │  (W_in/W_res/W_out, tanh)  │
                 └───────────┬────────────────┘
                             │ gated by
                             ▼
                 ┌────────────────────────────┐
                 │  Oscillatory Gating        │
                 │  (global rhythm σ(Wg·g(t)))│
                 └───────────┬────────────────┘
                             │
                             ▼
                 ┌────────────────────────────┐
                 │  Reasoning Engine          │
                 │  - Energy descent          │
                 │  - Metacognition           │
                 │  - Logic gates             │
                 │  - Value system            │
                 └───────────┬────────────────┘
                             │                ┌───────────────────────┐
                             │                │  Geometric Memory     │
                             │                │  (Attractor deposition│
                             │                │   via trajectories)   │
                             │                └──────────┬────────────┘
                             ▼                           │ concept_map
                 ┌────────────────────────────┐          │
                 │  Sanskrit Bridge & Grammar │◀─────────┘
                 │  (structure thought)       │
                 └───────────┬────────────────┘
                             │ structured insight
                             ▼
                 ┌────────────────────────────┐
                 │  AXIOM (LLM Merger)        │
                 │  - Value filter            │
                 │  - LLMBridge (llama-cpp)   │
                 │  - Grounded prompt         │
                 └───────────┬────────────────┘
                             │ fluent output
                             ▼
                          Response
```

## Key Modules and Responsibilities
- System orchestrator: [urcm/core/system.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/urcm/core/system.py)
- Data models: [urcm/core/data_models.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/urcm/core/data_models.py)
- Phoneme-to-frequency pipeline: [urcm/core/phoneme_mapper.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/urcm/core/phoneme_mapper.py)
- Resonance encoder: [urcm/core/resonance_encoder.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/urcm/core/resonance_encoder.py)
- Oscillatory gating: [urcm/core/oscillatory_gating.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/urcm/core/oscillatory_gating.py)
- Reasoning engine: [urcm/core/reasoning.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/urcm/core/reasoning.py)
- Value system: [urcm/core/values.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/urcm/core/values.py)
- Grammar (Paninian): [urcm/core/sanskrit_grammar.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/urcm/core/sanskrit_grammar.py)
- Knowledge ingestion & memory: [urcm/core/ingest.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/urcm/core/ingest.py)
- Error recovery: [urcm/core/error_handling.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/urcm/core/error_handling.py)
- Performance suite: [urcm/core/performance.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/urcm/core/performance.py)
- Multimodal encoders (Phase 2): [urcm/core/multimodal.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/urcm/core/multimodal.py)

## Tech Stack
- Python + numpy core computations; deterministic vector operations
- Tests: pytest, Hypothesis property-based tests
- LLM integration: llama-cpp-python via [urcm/core/llm_bridge.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/urcm/core/llm_bridge.py)
- Demo/UI: Streamlit app [demo_app.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/demo_app.py)
- Utility libs used: requests, beautifulsoup4 (content ingestion), psutil (perf), colorama (CLI styling)
- Declared in [requirements.txt](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/requirements.txt)

## Reasoning Flow
- Initialize from query concept vector (deterministic lookup; partial matches allowed)
- Predict next state via W_res and tanh; measure energy and decode nearest concept
- Metacognition adapts learning rate and injects frustration noise when stuck
- Apply gradient descent toward nearest codebook vectors with constraints:
  - User constraints
  - Axiomatic constraints (values)
  - Logic gate gradients
- Normalize and accumulate trajectory; translate and structure into sentence
- See [reasoning.py step](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/urcm/core/reasoning.py#L160)

## Safety & Values
- Constitutional axioms (truth, safety, harm avoidance) embedded as attractors/repulsors
- Valence evaluation and alignment gradient steer away from harmful states
- Pre-generation filter in AXIOM refuses unsafe queries
- See [values.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/urcm/core/values.py)

## Knowledge Accretion
- Deterministic 512-dim concept vectors via MD5-seeded normal draws (unit-norm)
- Sentence ingestion deposits trajectories into W_res, deepening attractor basins
- Brain stored in .pkl (concept_map, L2 weights), enabling reproducible runs
- See [ingest.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/urcm/core/ingest.py)

## LLM Merger (AXIOM)
- Merges URCM’s structured insight with LLM fluency under value constraints
- LLMBridge loads local GGUF via llama-cpp; falls back to mock mode if missing
- AXIOM prompts include structured thought and valence score for grounded outputs
- See [axiom.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/axiom.py), [llm_bridge.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/urcm/core/llm_bridge.py)

## Multimodal (Phase 2)
- VisualEncoder: hash-derived features projected to concept space (placeholder for ViT/CNN)
- AudioProcessor: filename-derived “transcription” mapped via phoneme pipeline
- VideoProcessor: simple fusion of visual+audio embeddings
- See [multimodal.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/urcm/core/multimodal.py)

## Performance & Testing
- Performance suite measures compression, latency, throughput; compare to token systems  
  - [benchmark_performance.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/benchmark_performance.py)
- Property-based tests across data models, latent space, gating, performance  
  - [tests/test_data_models.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/tests/test_data_models.py)
  - [tests/test_performance.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/tests/test_performance.py)
  - [tests/test_complete_system_integration.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/tests/test_complete_system_integration.py)
  - [tests/test_isre_integration.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/tests/test_isre_integration.py)

## Scripts & Usage
- Train/ingest: [train_urcm.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/train_urcm.py), [train_identity.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/train_identity.py)
- Validate: [run_final_validation.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/run_final_validation.py), [verify_urcm.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/verify_urcm.py)
- Demo: [demo_app.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/demo_app.py), AXIOM CLI via [axiom.py](file:///C:/Users/kriti/OneDrive/Unified%20%E2%80%9C%CE%BC-Resonance%E2%80%9D%20Cognitive%20Mesh%20(URCM)/axiom.py)

## Future Extensions
- Replace simulated encoders with real ViT/CNN and Whisper/STT
- Expand logic gates and μ-convergence diagnostics for complex reasoning
- Distributed mesh consensus across nodes for collective resonance

