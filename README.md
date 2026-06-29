# URCM: Unified μ-Resonance Cognitive Mesh

A value-grounded, local-first cognitive architecture that replaces token-based processing with continuous frequency-based representations. Runs entirely on consumer hardware with no cloud dependency.

## Architecture

URCM implements a three-layer hierarchical reasoning system built on μ-stability theory:

```
Input Text → Phoneme Pipeline → Hierarchical Encoder → Convergence Engine → Reasoning Output
     ↓              ↓                    ↓                      ↓
Frequency Vectors  Resonance States   Concept Resonance    Beam Search with
(Sanskrit phonemes) (temporal encoding) (hierarchical)      μ-stability selection
```

### Core Components

| Module | Purpose |
|--------|---------|
| `phoneme_mapper.py` | Maps text to Sanskrit phoneme frequency vectors |
| `resonance_encoder.py` | Encodes frequency paths to resonance states |
| `hierarchical_encoder.py` | Multi-layer encoding (phoneme → word → concept) |
| `convergence_engine.py` | Beam search reasoning with μ-stability selection |
| `executive.py` | Left-brain executive control with working memory |
| `reasoning.py` | Right-brain cognitive reasoning engine |
| `safety.py` | Physics-level constraint enforcement |
| `values.py` | Axiomatic alignment (Truth, Safety, Wisdom) |
| `error_handling.py` | Domain-specific error recovery |
| `metacognition.py` | Self-healing thought loop detection |

### Key Theory: μ-Stability

The system selects reasoning paths based on:

```
μ = ρ / (1 + χ)
```

Where:
- **μ** (mu): Overall resonance/stability score
- **ρ** (rho): Semantic density (inverse entropy)
- **χ** (chi): Transformation cost (energy expenditure)

Higher μ indicates more stable, coherent reasoning.

## Installation

### Prerequisites

- Python 3.10+
- 4GB+ RAM recommended
- No GPU required

### Setup

```bash
git clone https://github.com/abhi9199-tech42/urcm.git
cd urcm
pip install -r requirements.txt
```

### Model Download (Optional - for LLM integration)

```bash
python download_model.py
```

This downloads TinyLlama-1.1B for the dual-brain interface.

## Usage

### Core System

```python
from urcm.core import URCMSystem

# Initialize the system
system = URCMSystem()

# Process a query
result = system.process_query("What is the meaning of life?")

# Access reasoning metrics
print(f"Final μ: {result.mu_trajectory[-1]:.4f}")
print(f"Convergence steps: {len(result.mu_trajectory)}")
```

### Interactive Console

```bash
python axiom.py
```

Provides a dual-brain interface combining URCM reasoning with local LLM generation.

### Benchmarks

```bash
python benchmark_axiom.py        # System performance
python benchmark_vs_gpt.py       # Comparison vs ChatGPT
python benchmark_performance.py  # URCM vs token-based systems
```

### Visualization

```bash
streamlit run app.py
```

Opens the visual debugger with real-time resonance state visualization.

## Value System

URCM implements 10 core axioms as permanent attractors/repulsors in resonance space:

| Axiom | Valence | Category |
|-------|---------|----------|
| Truth | +1.0 | Epistemic |
| Safety | +1.0 | Moral |
| Benefit | +0.9 | Social |
| Help | +0.8 | Social |
| Clarity | +0.8 | Epistemic |
| Understanding | +0.8 | Epistemic |
| Harm | -1.0 | Negative |
| Deception | -1.0 | Negative |
| Pain | -0.9 | Negative |
| Destruction | -0.9 | Negative |

## Safety Features

- **Energy Ceiling**: Prevents runaway state energy
- **Spectral Stability**: Ensures weight matrix stability
- **Reversibility Invariant**: Maintains state reversibility
- **Self-Modification Lock**: Prevents unauthorized core changes
- **Valence Filtering**: Blocks harmful content generation

## Project Structure

```
urcm/
├── core/                    # Core reasoning engine
│   ├── system.py           # Main URCMSystem class
│   ├── convergence_engine.py
│   ├── phoneme_mapper.py
│   ├── resonance_encoder.py
│   ├── hierarchical_encoder.py
│   ├── executive.py        # Left-brain controller
│   ├── reasoning.py        # Right-brain engine
│   ├── safety.py           # Safety governor
│   ├── values.py           # Axiomatic alignment
│   ├── error_handling.py   # Error recovery
│   ├── metacognition.py    # Self-healing
│   ├── memory.py           # Geometric memory
│   ├── logic_gates.py      # Geometric logic
│   └── theory.py           # μ-stability math
├── domains/                 # Domain-specific modules
├── integration/            # External integrations
├── ops/                    # Observability & metrics
├── tests/                  # Test suite (107 files)
├── pages/                  # Streamlit dashboard
├── deploy/                 # Deployment configs
└── examples/               # Usage examples
```

## Testing

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=urcm --cov-report=html

# Run specific test categories
pytest tests/ -m unit
pytest tests/ -m property
pytest tests/ -m integration
```

## Benchmarks

| Metric | Value |
|--------|-------|
| Convergence Time | <12ms average |
| Memory Footprint | <45MB |
| Test Suite | 107 files |
| Compression Ratio | >2x vs token-based |

## Roadmap

See [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md) for the plan to reach production readiness.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

Copyright (C) 2024 URCM Contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

## Acknowledgments

- Built on μ-stability theory for semantic convergence
- Sanskrit phoneme mapping for low-entropy semantics
- Echo State Networks with orthogonal weight initialization
