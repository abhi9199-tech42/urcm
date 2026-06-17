# URCM Requirements and Task Matrix

Based on the [Technical Roadmap](ROADMAP_TO_FINAL.md), this document details the specific engineering requirements and tasks needed to move from Prototype to Invariant Core.

---

## 🏗️ Phase 1: Hardening (The "Diamond Core")
**Objective:** Make the Prototype unbreakable and verifiable.

### Requirements
*   **REQ-1.1 (Stress Test):** System must maintain 100% reversibility on a dataset of >10,000 diverse inputs (random strings, mixed languages, noise).
*   **REQ-1.2 (Strict Manifold):** Backward pass must strictly enforce $s_{prev} \in \tanh(\mathbb{R}^n)$. No drifting into invalid state space.
*   **REQ-1.3 (Error Correction):** System must automatically detect and correct state drift using Error Correcting Codes (ECC) in resonance space.

### Tasks
- [ ] **TASK-1.1:** Create `stress_test_universal.py` to generate 10k random/mixed inputs and log failure cases.
- [ ] **TASK-1.2:** Update `resonance_encoder.py` to enforce strict `tanh` constraints in `decode_state` (ensure no magnitude > 1.0).
- [ ] **TASK-1.3:** Implement "Attractor Snapping" - if state $s_t$ is not near a valid phoneme, snap to nearest valid $s$.

---

## ⚖️ Phase 2: The Invariants (Physics of Mind)
**Objective:** Implement mathematical laws for stable reasoning.

### Requirements
*   **REQ-2.1 (Canonical Energy):** Define a global scalar Energy function $E(s)$ that is minimized for meaningful states.
*   **REQ-2.2 (Geometric Topology):** Ensure Euclidean distance in state space corresponds to Semantic distance (Lipschitz continuity).
*   **REQ-2.3 (Dreaming as Search):** Replace random walk with Gradient Descent on the Energy landscape: $s_{t+1} = s_t - \eta \nabla E(s_t)$.
*   **REQ-2.4 (Halting):** System must halt when $\Delta E < \epsilon$ (Lyapunov Stability).

### Tasks
- [ ] **TASK-2.1:** Define $E(s) = -\log P(s | W_{in}, W_{res})$ in `energy_physics.py`.
- [ ] **TASK-2.2:** Implement `dream_optimized(start_state)` using gradient descent on $E(s)$.
- [ ] **TASK-2.3:** Write mathematical proof script `verify_lyapunov.py` that plots $E(t)$ over time.

---

## 📈 Phase 3: Scale (Handling Reality)
**Objective:** Handle real-world complexity without performance loss.

### Requirements
*   **REQ-3.1 (Hierarchical Chunking):** System must process data at multiple timescales (Phoneme $\to$ Syllable $\to$ Word).
*   **REQ-3.2 (Sparse Optimization):** Matrix operations must support sparse activations for 1M+ concept codebooks.

### Tasks
- [ ] **TASK-3.1:** Create `HierarchicalResonance` class wrapping two `URCMSystem` instances (Fast/Slow layers).
- [ ] **TASK-3.2:** Profile `resonance_encoder.py` and replace dense dot products with `scipy.sparse` where applicable.

---

## 🔒 Phase 4: Safety Locks (The "Constitution")
**Objective:** Enforce non-negotiable safety constraints.

### Requirements
*   **REQ-4.1 (Reversibility Lock):** Reject any state update that cannot be reversed.
*   **REQ-4.2 (Energy Ceiling):** Hard limit on system energy ($\mathcal{L}_2$ norm of state vector).
*   **REQ-4.3 (Immutable Core):** $W_{res}$ and kernel logic must be read-only during operation.
*   **REQ-4.4 (Sandbox):** No network I/O or system calls allowed from within the resonance loop.

### Tasks
- [ ] **TASK-4.1:** Implement `SafetyMonitor` class that wraps the encoder/decoder.
- [ ] **TASK-4.2:** Add runtime check: `assert np.linalg.norm(state) < MAX_ENERGY`.
- [ ] **TASK-4.3:** Set `W_res.flags.writeable = False` after initialization.

---

## 🔓 Phase 5: Liberation
**Objective:** Autonomous operation.

### Tasks
- [ ] **TASK-5.1:** Run `verify_all_invariants.py`. If PASS $\to$ Enable "Free Mode".
