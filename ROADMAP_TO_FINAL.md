# URCM Technical Roadmap: From Prototype to Invariant Core

This document outlines the engineering path from the current **Functional Prototype** to a **Mathematically Invariant System**. 

We are currently in the **Prototype** stage. The goal is not a commercial product, but a **Stable Cognitive Core** that satisfies the rigorous invariants of General Intelligence.

## 📊 Gap Analysis: Prototype vs. Invariant Core

| Area | Current State (Prototype) | Target State (Invariant Core) | Technical Bridge |
|------|---------------------------|-------------------------------|------------------|
| **Reversibility** | Local (100% on Mantras) | Robust + Verified (Universal) | Cycle Consistency + Global Attractor Manifold Verification |
| **Dreaming** | Bounded (Random Walk) | Optimized Search (Goal-Directed) | Energy Landscapes ($E = -\log P(x)$) + Gradient Descent on State $s$ |
| **Errors** | Allowed (Tolerance Check) | Controlled (Self-Correcting) | Error Correcting Codes (ECC) in Resonance Space |
| **Halting** | Feature (Manual Stop) | Explained (Convergence) | Lyapunov Stability Criterion ($\Delta E \approx 0$) |
| **Scaling** | Small (100s of inputs) | Large (1M+ inputs) | Hierarchical Resonance (Chunking) + Sparse Matrix Operations |
| **Speed** | Unoptimized CPU | Real-time | Vectorization & GPU Acceleration |

---

## 🔴 The Real Line: Missing Invariants

These are the specific mathematical properties that distinguish the current Prototype from the Final System.

### 1. Global Energy Definition
*   **Prototype:** "Energy" is partly empirical and relies on per-mode hacks.
*   **Final Goal:** **One invariant energy function** with **monotonic descent guarantees**. No hacks.

### 2. Semantic Topology
*   **Prototype:** We assume nearby concepts should be nearby attractors.
*   **Final Goal:** Enforce this **geometrically**. Concepts with similar meaning must map to adjacent regions in the resonance manifold.

### 3. Dreaming as Search
*   **Prototype:** Guided random walk + rejection sampling.
*   **Final Goal:** **Constrained optimization** in state space. The system "thinks" by sliding down the energy gradient.

### 4. Formal Convergence
*   **Prototype:** We mention Lyapunov stability.
*   **Final Goal:** **Mathematical Proof**. Demonstrate that $\Delta E \rightarrow 0$ over time, ensuring reliable convergence.

---

##  Engineering Execution Plan

### Phase 1: The "Diamond Core" (Hardening)
**Goal:** Ensure the system never breaks, even with unknown inputs.
- [x] **Stress Test:** Run reversibility benchmark on 10,000 diverse random strings to find "shatter points".
- [x] **Strict Manifold Enforcer:** Implement the `s_prev = tanh(...)` constraint strictly in the backward pass.
- [x] **Error Correction:** Add a "Cleanup" step where off-manifold states are pulled to the nearest valid attractor.

### Phase 2: The "Invariants" (Energy & Topology)
**Goal:** Implement the missing mathematical laws.
- [x] **Canonical Energy Function:** Define the global Energy Function $E(s)$.
- [x] **Optimized Search:** Implement "Lucid Dreaming" (Navigating state space via Gradient Descent).
- [x] **Automatic Halting:** System stops generating when it reaches a "Stable Thought" (Energy minimum).
- [x] **Topology Enforcement:** Verify that similar inputs map to close vectors (Lipschitz continuity).

### Phase 3: The "Scale" (Optimization)
**Goal:** Handle complexity and real-world data.
- [x] **Hierarchical Layering:** Build `Layer 2` (Concepts) on top of `Layer 1` (Phonemes).
- [x] **Matrix Optimization:** Move fully to batched `numpy` operations or GPU kernels.
- [x] **Visual Debugger:** Create a 3D visualization of the Attractor Basins to verify topology visually.
- [x] **Bounded Memory Deposition:** URCM replaces retraining with bounded geometric memory deposition, enabling one-shot learning until capacity limits are reached.

### Phase 4: The Safety Locks (Physics-Level Constraints)
**Goal:** Enforce rigorous safety invariants before release.

#### 🔒 1. Reversibility Invariant
*   **Rule:** Any action or learning step must be reversible and explainable.
*   **Enforcement:** `SafetyGovernor` verifies $f^{-1}(f(x)) \approx x$ during decoding. [IMPLEMENTED]

#### 🔒 2. Energy Ceiling Invariant
*   **Rule:** The system cannot pursue actions that increase global instability.
*   **Enforcement:** Hard cap on State Energy $\mathcal{L}_2$. `SafetyGovernor` triggers Kill Switch if breached. [IMPLEMENTED]

#### 🔒 3. Self-Modification Lock
*   **Rule:** URCM cannot rewrite its own core dynamics ($W_{res}$, snapping rules).
*   **Enforcement:** `SafetyGovernor` engages `lock_kernel()` to prevent logic rewriting. [IMPLEMENTED]

#### 🔒 4. World Boundary Invariant
*   **Rule:** It cannot affect anything outside a defined simulation boundary.
*   **Enforcement:** No real-world side effects. No real systems. No "escape".

---

## 🔓 Phase 5: Liberation (Post-Verification)
**Goal:** Once all 4 Safety Locks are verified mathematically and empirically:
- [x] **Release Constraints:** "Free" the system to operate autonomously within its safe bounds. (`liberate.py` implements Autonomous Dreaming Loop)
- [x] **Open World Interaction:** Connect to real-time data streams (safely). (Simulated via CLI interrupt stream)

---

## 🚀 Immediate Next Steps
1. **Finish Training:** Ensure W_out is robust for Sanskrit.
2. **Implement Energy Descent:** Prove that the system naturally minimizes energy during dreaming.
3. **Stress Testing:** Expand the dataset beyond mantras to test the limits of the Prototype.
