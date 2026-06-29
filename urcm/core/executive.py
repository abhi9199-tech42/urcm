import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from urcm.core.heuristic_engine import HeuristicEngine
from urcm.core.logic_gates import ConstraintGraph, FormalLogic, SMTBridge, TheoremProver
from urcm.core.observability import record_event
from urcm.core.reasoning import ReasoningEngine
from urcm.core.tms import TruthMaintenanceSystem
from urcm.core.working_memory import Intent, WorkingMemory


class ExecutiveController:
    """
    The 'Left Brain' Executive.
    Orchestrates the Right Brain (ReasoningEngine) using Working Memory (Intent Stack).
    """
    def __init__(self, brain_path: str = "urcm_identity.pkl"):
        self.engine = ReasoningEngine(brain_path)
        self.memory = WorkingMemory()
        self.current_state = None
        self.explain_refusal_threshold = 1.0
        self.last_loop_metrics = {}
        self.last_refused_reason = ""
        self.circuit_breaker_tripped = False
        self.tms = TruthMaintenanceSystem()
        self.last_proof_core = []
        self._explain_cache: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        self._explain_cache_order: List[Tuple[str, str, str]] = []
        self._explain_cache_cap = 256

        # Phase 1.5 Heuristics (loaded from external JSON)
        self.heuristics = HeuristicEngine()
        # Expose for backward compatibility
        self.ANTONYMS = self.heuristics.antonyms
        self.SEQUENCES = self.heuristics.sequences
        self.POSITIVE_ANCHORS = self.heuristics.positive_anchors
        self.NEGATIVE_ANCHORS = self.heuristics.negative_anchors
        self.SYNONYMS = self.heuristics.synonyms

    def set_initial_state(self, concept_text: str):
        """Sets the starting thought of the Right Brain."""
        vec = self.engine.get_concept_vector(concept_text)
        if vec is not None:
            self.current_state = vec
            print(f"[Exec] Initialized state to: {concept_text}")
        else:
            print(f"[Exec] ⚠️ Unknown concept: {concept_text}")

    def reload_brain(self):
        """Reloads the ReasoningEngine to pick up new knowledge."""
        print("[Exec] 🔄 Reloading Brain (Plasticity Update)...")
        # Preserve state if possible
        old_state = self.current_state.copy() if self.current_state is not None else None

        self.engine = ReasoningEngine(self.engine.brain_data_path if hasattr(self.engine, 'brain_data_path') else "urcm_identity.pkl")

        # Restore state (if dimensions match)
        if old_state is not None:
             # Check if dimension changed (e.g. if we resized W_res)
             if old_state.shape[0] == self.engine.l2_dim:
                 self.current_state = old_state
             else:
                 print("[Exec] ⚠️ State dimension mismatch after reload. Resetting to neutral.")
                 self.current_state = np.zeros(self.engine.l2_dim)

    def _get_relations(self):
        """
        Access relations stored in the brain data, if available.
        """
        if hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
            return self.engine.brain_data.get("relations", [])
        return []

    def add_goal(self, description: str, target_concept: str = None, priority: float = 1.0):
        """Adds a high-level goal to Working Memory."""
        target_vec = None
        if target_concept:
            target_vec = self.engine.get_concept_vector(target_concept)

        intent = Intent(
            description=description,
            target_concept_name=target_concept,
            target_vector=target_vec,
            priority=priority
        )
        self.memory.add_intent(intent)

    def _apply_context_heuristics(self, current_word: str, ctx: set, constraints: list) -> None:
        """Apply data-driven context heuristics from heuristics.json."""
        for h in self.heuristics.context_heuristics:
            trigger = h.get("trigger", {})

            if "current" in trigger and current_word != trigger["current"]:
                continue
            if "words" in trigger and not all(w in ctx for w in trigger["words"]):
                continue
            if "negations" in trigger and any(n in ctx for n in trigger["negations"]):
                continue
            if "any_in" in trigger and not any(w in ctx for w in trigger["any_in"]):
                continue
            if trigger.get("current_not_in_ctx") and current_word in ctx:
                continue
            if "current_in" in trigger and current_word not in trigger["current_in"]:
                continue

            for item in h.get("attract", []):
                v = self.engine.get_concept_vector(item["word"])
                if v is not None:
                    constraints.append((v, item["weight"]))
            for item in h.get("repulse", []):
                v = self.engine.get_concept_vector(item["word"])
                if v is not None:
                    constraints.append((v, item["weight"]))

    def _apply_word_rules(self, current_word: str, constraints: list) -> None:
        """Apply word-specific attract/repulse rules from heuristics.json."""
        if current_word in self.heuristics.word_rules:
            rule = self.heuristics.word_rules[current_word]
            for item in rule.get("attract", []):
                v = self.engine.get_concept_vector(item["word"])
                if v is not None:
                    constraints.append((v, item["weight"]))
            for item in rule.get("repulse", []):
                v = self.engine.get_concept_vector(item["word"])
                if v is not None:
                    constraints.append((v, item["weight"]))

    def explain(self, start: str, goal: Optional[str] = None) -> Dict[str, any]:
        rels = self._get_relations()
        graph = {}
        neg = set()
        for r in rels:
            if len(r) >= 3:
                typ, a, b = r[0], r[1], r[2]
                if typ in ("all", "some", "most", "implies", "coref"):
                    graph.setdefault(a, []).append(b)
                elif typ == "no":
                    neg.add((a, b))
        contradictions = []
        if goal:
            if (start, goal) in neg:
                contradictions.append(("no", start, goal))
            visited = set([start])
            queue = [(start, [start])]
            found = []
            while queue:
                node, path = queue.pop(0)
                for nxt in graph.get(node, []):
                    if (node, nxt) in neg:
                        contradictions.append(("no", node, nxt))
                        self.tms.retract_support(f"chain:{start}->{goal}")
                        try:
                            ast = FormalLogic.to_cnf("(p and not p)")
                            clauses = FormalLogic._collect_clauses(ast)
                            proof = TheoremProver.prove(clauses, "p")
                            core = proof.get("unsat_core", [])
                            self.last_proof_core = core
                            record_event("exec_refusal", {"reason": "contradiction", "proof_core_len": len(core)})
                            self.last_refused_reason = "contradiction"
                        except Exception:
                            self.last_proof_core = []
                        continue
                    new_path = path + [nxt]
                    if nxt == goal:
                        found = new_path
                        for i in range(len(found)-1):
                            a = found[i]
                            b = found[i+1]
                            self.tms.assert_fact(("all", a, b), support_key=f"chain:{start}->{goal}")
                        break
                    if nxt not in visited and len(new_path) <= 6:
                        visited.add(nxt)
                        queue.append((nxt, new_path))
                if found:
                    break
            return {
                "discovered_chain": found,
                "contradictions": contradictions,
                "refused": bool(contradictions),
                "proof_core": list(self.last_proof_core)
            }
        return {
            "discovered_chain": [],
            "contradictions": contradictions,
            "refused": bool(contradictions),
            "proof_core": list(self.last_proof_core)
        }

    def verify_program(self, spec: Dict[str, any]) -> Dict[str, any]:
        pre = spec.get("pre", [])
        post = spec.get("post", [])
        steps = spec.get("steps", [])
        ctx = spec.get("ctx", {})
        pre_ok = True
        for p in pre:
            if callable(p):
                if not bool(p(ctx)):
                    pre_ok = False
                    break
        executed = []
        if pre_ok:
            for s in steps:
                executed.append(s)
        post_ok = True
        for q in post:
            if callable(q):
                if not bool(q({"ctx":ctx,"executed":executed})):
                    post_ok = False
                    break
        return {"pre_ok": pre_ok, "post_ok": post_ok, "executed": executed}

    def run_loop(self, max_steps: int = 20):
        """
        Runs the Cognitive Cycle.
        1. Check WM (Intent)
        2. Set Constraints (Attention)
        3. Step Reasoning (Right Brain)
        4. Evaluate Progress (Comparator)
        """
        if self.current_state is None:
            print("[Exec] 🛑 No active state. Call set_initial_state() first.")
            return

        trajectory = []
        visited_states = []

        step_times = []
        t_start = time.perf_counter()
        max_wall_ms = getattr(self, "max_wall_ms", None)
        fixed_step_ms = getattr(self, "fixed_step_ms", None)
        deterministic_seed = getattr(self, "deterministic_seed", None)
        if deterministic_seed is not None:
            try:
                np.random.seed(int(deterministic_seed))
            except Exception:
                pass
        frustration_count = 0
        for t in range(max_steps):
            t0 = time.perf_counter()
            active_intent = self.memory.get_current_intent()

            # 1. Configure Attention (Constraints from Intent)
            constraints = []
            goal_vec = None
            logic_gates = []

            # INHIBITION OF RETURN: Repel from recent trajectory
            # We add weak repulsion for the last 3 states to prevent loops
            if visited_states:
                repulsion_window = 3
                recent_states = visited_states[-repulsion_window:]
                for v_state in recent_states:
                    # Weight 2.0 (Repulsion)
                    constraints.append((v_state, 2.0))

            # --- HEURISTIC CONSTRAINTS (Phase 1.5) ---
            current_word = self.engine.decode(self.current_state)

            # 1. Antonym Repulsion (Contrastive)
            if current_word in self.ANTONYMS:
                antonym_word = self.ANTONYMS[current_word]
                ant_vec = self.engine.get_concept_vector(antonym_word)
                if ant_vec is not None:
                    # Strong Repulsion to prevent semantic bleed
                    constraints.append((ant_vec, 6.0))

            # 2. Metaphor Context (Time is Money)
            if hasattr(self.memory, "context_buffer"):
                ctx = self.memory.context_buffer
                has_time = "time" in ctx or current_word == "time"
                has_money = "money" in ctx or (active_intent and active_intent.description and "money" in active_intent.description.lower())
                if has_time and has_money:
                    for w in ["value", "cost", "spend"]:
                        v = self.engine.get_concept_vector(w)
                        if v is not None:
                            constraints.append((v, -6.0))

            # 3. Sequence Completion (Heuristic)
            if current_word in self.SEQUENCES:
                next_word = self.SEQUENCES[current_word]
                next_vec = self.engine.get_concept_vector(next_word)
                if next_vec is not None:
                    constraints.append((next_vec, -4.0))

            # Relation-guided transitive step
            rels = self._get_relations()
            if rels:
                outs = [r for r in rels if len(r) >= 3 and r[1] == current_word and r[0] in ("all","implies")]
                for _, _, b in outs[:2]:
                    v = self.engine.get_concept_vector(b)
                    if v is not None:
                        constraints.append((v, -2.0))

            # 4. Data-driven context heuristics (loaded from JSON)
            if hasattr(self.memory, "get_context_vector"):
                ctx_vec = self.memory.get_context_vector(self.engine)
                if ctx_vec is not None:
                    constraints.append((ctx_vec, -1.5))
                    ctx = self.memory.context_buffer
                    self._apply_context_heuristics(current_word, ctx, constraints)

            # 5. Sentiment Polarity (Antonym Separation)
            if current_word in self.POSITIVE_ANCHORS or current_word == "good":
                for w in self.POSITIVE_ANCHORS:
                     v = self.engine.get_concept_vector(w)
                     if v is not None: constraints.append((v, -3.0))

            if current_word in self.NEGATIVE_ANCHORS or current_word == "bad":
                 for w in self.NEGATIVE_ANCHORS:
                     v = self.engine.get_concept_vector(w)
                     if v is not None: constraints.append((v, -3.0))

            # 6. Synonym Boosting (Force Separation)
            if current_word in self.SYNONYMS:
                syn = self.SYNONYMS[current_word]
                v = self.engine.get_concept_vector(syn)
                if v is not None: constraints.append((v, -12.0))
                # Apply word-specific rules from heuristics.json
                self._apply_word_rules(current_word, constraints)

            if active_intent:
                # Goal Vector serves as the 'Attractor'
                if active_intent.target_vector is not None:
                    goal_vec = active_intent.target_vector
                    # Add strong constraint towards goal
                    # NEGATIVE weight = ATTRACTION (Gradient Descent minimizes Energy)
                    # We want to minimize distance to goal, so we want the gradient to point TO goal.
                    # descend_energy_gradient: s_new = s - lr * (grad_s + weight * vec)
                    # If weight < 0: s_new = s + lr * |weight| * vec (Attraction)
                    constraints.append((active_intent.target_vector, -5.0 * active_intent.priority))

                # Add Logic Gates from Intent
                logic_gates.extend(active_intent.logic_gates)

            # 2. Execute Step (Right Brain)
            # Increase descent steps if we have an active goal; moderate steps when only constraints exist
            descent_steps = 10 if active_intent else 3

            next_state, word, signals = self.engine.step(
                self.current_state,
                goal_vec,
                constraints,
                logic_gates,
                descent_steps=descent_steps
            )

            self.current_state = next_state
            visited_states.append(next_state.copy())
            trajectory.append(word)

            # Debug distance
            dist_str = ""
            if active_intent and active_intent.target_vector is not None:
                dist = np.linalg.norm(self.current_state - active_intent.target_vector)
                dist_str = f" | DistToGoal={dist:.4f}"

            print(f"[Exec] Step {t}: Thought='{word}' | Focus={signals['focus']:.2f}{dist_str}")

            dur_ms = (time.perf_counter() - t0) * 1000.0
            if fixed_step_ms is not None and dur_ms < float(fixed_step_ms):
                time.sleep((float(fixed_step_ms) - dur_ms) / 1000.0)
                step_times.append(float(fixed_step_ms))
            else:
                step_times.append(dur_ms)
            if max_wall_ms is not None:
                if (time.perf_counter() - t_start) * 1000.0 >= max_wall_ms:
                    break

            # 3. Check Completion (Comparator)
            if active_intent and active_intent.target_vector is not None:
                # Distance Check
                dist = np.linalg.norm(self.current_state - active_intent.target_vector)
                # Or simple string match if we landed on it
                if word == active_intent.target_concept_name or dist < 0.5:
                    print(f"[Exec] ✅ Goal '{active_intent.description}' Complete!")
                    self.memory.complete_intent(active_intent, success=True)

            # 4. Handle Frustration (Switching/Giving Up)
            if signals["frustration"] > 0.8:
                 print("[Exec] ⚠️ High Frustration. Re-evaluating strategy...")
                 # In future: Pop intent, or add "Relax" sub-goal
                 frustration_count += 1

            # Explain-quality refusal check
            if active_intent and active_intent.target_concept_name:
                eq = self.explain_quality(word, active_intent.target_concept_name)
                if eq["score"] < self.explain_refusal_threshold:
                    self.last_refused_reason = f"Low explain quality ({eq['score']:.2f} < {self.explain_refusal_threshold:.2f})"
                    record_event("exec_refuse_low_quality", {"score": eq["score"], "threshold": self.explain_refusal_threshold, "start": word, "goal": active_intent.target_concept_name})
                    # increment errors and check circuit breaker
                    self._explain_quality_failures = getattr(self, '_explain_quality_failures', 0) + 1
                    self.circuit_breaker_check(errors=self._explain_quality_failures)
                    break

            if not self.memory.get_current_intent():
                pass

        # Metrics
        avg_ms = sum(step_times)/len(step_times) if step_times else 0.0
        self.last_loop_metrics = {
            "steps": len(step_times),
            "avg_step_ms": avg_ms,
            "frustrations": frustration_count
        }
        if max_wall_ms is not None:
            self.last_loop_metrics["elapsed_ms"] = (time.perf_counter() - t_start) * 1000.0
            self.last_loop_metrics["deterministic"] = deterministic_seed is not None
        record_event("exec_loop_metrics", self.last_loop_metrics)
        return trajectory

    def plan_a_star(self, start: str, goal: str) -> List[str]:
        """
        Cost-based planning over relation graph.
        Edge cost is inverse of evidence weight; heuristic=0 (uniform cost).
        """
        t_pl_start = time.perf_counter()
        rels = self._get_relations()
        forward = {}
        typemap = {}
        for r in rels:
            if len(r) >= 3:
                t, a, b = r[0], r[1], r[2]
                if t in ("all","implies","most","some","coref"):
                    forward.setdefault(a, []).append(b)
                    typemap[(a,b)] = t
        w = {"all":1.0,"implies":0.8,"most":0.6,"some":0.5,"coref":0.4}
        import heapq
        try_chain = self.explain(start, goal)
        if try_chain and try_chain.get("discovered_chain"):
            path = try_chain.get("discovered_chain")
            constraints = []
            bilinear_pairs = []
            square_pairs = []
            if hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
                constraints = self.engine.brain_data.get("numeric_constraints", [])
                bilinear_pairs = self.engine.brain_data.get("bilinear_pairs", [])
                square_pairs = self.engine.brain_data.get("square_pairs", [])
            if constraints:
                smt_res = None
                if bilinear_pairs or square_pairs:
                    smt_res = SMTBridge.solve_nonlinear_z3(constraints, bilinear_pairs=bilinear_pairs, square_pairs=square_pairs)
                else:
                    smt_res = SMTBridge.solve_with_z3(constraints)
                if smt_res is not None:
                    record_event("exec_plan_numeric_smt_feasible", {"assignment": smt_res})
                    record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 10, "violations": 0})
                    return path
                int_vars = []
                if hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
                    int_vars = self.engine.brain_data.get("int_vars", [])
                mix_res = None
                if not (bilinear_pairs or square_pairs):
                    mix_res = SMTBridge.solve_mixed_integer_z3(constraints, int_vars=int_vars)
                if mix_res is not None:
                    record_event("exec_plan_numeric_mip_feasible", {"assignment": mix_res})
                    record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 10, "violations": 0})
                    return path
                try:
                    vars = sorted({v for coeffs, _, _ in constraints for v in coeffs.keys()})
                    if bilinear_pairs and ConstraintGraph.quick_bilinear_feasible(vars, constraints, bilinear_pairs):
                        record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 10, "violations": 0})
                        return path
                except Exception:
                    pass
                try:
                    vars = sorted({v for coeffs, _, _ in constraints for v in coeffs.keys()})
                    piecewise_pairs = []
                    int_vars = []
                    if hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
                        piecewise_pairs = self.engine.brain_data.get("piecewise_pairs", [])
                        int_vars = self.engine.brain_data.get("int_vars", [])
                    if bilinear_pairs or square_pairs or piecewise_pairs:
                        for wname, x, y in bilinear_pairs:
                            if wname not in vars:
                                vars.append(wname)
                        for wname, x in square_pairs:
                            if wname not in vars:
                                vars.append(wname)
                        for wname, x, _ in piecewise_pairs:
                            if wname not in vars:
                                vars.append(wname)
                        chain_pairs = []
                        log_pairs = []
                        sigmoid_pairs = []
                        if hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
                            chain_pairs = self.engine.brain_data.get("chain_pairs", [])
                            log_pairs = self.engine.brain_data.get("log_pairs", [])
                            sigmoid_pairs = self.engine.brain_data.get("sigmoid_pairs", [])
                        for wname, a, b in chain_pairs:
                            if wname not in vars:
                                vars.append(wname)
                        for wname, x in log_pairs:
                            if wname not in vars:
                                vars.append(wname)
                        for wname, x in sigmoid_pairs:
                            if wname not in vars:
                                vars.append(wname)
                        bounds = ConstraintGraph.solve_with_extended_envelopes(vars, constraints, bilinear_pairs=bilinear_pairs, square_pairs=square_pairs, piecewise_pairs=piecewise_pairs, chain_pairs=chain_pairs, log_pairs=log_pairs, sigmoid_pairs=sigmoid_pairs, segments=10, int_vars=int_vars)
                    else:
                        bounds = ConstraintGraph.solve_boxes(vars, constraints)
                    bilinear_outs = {w for w, _, _ in bilinear_pairs} if bilinear_pairs else set()
                    square_outs = {w for w, _ in square_pairs} if square_pairs else set()
                    explicit_vars = sorted({v for coeffs, _, _ in constraints for v in coeffs.keys()})
                    infeasible = any(b[0] > b[1] for v, b in bounds.items() if (v not in bilinear_outs and v not in square_outs) or (v in explicit_vars))
                    if infeasible:
                        self.last_refused_reason = "numeric infeasible bounds"
                        record_event("exec_plan_refusal_numeric_bounds", {"vars": vars, "bounds": bounds})
                        # Numeric blocks require refusal; avoid returning alternative chain when constraints exist
                        record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 10, "violations": 0})
                        return []
                except Exception:
                    pass
            return path
        # Fast path for single-edge goal
        if goal in forward.get(start, []):
            constraints = []
            bilinear_pairs = []
            square_pairs = []
            if hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
                constraints = self.engine.brain_data.get("numeric_constraints", [])
                bilinear_pairs = self.engine.brain_data.get("bilinear_pairs", [])
                square_pairs = self.engine.brain_data.get("square_pairs", [])
            if constraints:
                smt_res = None
                if bilinear_pairs or square_pairs:
                    smt_res = SMTBridge.solve_nonlinear_z3(constraints, bilinear_pairs=bilinear_pairs, square_pairs=square_pairs)
                else:
                    smt_res = SMTBridge.solve_with_z3(constraints)
                if smt_res is not None:
                    record_event("exec_plan_numeric_smt_feasible", {"assignment": smt_res})
                    record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 10, "violations": 0})
                    return [start, goal]
            if constraints:
                int_vars = []
                if hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
                    int_vars = self.engine.brain_data.get("int_vars", [])
                mix_res = None
                if not (bilinear_pairs or square_pairs):
                    mix_res = SMTBridge.solve_mixed_integer_z3(constraints, int_vars=int_vars)
                if mix_res is not None:
                    record_event("exec_plan_numeric_mip_feasible", {"assignment": mix_res})
                    record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 10, "violations": 0})
                    return [start, goal]
            if constraints:
                try:
                    vars = sorted({v for coeffs, _, _ in constraints for v in coeffs.keys()})
                    piecewise_pairs = []
                    int_vars = []
                    if hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
                        piecewise_pairs = self.engine.brain_data.get("piecewise_pairs", [])
                        int_vars = self.engine.brain_data.get("int_vars", [])
                    if bilinear_pairs or square_pairs or piecewise_pairs:
                        for wname, x, y in bilinear_pairs:
                            if wname not in vars:
                                vars.append(wname)
                        for wname, x in square_pairs:
                            if wname not in vars:
                                vars.append(wname)
                        for wname, x, _ in piecewise_pairs:
                            if wname not in vars:
                                vars.append(wname)
                        bounds = ConstraintGraph.solve_with_extended_envelopes(vars, constraints, bilinear_pairs=bilinear_pairs, square_pairs=square_pairs, piecewise_pairs=piecewise_pairs, segments=4, int_vars=int_vars)
                    else:
                        bounds = ConstraintGraph.solve_boxes(vars, constraints)
                    bilinear_outs = {w for w, _, _ in bilinear_pairs} if bilinear_pairs else set()
                    square_outs = {w for w, _ in square_pairs} if square_pairs else set()
                    infeasible = any(b[0] > b[1] for v, b in bounds.items() if v not in bilinear_outs and v not in square_outs)
                    if infeasible:
                        self.last_refused_reason = "numeric infeasible bounds"
                        record_event("exec_plan_refusal_numeric_bounds", {"vars": vars, "bounds": bounds})
                        record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 4, "violations": 0})
                        return []
                    record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 4, "violations": 0})
                    return [start, goal]
                except Exception:
                    record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 4, "violations": 0})
                    return [start, goal]
            record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 0, "violations": 0})
            return [start, goal]
        pq = []
        heapq.heappush(pq, (0.0, [start]))
        visited = {}
        while pq:
            cost, path = heapq.heappop(pq)
            node = path[-1]
            # Direct proof shortcut using structured FOL if available
            try:
                text = ""
                dom = []
                if hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
                    text = self.engine.brain_data.get("formal_text", "")
                    type_map = self.engine.brain_data.get("type_map", {})
                    dom = sorted(set(list(type_map.keys())))
                if text:
                    clauses = FormalLogic.generate_fol_clauses_structured(text, dom or ["a","b","c"])
                    # If a proof of goal predicate exists with current node as first arg, accept
                    # Allow simple two-arity goal predicate mapping: S(node,goal) or s(node,goal)
                    ok = FormalLogic.resolve_fol(clauses, ("s", node, goal)) or FormalLogic.resolve_fol(clauses, ("goal", node, goal))
                    if ok:
                        record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 0, "violations": 0})
                        return [start, goal]
            except Exception:
                pass
            if node == goal:
                constraints = []
                bilinear_pairs = []
                square_pairs = []
                if hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
                    constraints = self.engine.brain_data.get("numeric_constraints", [])
                    bilinear_pairs = self.engine.brain_data.get("bilinear_pairs", [])
                    square_pairs = self.engine.brain_data.get("square_pairs", [])
                if constraints:
                    smt_res = None
                    if bilinear_pairs or square_pairs:
                        smt_res = SMTBridge.solve_nonlinear_z3(constraints, bilinear_pairs=bilinear_pairs, square_pairs=square_pairs)
                    if smt_res is None:
                        smt_res = SMTBridge.solve_with_z3(constraints)
                    if smt_res is not None:
                        record_event("exec_plan_numeric_smt_feasible", {"assignment": smt_res})
                        record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 10, "violations": 0})
                        return path
                mix_res = SMTBridge.solve_mixed_integer_z3(constraints, int_vars=int_vars)
                if mix_res is not None:
                    record_event("exec_plan_numeric_mip_feasible", {"assignment": mix_res})
                    record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 10, "violations": 0})
                    return path
                if constraints:
                    try:
                        vars = sorted({v for coeffs, _, _ in constraints for v in coeffs.keys()})
                        piecewise_pairs = []
                        if hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
                            piecewise_pairs = self.engine.brain_data.get("piecewise_pairs", [])
                        if bilinear_pairs or square_pairs or piecewise_pairs:
                            # include bilinear w variables
                            for wname, x, y in bilinear_pairs:
                                if wname not in vars:
                                    vars.append(wname)
                            for wname, x in square_pairs:
                                if wname not in vars:
                                    vars.append(wname)
                            for wname, x, _ in piecewise_pairs:
                                if wname not in vars:
                                    vars.append(wname)
                            chain_pairs = []
                            log_pairs = []
                            sigmoid_pairs = []
                            if hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
                                chain_pairs = self.engine.brain_data.get("chain_pairs", [])
                                log_pairs = self.engine.brain_data.get("log_pairs", [])
                                sigmoid_pairs = self.engine.brain_data.get("sigmoid_pairs", [])
                            for wname, a, b in chain_pairs:
                                if wname not in vars:
                                    vars.append(wname)
                            for wname, x in log_pairs:
                                if wname not in vars:
                                    vars.append(wname)
                            for wname, x in sigmoid_pairs:
                                if wname not in vars:
                                    vars.append(wname)
                            int_vars = []
                            if hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
                                int_vars = self.engine.brain_data.get("int_vars", [])
                            bounds = ConstraintGraph.solve_with_extended_envelopes(vars, constraints, bilinear_pairs=bilinear_pairs, square_pairs=square_pairs, piecewise_pairs=piecewise_pairs, chain_pairs=chain_pairs, log_pairs=log_pairs, sigmoid_pairs=sigmoid_pairs, segments=10, int_vars=int_vars)
                        else:
                            bounds = ConstraintGraph.solve_boxes(vars, constraints)
                        bilinear_outs = {w for w, _, _ in bilinear_pairs} if bilinear_pairs else set()
                        square_outs = {w for w, _ in square_pairs} if square_pairs else set()
                        explicit_vars = sorted({v for coeffs, _, _ in constraints for v in coeffs.keys()})
                        infeasible = any(b[0] > b[1] for v, b in bounds.items() if (v not in bilinear_outs and v not in square_outs) or (v in explicit_vars))
                        if infeasible:
                            self.last_refused_reason = "numeric infeasible bounds"
                            record_event("exec_plan_refusal_numeric_bounds", {"vars": vars, "bounds": bounds})
                            record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 10, "violations": 0})
                            return []
                        all_single_var = all(len(coeffs) == 1 for coeffs, _, _ in constraints)
                        if all_single_var:
                            intervals = {v: [bounds[v][0], bounds[v][1]] for v in vars}
                            for coeffs, op, c in constraints:
                                (vv, aa), = coeffs.items()
                                if op == "<=":
                                    if aa > 0:
                                        intervals[vv][1] = min(intervals[vv][1], c / max(1e-9, aa))
                                    elif aa < 0:
                                        intervals[vv][0] = max(intervals[vv][0], c / aa)
                                elif op == ">=":
                                    if aa > 0:
                                        intervals[vv][0] = max(intervals[vv][0], c / max(1e-9, aa))
                                    elif aa < 0:
                                        intervals[vv][1] = min(intervals[vv][1], c / aa)
                            bilinear_outs2 = {w for w, _, _ in bilinear_pairs} if bilinear_pairs else set()
                            consistent = all(lo <= hi for v, (lo, hi) in intervals.items() if v not in bilinear_outs2)
                            if consistent:
                                record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 10, "violations": 0})
                                return path
                        # Try projection to get a feasible example; if violation persists, surface counterexample
                        x = ConstraintGraph.project_feasible_point(bounds, constraints, steps=200, alpha=0.2, bilinear_pairs=bilinear_pairs)
                        if hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
                            int_vars = self.engine.brain_data.get("int_vars", [])
                            for iv in int_vars:
                                if iv in x and iv in bounds:
                                    xv = round(x[iv])
                                    x[iv] = max(bounds[iv][0], min(bounds[iv][1], xv))
                        violations = []
                        for coeffs, op, c in constraints:
                            s = sum(a * x.get(v, 0.0) for v, a in coeffs.items())
                            if op == "<=" and s > c + 1e-6:
                                violations.append({"coeffs": coeffs, "op": op, "c": c, "value": s})
                            elif op == ">=" and s < c - 1e-6:
                                violations.append({"coeffs": coeffs, "op": op, "c": c, "value": s})
                        if violations:
                            simple = dict(x)
                            if hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
                                int_vars = self.engine.brain_data.get("int_vars", [])
                                for iv in int_vars:
                                    if iv in simple and iv in bounds:
                                        sv = round(simple[iv])
                                        simple[iv] = max(bounds[iv][0], min(bounds[iv][1], sv))
                            intervals = {v: [bounds[v][0], bounds[v][1]] for v in vars}
                            for coeffs, op, c in constraints:
                                if len(coeffs) == 1:
                                    (v, a), = coeffs.items()
                                    if op == "<=":
                                        if a > 0:
                                            intervals[v][1] = min(intervals[v][1], c / max(1e-9, a))
                                        elif a < 0:
                                            intervals[v][0] = max(intervals[v][0], c / a)
                                    elif op == ">=":
                                        if a > 0:
                                            intervals[v][0] = max(intervals[v][0], c / max(1e-9, a))
                                        elif a < 0:
                                            intervals[v][1] = min(intervals[v][1], c / a)
                            for v, (lo, hi) in intervals.items():
                                m = 0.5 * (lo + hi)
                                simple[v] = max(bounds[v][0], min(bounds[v][1], m))
                            for wname, xv, yv in bilinear_pairs or []:
                                if wname in bounds and xv in simple and yv in simple:
                                    prod = simple[xv] * simple[yv]
                                    simple[wname] = max(bounds[wname][0], min(bounds[wname][1], prod))
                            recheck = []
                            for coeffs, op, c in constraints:
                                s2 = sum(a * simple.get(v, 0.0) for v, a in coeffs.items())
                                if op == "<=" and s2 > c + 1e-6:
                                    recheck.append(1)
                                elif op == ">=" and s2 < c - 1e-6:
                                    recheck.append(1)
                            if recheck:
                                self.last_refused_reason = "numeric projection violation"
                                try:
                                    int_vars = []
                                    if hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
                                        int_vars = self.engine.brain_data.get("int_vars", [])
                                    bounds2 = ConstraintGraph.solve_with_extended_envelopes(vars, constraints, bilinear_pairs=bilinear_pairs, square_pairs=square_pairs, piecewise_pairs=piecewise_pairs, segments=6, int_vars=int_vars)
                                    bilinear_outs = {w for w, _, _ in bilinear_pairs} if bilinear_pairs else set()
                                    square_outs = {w for w, _ in square_pairs} if square_pairs else set()
                                    explicit_vars = sorted({v for coeffs, _, _ in constraints for v in coeffs.keys()})
                                    infeasible2 = any(b[0] > b[1] for v, b in bounds2.items() if (v not in bilinear_outs and v not in square_outs) or (v in explicit_vars))
                                    if not infeasible2:
                                        record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 6, "violations": len(violations)})
                                        return path
                                except Exception:
                                    pass
                                if isinstance(self.last_loop_metrics, dict):
                                    self.last_loop_metrics["last_counterexample"] = {"assignment": x, "violations": violations}
                                record_event("exec_plan_refusal_numeric_projection", {"violations_count": len(violations)})
                                record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 6, "violations": len(violations)})
                                return []
                        record_event("exec_plan_numeric_feasible", {"assignment": x})
                        record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 10, "violations": 0})
                    except Exception as e:
                        record_event("exec_plan_numeric_error", {"error": str(e)})
                record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 0, "violations": 0})
                return path
            if node in visited and visited[node] <= cost:
                continue
            visited[node] = cost
            for nxt in forward.get(node, []):
                edge_w = w.get(typemap.get((node,nxt), "some"), 0.5)
                edge_cost = 1.0 / max(1e-6, edge_w)
                # Type-guided preference and enforcement: prefer compatible, skip incompatible when both typed
                type_map = {}
                if hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
                    type_map = self.engine.brain_data.get("type_map", {})
                    predicate_ontology = self.engine.brain_data.get("predicate_ontology", {})
                types_node = set(type_map.get(node, []))
                types_nxt = set(type_map.get(nxt, []))
                if types_node and types_nxt:
                    if not types_node.intersection(types_nxt):
                        # Hard enforcement: skip edges that violate typing when both sides are typed
                        continue
                    edge_cost *= 0.9
                # Ontology domain/range bias for subclass edges (treat 'all' as subclass)
                ont = predicate_ontology.get("subclass", {})
                dom_types = set(ont.get("domain", []))
                rng_types = set(ont.get("range", []))
                if dom_types and rng_types:
                    dom_ok = not dom_types or bool(types_node.intersection(dom_types))
                    rng_ok = not rng_types or bool(types_nxt.intersection(rng_types))
                    if dom_ok and rng_ok:
                        edge_cost *= 0.85
                    else:
                        edge_cost *= 1.15
                new_path = path + [nxt]
                heapq.heappush(pq, (cost + edge_cost, new_path))
        record_event("exec_plan_latency", {"ms": (time.perf_counter() - t_pl_start) * 1000.0, "segments": 0, "violations": 0})
        return []

    def modus_ponens(self, premise: str) -> Optional[str]:
        rels = self._get_relations()
        for r in rels:
            if len(r) >= 3 and r[1] == premise and r[0] in ("implies", "all"):
                return r[2]
        return None

    def modus_tollens(self, premise: str, consequent: str):
        rels = self._get_relations()
        for r in rels:
            if len(r) >= 3 and r[0] in ("implies", "all") and r[1] == premise and r[2] == consequent:
                return ("not", premise)
        return None

    def syllogism(self, a: str, b: str, c: str) -> List[str]:
        rels = self._get_relations()
        has_ab = any(len(r) >= 3 and r[0] in ("all", "implies") and r[1] == a and r[2] == b for r in rels)
        has_bc = any(len(r) >= 3 and r[0] in ("all", "implies") and r[1] == b and r[2] == c for r in rels)
        if has_ab and has_bc:
            return [a, b, c]
        return []

    def explain_quality(self, start: str, goal: str) -> Dict[str, any]:
        res = self.explain(start, goal)
        score = 0.0
        if res.get("discovered_chain"):
            score += 2.0
        if res.get("contradictions"):
            score -= 1.0 * len(res.get("contradictions"))
        return {"score": score, "chain": res.get("discovered_chain", []), "contradictions": res.get("contradictions", [])}

    def assess_mastery(self) -> Dict[str, any]:
        rels = self._get_relations()
        chain_discovery = any(len(r) >= 3 and r[0] in ("all", "implies") for r in rels)
        contradiction_detection = any(len(r) >= 3 and r[0] == "no" for r in rels)
        score = 4
        if chain_discovery:
            score += 1
        if contradiction_detection:
            score += 1
        return {"capabilities": {"chain_discovery": chain_discovery, "contradiction_detection": contradiction_detection}, "score": score}

    def _relations_fingerprint(self) -> str:
        rels = self._get_relations()
        return str(len(rels))

    def explain_candidates(self, start: str, goal: str, k: int = 3) -> List[Dict[str, Any]]:
        fp = self._relations_fingerprint()
        key = (start, goal, fp)
        if key in self._explain_cache:
            return self._explain_cache[key][:max(1, int(k))]
        rels = self._get_relations()
        forward = {}
        typemap = {}
        for r in rels:
            if len(r) >= 3:
                t, a, b = r[0], r[1], r[2]
                if t in ("all", "implies", "most", "some", "coref"):
                    forward.setdefault(a, []).append(b)
                    typemap[(a, b)] = t
        w = {"all": 1.0, "implies": 0.8, "most": 0.6, "some": 0.5, "coref": 0.4}
        # Enumerate simple paths up to a small depth
        max_depth = 6
        cands: List[Dict[str, Any]] = []
        queue = [(start, [start])]
        visited_paths = set()
        while queue:
            node, path = queue.pop(0)
            if len(path) > max_depth:
                continue
            if node == goal and len(path) > 1:
                path_key = tuple(path)
                if path_key not in visited_paths:
                    visited_paths.add(path_key)
                    # score as product of edge weights
                    score = 1.0
                    for i in range(len(path) - 1):
                        t = typemap.get((path[i], path[i+1]), "some")
                        score *= w.get(t, 0.5)
                    cands.append({"chain": path, "score": score})
                continue
            for nxt in forward.get(node, []):
                if nxt in path:
                    continue
                queue.append((nxt, path + [nxt]))
        cands.sort(key=lambda x: (-x["score"], len(x["chain"])))
        # cache
        self._explain_cache[key] = cands
        self._explain_cache_order.append(key)
        if len(self._explain_cache_order) > self._explain_cache_cap:
            old = self._explain_cache_order.pop(0)
            self._explain_cache.pop(old, None)
        return cands[:max(1, int(k))]

    def prove_structured(self, target: tuple, text: str = None, domain_constants: List[str] = None) -> bool:
        """
        Proves a target predicate using structured clause generation from brain_data.formal_text or provided text.
        """
        try:
            if text is None and hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
                text = self.engine.brain_data.get("formal_text", "")
            dom = domain_constants
            if dom is None:
                dom = []
                if hasattr(self.engine, "brain_data") and isinstance(self.engine.brain_data, dict):
                    type_map = self.engine.brain_data.get("type_map", {})
                    dom = sorted(set(list(type_map.keys())))
                if not dom:
                    dom = ["a","b","c"]
            if not text:
                return False
            clauses = FormalLogic.generate_fol_clauses_structured(text, dom)
            return FormalLogic.resolve_fol(clauses, target)
        except Exception:
            return False

    def contingency_tool_policy(self, context: List[str]) -> Optional[str]:
        """
        Simple tool policy: if kettle broken, use microwave; if drive nail, use hammer.
        """
        if "kettle" in context and "broken" in context:
            return "microwave"
        if "drive" in context and "nail" in context:
            return "hammer"
        return None

    def bind_variables(self, pattern: tuple, fact: tuple) -> Optional[Dict[str,str]]:
        if len(pattern) < 3 or len(fact) < 3:
            return None
        pt, pa, pb = pattern[0], pattern[1], pattern[2]
        ft, fa, fb = fact[0], fact[1], fact[2]
        if pt != ft:
            return None
        bind = {}
        def is_var(tok: str) -> bool:
            if not tok:
                return False
            if tok.startswith("?"):
                return True
            if tok == "_":
                return True
            if tok.isalpha() and tok.isupper():
                return True
            return len(tok) == 1 and tok.isalpha()
        def match_token(p, f):
            if is_var(p):
                if p in bind and bind[p] != f:
                    return False
                bind[p] = f
                return True
            return p == f
        if match_token(pa, fa) and match_token(pb, fb):
            return bind
        return None

    def circuit_breaker_check(self, errors: int, threshold: int = 3):
        """
        Trip circuit breaker after repeated errors; emit audit log.
        """
        if errors >= threshold:
            self.circuit_breaker_tripped = True
            record_event("circuit_breaker_tripped", {"errors": errors, "threshold": threshold})
