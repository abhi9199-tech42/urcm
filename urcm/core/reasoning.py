import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Optional
from urcm.core.hierarchical_encoder import HierarchicalEncoder
from urcm.core.values import ValueSystem
from urcm.core.sanskrit_bridge import SanskritBridge
from urcm.core.sanskrit_grammar import SanskritGrammar
from urcm.core.metacognition import MetacognitiveMonitor
from urcm.core.logic_gates import GeometricLogic

class ReasoningEngine:
    """
    Implements Cognitive Reasoning (Inference) via Energy Minimization.
    Now enhanced with Metacognition (Self-Correction), Grammar (Structure),
    and Geometric Logic (Gates).
    """
    
    def __init__(self, brain_path: str = "urcm_identity.pkl"):
        self.brain_data_path = brain_path
        # Load Brain
        if not os.path.exists(brain_path):
            raise FileNotFoundError(f"Brain file {brain_path} not found.")
            
        with open(brain_path, "rb") as f:
            self.brain_data = pickle.load(f)
            
        self.l2_dim = self.brain_data["l2_W_res"].shape[0]
        self.hierarchy = HierarchicalEncoder(l2_res_dim=self.l2_dim)
        
        # Load Weights
        self.hierarchy.layer2.W_res = self.brain_data["l2_W_res"]
        
        # FIX: Load ALL trained weights (W_in, W_out) if available in brain_data.
        # This prevents "Dimension Mismatch" warnings and ensures stability of L2 projections.
        if "l2_W_in" in self.brain_data:
            self.hierarchy.layer2.W_in = self.brain_data["l2_W_in"]
        if "l2_W_out" in self.brain_data:
            self.hierarchy.layer2.W_out = self.brain_data["l2_W_out"]
            # Recalculate inverse if needed (though W_res is main driver)
            
        # Ensure concept_map has stable vectors
        # If brain was initialized with random vectors due to mismatch, 
        # we must ensure they persist or are consistent.
        # But here we just load what's in the pickle.
        self.concept_map = self.brain_data["concept_map"]
        
        # FIX: Check if concept_map values match L2 dim. If not, re-project or warn.
        # The logs showed "Weight dimension mismatch. Using random initialization." 
        # This happens in run_internet_goal.py, but here we are just loading.
        # If the pickle has bad dimensions, we are in trouble.
        
        # Check first key
        first_key = next(iter(self.concept_map))
        if self.concept_map[first_key].shape[0] != self.l2_dim:
             print(f"⚠️ Dimension Mismatch in Brain! Map={self.concept_map[first_key].shape[0]}, W_res={self.l2_dim}")
             # We must resize W_res or Map.
             # Since Map is ground truth for concepts, we resize W_res.
             new_dim = self.concept_map[first_key].shape[0]
             self.l2_dim = new_dim
             # Re-init W_res randomly OR load from weights file
             self.l2_dim = new_dim
             self.hierarchy = HierarchicalEncoder(l2_res_dim=self.l2_dim)
             # Try to load weights again now that dim is correct
             if os.path.exists("urcm_weights.pkl"):
                 with open("urcm_weights.pkl", "rb") as f:
                     weights = pickle.load(f)
                     if "l2_W_res" in weights and weights["l2_W_res"].shape[0] == self.l2_dim:
                         self.hierarchy.layer2.W_res = weights["l2_W_res"]
                         print(f"✅ Recovered W_res from urcm_weights.pkl with correct dim {self.l2_dim}")
                     else:
                         self.hierarchy.layer2.W_res = np.random.randn(self.l2_dim, self.l2_dim) * 0.01
                         print(f"⚠️ Re-initialized W_res to {self.l2_dim}x{self.l2_dim} (Random)")
             else:
                 self.hierarchy.layer2.W_res = np.random.randn(self.l2_dim, self.l2_dim) * 0.01
                 print(f"⚠️ Re-initialized W_res to {self.l2_dim}x{self.l2_dim} (Random - No weights file)")
        
        # Helper: Reverse Map
        self.inv_map = {}
        for k, v in self.concept_map.items():
            # Convert numpy array to tuple for hashing
            self.inv_map[tuple(v)] = k
            
        # Initialize Value System (Moral Compass)
        self.values = ValueSystem(self.concept_map)

        # Initialize Sanskrit Bridge (Vocabulary)
        self.bridge = SanskritBridge()
        
        # Initialize Sanskrit Grammar (Structure)
        self.grammar = SanskritGrammar()
        
        # Initialize Metacognitive Monitor (Control)
        self.monitor = MetacognitiveMonitor()
        
        # Initialize Logic Gates (Steering)
        self.logic = GeometricLogic(self.concept_map)
            
    def get_concept_vector(self, word: str) -> np.ndarray:
        w = word.lower()
        v = self.concept_map.get(w)
        if v is not None:
            return v
        dim = self.l2_dim
        new_v = np.random.randn(dim)
        n = np.linalg.norm(new_v)
        if n > 0:
            new_v = new_v / n
        self.concept_map[w] = new_v
        return new_v

    def decode(self, vec: np.ndarray) -> str:
        """Finds nearest word."""
        best_word = "?"
        best_dist = float('inf')
        for word, w_vec in self.concept_map.items():
            dist = np.linalg.norm(vec - w_vec)
            if dist < best_dist:
                best_dist = dist
                best_word = word
        return best_word

    def solve(self, 
              query_text: str, 
              constraints: List[Tuple[str, float]], 
              logic_gates: List[Dict] = [], # [{"type": "NOT", "operands": ["war"], "weight": 2.0}]
              steps: int = 10) -> List[str]:
        """
        Runs the reasoning process with Constraints AND Logic Gates.
        """
        print(f"Reasoning: Query='{query_text}' | Constraints={constraints}")
        if logic_gates:
            print(f"  Logic Gates: {logic_gates}")
        
        # 1. Initialize State from Query
        start_vec = self.get_concept_vector(query_text)
        if start_vec is None:
            return [f"Unknown concept: {query_text}"]
            
        current_state = start_vec.copy()
        trajectory = [self.decode(current_state)]
        
        # 2. Prepare Constraints (Vector Space)
        vector_constraints = []
        
        # A. User Constraints (Simple Weights)
        for word, weight in constraints:
            vec = self.get_concept_vector(word)
            if vec is not None:
                vector_constraints.append((vec, weight))
            else:
                print(f"⚠️ Constraint '{word}' not found in brain.")
                
        # B. Axiomatic Constraints (The "Super-Ego")
        for name, valence in self.values.valences.items():
            if name in self.values.axioms:
                vec = self.values.axioms[name]
                # Invert Valence to get Constraint Weight
                weight = -1.0 * valence * 2.0 # Multiplier 2.0 for strong moral grounding
                vector_constraints.append((vec, weight))
        
    def step(self, 
             current_state: np.ndarray, 
             goal_vec: Optional[np.ndarray], 
             constraints: List[Tuple[np.ndarray, float]], 
             logic_gates: List[Dict],
             descent_steps: int = 3) -> Tuple[np.ndarray, str, Dict]:
        """
        Executes a SINGLE cognitive step (Reasoning + Metacognition + Logic).
        """
        # A. Standard Associative Flow (Recall)
        next_state_prediction = np.tanh(np.dot(current_state, self.hierarchy.layer2.W_res))
        
        # --- METACOGNITION CHECK ---
        energy = self.hierarchy.layer2.get_global_energy(next_state_prediction)
        predicted_word = self.decode(next_state_prediction)
        
        signals = self.monitor.get_control_signals(
            current_state=next_state_prediction, 
            current_energy=energy, 
            current_word=predicted_word,
            goal_state=goal_vec
        )
        
        # 4. Apply Corrections
        learning_rate = 0.1
        if signals["focus"] > 0:
            learning_rate *= (1.0 + signals["focus"])
            
        if signals["frustration"] > 0:
            noise = np.random.normal(0, signals["frustration"], next_state_prediction.shape)
            next_state_prediction += noise
            # Re-Normalize to prevent Energy Explosion (Safety Breach)
            norm = np.linalg.norm(next_state_prediction)
            if norm > 0:
                next_state_prediction = next_state_prediction / norm
            # print("  [Meta] ⚡ Injected Frustration Noise & Renormalized")
        
        # B. Apply Constraints (Inference/Adaptation)
        refined_state = self.hierarchy.layer2.descend_energy_gradient(
            state=next_state_prediction,
            codebook_vectors=self.concept_map, 
            steps=descent_steps, 
            learning_rate=learning_rate,
            constraints=constraints
        )
        
        # C. Apply Logic Gates (Geometric Steering)
        for gate in logic_gates:
            grad = self.logic.apply_constraint(
                refined_state, 
                gate["type"], 
                gate["operands"], 
                weight=gate.get("weight", 1.0)
            )
            refined_state += grad * 0.1 
            
        # Re-Normalize after Logic
        norm = np.linalg.norm(refined_state)
        if norm > 0:
            refined_state = refined_state / norm
        
        final_word = self.decode(refined_state)
        
        return refined_state, final_word, signals

    def solve(self, 
              query_text: str, 
              constraints: List[Tuple[str, float]], 
              logic_gates: List[Dict] = [], 
              steps: int = 10) -> List[str]:
        """
        Runs the reasoning process with Constraints AND Logic Gates.
        Includes Inhibition of Return (Repulsion) to prevent loops.
        """
        print(f"Reasoning: Query='{query_text}' | Constraints={constraints}")
        if logic_gates:
            print(f"  Logic Gates: {logic_gates}")
        
        # 1. Initialize State from Query
        start_vec = self.get_concept_vector(query_text)
        if start_vec is None:
            return [f"Unknown concept: {query_text}"]
            
        current_state = start_vec.copy()
        trajectory = [self.decode(current_state)]
        visited_vectors = [current_state.copy()] # Track history for Repulsion
        
        # 2. Prepare Constraints (Vector Space)
        base_constraints = []
        
        # A. User Constraints (Simple Weights)
        for word, weight in constraints:
            vec = self.get_concept_vector(word)
            if vec is not None:
                base_constraints.append((vec, weight))
            else:
                print(f"⚠️ Constraint '{word}' not found in brain.")
                
        # B. Axiomatic Constraints (The "Super-Ego")
        for name, valence in self.values.valences.items():
            if name in self.values.axioms:
                vec = self.values.axioms[name]
                weight = -1.0 * valence * 2.0 
                base_constraints.append((vec, weight))
        
        # 3. Run Dynamics
        goal_vec = start_vec.copy() 
        
        for t in range(steps):
            # Dynamic Constraints: Add Repulsion for Visited States
            # Weight = +2.0 (Push away)
            step_constraints = base_constraints.copy()
            for v_vec in visited_vectors:
                step_constraints.append((v_vec, 2.0))
            
            current_state, word, signals = self.step(
                current_state, goal_vec, step_constraints, logic_gates
            )
            
            if signals["status"] != "stable":
                # print(f"  [Meta] Alert at step {t}: {signals['status']} (Focus={signals['focus']:.2f}, Frust={signals['frustration']:.2f})")
                if signals["frustration"] > 0:
                     pass # handled in step
                     # print("  [Meta] ⚡ Injected Frustration Noise & Renormalized")

            trajectory.append(word)
            visited_vectors.append(current_state.copy())
            
        # 1. Translate to Sanskrit Concepts (Right Brain -> Vocabulary)
        sanskrit_trajectory = self.bridge.translate_trajectory(trajectory)
        
        # 2. Structure into Paninian Sentence (Vocabulary -> Left Brain Structure)
        structured_thought = self.grammar.structure_thought(sanskrit_trajectory)
        
        print(f"🕉️ Structured Thought: {structured_thought}")
        
        # Return both the raw trajectory (for debug) and the structured one
        sanskrit_trajectory.append(f"[Structure]: {structured_thought}")
            
        return sanskrit_trajectory
