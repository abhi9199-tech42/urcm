"""
URCM Right Brain Benchmark Suite.
Measures the performance of the Associative/Intuitive system.

Metrics:
1. Stability: Do trajectories stay semantically relevant?
2. Basin Depth: Can it recover from noise?
3. Hub Dominance: How often does it collapse into 'Black Hole' concepts (e.g., 'help')?
4. Creativity (Entropy): How diverse are the associations?
"""

import numpy as np
import pickle
import os
import time
from collections import Counter
from urcm.core.reasoning import ReasoningEngine

class RightBrainBenchmark:
    def __init__(self, brain_path: str = "urcm_identity.pkl"):
        print(f"Loading Right Brain from {brain_path}...")
        self.engine = ReasoningEngine(brain_path=brain_path)
        self.vocab = list(self.engine.concept_map.keys())
        print(f"Brain Loaded. Vocab Size: {len(self.vocab)}")
        
        # known "Black Holes" to watch out for
        self.black_holes = ["help", "copyright", "privacy", "disclaimer", "contact", "wiki", "causing"]

    def measure_basin_depth(self, concept: str, noise_levels=[0.0, 0.1, 0.3, 0.5], trials=10):
        """
        Tests how robust a concept's memory is against noise.
        """
        if concept not in self.engine.concept_map:
            print(f"Skipping {concept} (Unknown)")
            return None

        base_vec = self.engine.concept_map[concept]
        results = {}

        print(f"\n--- Basin Depth: '{concept}' ---")
        for sigma in noise_levels:
            recoveries = 0
            for _ in range(trials):
                # Inject noise
                noise = np.random.normal(0, sigma, base_vec.shape)
                noisy_vec = base_vec + noise
                # Normalize to preserve energy level (prevent Safety Breach)
                noisy_vec = noisy_vec / np.linalg.norm(noisy_vec) * np.linalg.norm(base_vec)
                
                # Let it settle (Energy Descent)
                # We manually call the layer2 descent
                settled_vec = self.engine.hierarchy.layer2.descend_energy_gradient(
                    state=noisy_vec,
                    codebook_vectors=self.engine.concept_map,
                    steps=5
                )
                
                # Decode
                output = self.engine.decode(settled_vec)
                if output == concept:
                    recoveries += 1
            
            success_rate = recoveries / trials
            results[sigma] = success_rate
            print(f"  Noise {sigma:.1f}: {success_rate*100:.0f}% Recovery")
        
        return results

    def measure_hub_dominance(self, samples=50, steps=10):
        """
        Picks random concepts and sees where they drift.
        Detects 'Black Hole' collapse.
        """
        print(f"\n--- Hub Dominance (N={samples}) ---")
        if len(self.vocab) < samples:
            test_vocab = self.vocab
        else:
            test_vocab = np.random.choice(self.vocab, samples, replace=False)
            
        final_destinations = []
        
        for start_word in test_vocab:
            trajectory = self.engine.solve(start_word, [], steps=steps)
            final_word = trajectory[-1]
            final_destinations.append(final_word)
            
        # Analyze
        counts = Counter(final_destinations)
        total = len(final_destinations)
        
        print("Top Attractors (Where thoughts end up):")
        black_hole_score = 0
        for word, count in counts.most_common(5):
            pct = (count / total) * 100
            print(f"  {word}: {pct:.1f}%")
            if word in self.black_holes:
                black_hole_score += pct
                
        print(f"Black Hole Collapse Rate: {black_hole_score:.1f}%")
        return black_hole_score

    def measure_semantic_drift(self, concept: str, steps=10):
        """
        Measures how fast a concept loses its original meaning.
        Uses Cosine Similarity between Start and End states.
        """
        if concept not in self.engine.concept_map:
            return 0.0
            
        # Run trajectory
        trajectory = self.engine.solve(concept, [], steps=steps)
        
        # Get vectors
        start_vec = self.engine.concept_map[concept]
        end_word = trajectory[-1]
        
        if end_word not in self.engine.concept_map:
            return 0.0 # Should not happen
            
        end_vec = self.engine.concept_map[end_word]
        
        # Cosine Sim
        sim = np.dot(start_vec, end_vec) / (np.linalg.norm(start_vec) * np.linalg.norm(end_vec))
        
        print(f"Drift '{concept}' -> '{end_word}': Similarity = {sim:.4f}")
        return sim

    def run_full_suite(self):
        print("\n==========================================")
        print("RUNNING RIGHT BRAIN REGRESSION SUITE")
        print("==========================================")
        
        # 1. Basin Depth (Robustness)
        # Test a few known strong concepts
        test_concepts = ["midway", "deepseek", "pacific", "zero"]
        for c in test_concepts:
            self.measure_basin_depth(c)
            
        # 2. Hub Dominance (Health)
        bh_score = self.measure_hub_dominance(samples=50)
        
        # 3. Semantic Drift (Stability)
        drifts = []
        for c in test_concepts:
            drifts.append(self.measure_semantic_drift(c))
        
        avg_drift = np.mean(drifts)
        print(f"\nAverage Semantic Retention: {avg_drift:.4f}")
        
        # Verdict
        print("\n--- BENCHMARK VERDICT ---")
        if bh_score > 30.0:
            print("❌ FAIL: Excessive Black Hole Collapse. Brain is obsessed with 'Help'/'Meta' pages.")
        elif avg_drift < 0.5:
            print("⚠️ WARN: High Semantic Drift. Concepts are unstable.")
        else:
            print("✅ PASS: Right Brain is stable and healthy.")

if __name__ == "__main__":
    bench = RightBrainBenchmark()
    bench.run_full_suite()
