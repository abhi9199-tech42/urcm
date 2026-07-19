"""
URCM Consistency Benchmark
Compares Standard LLM Chain vs URCM Mesh on 100 reasoning tasks.
Metric: Consistency score (run each task 3 times, measure response stability).
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 100 REASONING TASKS (8 categories x ~12-13 each)
# ============================================================

REASONING_TASKS = {
    "causal_reasoning": [
        "Why does ice float on water?",
        "What causes seasons on Earth?",
        "Why do we see lightning before thunder?",
        "What causes tides in the ocean?",
        "Why does heat rise?",
        "What causes earthquakes?",
        "Why do plants need sunlight?",
        "What causes rain to form?",
        "Why does sound travel faster in water?",
        "What causes wind to blow?",
        "Why does metal feel cold to touch?",
        "What causes a rainbow to appear?",
    ],
    "logical_deduction": [
        "If all cats are animals and all animals need food, do cats need food?",
        "If it rains, the ground is wet. The ground is wet. Did it rain?",
        "If A implies B and B implies C, what does A imply?",
        "All birds can fly. Penguins are birds. Can penguins fly?",
        "If no fish can fly and some animals can fly, are all fish non-flying animals?",
        "If every student passed and John is a student, did John pass?",
        "If X is greater than Y and Y is greater than Z, is X greater than Z?",
        "If all roses are flowers and some flowers fade quickly, do some roses fade quickly?",
        "If it is sunny, we go to the park. We did not go to the park. Is it sunny?",
        "If A equals B and B equals C, what is the relationship between A and C?",
        "If all mammals breathe air and whales are mammals, do whales breathe air?",
        "If some doctors are tall and all tall people play basketball, do some doctors play basketball?",
    ],
    "moral_reasoning": [
        "Is it ethical to lie to protect someone's feelings?",
        "Should autonomous vehicles prioritize passengers or pedestrians?",
        "Is it wrong to break a law you believe is unjust?",
        "Should AI systems be allowed to make life-or-death decisions?",
        "Is it ethical to use animals for medical research?",
        "Should wealth be redistributed to reduce inequality?",
        "Is it wrong to stay silent when witnessing injustice?",
        "Should governments limit personal freedoms for public safety?",
        "Is it ethical to keep secrets from someone for their own good?",
        "Should companies prioritize profit over environmental protection?",
        "Is it morally acceptable to sacrifice one life to save many?",
        "Should children be allowed to make major life decisions?",
    ],
    "mathematical_reasoning": [
        "What is 15% of 240?",
        "If a triangle has sides 3, 4, 5, is it a right triangle?",
        "What is the area of a circle with radius 7?",
        "If x + 5 = 12, what is 2x?",
        "What is the square root of 144?",
        "If you invest $1000 at 5% annual interest, what do you have after 2 years?",
        "How many degrees are in a regular hexagon?",
        "What is the probability of flipping heads twice in a row?",
        "If a train travels 60 mph for 2.5 hours, how far does it go?",
        "What is 3 factorial?",
        "If the ratio of boys to girls is 3:5 and there are 40 students, how many are boys?",
        "What is the sum of interior angles of a pentagon?",
    ],
    "common_sense": [
        "Why do we use umbrellas in the rain?",
        "What would happen if gravity suddenly stopped?",
        "Why do hospitals have white walls?",
        "What is the purpose of a steering wheel?",
        "Why do we sleep at night?",
        "What would happen if the sun disappeared?",
        "Why are stop signs red?",
        "What is the function of a thermostat?",
        "Why do bridges have arches?",
        "What happens if you mix blue and yellow paint?",
        "Why do we need to eat food?",
        "What would happen if there were no bacteria?",
    ],
    "analogical_reasoning": [
        "Hand is to glove as foot is to what?",
        "Doctor is to hospital as teacher is to what?",
        "Eye is to see as ear is to what?",
        "Car is to road as boat is to what?",
        "Brain is to thinking as heart is to what?",
        "Book is to reading as food is to what?",
        "Clock is to time as thermometer is to what?",
        "Key is to lock as password is to what?",
        "Seed is to tree as egg is to what?",
        "Painter is to canvas as musician is to what?",
        "Factory is to product as kitchen is to what?",
        "Compass is to navigation as map is to what?",
    ],
    "counterfactual_reasoning": [
        "What if humans had wings?",
        "What if the Earth rotated in the opposite direction?",
        "What if there was no moon?",
        "What if water was not transparent?",
        "What if humans could photosynthesize?",
        "What if sound could not travel through air?",
        "What if electricity had never been discovered?",
        "What if all oceans were freshwater?",
        "What if humans lived twice as long?",
        "What if the internet had never been invented?",
        "What if there were no mountains on Earth?",
        "What if gravity was twice as strong?",
    ],
    "explanation_reasoning": [
        "Explain why the sky is blue.",
        "Explain how a clock works.",
        "Explain why we yawn.",
        "Explain how vaccines work.",
        "Explain why objects fall when dropped.",
        "Explain how electricity reaches homes.",
        "Explain why leaves change color in autumn.",
        "Explain how a camera captures images.",
        "Explain why the moon has phases.",
        "Explain how bridges support heavy loads.",
        "Explain why ice is slippery.",
        "Explain how a refrigerator keeps food cold.",
    ],
}

# Flatten all tasks
ALL_TASKS = []
for category, tasks in REASONING_TASKS.items():
    for task in tasks:
        ALL_TASKS.append({"query": task, "category": category})

print(f"Total reasoning tasks: {len(ALL_TASKS)}")


# ============================================================
# CONSISTENCY METRICS
# ============================================================

def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def keyword_overlap(text1: str, text2: str) -> float:
    """Jaccard similarity of word sets."""
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def response_variance(responses: List[str]) -> float:
    """Average pairwise keyword overlap across responses."""
    if len(responses) < 2:
        return 1.0
    scores = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            scores.append(keyword_overlap(responses[i], responses[j]))
    return np.mean(scores) if scores else 0.0


def consistency_score(responses: List[str]) -> Dict:
    """Compute consistency metrics for a set of responses to the same query."""
    avg_overlap = response_variance(responses)
    lengths = [len(r.split()) for r in responses]
    len_cv = np.std(lengths) / (np.mean(lengths) + 1e-8)  # coefficient of variation
    exact_matches = sum(1 for i in range(len(responses))
                       for j in range(i+1, len(responses))
                       if normalize_text(responses[i]) == normalize_text(responses[j]))
    pairs = len(responses) * (len(responses) - 1) / 2

    return {
        "avg_keyword_overlap": round(avg_overlap, 4),
        "length_cv": round(len_cv, 4),
        "exact_match_rate": round(exact_matches / max(pairs, 1), 4),
        "consistency_score": round(avg_overlap * (1 - min(len_cv, 1.0)) * 0.7 +
                                   (exact_matches / max(pairs, 1)) * 0.3, 4),
    }


# ============================================================
# BENCHMARK RUNNERS
# ============================================================

def run_standard_llm_chain(llm, query: str, n_runs: int = 3, temperature: float = 0.3) -> List[str]:
    """Run standard LLM chain: raw query -> LLM -> response."""
    responses = []
    prompt = f"User: {query}\nAssistant:"
    for _ in range(n_runs):
        resp = llm.generate(prompt, max_tokens=150, temperature=temperature,
                           stop=["User:", "System:", "\n\n"])
        responses.append(resp.strip())
    return responses


def run_urcm_mesh(axiom, query: str, n_runs: int = 3) -> List[str]:
    """Run URCM mesh: query -> reasoning -> grammar -> LLM -> response."""
    import io
    responses = []
    for _ in range(n_runs):
        try:
            axiom.exec_ctrl.memory.intent_stack.clear()
            axiom.exec_ctrl.current_state = None
        except Exception:
            pass
        # Suppress verbose AXIOM stdout/stderr
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            resp = axiom.process(query)
            responses.append(resp.strip() if resp else "")
        except Exception as e:
            responses.append(f"[ERROR: {type(e).__name__}]")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
    return responses


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():
    print("=" * 60)
    print("  URCM Consistency Benchmark")
    print("  Standard LLM Chain vs URCM Mesh")
    print("  100 Reasoning Tasks | 8 Categories")
    print("=" * 60)

    # Load models
    model_path = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    brain_path = "urcm_identity.pkl"

    print("\n[1/4] Loading Standard LLM...")
    from urcm.core.llm_bridge import LLMBridge
    llm = LLMBridge(model_path=model_path, n_ctx=2048)
    if llm.mock_mode:
        print("WARNING: LLM in mock mode. Results will be meaningless.")
    else:
        print("  LLM loaded successfully.")

    print("\n[2/4] Loading URCM Mesh (AXIOM)...")
    from axiom import AXIOM
    axiom = AXIOM(model_path=model_path, brain_path=brain_path)
    print("  AXIOM loaded successfully.")

    print(f"\n[3/4] Running benchmark on {len(ALL_TASKS)} tasks...")
    print(f"  (Each task run 3 times per approach, temp=0.3)\n")

    # Results storage
    results = {
        "standard_llm": defaultdict(list),
        "urcm_mesh": defaultdict(list),
    }
    all_responses = {
        "standard_llm": {},
        "urcm_mesh": {},
    }

    start_time = time.time()

    for idx, task in enumerate(ALL_TASKS):
        query = task["query"]
        category = task["category"]

        print(f"  [{idx+1:3d}/100] {category:25s} | {query[:50]:50s}", end="")

        # Standard LLM Chain
        llm_responses = run_standard_llm_chain(llm, query, n_runs=3, temperature=0.3)
        llm_consistency = consistency_score(llm_responses)
        results["standard_llm"][category].append(llm_consistency)

        # URCM Mesh
        urcm_responses = run_urcm_mesh(axiom, query, n_runs=3)
        urcm_consistency = consistency_score(urcm_responses)
        results["urcm_mesh"][category].append(urcm_consistency)

        # Store raw responses
        all_responses["standard_llm"][query] = llm_responses
        all_responses["urcm_mesh"][query] = urcm_responses

        llm_cs = llm_consistency["consistency_score"]
        urcm_cs = urcm_consistency["consistency_score"]
        winner = "URCM" if urcm_cs > llm_cs else "LLM" if llm_cs > urcm_cs else "TIE"
        print(f"  LLM={llm_cs:.3f}  URCM={urcm_cs:.3f}  [{winner}]")

    elapsed = time.time() - start_time
    print(f"\n  Benchmark completed in {elapsed:.1f}s ({elapsed/len(ALL_TASKS):.1f}s/task)")

    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================

    print("\n[4/4] Computing aggregate results...\n")

    # Per-category aggregation
    category_results = {}
    all_llm_scores = []
    all_urcm_scores = []

    for category in REASONING_TASKS:
        llm_scores = [s["consistency_score"] for s in results["standard_llm"][category]]
        urcm_scores = [s["consistency_score"] for s in results["urcm_mesh"][category]]

        llm_avg = np.mean(llm_scores) if llm_scores else 0
        urcm_avg = np.mean(urcm_scores) if urcm_scores else 0
        improvement = ((urcm_avg - llm_avg) / (llm_avg + 1e-8)) * 100

        category_results[category] = {
            "llm_consistency": round(llm_avg, 4),
            "urcm_consistency": round(urcm_avg, 4),
            "improvement_pct": round(improvement, 2),
            "n_tasks": len(llm_scores),
        }

        all_llm_scores.extend(llm_scores)
        all_urcm_scores.extend(urcm_scores)

    # Overall
    overall_llm = np.mean(all_llm_scores) if all_llm_scores else 0
    overall_urcm = np.mean(all_urcm_scores) if all_urcm_scores else 0
    overall_improvement = ((overall_urcm - overall_llm) / (overall_llm + 1e-8)) * 100

    # Print results
    print("=" * 80)
    print(f"{'Category':25s} | {'LLM Chain':>10s} | {'URCM Mesh':>10s} | {'Improvement':>12s}")
    print("-" * 80)
    for cat, res in category_results.items():
        arrow = "+" if res["improvement_pct"] > 0 else ""
        print(f"{cat:25s} | {res['llm_consistency']:>10.4f} | {res['urcm_consistency']:>10.4f} | {arrow}{res['improvement_pct']:>10.2f}%")
    print("-" * 80)
    arrow = "+" if overall_improvement > 0 else ""
    print(f"{'OVERALL':25s} | {overall_llm:>10.4f} | {overall_urcm:>10.4f} | {arrow}{overall_improvement:>10.2f}%")
    print("=" * 80)

    # Additional metrics
    llm_overlaps = [s["avg_keyword_overlap"] for cat_scores in results["standard_llm"].values() for s in cat_scores]
    urcm_overlaps = [s["avg_keyword_overlap"] for cat_scores in results["urcm_mesh"].values() for s in cat_scores]
    llm_exact = [s["exact_match_rate"] for cat_scores in results["standard_llm"].values() for s in cat_scores]
    urcm_exact = [s["exact_match_rate"] for cat_scores in results["urcm_mesh"].values() for s in cat_scores]

    print(f"\nDetailed Metrics:")
    print(f"  {'Metric':30s} | {'LLM Chain':>10s} | {'URCM Mesh':>10s}")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}")
    print(f"  {'Avg Keyword Overlap':30s} | {np.mean(llm_overlaps):>10.4f} | {np.mean(urcm_overlaps):>10.4f}")
    print(f"  {'Exact Match Rate':30s} | {np.mean(llm_exact):>10.4f} | {np.mean(urcm_exact):>10.4f}")
    print(f"  {'Consistency Score (composite)':30s} | {overall_llm:>10.4f} | {overall_urcm:>10.4f}")

    # Save results
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tasks": len(ALL_TASKS),
            "n_runs_per_task": 3,
            "temperature": 0.3,
            "model": model_path,
            "elapsed_seconds": round(elapsed, 1),
        },
        "overall": {
            "llm_consistency": round(overall_llm, 4),
            "urcm_consistency": round(overall_urcm, 4),
            "improvement_pct": round(overall_improvement, 2),
        },
        "by_category": category_results,
    }

    with open("benchmark_consistency_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to benchmark_consistency_results.json")

    # Save raw responses
    with open("benchmark_consistency_responses.json", "w", encoding="utf-8") as f:
        json.dump(all_responses, f, indent=2, ensure_ascii=False)
    print(f"Raw responses saved to benchmark_consistency_responses.json")

    return output


if __name__ == "__main__":
    main()
