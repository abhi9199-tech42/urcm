"""
Fast URCM Brain Training from SCDL-RAG Data.
Saves after each data source to avoid losing progress.
"""

import json, os, sys, time, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from urcm.core.ingest import KnowledgeIngestion

SCDL = r"C:\Users\kriti\OneDrive\Contradiction-Aware Retrieval Architecture for Reliable AI Reasoning"
BRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urcm_scdl_brain.pkl")
L2 = 512

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    t0 = time.time()
    print("=" * 60)
    print("  URCM BRAIN TRAINING FROM SCDL-RAG DATA (FAST)")
    print("=" * 60)

    brain = KnowledgeIngestion(brain_path=BRAIN, l2_dim=L2)
    total = 0

    # 1. Vector Store (limit to 150 passages for speed)
    vs_path = os.path.join(SCDL, "data", "vector_store.json")
    if os.path.exists(vs_path):
        print("\n[1/3] Vector Store...")
        data = load_json(vs_path)
        count = 0
        for item in data[:150]:
            text = item.get("content", "").strip()
            if text and len(text) > 20:
                brain.ingest_text(text)
                count += 1
                total += 1
        brain.save()
        print(f"  {count} passages ingested. Brain saved.")
    else:
        print("  Skipped (not found)")

    # 2. HotpotQA
    hp_path = os.path.join(SCDL, "experiments", "data", "hotpotqa_samples.json")
    if os.path.exists(hp_path):
        print("\n[2/3] HotpotQA...")
        data = load_json(hp_path)
        count = 0
        for item in data[:100]:
            q = item.get("question", "")
            a = item.get("answer", "")
            titles = item.get("supporting_titles", [])
            ctx = ". ".join(str(t) for t in titles if t)
            if ctx:
                brain.ingest_text(ctx)
                total += 1
            if q and a:
                brain.ingest_text(f"{q} {a}")
                total += 1
                count += 1
        brain.save()
        print(f"  {count} QA pairs ingested. Brain saved.")
    else:
        print("  Skipped (not found)")

    # 3. SQuAD (limited)
    sq_path = os.path.join(SCDL, "experiments", "data", "train-v2.0.json")
    if os.path.exists(sq_path):
        print("\n[3/3] SQuAD v2.0...")
        data = load_json(sq_path)
        count = 0
        for article in data.get("data", [])[:50]:  # Limit articles
            for para in article.get("paragraphs", []):
                ctx = para.get("context", "")
                for qa in para.get("qas", [])[:2]:  # Limit QAs per paragraph
                    if qa.get("is_impossible", False):
                        continue
                    answers = qa.get("answers", [])
                    if answers:
                        ans = answers[0].get("text", "")
                        if ctx:
                            brain.ingest_text(ctx)
                            total += 1
                        brain.ingest_text(f"{qa.get('question','')} {ans}")
                        total += 1
                        count += 1
        brain.save()
        print(f"  {count} QA pairs ingested. Brain saved.")
    else:
        print("  Skipped (not found)")

    t = time.time() - t0
    sz = os.path.getsize(BRAIN) / 1024 if os.path.exists(BRAIN) else 0
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Brain:       {BRAIN}")
    print(f"  Size:        {sz:.0f} KB")
    print(f"  Concepts:    {len(brain.concept_map)}")
    print(f"  Relations:   {len(brain.relations)}")
    print(f"  Total texts: {total}")
    print(f"  Time:        {t:.1f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()
