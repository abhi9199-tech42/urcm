"""
URCM Brain Trainer — Multi-Dataset Edition.
Downloads & ingests from HuggingFace datasets + local SCDL-RAG data.
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from urcm.core.ingest import KnowledgeIngestion

L2 = 512
BRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urcm_trained_brain.pkl")
SCDL = r"C:\Users\kriti\OneDrive\Contradiction-Aware Retrieval Architecture for Reliable AI Reasoning"


def safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode())


def train_wikipedia(ing, max_items=300):
    """Download Wikipedia articles from HuggingFace."""
    from datasets import load_dataset
    safe_print("\n[1/6] Wikipedia (300 articles)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True, trust_remote_code=True)
    count = 0
    for item in ds:
        if count >= max_items:
            break
        text = item.get("text", "")
        if len(text) > 100:
            # Take first 500 chars to keep it fast
            ing.ingest_text(text[:500])
            count += 1
            if count % 50 == 0:
                safe_print(f"  {count}/{max_items}...")
    ing.save()
    safe_print(f"  Done: {count} articles")
    return count


def train_squad(ing, max_items=300):
    """SQuAD v2.0 reading comprehension."""
    from datasets import load_dataset
    safe_print("\n[2/6] SQuAD v2.0 (300 QA pairs)...")
    ds = load_dataset("rajpurkar/squad_v2", split="train", streaming=True)
    count = 0
    for item in ds:
        if count >= max_items:
            break
        ctx = item.get("context", "")
        q = item.get("question", "")
        answers = item.get("answers", {})
        ans_list = answers.get("answer", [])
        if ans_list and ctx:
            ing.ingest_text(ctx[:400])
            ing.ingest_text(f"{q} {ans_list[0]}")
            count += 1
    ing.save()
    safe_print(f"  Done: {count} QA pairs")
    return count


def train_hotpotqa(ing, max_items=200):
    """HotpotQA multi-hop reasoning."""
    from datasets import load_dataset
    safe_print("\n[3/6] HotpotQA (200 questions)...")
    ds = load_dataset("hotpot_qa", "distractor", split="train", streaming=True)
    count = 0
    for item in ds:
        if count >= max_items:
            break
        q = item.get("question", "")
        a = item.get("answer", "")
        support = item.get("supporting_facts", [])
        # Build context from titles
        titles = list(set(s[0] for s in support))[:3]
        ctx = ". ".join(titles)
        if q and a:
            if ctx:
                ing.ingest_text(ctx)
            ing.ingest_text(f"{q} {a}")
            count += 1
    ing.save()
    safe_print(f"  Done: {count} questions")
    return count


def train_scifact(ing, max_items=200):
    """SciFact — scientific claim verification."""
    from datasets import load_dataset
    safe_print("\n[4/6] SciFact (200 claims)...")
    ds = load_dataset("allenai/scifact", split="train", streaming=True)
    count = 0
    for item in ds:
        if count >= max_items:
            break
        claim = item.get("claim", "")
        rationale = item.get("rationale", "")
        if claim:
            ing.ingest_text(claim)
            count += 1
        if rationale:
            ing.ingest_text(str(rationale)[:400])
    ing.save()
    safe_print(f"  Done: {count} claims")
    return count


def train_boolq(ing, max_items=200):
    """BoolQ — boolean questions with passages."""
    from datasets import load_dataset
    safe_print("\n[5/6] BoolQ (200 QA pairs)...")
    ds = load_dataset("google/boolq", split="train", streaming=True)
    count = 0
    for item in ds:
        if count >= max_items:
            break
        q = item.get("question", "")
        p = item.get("passage", "")
        a = item.get("answer", False)
        if q and p:
            ing.ingest_text(p[:400])
            answer_str = "yes" if a else "no"
            ing.ingest_text(f"{q} {answer_str}")
            count += 1
    ing.save()
    safe_print(f"  Done: {count} QA pairs")
    return count


def train_scdl_local(ing, max_items=100):
    """Local SCDL-RAG vector store."""
    safe_print("\n[6/6] SCDL-RAG Vector Store (local)...")
    vs_path = os.path.join(SCDL, "data", "vector_store.json")
    if not os.path.exists(vs_path):
        safe_print("  Skipped (not found)")
        return 0
    with open(vs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    count = 0
    for item in data[:max_items]:
        text = item.get("content", "").strip()
        if text and len(text) > 20:
            ing.ingest_text(text[:500])
            count += 1
    ing.save()
    safe_print(f"  Done: {count} passages")
    return count


def main():
    t0 = time.time()
    safe_print("=" * 60)
    safe_print("  URCM BRAIN TRAINER — MULTI-DATASET")
    safe_print("=" * 60)

    ing = KnowledgeIngestion(brain_path=BRAIN, l2_dim=L2)
    total = 0

    # Train on all datasets
    try:
        total += train_wikipedia(ing, 300)
    except Exception as e:
        safe_print(f"  Wikipedia failed: {e}")

    try:
        total += train_squad(ing, 300)
    except Exception as e:
        safe_print(f"  SQuAD failed: {e}")

    try:
        total += train_hotpotqa(ing, 200)
    except Exception as e:
        safe_print(f"  HotpotQA failed: {e}")

    try:
        total += train_scifact(ing, 200)
    except Exception as e:
        safe_print(f"  SciFact failed: {e}")

    try:
        total += train_boolq(ing, 200)
    except Exception as e:
        safe_print(f"  BoolQ failed: {e}")

    try:
        total += train_scdl_local(ing, 100)
    except Exception as e:
        safe_print(f"  SCDL local failed: {e}")

    # Final save
    ing.save()

    t = time.time() - t0
    sz = os.path.getsize(BRAIN) / 1024 if os.path.exists(BRAIN) else 0
    safe_print("\n" + "=" * 60)
    safe_print("  TRAINING COMPLETE")
    safe_print("=" * 60)
    safe_print(f"  Brain:       {BRAIN}")
    safe_print(f"  Size:        {sz:.0f} KB")
    safe_print(f"  Concepts:    {len(ing.concept_map)}")
    safe_print(f"  Relations:   {len(ing.relations)}")
    safe_print(f"  Total texts: {total}")
    safe_print(f"  Time:        {t:.1f}s")
    safe_print("=" * 60)


if __name__ == "__main__":
    main()
