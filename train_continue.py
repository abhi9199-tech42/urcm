"""Continue training: add SQuAD, HotpotQA, SciFact, BoolQ on top of existing brain."""
import os, sys, time
sys.path.insert(0, ".")
from urcm.core.ingest import KnowledgeIngestion

BRAIN = "urcm_trained_brain.pkl"
L2 = 512

def p(msg):
    try: print(msg)
    except: print(msg.encode("ascii","replace").decode())

def main():
    t0 = time.time()
    ing = KnowledgeIngestion(brain_path=BRAIN, l2_dim=L2)
    p(f"Loaded brain: {len(ing.concept_map)} concepts")
    total = 0

    # SQuAD
    try:
        p("\n[1/4] SQuAD v2.0...")
        from datasets import load_dataset
        ds = load_dataset("rajpurkar/squad_v2", split="train", streaming=True)
        c = 0
        for item in ds:
            if c >= 300: break
            ctx = item.get("context","")
            q = item.get("question","")
            ans = item.get("answers",{}).get("answer",[])
            if ans and ctx:
                ing.ingest_text(ctx[:400])
                ing.ingest_text(f"{q} {ans[0]}")
                c += 1; total += 1
        ing.save()
        p(f"  Done: {c}")
    except Exception as e:
        p(f"  Failed: {e}")

    # HotpotQA
    try:
        p("\n[2/4] HotpotQA...")
        from datasets import load_dataset
        ds = load_dataset("hotpot_qa", "distractor", split="train", streaming=True)
        c = 0
        for item in ds:
            if c >= 200: break
            q = item.get("question","")
            a = item.get("answer","")
            sup = item.get("supporting_facts",[])
            titles = list(set(s[0] for s in sup))[:3]
            if q and a:
                if titles: ing.ingest_text(". ".join(titles))
                ing.ingest_text(f"{q} {a}")
                c += 1; total += 1
        ing.save()
        p(f"  Done: {c}")
    except Exception as e:
        p(f"  Failed: {e}")

    # SciFact
    try:
        p("\n[3/4] SciFact...")
        from datasets import load_dataset
        ds = load_dataset("allenai/scifact", split="train", streaming=True)
        c = 0
        for item in ds:
            if c >= 200: break
            claim = item.get("claim","")
            rat = item.get("rationale","")
            if claim:
                ing.ingest_text(claim); c += 1; total += 1
            if rat:
                ing.ingest_text(str(rat)[:400]); total += 1
        ing.save()
        p(f"  Done: {c}")
    except Exception as e:
        p(f"  Failed: {e}")

    # BoolQ
    try:
        p("\n[4/4] BoolQ...")
        from datasets import load_dataset
        ds = load_dataset("google/boolq", split="train", streaming=True)
        c = 0
        for item in ds:
            if c >= 200: break
            q = item.get("question","")
            psg = item.get("passage","")
            a = item.get("answer", False)
            if q and psg:
                ing.ingest_text(psg[:400])
                ing.ingest_text(f"{q} {'yes' if a else 'no'}")
                c += 1; total += 1
        ing.save()
        p(f"  Done: {c}")
    except Exception as e:
        p(f"  Failed: {e}")

    ing.save()
    t = time.time() - t0
    sz = os.path.getsize(BRAIN) / 1024
    p(f"\n{'='*50}")
    p(f"  Concepts:  {len(ing.concept_map)}")
    p(f"  Relations: {len(ing.relations)}")
    p(f"  Size:      {sz:.0f} KB")
    p(f"  Added:     {total} texts")
    p(f"  Time:      {t:.1f}s")
    p(f"{'='*50}")

if __name__ == "__main__":
    main()
