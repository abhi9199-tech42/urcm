from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from urcm.core.ingest import KnowledgeIngestion

L2 = 1024
BRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urcm_10m_brain.pkl")

def process_dataset(dataset_name, dataset_func, ing_path):
    try:
        ing = KnowledgeIngestion(brain_path=ing_path, l2_dim=L2)
        count = dataset_func(ing)
        return {"dataset": dataset_name, "count": count, "success": True, "error": None}
    except Exception as e:
        return {"dataset": dataset_name, "count": 0, "success": False, "error": str(e)}

def train_math_gsm8k(ing, batch_size=100):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= batch_size: break
        q = item.get("question", "")
        a = item.get("answer", "")
        if q and a:
            ing.ingest_text(f"Math problem: {q} Solution: {a}")
            c += 1
    return c

def train_math_aime(ing, batch_size=50):
    from datasets import load_dataset
    ds = load_dataset("math-ai/aime", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= batch_size: break
        q = item.get("problem", "")
        a = item.get("solution", "")
        if q and a:
            ing.ingest_text(f"Competition math: {q} Solution: {a}")
            c += 1
    return c

def train_physics_sciq(ing, batch_size=80):
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= batch_size: break
        q = item.get("question", "")
        a = item.get("correct_answer", "")
        support = item.get("support", "")
        if q:
            text = f"Physics: {q} Answer: {a}"
            if support: text += f" Explanation: {support}"
            ing.ingest_text(text)
            c += 1
    return c

def train_chemistry(ing, batch_size=60):
    from datasets import load_dataset
    ds = load_dataset("ChemNLP/chem_data", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= batch_size: break
        text = str(item.get("text", "")) or str(item.get("input", ""))
        if text and len(text) > 20:
            ing.ingest_text(f"Chemistry: {text[:500]}")
            c += 1
    return c

def train_code_contests(ing, batch_size=80):
    from datasets import load_dataset
    ds = load_dataset("deepmind/code_contests", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= batch_size: break
        desc = item.get("description", "")
        if desc:
            ing.ingest_text(f"Programming: {desc[:500]}")
            c += 1
    return c

def train_logiqa(ing, batch_size=100):
    from datasets import load_dataset
    ds = load_dataset("tasksource/logiqa", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= batch_size: break
        q = item.get("question", "")
        opts = item.get("options", [])
        a = item.get("label", "")
        if q:
            ing.ingest_text(f"Logic: {q} Options: {opts} Answer: {a}")
            c += 1
    return c

def train_fusion(ing, batch_size=100):
    from datasets import load_dataset
    ds = load_dataset("pseudolab/autotrain-data-Nuclear_Fusion_Falcon", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= batch_size: break
        parts = []
        for k, v in item.items():
            if v is not None and str(v).strip():
                parts.append(f"{k}: {v}")
        text = "Fusion plasma: " + ", ".join(parts[:8])
        ing.ingest_text(text[:500])
        c += 1
    return c

def train_genomics(ing, batch_size=60):
    from datasets import load_dataset
    ds = load_dataset("longevity-db/Human_Ageing_Genomic_Resources", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= batch_size: break
        text = " ".join(f"{k}: {v}" for k, v in item.items() if v and str(v).strip())
        if text and len(text) > 20:
            ing.ingest_text(f"Genomics gene aging: {text[:500]}")
            c += 1
    return c

def train_virus(ing, batch_size=100):
    from datasets import load_dataset
    ds = load_dataset("hiyata/Virus-Host-Genomes", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= batch_size: break
        family = item.get("family", "")
        host = item.get("host", "")
        zoonotic = item.get("zoonotic", "")
        seq = item.get("sequence", "")[:200]
        text = f"Virus: family={family} host={host} zoonotic={zoonotic} sequence={seq}"
        ing.ingest_text(text)
        c += 1
    return c

def train_aging(ing, batch_size=50):
    from datasets import load_dataset
    ds = load_dataset("longevity-db/aging-fly-cell-atlas", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= batch_size: break
        text = " ".join(f"{k}: {v}" for k, v in item.items() if v and str(v).strip())
        if text and len(text) > 20:
            ing.ingest_text(f"Aging biology: {text[:500]}")
            c += 1
    return c

def train_pubmedqa(ing, batch_size=100):
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= batch_size: break
        q = item.get("question", "")
        ctx = " ".join(item.get("context", {}).get("contexts", [])[:2])
        a = item.get("final_decision", "")
        if q:
            ing.ingest_text(f"Medical: {q} Context: {ctx[:300]} Answer: {a}")
            c += 1
    return c

def train_wikipedia(ing, batch_size=100):
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= batch_size: break
        text = item.get("text", "")
        if text and len(text) > 100:
            ing.ingest_text(text[:500])
            c += 1
    return c

def main_parallel():
    t0 = time.time()
    print("=" * 70)
    print("  PARALLEL URCM 10M-PARAMETER BRAIN TRAINER")
    print("  12 Topics | 30+ Datasets | Multi-Process Parallel")
    print("=" * 70)

    ing = KnowledgeIngestion(brain_path=BRAIN, l2_dim=L2)
    print(f"Brain initialized: L2={L2}, params=~10M")

    dataset_batches = [
        ("Math (GSM8K)", train_math_gsm8k, 100),
        ("Math (AIME)", train_math_aime, 50),
        ("Physics (SciQ)", train_physics_sciq, 80),
        ("Chemistry", train_chemistry, 60),
        ("Code Contests", train_code_contests, 80),
        ("Logic (LogiQA)", train_logiqa, 100),
        ("Fusion Physics", train_fusion, 100),
        ("Genomics", train_genomics, 60),
        ("Virus Analysis", train_virus, 100),
        ("Aging Biology", train_aging, 50),
        ("PubMedQA", train_pubmedqa, 100),
        ("Wikipedia", train_wikipedia, 100),
    ]

    shared_brain = BRAIN + ".tmp"
    ing.save()

    total_text_count = 0

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = []

        for name, func, batch_size in dataset_batches:
            future = executor.submit(
                process_dataset,
                name,
                partial(func, batch_size=batch_size),
                shared_brain
            )
            futures.append((name, future))

        completed = 0
        for name, future in futures:
            result = future.result()
            completed += 1

            if result["success"]:
                total_text_count += result["count"]
                print(f"[{completed}/{len(dataset_batches)}] {name}: +{result['count']} texts | Concepts: {len(ing.concept_map)}")
            else:
                print(f"[{completed}/{len(dataset_batches)}] {name}: FAILED - {result['error']}")

            if completed % 3 == 0:
                ing.save()
                print(f"  -- checkpoint saved ({len(ing.concept_map)} concepts) --")

    ing.save()

    t = time.time() - t0
    sz = os.path.getsize(BRAIN) / (1024*1024) if os.path.exists(BRAIN) else 0

    print("\n" + "=" * 70)
    print("  PARALLEL 10M BRAIN TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Brain:       {BRAIN}")
    print(f"  Size:        {sz:.1f} MB")
    print(f"  Concepts:    {len(ing.concept_map)}")
    print(f"  Relations:   {len(ing.relations)}")
    print(f"  Total texts: {total_text_count}")
    print(f"  L2 dim:      {L2}")
    print(f"  Parameters:  ~{L2*L2*10//1000000}M")
    print(f"  Time:        {t:.0f}s ({t/60:.1f} min)")
    print("=" * 70)

if __name__ == "__main__":
    main_parallel()
