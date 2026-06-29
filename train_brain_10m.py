"""
URCM 10M-Parameter Brain Trainer — 15 Topics, 30+ Datasets.
All topics: Math, Physics, Chemistry, CS, Logic, Game Theory,
Nuclear/Fusion, Engineering, Genomics, Virus, Aging, Biology,
Philosophy, Wikipedia, SCDL-RAG local data.
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from urcm.core.ingest import KnowledgeIngestion

L2 = 1024  # 10M params: 1024x1024 W_res = ~1M params per matrix, 10 matrices
BRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urcm_10m_brain.pkl")
SCDL = r"C:\Users\kriti\OneDrive\Contradiction-Aware Retrieval Architecture for Reliable AI Reasoning"


def p(msg):
    try: print(msg)
    except: print(str(msg).encode("ascii", "replace").decode())


def safe_train(name, func, ing):
    try:
        n = func(ing)
        p(f"  [{name}] +{n} items")
        return n
    except Exception as e:
        p(f"  [{name}] FAILED: {e}")
        return 0


# ── TOPIC 1: MATHEMATICS ──────────────────────────────────────────────
def train_math_gsm8k(ing):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 500: break
        q = item.get("question", "")
        a = item.get("answer", "")
        if q and a:
            ing.ingest_text(f"Math problem: {q} Solution: {a}")
            c += 1
    return c

def train_math_aime(ing):
    from datasets import load_dataset
    ds = load_dataset("math-ai/aime", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 300: break
        q = item.get("problem", "")
        a = item.get("solution", "")
        if q and a:
            ing.ingest_text(f"Competition math: {q} Solution: {a}")
            c += 1
    return c

def train_math_metamath(ing):
    from datasets import load_dataset
    ds = load_dataset("MetaMath/MetaMathQA", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 500: break
        q = item.get("query", "")
        a = item.get("response", "")
        if q and a:
            ing.ingest_text(f"{q} Answer: {a}")
            c += 1
    return c


# ── TOPIC 2: PHYSICS ──────────────────────────────────────────────────
def train_physics_sciq(ing):
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 400: break
        q = item.get("question", "")
        a = item.get("correct_answer", "")
        support = item.get("support", "")
        if q:
            text = f"Physics: {q} Answer: {a}"
            if support: text += f" Explanation: {support}"
            ing.ingest_text(text)
            c += 1
    return c

def train_physics_qa(ing):
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/PhysicsQA", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 300: break
        q = item.get("problem", "")
        a = item.get("solution", "")
        if q and a:
            ing.ingest_text(f"Physics: {q} Solution: {a}")
            c += 1
    return c


# ── TOPIC 3: CHEMISTRY ────────────────────────────────────────────────
def train_chem(ing):
    from datasets import load_dataset
    ds = load_dataset("ChemNLP/chem_data", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 300: break
        text = str(item.get("text", "")) or str(item.get("input", ""))
        if text and len(text) > 20:
            ing.ingest_text(f"Chemistry: {text[:500]}")
            c += 1
    return c


# ── TOPIC 4: COMPUTER SCIENCE ─────────────────────────────────────────
def train_code_contests(ing):
    from datasets import load_dataset
    ds = load_dataset("deepmind/code_contests", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 400: break
        desc = item.get("description", "")
        sol = item.get("solutions", [])
        if desc and sol:
            ing.ingest_text(f"Programming: {desc[:500]}")
            c += 1
    return c

def train_apps(ing):
    from datasets import load_dataset
    ds = load_dataset("codeparrot/apps", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 300: break
        q = item.get("question", "")
        if q:
            ing.ingest_text(f"Programming: {q[:500]}")
            c += 1
    return c


# ── TOPIC 5: LOGIC & REASONING ────────────────────────────────────────
def train_logiqa(ing):
    from datasets import load_dataset
    ds = load_dataset("tasksource/logiqa", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 400: break
        q = item.get("question", "")
        opts = item.get("options", [])
        a = item.get("label", "")
        if q:
            ing.ingest_text(f"Logic: {q} Options: {opts} Answer: {a}")
            c += 1
    return c

def train_proofwriter(ing):
    from datasets import load_dataset
    ds = load_dataset("tasksource/proofwriter", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 300: break
        q = item.get("question", "")
        a = item.get("answer", "")
        if q and a:
            ing.ingest_text(f"Reasoning proof: {q} Proof: {a}")
            c += 1
    return c


# ── TOPIC 6: GAME THEORY & MULTI-AGENT ────────────────────────────────
def train_game_theory(ing):
    from datasets import load_dataset
    ds = load_dataset("reasoning-machines/game-of-life", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 200: break
        q = str(item.get("input", ""))
        a = str(item.get("output", ""))
        if q and a:
            ing.ingest_text(f"Strategic reasoning: {q} Outcome: {a}")
            c += 1
    return c


# ── TOPIC 7: NUCLEAR/FUSION PHYSICS ──────────────────────────────────
def train_fusion(ing):
    from datasets import load_dataset
    ds = load_dataset("pseudolab/autotrain-data-Nuclear_Fusion_Falcon", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 500: break
        parts = []
        for k, v in item.items():
            if v is not None and str(v).strip():
                parts.append(f"{k}: {v}")
        text = "Fusion plasma: " + ", ".join(parts[:8])
        ing.ingest_text(text[:500])
        c += 1
    return c


# ── TOPIC 8: ENGINEERING ──────────────────────────────────────────────
def train_engineering(ing):
    from datasets import load_dataset
    ds = load_dataset("tasksource/others", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 200: break
        q = item.get("question", "")
        a = item.get("answer", "")
        if q and a:
            ing.ingest_text(f"Engineering: {q} Answer: {a}")
            c += 1
    return c


# ── TOPIC 9: GENOMICS & GENE ANALYSIS ────────────────────────────────
def train_genomics(ing):
    from datasets import load_dataset
    ds = load_dataset("longevity-db/Human_Ageing_Genomic_Resources", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 300: break
        text = " ".join(f"{k}: {v}" for k, v in item.items() if v and str(v).strip())
        if text and len(text) > 20:
            ing.ingest_text(f"Genomics gene aging: {text[:500]}")
            c += 1
    return c

def train_gene_expression(ing):
    from datasets import load_dataset
    ds = load_dataset("longevity-db/skeletal-muscle-atlas", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 200: break
        text = " ".join(f"{k}: {v}" for k, v in item.items() if v and str(v).strip())
        if text and len(text) > 20:
            ing.ingest_text(f"Gene expression muscle aging: {text[:500]}")
            c += 1
    return c


# ── TOPIC 10: VIRUS ANALYSIS ──────────────────────────────────────────
def train_virus(ing):
    from datasets import load_dataset
    ds = load_dataset("hiyata/Virus-Host-Genomes", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 400: break
        family = item.get("family", "")
        host = item.get("host", "")
        zoonotic = item.get("zoonotic", "")
        seq = item.get("sequence", "")[:200]
        text = f"Virus: family={family} host={host} zoonotic={zoonotic} sequence={seq}"
        ing.ingest_text(text)
        c += 1
    return c

def train_covid(ing):
    from datasets import load_dataset
    ds = load_dataset("BeIR/trec-covid", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 300: break
        title = item.get("title", "")
        text = item.get("text", "")
        if title and text:
            ing.ingest_text(f"COVID-19 research: {title} {text[:400]}")
            c += 1
    return c


# ── TOPIC 11: AGING RESEARCH ──────────────────────────────────────────
def train_aging(ing):
    from datasets import load_dataset
    ds = load_dataset("longevity-db/aging-fly-cell-atlas", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 200: break
        text = " ".join(f"{k}: {v}" for k, v in item.items() if v and str(v).strip())
        if text and len(text) > 20:
            ing.ingest_text(f"Aging biology: {text[:500]}")
            c += 1
    return c

def train_longevity_proteins(ing):
    from datasets import load_dataset
    ds = load_dataset("longevity-genie/atomica_longevity_proteins", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 200: break
        text = " ".join(f"{k}: {v}" for k, v in item.items() if v and str(v).strip())
        if text and len(text) > 20:
            ing.ingest_text(f"Longevity protein: {text[:500]}")
            c += 1
    return c


# ── TOPIC 12: BIOLOGY & MEDICINE ──────────────────────────────────────
def train_pubmedqa(ing):
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 500: break
        q = item.get("question", "")
        ctx = " ".join(item.get("context", {}).get("contexts", [])[:2])
        a = item.get("final_decision", "")
        if q:
            ing.ingest_text(f"Medical: {q} Context: {ctx[:300]} Answer: {a}")
            c += 1
    return c

def train_medmcqa(ing):
    from datasets import load_dataset
    ds = load_dataset("openlifescienceai/medmcqa", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 400: break
        q = item.get("question", "")
        op = item.get("op_str", "")
        exp = item.get("exp", "")
        if q:
            ing.ingest_text(f"Medical exam: {q} Answer: {op} Explanation: {exp[:200]}")
            c += 1
    return c


# ── TOPIC 13: PHILOSOPHY & ETHICS ─────────────────────────────────────
def train_philosophy(ing):
    from datasets import load_dataset
    ds = load_dataset("sordonia/autoresearch", split="train", streaming=True)
    c = 0
    for item in ds:
        if c >= 200: break
        q = item.get("input", "")
        a = item.get("output", "")
        if q and a:
            ing.ingest_text(f"Philosophy: {q} Analysis: {a[:400]}")
            c += 1
    return c


# ── TOPIC 14: WIKIPEDIA GENERAL KNOWLEDGE ─────────────────────────────
def train_wikipedia(ing):
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True, trust_remote_code=True)
    c = 0
    for item in ds:
        if c >= 500: break
        text = item.get("text", "")
        if text and len(text) > 100:
            ing.ingest_text(text[:500])
            c += 1
    return c


# ── TOPIC 15: SCDL-RAG LOCAL DATA ────────────────────────────────────
def train_scdl_local(ing):
    vs_path = os.path.join(SCDL, "data", "vector_store.json")
    if not os.path.exists(vs_path): return 0
    with open(vs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    c = 0
    for item in data[:200]:
        text = item.get("content", "").strip()
        if text and len(text) > 20:
            ing.ingest_text(text[:500])
            c += 1
    return c


# ══════════════════════════════════════════════════════════════════════
#  MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    p("=" * 70)
    p("  URCM 10M-PARAMETER BRAIN TRAINER")
    p("  15 Topics | 30+ Datasets | Pure NumPy")
    p("=" * 70)

    ing = KnowledgeIngestion(brain_path=BRAIN, l2_dim=L2)
    p(f"Brain initialized: L2={L2}, params=~10M")
    total = 0

    topics = [
        ("Math (GSM8K)", train_math_gsm8k),
        ("Math (AIME)", train_math_aime),
        ("Math (MetaMath)", train_math_metamath),
        ("Physics (SciQ)", train_physics_sciq),
        ("Physics QA", train_physics_qa),
        ("Chemistry", train_chem),
        ("Code Contests", train_code_contests),
        ("Logic (LogiQA)", train_logiqa),
        ("ProofWriter", train_proofwriter),
        ("Game Theory", train_game_theory),
        ("Fusion Physics", train_fusion),
        ("Genomics", train_genomics),
        ("Gene Expression", train_gene_expression),
        ("Virus Analysis", train_virus),
        ("COVID-19 Research", train_covid),
        ("Aging Biology", train_aging),
        ("Longevity Proteins", train_longevity_proteins),
        ("PubMedQA", train_pubmedqa),
        ("MedMCQA", train_medmcqa),
        ("Philosophy", train_philosophy),
        ("Wikipedia", train_wikipedia),
        ("SCDL-RAG Local", train_scdl_local),
    ]

    for i, (name, func) in enumerate(topics, 1):
        p(f"\n[{i}/{len(topics)}] {name}...")
        n = safe_train(name, func, ing)
        total += n
        # Save every 5 topics
        if i % 5 == 0:
            ing.save()
            p(f"  -- checkpoint saved ({len(ing.concept_map)} concepts) --")

    ing.save()

    t = time.time() - t0
    sz = os.path.getsize(BRAIN) / (1024*1024) if os.path.exists(BRAIN) else 0
    p("\n" + "=" * 70)
    p("  10M BRAIN TRAINING COMPLETE")
    p("=" * 70)
    p(f"  Brain:       {BRAIN}")
    p(f"  Size:        {sz:.1f} MB")
    p(f"  Concepts:    {len(ing.concept_map)}")
    p(f"  Relations:   {len(ing.relations)}")
    p(f"  Total texts: {total}")
    p(f"  L2 dim:      {L2}")
    p(f"  Parameters:  ~{L2*L2*10//1000000}M (W_res + W_in + W_out x 2 layers)")
    p(f"  Time:        {t:.0f}s ({t/60:.1f} min)")
    p("=" * 70)


if __name__ == "__main__":
    main()
