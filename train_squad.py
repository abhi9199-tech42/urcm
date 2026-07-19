"""Download high-quality QA dataset (SQuAD 2.0) and train URCM with it."""
import sys, os, json, urllib.request, pickle, time, argparse
import numpy as np

base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base)
os.chdir(base)

import builtins
_orig = builtins.__import__
def _skip(name, *a, **kw):
    if name.startswith("isre"): raise ImportError()
    return _orig(name, *a, **kw)
builtins.__import__ = _skip

# --- Download SQuAD 2.0 dev set (smaller, ~12MB) ---
squad_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
squad_file = os.path.join(base, "squad_train.json")

if not os.path.exists(squad_file):
    print("Downloading SQuAD 2.0 dev set...")
    urllib.request.urlretrieve(squad_url, squad_file)
    print(f"Downloaded {squad_file}")
else:
    print(f"Already have {squad_file}")

# --- Convert SQuAD to URCM training text ---
print("Converting SQuAD QA pairs to training text...")
with open(squad_file, "r", encoding="utf-8") as f:
    squad = json.load(f)

qa_texts = []
for article in squad["data"]:
    title = article.get("title", "")
    for para in article["paragraphs"]:
        context = para["context"]
        # Add context as training text
        qa_texts.append(context)
        for qa in para["qas"]:
            question = qa["question"]
            if not qa["is_impossible"]:
                for ans in qa["answers"]:
                    answer = ans["text"]
                    # Format as QA statement
                    qa_texts.append(f"Question: {question} Answer: {answer}")
                    # Also add as standalone statements
                    qa_texts.append(f"{question} {answer}")
            else:
                # Unanswerable questions - still useful for training
                qa_texts.append(question)
                qa_texts.append(f"The question '{question}' cannot be answered from the given context.")

print(f"Generated {len(qa_texts)} training texts from SQuAD 2.0")

# Remove duplicates
qa_texts = list(set(qa_texts))
print(f"After dedup: {len(qa_texts)} unique texts")

# --- Import URCM and train ---
from urcm.core.system import URCMSystem
from urcm.core.data_models import FrequencyPath

print("\nInitializing URCM System...")
system = URCMSystem()
system.pipeline.frequency_mapper.smoothness_weight = 0.0
print("System initialized (smoothing disabled).")

# Prepare training data
import random
random.seed(42)
random.shuffle(qa_texts)
train_texts = qa_texts[:5000]  # Use 5K samples for this run
print(f"Training on {len(train_texts)} texts")

# Also keep the original TEST_DATA_100 + Sanskrit data
from tests.comprehensive_test_data import TEST_DATA_100

# Add TEST_DATA_100 statements
for cat, statements in TEST_DATA_100.items():
    train_texts.extend(statements)
print(f"After adding TEST_DATA_100: {len(train_texts)} texts")

# Add Sanskrit mantras (for phoneme coverage)
sanskrit_mantras = [
    "om namah shivaya", "om mani padme hum", "gayatri mantra",
    "om bhur bhuva svah", "om shantih shantih shantih",
    "aham brahmasmi", "tat tvam asi", "satyam eva jayate",
    "om ganeshaya namah", "om namo narayanaya",
]
for m in sanskrit_mantras:
    train_texts.extend([m] * 20)
print(f"After Sanskrit: {len(train_texts)} texts")

# --- Training ---
def path_generator(texts, system, batch_size=50):
    """Yield batches of FrequencyPaths from text."""
    current_batch = []
    for text in texts:
        if not text or not text.strip():
            continue
        try:
            path = system.pipeline.process_text(text)
            current_batch.append(path)
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []
        except Exception as e:
            continue
    if current_batch:
        yield current_batch

parser = argparse.ArgumentParser()
parser.add_argument("--cycles", type=int, default=10)
parser.add_argument("--noise", type=float, default=0.005)
args = parser.parse_args()

print(f"\nTraining URCM with {args.cycles} cycles, noise={args.noise}...")
t0 = time.time()

# Phase 1: Fast one-shot denoising
print("\nPhase 1: One-shot pre-training...")
gen = lambda: path_generator(train_texts, system)
mse = system.encoder.train_decoder_fast_denoising(gen, noise_level=0.0001, ridge_alpha=1e-6)
print(f"  Phase 1 MSE: {mse:.6f}")

# Phase 2: DAgger dreaming
print(f"\nPhase 2: DAgger dreaming ({args.cycles} cycles)...")
mse = system.encoder.train_decoder_incremental(gen, iterations=args.cycles,
                                                ridge_alpha=1e-6, noise_level=args.noise)
print(f"  Final MSE: {mse:.6f}")

# Save weights
weights = {
    "W_in": system.encoder.W_in, "W_res": system.encoder.W_res,
    "W_out": system.encoder.W_out, "bias": system.encoder.bias,
    "W_res_inv": system.encoder.W_res_inv
}
weight_path = os.path.join(base, "urcm_weights_squad.pkl")
with open(weight_path, "wb") as f:
    pickle.dump(weights, f)
print(f"\nWeights saved to {weight_path}")

# Validation
print("\n--- Validation ---")
mapper = system.pipeline.frequency_mapper
test_queries = [
    ("quantum physics", "quantum"),
    ("neural networks", "neural"),
    ("what is superposition in quantum mechanics", "superposition"),
    ("how does photosynthesis work", "photosynthesis"),
    ("who wrote to be or not to be", "shakespeare"),
]
for query, expected in test_queries:
    try:
        path = system.pipeline.process_text(query)
        r_state = system.encoder.get_resonance_state(path)
        decoded = system.encoder.decode_state(r_state, steps=len(path.vectors),
                                               phoneme_vectors=mapper.phoneme_vectors)
        phonemes = []
        for vec in decoded:
            best = min(mapper.phoneme_vectors.keys(),
                      key=lambda p: np.linalg.norm(vec - mapper.phoneme_vectors[p]))
            phonemes.append(best)
        print(f"  Input: {query}")
        print(f"  Output: {'-'.join(phonemes[:10])}")
    except Exception as e:
        print(f"  Input: {query} - Error: {e}")

print(f"\nDone in {time.time()-t0:.0f}s")
