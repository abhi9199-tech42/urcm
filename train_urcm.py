import numpy as np
import pickle
import os
import sys
import time
import argparse

# Add root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from urcm.core.system import URCMSystem
from tests.comprehensive_test_data import TEST_DATA_100
from verify_scale_test import generate_extra_inputs
from urcm.core.data_models import FrequencyPath

def train_system():
    parser = argparse.ArgumentParser(description="URCM Training System")
    parser.add_argument("--duration_minutes", type=int, default=0, help="Run training loop for N minutes (0 for standard 3 cycles)")
    parser.add_argument("--extended_cycles", type=int, default=0, help="Run specific number of extra cycles")
    parser.add_argument("--noise_level", type=float, default=0.005, help="Noise level for DAgger training (default: 0.005)")
    args = parser.parse_args()

    print("🚀 Starting URCM Training Session...")
    
    # 1. Initialize System
    system = URCMSystem()
    # Disable smoothing for training to ensure we learn exact mappings
    system.pipeline.frequency_mapper.smoothness_weight = 0.0
    print("✅ System Initialized (Smoothing Disabled for Training).")
    
    # 2. Prepare Data
    # Use the 200 inputs from scale test
    text_inputs = generate_extra_inputs(TEST_DATA_100, 200)
    
    # CRITICAL: Add Sanskrit Mantras for Sequence Training
    # W_out needs to learn the trajectories of Sanskrit phonemes, not just atomic states.
    sanskrit_inputs = [
        "oṃ namaḥ śivāya",
        "oṃ maṇipadme hūṃ",
        "gāyatrī mantra",
        "oṃ bhūr bhuvaḥ svaḥ tat savitur vareṇyaṃ bhargo devasya dhīmahi dhiyo yo naḥ pracodayāt",
        "oṃ tryambakaṃ yajāmahe sugandhiṃ puṣṭivardhanam",
        "urvārukamiva bandhanān mṛtyormukṣīya māmṛtāt",
        "oṃ śāntiḥ śāntiḥ śāntiḥ",
        "ahaṃ brahmāsmi",
        "tat tvam asi",
        "satyam eva jayate",
        "oṃ gaṇeśāya namaḥ",
        "oṃ namo nārāyaṇāya",
        "oṃ namo bhagavate vāsudevāya",
        "hare kṛṣṇa hare kṛṣṇa kṛṣṇa kṛṣṇa hare hare",
        "hare rāma hare rāma rāma rāma hare hare"
    ]
    # Replicate them to give them weight
    for _ in range(20):
        text_inputs.extend(sanskrit_inputs)
        
    print(f"➕ Added {len(sanskrit_inputs) * 20} Sanskrit Mantra sequences.")
    
    # CRITICAL: Add "Atomic" Phoneme Training Data for ALL phonemes
    print("➕ Augmenting with Atomic Phoneme Data (All 50+ Sanskrit Phonemes)...")
    mapper = system.pipeline.frequency_mapper
    
    atomic_paths = []
    
    # Create pure atomic paths for every single phoneme in the codebook
    # This ensures W_out has a "pin" for every possible symbol.
    for phoneme in mapper.SANSKRIT_PHONEMES:
        try:
            # Create a path with this phoneme repeated 
            # We repeat it to simulate a stable state holding this sound
            # But importantly, the first step is s_0 -> s_1(x)
            vec = mapper.map_phoneme(phoneme)
            
            # Create a FrequencyPath manually
            # Repeat 5 times
            vectors = np.array([vec] * 5)
            
            path = FrequencyPath(
                vectors=vectors,
                smoothness_score=0.0,
                phoneme_mapping=[(phoneme, i) for i in range(5)]
            )
            
            # Add 20 copies of each atomic path to weight them heavily
            atomic_paths.extend([path] * 20)
            
        except Exception as e:
            print(f"  ⚠️ Could not map phoneme {phoneme}: {e}")
            
    print(f"✅ Created {len(atomic_paths)} atomic training paths.")

    # 4. Import Multilingual Understanding from ISRE
    print("🌍 Importing Multilingual Understanding from ISRE...")
    try:
        isre_path = os.environ.get("URCM_ISRE_PATH", os.path.join(os.path.dirname(__file__), "..", "ISRE"))
        if isre_path not in sys.path:
            sys.path.append(isre_path)
        
        from isre.compression.text import ConceptMapper
        isre_mapper = ConceptMapper()
        
        # Extract multilingual concepts
        multilingual_inputs = []
        if hasattr(isre_mapper, '_semantic_map'):
             # Add keys (e.g., 'pomme', 'manzana') as inputs
             # This teaches URCM to resonate with these foreign words
             multilingual_inputs.extend(list(isre_mapper._semantic_map.keys()))
             # Add values (e.g., 'fruit') to reinforce the concept
             multilingual_inputs.extend(list(isre_mapper._semantic_map.values()))
        
        # Remove duplicates
        multilingual_inputs = list(set(multilingual_inputs))
        
        # Add to training data (replicate 10 times)
        for _ in range(10):
            text_inputs.extend(multilingual_inputs)
            
        print(f"✅ Imported {len(multilingual_inputs)} multilingual concepts from ISRE.")
        
    except Exception as e:
        print(f"⚠️ Failed to import ISRE data: {e}")
    
    # 3. Define Data Generator
    def path_generator():
        """Yields batches of FrequencyPaths."""
        batch_size = 50
        
        # 1. Yield Atomic Paths first (Base Knowledge)
        current_batch = []
        for path in atomic_paths:
            current_batch.append(path)
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []
        if current_batch:
            yield current_batch
            
        # 2. Yield Text Paths (Sequence Knowledge)
        current_batch = []
        for text in text_inputs:
            if not text or not text.strip(): continue
            try:
                # Get FrequencyPath
                path = system.pipeline.process_text(text)
                current_batch.append(path)
                
                if len(current_batch) >= batch_size:
                    yield current_batch
                    current_batch = []
            except Exception as e:
                continue
        if current_batch:
            yield current_batch
    
    # 4. Train Decoder (Readout Layer)
    print("\n🧠 Training Readout Layer (Hybrid Strategy)...")
    
    # Phase 1: Fast One-Shot Denoising
    # Use minimal noise (1e-4) to allow for float precision but enforce strictness
    print("Phase 1: Fast One-Shot Pre-training (Noise=1e-4)...")
    mse = system.encoder.train_decoder_fast_denoising(path_generator, noise_level=0.0001, ridge_alpha=1e-6)
    print(f"  Phase 1 MSE (Approx): {mse:.6f}")
    
    # Phase 2: Multi-Cycle Iterative Dreaming
    print("Phase 2: DAgger Dreaming (Fine-tuning)...")
    
    # Determine mode
    if args.duration_minutes > 0:
        print(f"⏳ Running Continuous Training Loop for {args.duration_minutes} minutes...")
        start_time = time.time()
        end_time = start_time + (args.duration_minutes * 60)
        cycle_count = 0
        
        while time.time() < end_time:
            cycle_count += 1
            remaining = int((end_time - time.time()) / 60)
            print(f"\n🔄 Continuous Cycle {cycle_count} (Time Remaining: {remaining}m)...")
            
            # Run 1 cycle at a time
            mse = system.encoder.train_decoder_incremental(path_generator, iterations=1, ridge_alpha=1e-6, noise_level=args.noise_level)
            
            # Save weights periodically (every cycle)
            root_dir = os.path.dirname(os.path.abspath(__file__))
            weight_path = os.path.join(root_dir, "urcm_weights.pkl")
            
            weights = {
                "W_in": system.encoder.W_in,
                "W_res": system.encoder.W_res,
                "W_out": system.encoder.W_out,
                "bias": system.encoder.bias,
                "W_res_inv": system.encoder.W_res_inv
            }
            with open(weight_path, "wb") as f:
                pickle.dump(weights, f)
            print(f"  💾 Weights checkpoint saved to {weight_path}")

    elif args.extended_cycles > 0:
        print(f"🔄 Running Extended Training for {args.extended_cycles} cycles...")
        mse = system.encoder.train_decoder_incremental(path_generator, iterations=args.extended_cycles, ridge_alpha=1e-6, noise_level=args.noise_level)
        
    else:
        # Default behavior (3 cycles)
        print("🔄 Running Standard Training (3 cycles)...")
        mse = system.encoder.train_decoder_incremental(path_generator, iterations=3, ridge_alpha=1e-6, noise_level=args.noise_level)
    
    # 5. Save Weights (Final Save)
    print("\n💾 Saving FINAL trained weights...")
    weights = {
        "W_in": system.encoder.W_in,
        "W_res": system.encoder.W_res,
        "W_out": system.encoder.W_out,
        "bias": system.encoder.bias,
        "W_res_inv": system.encoder.W_res_inv
    }
    
    # Save to project root
    # train_urcm.py is in project root, so we just use its directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(root_dir, "urcm_weights.pkl")
    
    with open(weight_path, "wb") as f:
        pickle.dump(weights, f)
        
    print(f"✅ Weights saved to {weight_path}")
    
    # 6. Quick Validation
    print("\n--- Validation Check ---")
    test_text = "quantum physics"
    path = system.pipeline.process_text(test_text)
    
    # Use get_resonance_state to return a proper object, not just a numpy array
    r_state = system.encoder.get_resonance_state(path)
    
    decoded_vecs = system.encoder.decode_state(
        r_state, 
        steps=len(path.vectors),
        phoneme_vectors=mapper.phoneme_vectors # Enable Snapping for validation
    )
    
    # Decode to phonemes
    decoded_phonemes = []
    for vec in decoded_vecs:
        best_match = None
        best_dist = float('inf')
        for cand_p, cand_vec in mapper.phoneme_vectors.items():
            dist = np.linalg.norm(vec - cand_vec)
            if dist < best_dist:
                best_dist = dist
                best_match = cand_p
        decoded_phonemes.append(best_match)
        
    print(f"Input: {test_text}")
    print(f"Output: {'-'.join(decoded_phonemes)}")
    
    if mse < 0.1:
        print("✅ Training SUCCESSFUL (Low MSE)")
    else:
        print("⚠️ Training completed with HIGH MSE (Underfitting?)")

if __name__ == "__main__":
    train_system()
