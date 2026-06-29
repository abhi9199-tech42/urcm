
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from urcm.core.system import URCMSystem

def measure_reversibility():
    print("🧪 URCM Reversibility Benchmark")
    print("==============================")
    
    # 1. Initialize System
    system = URCMSystem()
    # Disable smoothing for exact discrete test
    system.pipeline.frequency_mapper.smoothness_weight = 0.0
    print("✅ System Initialized (Smoothing Disabled).")
    
    # 2. Define Test Set (Sanskrit Mantras)
    test_inputs = [
        "oṃ namaḥ śivāya",
        "oṃ maṇipadme hūṃ",
        "gāyatrī mantra",
        "ahaṃ brahmāsmi",
        "tat tvam asi",
        "satyam eva jayate",
        "oṃ gaṇeśāya namaḥ",
        "hare kṛṣṇa hare kṛṣṇa",
        "kṛṣṇa kṛṣṇa hare hare",
        "hare rāma hare rāma",
        "rāma rāma hare hare"
    ]
    
    print(f"📊 Running benchmark on {len(test_inputs)} sequences...")
    
    total_phonemes = 0
    correct_phonemes = 0
    perfect_sequences = 0
    
    phoneme_vectors = system.pipeline.frequency_mapper.phoneme_vectors
    
    # Helper to get char from vector
    vec_to_char = {}
    for k, v in phoneme_vectors.items():
        vec_to_char[v.tobytes()] = k

    for idx, text in enumerate(test_inputs):
        print(f"\n[{idx+1}/{len(test_inputs)}] Testing: '{text}'")
        
        # 1. Forward Pass (Encode)
        freq_path = system.pipeline.process_text(text)
        final_state = system.encoder.get_resonance_state(freq_path)
        
        ground_truth_len = len(freq_path.vectors)
        ground_truth_chars = []
        for vec in freq_path.vectors:
            ground_truth_chars.append(vec_to_char.get(vec.tobytes(), "?"))
            
        # 2. Backward Pass (Decode - Free Dreaming)
        # We catch the output vectors
        decoded_vecs = system.encoder.decode_state(
            final_state, 
            steps=ground_truth_len,
            track_energy=False, # Faster
            phoneme_vectors=phoneme_vectors,
            ground_truth_vectors=freq_path.vectors,
            force_ground_truth=False # IMPORTANT: Test actual performance
        )
        
        # 3. Compare
        decoded_chars = []
        # decode_state returns vectors in FORWARD order [x_1, x_2, ..., x_T]
        # (It builds the list by inserting at 0 during backward iteration)
        
        decoded_vecs_ordered = decoded_vecs
        
        seq_matches = 0
        
        print(f"  GT : {'-'.join(ground_truth_chars)}")
        
        decoded_chars_str = []
        
        for i in range(ground_truth_len):
            if i >= len(decoded_vecs_ordered):
                break
                
            v_out = decoded_vecs_ordered[i]
            v_gt = freq_path.vectors[i]
            
            # Find nearest char for output
            best_c = "?"
            best_d = float('inf')
            for k, v in phoneme_vectors.items():
                d = np.linalg.norm(v_out - v)
                if d < best_d:
                    best_d = d
                    best_c = k
            
            decoded_chars_str.append(best_c)
            
            # Strict match check (using Snapping, they should be identical if correct)
            # Or use name comparison
            if best_c == ground_truth_chars[i]:
                correct_phonemes += 1
                seq_matches += 1
        
        print(f"  OUT: {'-'.join(decoded_chars_str)}")
        
        total_phonemes += ground_truth_len
        if seq_matches == ground_truth_len:
            perfect_sequences += 1
            print("  ✅ Perfect Recall")
        else:
            print(f"  ❌ Errors: {ground_truth_len - seq_matches}/{ground_truth_len}")

    # Summary
    accuracy = (correct_phonemes / total_phonemes) * 100 if total_phonemes > 0 else 0
    seq_accuracy = (perfect_sequences / len(test_inputs)) * 100
    
    print("\n==============================")
    print("📈 FINAL REVERSIBILITY REPORT")
    print("==============================")
    print(f"Total Phonemes: {total_phonemes}")
    print(f"Correctly Recovered: {correct_phonemes}")
    print(f"Phoneme Accuracy: {accuracy:.2f}%")
    print(f"Perfect Sequence Rate: {seq_accuracy:.2f}%")
    print("==============================")
    
    if accuracy > 99.0:
        print("🚀 SYSTEM STATUS: LAUNCH READY")
    else:
        print("⚠️ SYSTEM STATUS: TRAINING REQUIRED")

if __name__ == "__main__":
    measure_reversibility()
