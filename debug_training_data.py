
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from urcm.core.system import URCMSystem
from tests.comprehensive_test_data import TEST_DATA_100
from verify_scale_test import generate_extra_inputs
from urcm.core.data_models import FrequencyPath

def debug_data():
    system = URCMSystem()
    system.pipeline.frequency_mapper.smoothness_weight = 0.0
    
    text_inputs = generate_extra_inputs(TEST_DATA_100, 200)
    sanskrit_inputs = [
        "oṃ namaḥ śivāya",
        "oṃ maṇipadme hūṃ",
    ]
    for _ in range(5):
        text_inputs.extend(sanskrit_inputs)

    # Generator
    def path_generator():
        # 1. Text Inputs
        for text in text_inputs:
            yield system.pipeline.process_text(text)
            
        # 2. Atomic
        mapper = system.pipeline.frequency_mapper
        for phoneme in mapper.SANSKRIT_PHONEMES:
            vec = mapper.map_phoneme(phoneme)
            vectors = np.array([vec] * 5)
            path = FrequencyPath(
                vectors=vectors,
                smoothness_score=0.0,
                phoneme_mapping=[(phoneme, i) for i in range(5)]
            )
            # Yield 1 copy
            yield path

    print("Generating and inspecting first 10 paths...")
    gen = path_generator()
    
    mapper = system.pipeline.frequency_mapper
    # Reverse map
    vec_to_char = {}
    for k, v in mapper.phoneme_vectors.items():
        vec_to_char[v.tobytes()] = k

    count = 0
    for path in gen:
        count += 1
        if count > 10: break
        
        print(f"\nPath {count} (Len {len(path.vectors)}):")
        # Print first few phonemes
        chars = []
        for v in path.vectors[:10]:
            c = vec_to_char.get(v.tobytes(), "?")
            chars.append(c)
        print("  " + " ".join(chars))
        
        # Check if vectors are normalized
        norms = np.linalg.norm(path.vectors, axis=1)
        print(f"  Norms: Min={np.min(norms):.4f}, Max={np.max(norms):.4f}")

if __name__ == "__main__":
    debug_data()
