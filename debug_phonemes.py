
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from urcm.core.system import URCMSystem

system = URCMSystem()
system.pipeline.frequency_mapper.smoothness_weight = 0.0
text = "oṃ namaḥ śivāya"
print(f"Processing text: '{text}'")
freq_path = system.pipeline.process_text(text)

print(f"Path length: {len(freq_path.vectors)}")

# Reverse lookup phonemes
phoneme_vectors = system.pipeline.frequency_mapper.phoneme_vectors
vec_to_char = {}
for k, v in phoneme_vectors.items():
    vec_tuple = tuple(np.round(v, 5))
    vec_to_char[vec_tuple] = k

print("Forward Path Phonemes:")
for i, vec in enumerate(freq_path.vectors):
    vec_tuple = tuple(np.round(vec, 5))
    char = vec_to_char.get(vec_tuple, "???")
    print(f"Step {i}: '{char}'")
