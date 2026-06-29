
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from urcm.core.system import URCMSystem

system = URCMSystem()
p_vecs = system.pipeline.frequency_mapper.phoneme_vectors

if 'i' in p_vecs and 'b' in p_vecs:
    vec_i = p_vecs['i']
    vec_b = p_vecs['b']
    dist = np.linalg.norm(vec_i - vec_b)
    print(f"Distance between 'i' and 'b': {dist}")
    print(f"Norm 'i': {np.linalg.norm(vec_i)}")
    print(f"Norm 'b': {np.linalg.norm(vec_b)}")
else:
    print("Phonemes not found")
