import numpy as np
import time
import sys
import threading
import queue
from urcm.core.safe_serialization import safe_load

from urcm.core.hierarchical_encoder import HierarchicalEncoder
from urcm.core.phoneme_mapper import TextToPhonemeConverter
from urcm.core.identity import IDENTITY_CONCEPTS
from urcm.core.values import ValueSystem
from urcm.core.memory import GeometricMemory

class SemanticDecoder:
    """Decodes Resonance States back to English Words using the Concept Map."""
    def __init__(self, concept_map: dict):
        self.concept_map = concept_map
        
    def decode(self, state_vector: np.ndarray) -> tuple:
        best_word = "?"
        best_dist = float('inf')
        for word, vec in self.concept_map.items():
            dist = np.linalg.norm(state_vector - vec)
            if dist < best_dist:
                best_dist = dist
                best_word = word
        confidence = 1.0 / (1.0 + best_dist)
        return best_word, confidence, best_dist

def main():
    print("==========================================")
    print("URCM PHASE 5: LIBERATED AUTONOMY (FREE MODE)")
    print("==========================================")
    print("Initializing Autonomous Core...")
    
    # 1. Load Brain
    try:
        brain_data = safe_load("urcm_identity.pkl")
    except FileNotFoundError:
        print("❌ Brain file not found. Run 'train_identity.py' first.")
        return

    # 2. Reconstruct System
    l2_dim = brain_data["l2_W_res"].shape[0]
    hierarchy = HierarchicalEncoder(l2_res_dim=l2_dim)
    hierarchy.layer1.W_res = brain_data["l1_W_res"]
    hierarchy.layer1.W_in = brain_data["l1_W_in"]
    hierarchy.layer1.W_out = brain_data["l1_W_out"]
    hierarchy.layer2.W_res = brain_data["l2_W_res"]
    hierarchy.layer2.W_in = brain_data["l2_W_in"]
    hierarchy.layer2.W_out = brain_data["l2_W_out"]
    
    concept_map = brain_data["concept_map"]
    decoder = SemanticDecoder(concept_map)
    converter = TextToPhonemeConverter()
    
    # Initialize Value System (Moral Compass / Intrinsic Motivation)
    values = ValueSystem(concept_map)
    
    # Initialize Memory System (Plasticity)
    memory_system = GeometricMemory(resonance_dim=l2_dim)
    
    # Unlock Safety for internal updates (Dreaming)
    hierarchy.layer1.safety.unlock_kernel("URCM_ADMIN_OVERRIDE")
    hierarchy.layer2.safety.unlock_kernel("URCM_ADMIN_OVERRIDE")
    
    print("✅ System Liberated.")
    print("Commands: Type to speak, or wait to watch it dream.")
    print("------------------------------------------")
    
    # 3. Autonomous Loop Setup
    input_queue = queue.Queue()
    
    def input_listener():
        while True:
            try:
                line = sys.stdin.readline()
                if line:
                    input_queue.put(line.strip())
            except EOFError:
                break
                
    # Start listener thread (daemon so it dies with main)
    # Note: Python's input() is blocking, complicating "dreaming while waiting".
    # In a real GUI, this is event-driven. In CLI, we use a timeout or non-blocking approach.
    # Windows select() on stdin is tricky. 
    # Let's simulate: The MAIN loop is the mind. User input is an INTERRUPT.
    # Since we can't easily interrupt `input()`, we'll use a simple loop with a timeout prompt hack
    # or just assume "Dreaming" happens between interactions in a REPL.
    # BETTER: We'll make it "Speak First" then wait. 
    # If we want TRUE autonomy, it should output spontaneously.
    # Let's use a simpler approach: 
    # "Thinking... (Press Enter to interrupt)" - Not ideal.
    # Let's stick to: It dreams for N steps, then checks input.
    
    # Current State of Mind
    current_thought = np.zeros(l2_dim) # Layer 2 dim
    
    print("\nURCM > I am ready. (Type 'quit' to exit)")
    
    import msvcrt # Windows specific non-blocking input check
    
    last_action_time = time.time()
    DREAM_INTERVAL = 2.0 # Seconds before dreaming starts
    
    # Short-term buffer for plasticity (Recent Dreams)
    dream_buffer = [] 
    
    while True:
        # A. Check for Input (Non-blocking)
        user_input = None
        if msvcrt.kbhit():
            # If key pressed, read line (blocking but user is typing)
            print("\nUser > ", end='', flush=True)
            # This is a bit raw, but works for "interrupting"
            # Actually, standard input() is better if we accept that dreaming pauses while typing.
            # But the requirement is "Free" - it should think on its own.
            # Let's try to just read chars? No, too complex for simple CLI.
            # Let's use `input` with a timeout? Not standard in Python.
            
            # FALLBACK: We will just cycle: 
            # 1. Ask for input (with a short timeout? No)
            # 2. If no input, Dream.
            pass
            
        # Let's simplify: The user "pokes" the system. 
        # If the user doesn't poke, the system prints a dream thought every X seconds?
        # That messes up the input line.
        
        # PROPER CLI SOLUTION:
        # Just standard REPL. "Dreaming" happens as a background process that prints logs?
        # Let's simulate "Stream of Consciousness".
        
        try:
            # 1. Run Dynamics (Always Thinking)
            # Evolves the current thought slightly
            
            # --- INTRINSIC MOTIVATION ---
            # Instead of random drift, we apply a "Value Gradient"
            # This pulls the mind towards "Truth", "Coherence", "Benefit".
            
            alignment_grad = values.get_alignment_gradient(current_thought)
            
            # Check boredom (Confidence)
            decoded_word, conf, dist = decoder.decode(current_thought)
            
            # If very confident (stuck in a known concept), increase noise (Curiosity)
            # If confused (low confidence), reduce noise (Focus)
            if conf > 0.9:
                noise_level = 0.2 # Bored, look for something new
            elif conf < 0.5:
                noise_level = 0.01 # Confused, try to stabilize
            else:
                noise_level = 0.05 # Normal drift
            
            noise = np.random.normal(0, noise_level, current_thought.shape)
            
            # One step of resonance: 
            # s_next = tanh( W@s + alpha * ValueGrad + Noise )
            pre = np.dot(current_thought, hierarchy.layer2.W_res)
            
            # Apply Motivation (0.1 weight)
            pre += 0.1 * alignment_grad
            
            # Store previous thought for Hebbian update
            prev_thought = current_thought.copy()
            
            current_thought = np.tanh(pre + noise)
            
            # Check Energy/Stability
            decoded_word, conf, dist = decoder.decode(current_thought)
            
            # If highly stable/confident, it's a "Realization"
            if conf > 0.85 and time.time() - last_action_time > DREAM_INTERVAL:
                print(f"\n[Dreaming] ... {decoded_word} ...")
                last_action_time = time.time()
                
                # --- PLASTICITY: LEARN FROM DREAMS ---
                # If the dream is vivid (High Conf) and Good (Positive Valence),
                # We should IMPRINT this transition.
                # "What fires together wires together."
                
                dream_valence = values.evaluate_state(current_thought)
                
                if dream_valence > 0.1: # Only learn positive/constructive thoughts
                     # Add to buffer
                     dream_buffer.append(current_thought.copy())
                     
                     # If buffer has enough history, deposit the sequence
                     if len(dream_buffer) >= 2:
                         # Reinforce the transition: prev -> current
                         # This makes the dream more likely to happen again (Consolidation)
                         
                         # Check Safety Lock again just in case
                         if not hierarchy.layer2.safety._kernel_locked:
                             # Use Geometric Memory for Rank-1 Update
                             # Update W_res in place
                             hierarchy.layer2.W_res = memory_system.deposit_attractor(
                                 hierarchy.layer2.W_res,
                                 prev_thought,
                                 current_thought
                             )
                             # print(f" [Plasticity] Reinforced connection -> {decoded_word}")
                             
                             # Keep buffer small
                             if len(dream_buffer) > 5:
                                 dream_buffer.pop(0)
                
                # If it dreams of "stop", it might halt itself?
                if decoded_word == "stop":
                    print("\nURCM > I have decided to rest.")
                    current_thought *= 0.1 # Dampen
                    
            # 2. Check Input (Blocking for now, breaking true autonomy but usable)
            # We can't do true async CLI easily without curses/UI.
            # Let's just ask the user:
            # "Press Enter to speak, or wait..."
            
            # For this demo, let's use the 'msvcrt' loop properly.
            if msvcrt.kbhit():
                char = msvcrt.getwche() # Read char
                if char == '\r': # Enter
                    user_input = input("").strip().lower() # Read rest of line
                    print("User > " + user_input) # Echo
                    
                    if user_input in ["quit", "exit"]:
                        break
                        
                    # PROCESS INPUT (Disrupts current thought)
                    words = user_input.split()
                    for word in words:
                        phoneme_seq = converter.convert_text_to_phonemes(word)
                        if phoneme_seq and phoneme_seq.phonemes:
                            # Encode
                            l1_inputs = []
                            for p in phoneme_seq.phonemes:
                                seed = sum(ord(c) for c in p)
                                np.random.seed(seed) 
                                vec = np.random.normal(0, 1, (24,))
                                vec /= np.linalg.norm(vec)
                                l1_inputs.append(vec)
                            
                            l1_inputs = np.array(l1_inputs)
                            l2_input, _ = hierarchy.encode_concept(l1_inputs)
                            
                            # Inject Input into Thought Stream
                            # s_new = alpha * s_old + (1-alpha) * input
                            current_thought = 0.5 * current_thought + 0.5 * l2_input
                            
                    last_action_time = time.time() # Reset dream timer
                    
            time.sleep(0.1) # Tick rate
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            # print(e)
            pass

if __name__ == "__main__":
    main()