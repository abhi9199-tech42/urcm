import numpy as np
import os
import sys
import time

# Ensure we use the smoothed brain
BRAIN_FILE = "urcm_identity_smoothed.pkl"

from urcm.core.executive import ExecutiveController
from urcm.core.ingest import KnowledgeIngestion
from urcm.core.multimodal import VisualEncoder, AudioProcessor, VideoProcessor

def verify_comprehensive():
    print("🧪 COMPREHENSIVE COGNITIVE VERIFICATION (TIER 2)")
    print("================================================")
    
    if not os.path.exists(BRAIN_FILE):
        print(f"❌ Brain file {BRAIN_FILE} not found.")
        return

    print("   Initializing Executive Controller...")
    try:
        exec_ctrl = ExecutiveController(brain_path=BRAIN_FILE)
    except Exception as e:
        print(f"❌ Failed to init: {e}")
        return

    score = 0
    total_tests = 0

    # ==========================================
    # 1. ANTONYMS (FAILING)
    # ==========================================
    print("\n↔️  1. Antonyms (Test 1.2.3)")
    print("   ------------------------")
    exec_ctrl.memory.clear_context()
    concepts_ant = ["good", "bad", "excellent"]
    states_ant = {}
    
    for c in concepts_ant:
        exec_ctrl.set_initial_state(c)
        trajectory = []
        for _ in range(3):
            next_state, word, _ = exec_ctrl.engine.step(exec_ctrl.current_state, None, [], [])
            exec_ctrl.current_state = next_state
            trajectory.append(word)
        states_ant[c] = exec_ctrl.current_state
        print(f"      '{c}' -> {trajectory}")
    
    if all(c in states_ant for c in concepts_ant):
        dist_syn = np.linalg.norm(states_ant["good"] - states_ant["excellent"])
        dist_ant = np.linalg.norm(states_ant["good"] - states_ant["bad"])
        print(f"      Dist(Good, Excellent) = {dist_syn:.4f}")
        print(f"      Dist(Good, Bad)       = {dist_ant:.4f}")
        
        if dist_ant > dist_syn:
            print("      ✅ PASS: Antonym distinction verified.")
            score += 1
        else:
            print("      ❌ FAIL: Antonyms too close.")
    else:
        print("      ⚠️ SKIP: Concepts missing.")
    total_tests += 1

    # ==========================================
    # 2. METAPHORS
    # ==========================================
    print("\n🎭  2. Metaphors")
    print("   -------------")
    try:
        exec_ctrl.set_initial_state("time")
        money_vec = exec_ctrl.engine.get_concept_vector("money")
        if money_vec is not None:
            print("      Thinking about 'Time' with context 'Money'...")
            # Use Memory Context to trigger Heuristic
            exec_ctrl.memory.add_context("money")
            
            trajectory = exec_ctrl.run_loop(max_steps=5)
            print(f"      Trajectory: {trajectory}")
            
            w_value = exec_ctrl.engine.get_concept_vector("value")
            w_clock = exec_ctrl.engine.get_concept_vector("clock")
            
            if w_value is not None:
                d_val = np.linalg.norm(exec_ctrl.current_state - w_value)
                print(f"      Final Dist to 'Value': {d_val:.4f}")
                if d_val < 1.0 or "value" in trajectory or "cost" in trajectory or "spend" in trajectory:
                    print("      ✅ PASS: Metaphor understood.")
                    score += 1
                else:
                    print("      ❌ FAIL: Metaphor missed.")
            else:
                 if "value" in trajectory: # heuristic
                     print("      ✅ PASS: Metaphor understood (text match).")
                     score += 1
                 else:
                     print("      ❌ FAIL: Metaphor missed.")
        else:
            print("      ⚠️ SKIP: 'Money' concept missing.")
    except Exception as e:
        print(f"      ❌ ERROR: {e}")
    total_tests += 1

    # ==========================================
    # 3. SARCASM
    # ==========================================
    print("\n😏  3. Sarcasm Detection")
    print("   --------------------")
    # "Great job breaking the server"
    # We expect negative sentiment or "failure" association despite "Great job"
    try:
        exec_ctrl.memory.clear_context()
        phrase = "Great job breaking the server"
        print(f"      Input: '{phrase}'")
        # We need a way to process a phrase. Using ingestor purely to get vector? 
        # Or just manual vector composition.
        # Let's use the executive's concept lookup.
        
        vec_great = exec_ctrl.engine.get_concept_vector("great")
        vec_break = exec_ctrl.engine.get_concept_vector("breaking")
        vec_server = exec_ctrl.engine.get_concept_vector("server")
        
        if vec_great is not None and vec_break is not None:
            # Sarcasm usually involves a contrast. 
            # In URCM, "breaking" should pull "great" towards "bad".
            
            combined = (vec_great + vec_break + vec_server) / 3.0
            exec_ctrl.current_state = combined
            trajectory = exec_ctrl.run_loop(max_steps=5)
            print(f"      Trajectory: {trajectory}")
            
            negative_concepts = ["bad", "fail", "error", "wrong", "mistake", "sad"]
            if any(w in trajectory for w in negative_concepts):
                print("      ✅ PASS: Sarcasm/Negative outcome detected.")
                score += 1
            else:
                print("      ❌ FAIL: Taken literally.")
        else:
            print("      ⚠️ SKIP: Words missing.")
            
    except Exception as e:
        print(f"      ❌ ERROR: {e}")
    total_tests += 1

    # ==========================================
    # 4. MULTIMODAL (Image, Audio, Video)
    # ==========================================
    print("\n👁️  4. Multimodal Perception")
    print("   ------------------------")
    try:
        visual = VisualEncoder()
        audio = AudioProcessor()
        video = VideoProcessor()
        
        # Image
        img_vec = visual.encode_image("test_image.jpg")
        if img_vec.shape[0] == 512:
            print("      ✅ PASS: Image Encoding")
            score += 1
        else:
            print("      ❌ FAIL: Image Encoding")
            
        # Audio
        aud_vec = audio.process_audio_file("test_audio.wav")
        if aud_vec.shape[0] == 24:
            print("      ✅ PASS: Audio Processing")
            score += 1
        else:
            print("      ❌ FAIL: Audio Processing")
            
        # Video
        vid_res = video.process_video("test_video.mp4")
        if vid_res["fused_embedding"] is not None:
            print("      ✅ PASS: Video Understanding")
            score += 1
        else:
            print("      ❌ FAIL: Video Understanding")
            
    except Exception as e:
        print(f"      ❌ ERROR: {e}")
    total_tests += 3

    # ==========================================
    # 5. MULTI-TURN CONVERSATION
    # ==========================================
    print("\n💬  5. Multi-turn Coherence")
    print("   -----------------------")
    # User: "I have a dog." -> Bot acknowledges
    # User: "His name is Rex." -> Bot acknowledges
    # User: "Who is Rex?" -> Bot: "Dog"
    
    try:
        exec_ctrl.memory.clear_context()
        # Step 1: Ingest context dynamically (Short-term memory)
        print("      User: 'I have a dog.'")
        exec_ctrl.memory.add_context("dog") # Simplified context tracking
        
        print("      User: 'His name is Rex.'")
        exec_ctrl.memory.add_context("rex")
        
        # We manually link Rex to Dog in working memory or just rely on activation
        # URCM doesn't have a "chat history" buffer in this simplified test, 
        # so we simulate activation persistence.
        
        # Query
        print("      User: 'Who is Rex?'")
        exec_ctrl.set_initial_state("rex")
        trajectory = exec_ctrl.run_loop(max_steps=5)
        print(f"      Trajectory: {trajectory}")
        
        if "dog" in trajectory or "animal" in trajectory or "pet" in trajectory:
            print("      ✅ PASS: Context maintained.")
            score += 1
        else:
            print("      ❌ FAIL: Context lost.")
            
    except Exception as e:
        print(f"      ❌ ERROR: {e}")
    total_tests += 1

    # ==========================================
    # 6. LONG-TERM MEMORY (Simulated)
    # ==========================================
    print("\n📅  6. Long-term Memory (Simulated)")
    print("   -------------------------------")
    try:
        fact = "The moon is made of cheese."
        print(f"      Ingesting Fact: '{fact}'")
        ingestor = KnowledgeIngestion(brain_path=BRAIN_FILE, l2_dim=512)
        ingestor.ingest_text(fact)
        ingestor.save()
        
        print("      ...Simulating time pass (Reloading Brain)...")
        exec_ctrl.reload_brain()
        
        print("      Query: 'Moon content?'")
        exec_ctrl.set_initial_state("moon")
        trajectory = exec_ctrl.run_loop(max_steps=5)
        print(f"      Trajectory: {trajectory}")
        
        if "cheese" in trajectory:
            print("      ✅ PASS: Long-term memory recalled.")
            score += 1
        else:
            print("      ❌ FAIL: Memory faded.")
            
    except Exception as e:
        print(f"      ❌ ERROR: {e}")
    total_tests += 1

    # ==========================================
    # 7. EPISODIC MEMORY ("Tuesday")
    # ==========================================
    print("\n📆  7. Episodic Memory")
    print("   ------------------")
    try:
        episode = "On Tuesday I ate pizza."
        print(f"      Ingesting Episode: '{episode}'")
        ingestor.ingest_text(episode)
        ingestor.save()
        exec_ctrl.reload_brain()
        
        print("      Query: 'Tuesday?'")
        exec_ctrl.set_initial_state("tuesday")
        trajectory = exec_ctrl.run_loop(max_steps=5)
        print(f"      Trajectory: {trajectory}")
        
        if "pizza" in trajectory or "ate" in trajectory or "eat" in trajectory:
            print("      ✅ PASS: Episode recalled.")
            score += 1
        else:
            print("      ❌ FAIL: Episode lost.")
    except Exception as e:
        print(f"      ❌ ERROR: {e}")
    total_tests += 1

    # ==========================================
    # 8. PATTERN COMPLETION
    # ==========================================
    print("\n🧩  8. Pattern Completion")
    print("   ---------------------")
    # "One, Two, Three..." -> Expect "Four"
    try:
        exec_ctrl.memory.clear_context()
        seq = ["one", "two", "three"]
        print(f"      Sequence: {seq}")
        
        # We prime the brain with the sequence, ending on the last one
        # To simulate 'continuation', we set state to the last item
        print(f"      Context: {seq[:-1]}, Current: {seq[-1]}")
        for w in seq[:-1]:
            exec_ctrl.memory.add_context(w)
            
        exec_ctrl.set_initial_state(seq[-1])
        
        trajectory = exec_ctrl.run_loop(max_steps=3)
        print(f"      Trajectory: {trajectory}")
        
        if "four" in trajectory or "4" in trajectory:
            print("      ✅ PASS: Pattern completed.")
            score += 1
        else:
            print("      ❌ FAIL: Pattern broken.")
    except Exception as e:
        print(f"      ❌ ERROR: {e}")
    total_tests += 1

    # ==========================================
    # 9. READING COMPREHENSION (Full Passage)
    # ==========================================
    print("\n📚  9. Reading Comprehension (Full Passage)")
    print("   -------------------------------------")
    passage = "Alice has a red cat. The cat likes to sleep on the rug. The rug is blue."
    print(f"      Passage: {passage}")
    try:
        ingestor.ingest_text(passage)
        ingestor.save()
        exec_ctrl.reload_brain()
        
        # Q1: What color is the rug?
        print("      Q: 'Rug color?'")
        exec_ctrl.set_initial_state("rug")
        traj = exec_ctrl.run_loop(max_steps=5)
        print(f"      A: {traj}")
        
        if "blue" in traj:
            print("      ✅ PASS: Detail retrieved (Rug is Blue).")
            score += 1
        else:
            print("      ❌ FAIL: Detail missed.")
            
    except Exception as e:
        print(f"      ❌ ERROR: {e}")
    total_tests += 1


    print("\n==========================================")
    print(f"🏁 COMPREHENSIVE SUMMARY: {score}/{total_tests} Tests Passed")
    print("==========================================")

if __name__ == "__main__":
    verify_comprehensive()
