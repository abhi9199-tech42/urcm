import time
import random
from urcm.core.web_sensor import WebSensor
from urcm.core.reasoning import ReasoningEngine

def run_internet_loop(start_topic: str, duration_minutes: int = 10):
    print(f"🚀 STARTING INTERNET LEARNING LOOP: '{start_topic}'")
    print(f"⏱️ Duration: {duration_minutes} minutes")
    
    sensor = WebSensor()
    # reasoner = ReasoningEngine() # Moved inside loop to refresh memory
    
    current_topic = start_topic
    start_time = time.time()
    
    visited_topics = set()
    
    while (time.time() - start_time) < (duration_minutes * 60):
        print(f"\n--- Cycle Start: {current_topic} ---")
        
        # 1. Sense (Fetch & Ingest)
        sensor.search_and_learn(current_topic)
        visited_topics.add(current_topic)
        
        # Reload brain to integrate new knowledge
        print("🔄 Integrating new knowledge...")
        reasoner = ReasoningEngine()
        
        # 2. Think (Reasoning / Dreaming)
        print("🧠 Thinking about what I learned...")
        
        # Extract a seed word from the topic (e.g., "Quantum Physics" -> "quantum")
        seed_word = current_topic.split()[-1].lower()
        
        # If seed not in brain, try to find a known word in the topic
        if seed_word not in reasoner.concept_map:
            # Fallback: find any known word in the topic
            found = False
            for w in current_topic.split():
                if w.lower() in reasoner.concept_map:
                    seed_word = w.lower()
                    found = True
                    break
            if not found:
                print("⚠️ Topic words unknown. Picking a random known concept.")
                seed_word = random.choice(list(reasoner.concept_map.keys()))

        # Generate association trajectory
        trajectory = reasoner.solve(seed_word, [], steps=5)
        print(f"💭 Stream of Consciousness: {trajectory}")
        
        # 3. Act (Decide next topic)
        # Pick the last concept in the trajectory that is NOT the seed and NOT visited
        next_topic = None
        for concept in reversed(trajectory):
            # Clean concept (remove Sanskrit tag if present for search)
            # "war [yuddha]" -> "war"
            clean_concept = concept.split(" [")[0].strip().lower()
            
            if clean_concept != seed_word and clean_concept not in visited_topics:
                next_topic = clean_concept
                break
        
        if not next_topic:
            print("🔄 Loop stuck. Picking random concept for novelty.")
            next_topic = random.choice(list(reasoner.concept_map.keys()))
            
        print(f"👉 Curiosity Driver: Switching focus to '{next_topic}'")
        current_topic = next_topic
        
        # Wait a bit to simulate reading/processing and avoid spamming
        time.sleep(2)

    print("\n✅ Learning Session Complete.")
    print(f"📚 Topics Explored: {visited_topics}")

if __name__ == "__main__":
    import sys
    duration = 20
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])
    
    # For demo purposes, we run a short loop (2 minutes) unless changed
    run_internet_loop("Artificial Intelligence", duration_minutes=duration)
