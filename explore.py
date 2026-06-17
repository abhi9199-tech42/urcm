"""
URCM Phase 8: Open-Ended Exploration (The Sandbox).
Connects the 'Web Sensor' to the 'Plasticity' loop.
"""

import time
import random
import pickle
import os
import sys
import numpy as np
from collections import deque

from urcm.core.web_sensor import WebSensor
from urcm.core.ingest import KnowledgeIngestion
from urcm.core.values import ValueSystem
from urcm.core.memory import GeometricMemory

def explore_sandbox(duration_minutes: int = 10, start_topic: str = "Cognitive_science"):
    print("==========================================")
    print(f"URCM EXPLORATION MODE (Sandbox: Wikipedia)")
    print(f"Goal: Learn about '{start_topic}' for {duration_minutes} minutes.")
    print("==========================================")
    
    # 1. Initialize Body (Sensor)
    sensor = WebSensor(sandbox_domains=["en.wikipedia.org"])
    
    # 2. Initialize Mind (Ingestion/Memory)
    # We use Ingestion class to handle text-to-concept mapping
    # But we need to persist it.
    brain_path = "urcm_identity.pkl"
    ingestor = KnowledgeIngestion(brain_path=brain_path, l2_dim=512)
    
    # Initialize Values (Curiosity Guide)
    values = ValueSystem(ingestor.concept_map)
    
    # 3. Exploration State
    frontier = deque() # Queue of URLs to visit
    start_url = sensor.search_start(start_topic)
    frontier.append(start_url)
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    pages_read = 0
    concepts_learned = 0
    
    print("\n🚀 Launching Exploration Loop...")
    
    while time.time() < end_time:
        if not frontier:
            print("Frontier empty. Restarting with random topic...")
            frontier.append("https://en.wikipedia.org/wiki/Special:Random")
            
        # A. Perceive (Fetch Page)
        url = frontier.popleft()
        perception = sensor.read_page(url)
        
        if not perception or not perception["text"]:
            continue
            
        print(f"🧠 Ingesting: {perception['title']} ({len(perception['text'])} chars)")
        
        # B. Digest (Ingest Text)
        # This updates the weights via 'deposit_sequence' inside ingest_text
        # We need to capture how many concepts were added.
        prev_vocab_size = len(ingestor.concept_map)
        
        # Ingest!
        # Note: ingest_text prints a lot, we might want to suppress it or let it scroll
        ingestor.ingest_text(perception["text"][:5000]) # Limit chunk size per page to stay fast
        
        new_vocab_size = len(ingestor.concept_map)
        learned = new_vocab_size - prev_vocab_size
        concepts_learned += learned
        pages_read += 1
        
        # C. Evaluate & Plan (Curiosity)
        # Which links should we follow?
        # A simple heuristic: Randomly pick 3 links to add to frontier (Breadth-First Exploration)
        # A smarter agent would pick links that match 'Interesting' keywords.
        
        candidates = perception["links"]
        if candidates:
            # Shuffle and pick a few
            random.shuffle(candidates)
            # Add up to 5 links to frontier
            for link in candidates[:5]:
                if link not in sensor.visited:
                    frontier.append(link)
                    
        print(f"   -> Learned {learned} new concepts.")
        print(f"   -> Frontier size: {len(frontier)}")
        print(f"   -> Time remaining: {int((end_time - time.time())/60)} min")
        
        # D. Save periodically (every 5 pages)
        if pages_read % 5 == 0:
            ingestor.save()
            
    # Final Save
    ingestor.save()
    print("\n==========================================")
    print("EXPLORATION COMPLETE.")
    print(f"Summary:")
    print(f"- Pages Read: {pages_read}")
    print(f"- New Concepts: {concepts_learned}")
    print(f"- Final Vocab Size: {len(ingestor.concept_map)}")
    print("==========================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="Artificial_intelligence", help="Starting topic")
    parser.add_argument("--minutes", type=int, default=10, help="Duration in minutes")
    args = parser.parse_args()
    
    explore_sandbox(duration_minutes=args.minutes, start_topic=args.topic)
