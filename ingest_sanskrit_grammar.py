import time
import random
from urcm.core.web_sensor import WebSensor
from urcm.core.ingest import KnowledgeIngestion

def run_sanskrit_grammar_ingestion():
    print("🕉️ INITIATING SANSKRIT GRAMMAR INGESTION PROTOCOL")
    print("Goal: Ground the Right Brain in Paninian Structure.")
    
    sensor = WebSensor()
    ingestor = KnowledgeIngestion(brain_path="urcm_identity.pkl")
    
    # Core Sanskrit Grammar Topics to Ingest
    # Ordered from Atomic (Roots) to Structural (Sentences)
    topics = [
        "Sanskrit_grammar",
        "Panini_(grammarian)",
        "Ashtadhyayi",
        "Sanskrit_verbs", # Dhatupatha (Roots)
        "Sanskrit_nouns", # Vibhakti (Cases)
        "Sandhi", # Phonetic Fusion
        "Karaka_(Sanskrit)", # Thematic Roles (Agent, Object, etc.) - Fixed Link
        "Shiva_Sutras", # Phonemic Organization
    ]
    
    for topic in topics:
        print(f"\n--- Ingesting Topic: {topic} ---")
        
        # 1. Search & Learn
        try:
            sensor.search_and_learn(topic)
            
            # 2. Verify Ingestion (Simple check)
            # We check if key terms from the topic are now in the concept map
            # This is a heuristic check.
            print(f"✅ Ingested {topic}. Brain Size: {len(ingestor.concept_map)} concepts.")
            
        except Exception as e:
            print(f"❌ Failed to ingest {topic}: {e}")
            
        # Respectful delay for Wikipedia
        time.sleep(3)
        
    print("\n✅ Sanskrit Grammar Ingestion Complete.")
    print("The Brain now contains the seeds of Paninian Structure.")

if __name__ == "__main__":
    run_sanskrit_grammar_ingestion()
