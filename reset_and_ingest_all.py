import os
import time
from urcm.core.ingest import KnowledgeIngestion
from ingest_sanskrit_grammar import run_sanskrit_grammar_ingestion

def main():
    print("WARNING: FULL BRAIN RESET INITIATED")
    if os.path.exists("urcm_identity.pkl"):
        os.remove("urcm_identity.pkl")
        print("Deleted old brain.")
        
    ingestor = KnowledgeIngestion(l2_dim=512)
    
    # 1. Ingest Core Identity/Philosophy (White Paper)
    print("\nIngesting White Paper...")
    ingestor.ingest_file("docs/URCM_WHITE_PAPER.md")
    
    # 2. Ingest General Knowledge (Corpus)
    print("\nIngesting Corpus...")
    ingestor.ingest_file("corpus.txt")
    
    # 3. Ingest Strategic Scenarios (Boosted 3x)
    print("\nIngesting Scenarios (Boosted 3x)...")
    for i in range(3):
        print(f"  Pass {i+1}...")
        ingestor.ingest_file("tests/data/isre_scenarios/ww2_scenarios.json")
        
    ingestor.save()
    
    # 4. Ingest Sanskrit Grammar (Paninian Structure)
    print("\n--- Phase 2: Sanskrit Grammar Ingestion ---")
    run_sanskrit_grammar_ingestion()
    
    print("\n✅ Reset and Ingestion Complete.")

if __name__ == "__main__":
    main()
