
import requests
import os
import time
from urcm.core.ingest import KnowledgeIngestion

def train_conversation_skills():
    print("🚀 AXIOM CONVERSATIONAL TRAINING (Human Basic Capability)")
    print("=======================================================")
    
    # Target: DailyDialog Dataset (English)
    # Source: Local Clone (temp_dataset_repo)
    
    data_files = [
        "temp_dataset_repo/data/en_train_human.txt",
        "temp_dataset_repo/data/en_dev_human.txt",
        "temp_dataset_repo/data/en_test_human.txt"
    ]
    
    all_lines = []
    
    print("📂 Loading DailyDialog from local clone...")
    for fpath in data_files:
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    all_lines.extend(lines)
                print(f"   ✅ Loaded {len(lines)} lines from {fpath}")
            except Exception as e:
                print(f"   ❌ Error reading {fpath}: {e}")
        else:
             # Fallback to 1k_part_data if main files are empty/missing
             alt_path = "temp_dataset_repo/data/1k_part_data/dialogues_text_En.txt"
             if os.path.exists(alt_path) and not all_lines:
                 print(f"   ⚠️ Main files not found. Trying {alt_path}...")
                 with open(alt_path, "r", encoding="utf-8") as f:
                     all_lines.extend(f.readlines())

    if not all_lines:
        print("❌ Could not find dataset files. Ensure 'temp_dataset_repo' is cloned.")
        return

    # Initialize Ingestion Engine
    print("\n🧠 Initializing URCM Ingestion Engine...")
    ingestor = KnowledgeIngestion(l2_dim=512)
    
    # Parse Data
    # DailyDialog format: "Speaker A... __eou__ Speaker B... __eou__"
    # We want to treat each dialogue as a "Concept Trajectory"
    
    lines = all_lines
    total_dialogues = len(lines)
    print(f"📄 Found {total_dialogues} dialogues.")
    
    print("\n▶️  STARTING INGESTION (Press Ctrl+C to stop safely)")
    
    count = 0
    start_time = time.time()
    
    try:
        for i, line in enumerate(lines):
            # Clean text: Remove __eou__ tokens and excessive whitespace
            clean_text = line.replace("__eou__", " ").replace("  ", " ").strip()
            
            if not clean_text:
                continue
                
            # Ingest
            # We treat the whole dialogue as a context block
            ingestor.ingest_text(clean_text)
            
            count += 1
            if count % 10 == 0:
                elapsed = time.time() - start_time
                rate = count / elapsed
                print(f"   [Progress] {i}/{total_dialogues} dialogues | {rate:.1f} dials/sec | Last: {clean_text[:50]}...")
                
            # Periodically save to disk to prevent data loss
            if count % 100 == 0:
                ingestor.save()
                print("   💾 Brain saved.")
                
            # Limit for demo purposes (User can run full training later)
            # Remove this break to run full dataset
            if count >= 200: 
                print("\n⚠️  Pausing after 200 dialogues for verification.")
                print("   (Run again without limit to train full dataset)")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Training Interrupted by User.")
        
    # Final Save
    ingestor.save()
    print(f"\n✅ Training Session Complete. Ingested {count} dialogues.")
    print("   The system now possesses basic conversational patterns.")

if __name__ == "__main__":
    train_conversation_skills()
