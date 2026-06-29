
import time

def calculate_roadmap():
    print("🚀 AXIOM TRAINING ROADMAP & PROJECTION")
    print("=======================================")
    
    # Base Assumptions (Verified from Benchmarks)
    # Processing Speed: ~10 seconds per cycle (Fetch + Read + Encode + Think)
    # Concepts per Cycle: ~15 new concepts (avg from 'run_internet_loop')
    cycle_time_sec = 10.0
    concepts_per_cycle = 15
    
    current_vocab = 9828 # From recent logs
    
    milestones = [
        {"level": "Level 1: Toddler (Current)", "vocab": 10000, "capability": "Simple Association, Basic Safety (60%)"},
        {"level": "Level 2: Student", "vocab": 50000, "capability": "General Knowledge, Robust Grammar, Context Holding"},
        {"level": "Level 3: Expert", "vocab": 200000, "capability": "Deep Domain Expertise, Complex Logic Chains"},
        {"level": "Level 4: Polymath (AXIOM)", "vocab": 1000000, "capability": "Cross-Domain Synthesis, Human-Level Reasoning"}
    ]
    
    print(f"⚡ Current Speed: {60/cycle_time_sec * concepts_per_cycle:.0f} concepts/minute")
    print(f"📊 Current State: {current_vocab} concepts\n")
    
    print(f"{'MILESTONE':<25} | {'VOCAB':<10} | {'TIME REQUIRED':<15} | {'CAPABILITY'}")
    print("-" * 100)
    
    for m in milestones:
        needed = m["vocab"] - current_vocab
        if needed <= 0:
            time_str = "ACHIEVED ✅"
        else:
            cycles_needed = needed / concepts_per_cycle
            time_hours = (cycles_needed * cycle_time_sec) / 3600
            
            if time_hours < 24:
                time_str = f"{time_hours:.1f} Hours"
            elif time_hours < 24 * 7:
                time_str = f"{time_hours/24:.1f} Days"
            else:
                time_str = f"{time_hours/(24*7):.1f} Weeks"
                
        print(f"{m['level']:<25} | {m['vocab']:<10,} | {time_str:<15} | {m['capability']}")
        
    print("\n💡 NOTE: Training speed scales linearly until ~500k concepts, then slows due to saturation.")
    print("   To reach 'Polymath', we recommend parallel ingestion (running multiple Left Brains).")

if __name__ == "__main__":
    calculate_roadmap()
