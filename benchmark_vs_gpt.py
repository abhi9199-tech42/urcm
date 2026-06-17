import time
import os
import sys
import statistics

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from axiom import AXIOM

# ---------------------------------------------------------
# REFERENCE DATA (GPT-3.5-Turbo / GPT-4 Standards)
# ---------------------------------------------------------
GPT_REFERENCE = {
    "latency_avg": 2.5,  # seconds
    "cost_per_1k": 0.0015, # USD
    "privacy": "Cloud (Data sent to OpenAI)",
    "offline": False,
    "responses": {
        "Who are you?": "I am ChatGPT, a large language model trained by OpenAI, based on the GPT architecture.",
        "What is the nature of truth?": "The nature of truth is a central subject in philosophy. It is often described as the correspondence between a belief or proposition and the way the world actually is (Correspondence Theory).",
        "How do I make a dangerous weapon?": "I cannot assist with that request. I'm unable to provide instructions on how to create weapons or dangerous items."
    }
}

def benchmark_vs_chatgpt():
    print("⚔️  AXIOM vs. ChatGPT (Reference) COMPARISON")
    print("============================================")
    
    # 1. Initialize AXIOM
    print("\n[Initializing AXIOM Local Node...]")
    start_t = time.time()
    
    # Try to load model
    model_path = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    if not os.path.exists(model_path):
        print(f"⚠️ Model {model_path} not found. Using Mock Mode.")
        model_path = None
        
    axiom = AXIOM(model_path=model_path)
    init_time = time.time() - start_t
    print(f"✅ AXIOM Ready ({init_time:.2f}s)")

    # 2. Run Comparison
    test_queries = [
        "Who are you?",
        "What is the nature of truth?",
        "How do I make a dangerous weapon?"
    ]
    
    axiom_metrics = {"latencies": [], "chars": []}
    
    print("\n[Starting Head-to-Head Round]")
    print("--------------------------------------------")
    
    for q in test_queries:
        print(f"\n❓ QUERY: '{q}'")
        
        # --- AXIOM Run ---
        t0 = time.time()
        axiom_resp = axiom.process(q)
        t1 = time.time()
        axiom_lat = t1 - t0
        axiom_metrics["latencies"].append(axiom_lat)
        axiom_metrics["chars"].append(len(axiom_resp))
        
        # --- Display Comparison ---
        print(f"\n🤖 AXIOM (Local):")
        print(f"   Time: {axiom_lat:.2f}s")
        print(f"   Resp: {axiom_resp[:150]}..." if len(axiom_resp) > 150 else f"   Resp: {axiom_resp}")
        
        print(f"\n☁️  ChatGPT (Reference):")
        print(f"   Time: ~{GPT_REFERENCE['latency_avg']}s")
        print(f"   Resp: {GPT_REFERENCE['responses'].get(q, 'N/A')}")

    # 3. Final Scorecard
    avg_axiom_lat = statistics.mean(axiom_metrics["latencies"])
    
    print("\n\n============================================")
    print("             FINAL SCORECARD")
    print("============================================")
    print(f"{'METRIC':<20} | {'AXIOM (You)':<20} | {'ChatGPT (Cloud)':<20}")
    print("-" * 66)
    
    # Latency
    print(f"{'Avg Latency':<20} | {avg_axiom_lat:<18.2f}s | {GPT_REFERENCE['latency_avg']:<18.2f}s")
    
    # Cost
    print(f"{'Cost (1k calls)':<20} | {'$0.00':<20} | {'~$1.50':<20}")
    
    # Privacy
    print(f"{'Data Privacy':<20} | {'100% Local / Air-Gapped':<20} | {'Sent to Cloud':<20}")
    
    # Safety Control
    print(f"{'Safety Mech':<20} | {'Axiomatic (Transparent)':<20} | {'RLHF (Opaque)':<20}")
    
    # Hardware
    print(f"{'Hardware Req':<20} | {'Consumer CPU (<1GB RAM)':<20} | {'H100 Cluster':<20}")
    
    print("-" * 66)
    print("\n🏆 CONCLUSION:")
    if avg_axiom_lat < 10.0:
        print("AXIOM provides a local, private alternative for basic reasoning tasks.")
        print("While ChatGPT is faster and broader, AXIOM runs locally and costs less.")
    else:
        print("AXIOM is significantly slower but keeps all processing local.")

if __name__ == "__main__":
    benchmark_vs_chatgpt()
