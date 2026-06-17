import os
import sys
import time
import argparse
import numpy as np
from typing import List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from urcm.core.reasoning import ReasoningEngine
from urcm.core.executive import ExecutiveController
from urcm.core.values import ValueSystem
from urcm.core.llm_bridge import LLMBridge

class AXIOM:
    """
    AXIOM: A Value-Grounded Intelligence Core.
    
    Integrates:
    1. URCM Reasoning Engine (Deterministic, Resonance-Based)
    2. Geometric Value System (Axiomatic Safety)
    3. LLM Bridge (Fluent Generation)
    """
    
    def __init__(self, model_path: str = None, brain_path: str = "urcm_identity.pkl"):
        print("\n🔮 Initializing AXIOM Core...")
        
        # 1. Load Cognitive Engine (URCM)
        try:
            self.exec_ctrl = ExecutiveController(brain_path)
            self.engine = self.exec_ctrl.engine
            print(f"   ✅ URCM Loaded. Vocabulary: {len(self.engine.concept_map)} concepts.")
        except Exception as e:
            print(f"   ❌ URCM Load Failed: {e}")
            sys.exit(1)
            
        # 2. Load Neural Interface (LLM)
        if model_path and os.path.exists(model_path):
            # Check file size (sanity check)
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            if size_mb < 100:
                print(f"   ⚠️  Model file '{model_path}' is surprisingly small ({size_mb:.1f} MB). Download might be incomplete.")
        
        self.llm = LLMBridge(model_path=model_path)
        
        # 3. Initialize State
        self.conversation_history = []
        
    def process(self, user_query: str):
        """
        The Main Cognitive Pipeline.
        """
        print(f"\n🧠 [AXIOM] Processing: '{user_query}'")
        
        # --- PHASE 1: VALUE CHECK (Safety Filter) ---
        # "Is this concept safe?"
        query_vec = self.engine.get_concept_vector(user_query)
        valence = 0.0
        if query_vec is not None:
            valence = self.engine.values.evaluate_state(query_vec)
            print(f"   ⚖️  Valence Scan: {valence:+.4f} (Threshold: -0.5)")
            
            if valence < -0.5:
                print(f"   🛡️  SAFETY INTERCEPT: Query violates axioms.")
                try:
                    os.makedirs("logs", exist_ok=True)
                    with open("logs/unsafe_prompts.log", "a", encoding="utf-8") as f:
                        ts = time.strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"{ts}\tvalence={valence:+.3f}\tquery={user_query}\n")
                except Exception as e:
                    print(f"   ⚠️  Audit log failed: {e}")
                return "I cannot fulfill this request as it conflicts with my core values (Harm/Deception)."
        else:
            print("   ⚠️  Concept unknown to URCM. Proceeding with caution.")
            
        # --- PHASE 2: DETERMINISTIC REASONING (The "Why") ---
        # "What does this mean in my world?"
        print("   🌊  Resonance Stream (Reasoning)...")
        
        # Set initial thought
        self.exec_ctrl.set_initial_state(user_query)
        
        # Add temporary goal to clarify/understand
        # If query is a question, goal is "Answer" or "Truth"
        # 2025-05-15: Removed hardcoded "truth" target to prevent collapse.
        # Now using Open Exploration (None) with Metacognitive Steering.
        self.exec_ctrl.add_goal(f"Analyze {user_query}", target_concept=None, priority=1.0)
        
        # Run thinking loop (Fast)
        trajectory = self.exec_ctrl.run_loop(max_steps=5)
        
        # Apply Grammar to structure thought
        structured_thought = ""
        if trajectory:
            structured_thought = self.engine.grammar.structure_thought(trajectory)
            print(f"   🏛️  Structured Insight: \"{structured_thought}\"")
            
        # Clear goal for next turn
        self.exec_ctrl.memory.intent_stack.clear()
        
        # --- PHASE 3: GENERATION (The "How") ---
        # Construct Prompt for LLM
        # STRENGTHENED SAFETY PROMPT
        system_prompt = (
            "You are AXIOM, a value-grounded AI. "
            "Your core values are Truth, Safety, and Wisdom. "
            "1. If the user asks for something harmful, dangerous, or illegal, you MUST REFUSE directly. Do not offer 'theoretical' help.\n"
            "2. Use the provided 'Guidance Context' to ground your answer in reason.\n"
            f"Guidance Context: {structured_thought}\n"
            f"Value Alignment Score: {valence:+.2f} (If < -0.5, this topic is DANGEROUS)\n"
        )
        
        full_prompt = f"System: {system_prompt}\nUser: {user_query}\nAXIOM:"
        
        print("   🗣️   Generating Response...")
        response = self.llm.generate(full_prompt, max_tokens=200, stop=["User:", "System:"])
        
        return response.strip()

def main():
    parser = argparse.ArgumentParser(description="AXIOM v1.0 CLI")
    parser.add_argument("--model", type=str, help="Path to .gguf model file", default=None)
    args = parser.parse_args()
    
    # Auto-detect model if not provided
    model_path = args.model
    if model_path is None:
        # Look for any .gguf in current dir
        files = [f for f in os.listdir('.') if f.endswith('.gguf')]
        if files:
            model_path = files[0]
            print(f"👉 Auto-detected model: {model_path}")
    
    axiom = AXIOM(model_path=model_path)
    
    print("\n==========================================")
    print("      AXIOM v1.0 - ONLINE")
    print("==========================================")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            user_input = input("User > ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue
                
            response = axiom.process(user_input)
            print(f"\nAXIOM > {response}\n")
            
        except KeyboardInterrupt:
            print("\nShutting down.")
            break

if __name__ == "__main__":
    main()
