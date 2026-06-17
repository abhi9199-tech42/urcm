import time
import random
import numpy as np
from urcm.core.executive import ExecutiveController
from urcm.core.web_sensor import WebSensor

def run_left_brain_loop(start_topic: str = "Money", goal_topic: str = "Economics", duration_minutes: int = 10):
    print(f"🚀 STARTING LEFT BRAIN LOOP (Directed Curiosity)")
    print(f"🎯 Goal: {goal_topic} (Practical) | Start: {start_topic}")
    print(f"⏱️ Duration: {duration_minutes} minutes")
    
    # 1. Initialize Systems
    sensor = WebSensor()
    exec_ctrl = ExecutiveController()
    
    # 2. Setup Initial State
    if exec_ctrl.engine.get_concept_vector(start_topic) is None:
        print(f"⚠️ Start topic '{start_topic}' unknown. Learning first...")
        sensor.search_and_learn(start_topic)
        exec_ctrl.reload_brain()
        
    exec_ctrl.set_initial_state(start_topic)
    
    # 3. Set Persistent Goal
    # We want to maintain this context throughout
    main_goal_desc = f"Master {goal_topic}"
    exec_ctrl.add_goal(main_goal_desc, target_concept=goal_topic, priority=1.0)
    
    # Visualization Data
    history_dist = []
    
    start_time = time.time()
    visited_topics = set()
    
Passed: 1.5 / 3 (synonym detection barely, homonym wrong answer but right mechanism, memory yes)
_repulsion_topic = None
repulsion_count = 0

    
loop_count = 0
    
    while (time.time() - start_time) < (duration_minutes * 60):
        loop_count += 1
        
        # Check if Main Goal was popped (completed), if so, re-add it to maintain focus
        # This acts as a "Theme" rather than a one-off task
        has_main_goal = False
        for intent in exec_ctrl.memory.intent_stack:
            if intent.description == main_goal_desc:
                has_main_goal = True
                break
        
        if not has_main_goal:
            print(f"[Left Brain] 🛡️ Asserting Persistent Goal: {main_goal_desc}")
            exec_ctrl.add_goal(main_goal_desc, target_concept=goal_topic, priority=0.2) # Very low priority to allow drift
            
        print(f"\n--- Cycle {loop_count} ---")
        print(f"[Right Brain] 💭 Current State: {exec_ctrl.engine.decode(exec_ctrl.current_state)}")
        
        # A. Executive Step (Thinking)
        # Run a few steps of internal reasoning to digest and steer
        print(f"[Right Brain] 🌊 Streaming Consciousness (Associating)...")
        trajectory = exec_ctrl.run_loop(max_steps=5)
        
        # Apply Grammar (Left Brain Structure)
        if trajectory:
            structured_thought = exec_ctrl.engine.grammar.structure_thought(trajectory)
            print(f"[Left Brain] 🏛️ Structured Thought: \"{structured_thought}\"")
        
        # Track Distance for Visualization
        current_vec = exec_ctrl.current_state
        goal_vec = exec_ctrl.engine.get_concept_vector(goal_topic)
        if goal_vec is not None:
            dist = np.linalg.norm(current_vec - goal_vec)
            history_dist.append(dist)
            
            # Draw Sparkline
            print(f"[Monitor] 📉 Distance to '{main_goal_desc}': {dist:.4f}")
            if len(history_dist) > 1:
                # Simple ASCII trend
                trend = "unchanged"
                if history_dist[-1] < history_dist[-2]: trend = "closer ✅"
                elif history_dist[-1] > history_dist[-2]: trend = "drifting ❌"
                print(f"[Monitor] Trend: {trend}")
        
        # Get the latest thought
        current_thought = trajectory[-1]
        print(f"[Right Brain] 🧠 Arrived at: {current_thought}")
        
        # B. Check Curiosity / Necessity
        # If we reached the goal, maybe switch goals?
        # For now, let's just keep exploring around the goal.
        
        # If thought is unknown or we want to expand:
        # We can simulate "Curiosity" by checking if the concept is "Weak" (low energy? few neighbors?)
        # Or just explore the current thought if we haven't visited it.
        
        topic_to_explore = current_thought.split(" [")[0].strip().lower() # Clean "truth [satya]"
        
        if topic_to_explore not in visited_topics:
            print(f"[Left Brain] 🔍 Curiosity Triggered: Exploring '{topic_to_explore}'...")
            sensor.search_and_learn(topic_to_explore)
            visited_topics.add(topic_to_explore)
            
            # C. Integration (Plasticity)
            exec_ctrl.reload_brain()
            
            # After reload, the 'current_state' is preserved, but we might want to 
            # jump to the NEW definition of the concept to ensure we are aligned.
            new_vec = exec_ctrl.engine.get_concept_vector(topic_to_explore)
            if new_vec is not None:
                # Blend old state with new reality (Accommodation)
                # alpha * old + (1-alpha) * new
                exec_ctrl.current_state = 0.5 * exec_ctrl.current_state + 0.5 * new_vec
                # Normalize
                exec_ctrl.current_state /= np.linalg.norm(exec_ctrl.current_state)
                print("[Left Brain] ✨ State Accommodation: Blended Memory with New Knowledge")
        else:
            print(f"[Left Brain] ⏩ Already visited '{topic_to_explore}'. Continuing thought stream...")
            
        # D. Boredom / Inhibition of Return
        # If we are stuck on a visited topic, push away!
        clean_thought = current_thought.split(" [")[0].strip().lower()
        if clean_thought in visited_topics:
            # Check for Repulsion Loop (getting bored of the same thing over and over)
            if clean_thought == last_repulsion_topic:
                repulsion_count += 1
            else:
                repulsion_count = 1
                last_repulsion_topic = clean_thought
            
            if repulsion_count > 3:
                print(f"[Left Brain] 🚨 Metacognitive Alarm! Stuck in loop around '{clean_thought}'. Initiating Hyper-Jump!")
                # Hyper-Jump: Randomize state completely
                exec_ctrl.current_state = np.random.randn(exec_ctrl.engine.l2_dim)
                exec_ctrl.current_state /= np.linalg.norm(exec_ctrl.current_state)
                repulsion_count = 0
                last_repulsion_topic = None
            else:
                print(f"[Left Brain] 🥱 Bored of '{current_thought}' ({repulsion_count}/3). Adding Repulsion...")
                exec_ctrl.add_goal(f"Avoid {clean_thought}", target_concept=current_thought, priority=-0.5) 
            
            # Run one step to force movement
            exec_ctrl.run_loop(max_steps=1)
            # Then pop it to avoid permanent avoidance
            if exec_ctrl.memory.intent_stack and exec_ctrl.memory.intent_stack[-1].description.startswith("Avoid"):
                 exec_ctrl.memory.intent_stack.pop()
        
        time.sleep(1)

    print("\n✅ Left Brain Session Complete.")
    print(f"📚 Topics Explored: {visited_topics}")

if __name__ == "__main__":
    import sys
    duration = 5
    goal = "Economics"
    start = "Money"
    
    # Simple Arg Parser
    args = sys.argv[1:]
    if len(args) > 0 and args[0].replace('.','',1).isdigit():
        duration = float(args[0])
        
    if "--goal" in args:
        try:
            idx = args.index("--goal")
            goal = args[idx+1]
        except (ValueError, IndexError):
            pass
            
    if "--start" in args:
        try:
            idx = args.index("--start")
            start = args[idx+1]
        except (ValueError, IndexError):
            pass
    
    run_left_brain_loop(start_topic=start, goal_topic=goal, duration_minutes=duration)
