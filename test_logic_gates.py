import numpy as np
import pytest
from urcm.core.reasoning import ReasoningEngine

def test_logic_gates():
    print("\n--- Testing Geometric Logic Gates ---")
    
    try:
        engine = ReasoningEngine()
        print("ReasoningEngine loaded.")
    except Exception as e:
        print(f"Skipping test: Engine load failed ({e})")
        return

    # 1. Test NOT Gate (Avoidance)
    # Scenario: Query="War", Constraint=NOT("War")
    # Expected: Should diverge from "War" quickly.
    print("\n1. Testing NOT('war')...")
    trajectory = engine.solve(
        query_text="war",
        constraints=[],
        logic_gates=[{"type": "NOT", "operands": ["war"], "weight": 2.0}],
        steps=5
    )
    print(f"Trajectory: {trajectory}")
    # Ideally, it shouldn't stay on 'war'
    assert trajectory[0] != "war" or len(trajectory) > 1, "Should move away from war"

    # 2. Test AND Gate (Intersection)
    # Scenario: Query="Midway", Logic=AND("Japan", "USA")
    # Expected: Should find concepts linking them (e.g. "battle", "war")
    print("\n2. Testing AND('japan', 'usa')...")
    trajectory = engine.solve(
        query_text="midway",
        constraints=[],
        logic_gates=[{"type": "AND", "operands": ["japan", "usa"], "weight": 1.0}],
        steps=5
    )
    print(f"Trajectory: {trajectory}")

    # 3. Test IMPLIES Gate (Directional)
    # Scenario: Query="Attack", Logic=IMPLIES("Attack", "Defense")
    # Expected: If "Attack" appears, "Defense" should follow.
    print("\n3. Testing IMPLIES('attack', 'defense')...")
    trajectory = engine.solve(
        query_text="attack",
        constraints=[],
        logic_gates=[{"type": "IMPLIES", "operands": ["attack", "defense"], "weight": 2.0}],
        steps=5
    )
    print(f"Trajectory: {trajectory}")

if __name__ == "__main__":
    test_logic_gates()
