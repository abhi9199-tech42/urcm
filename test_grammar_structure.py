from urcm.core.sanskrit_grammar import SanskritGrammar

def test_grammar():
    grammar = SanskritGrammar()
    
    # Test 1: Simple Sentence
    # Rama (Agent) -> Vana (Object) -> Gam (Action)
    trajectory = ["rama", "vana", "gam"]
    print(f"Raw Trajectory: {trajectory}")
    
    structured = grammar.structure_thought(trajectory)
    print(f"Structured: {structured}")
    
    # Expected: 
    # rama -> ramah
    # vana -> vanam
    # gam -> gamti
    # ramah vanam -> ramo vanam (Visarga Sandhi)
    # Result: ramo vanam gamti
    
    assert "ramo" in structured.lower()
    assert "vanam" in structured.lower()
    
    print("✅ Grammar Test Passed: Structure & Sandhi applied.")

if __name__ == "__main__":
    test_grammar()
