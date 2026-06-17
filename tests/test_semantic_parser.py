from urcm.core.logic_gates import SemanticParser, InductionEngine

def test_parse_all_are_edge():
    edge = SemanticParser.to_edge("All cats are animals.")
    assert edge == ("cats","animals")

def test_induction_transitive_closure():
    edges = [("a","b"), ("b","c")]
    closure = InductionEngine.transitive_closure(edges)
    assert ("a","c") in closure
