from urcm.core.logic_gates import Polytope

def test_language_grounding_parser():
    text = "all cats are animals. no foxes are cats. x <= 3 and y >= 1"
    rels = Polytope.parse_relations(text)
    nums = Polytope.parse_numeric_constraints(text)
    assert ("all","cats","animals") in rels
    assert ("no","foxes","cats") in rels
    assert any((c[0].get("x")==1.0 and c[1]=="<=" and c[2]==3.0) for c in nums)
    assert any((c[0].get("y")==1.0 and c[1]==">=" and c[2]==1.0) for c in nums)
