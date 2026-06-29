from urcm.core.logic_gates import Polytope

def test_parse_at_least_at_most():
    text = "x at least 3 and y at most 5"
    cs = Polytope.parse_numeric_constraints(text)
    assert any((c[0].get("x")==1.0 and c[1]==">=" and c[2]==3.0) for c in cs)
    assert any((c[0].get("y")==1.0 and c[1]=="<=" and c[2]==5.0) for c in cs)

def test_parse_long_phrases():
    text = "z less than or equal to 2 and w greater than or equal to 1"
    cs = Polytope.parse_numeric_constraints(text)
    assert any((c[0].get("z")==1.0 and c[1]=="<=" and c[2]==2.0) for c in cs)
    assert any((c[0].get("w")==1.0 and c[1]==">=" and c[2]==1.0) for c in cs)
