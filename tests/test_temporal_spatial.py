from urcm.core.logic_gates import TemporalLogic, SpatialLogic

def test_temporal_interval_relations():
    a = (1, 3)
    b = (2, 4)
    c = (5, 6)
    assert TemporalLogic.overlap(a,b) is True
    assert TemporalLogic.before(a,c) is True
    assert TemporalLogic.after(c,a) is True

def test_spatial_triangle_inequality():
    p1 = (0.0, 0.0)
    p2 = (1.0, 0.0)
    p3 = (1.0, 1.0)
    assert SpatialLogic.triangle_inequality(p1,p2,p3) is True
