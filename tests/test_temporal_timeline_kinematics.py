from urcm.core.logic_gates import SpatialLogic, TemporalLogic


def test_timeline_ordering_and_consistency():
    a = (1, 3)
    b = (3, 5)
    c = (2, 4)
    assert TemporalLogic.before(a,b) is True
    assert TemporalLogic.overlap(a,c) is True
    assert TemporalLogic.after(b,a) is True

def test_kinematics_suvat():
    s = SpatialLogic.suvat_displacement(0.0, 2.0, 3.0)  # 0 + 0.5*2*9 = 9
    v = SpatialLogic.suvat_velocity(1.0, 2.0, 3.0)      # 1 + 6 = 7
    assert abs(s - 9.0) < 1e-6
    assert abs(v - 7.0) < 1e-6
