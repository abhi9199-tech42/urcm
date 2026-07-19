from urcm.core.executive import ExecutiveController
from urcm.core.logic_gates import SpatialLogic
from urcm.core.tms import TruthMaintenanceSystem


def test_truth_maintenance_assert_and_retract():
    tms = TruthMaintenanceSystem()
    fact = ("all","socrates","mortal")
    tms.assert_fact(fact, support_key="rule1")
    assert tms.has(fact) is True
    tms.retract_support("rule1")
    assert tms.has(fact) is False

def test_program_synthesis_execution():
    # simple program: compute displacement and verify threshold
    ExecutiveController()
    def step_compute(ctx):
        ctx["s"] = SpatialLogic.suvat_displacement(2.0, 1.0, 3.0)  # 2*3 + 0.5*1*9 = 6 + 4.5 = 10.5
        return True
    def step_verify(ctx):
        return ctx["s"] > 10.0
    ctx = {}
    steps = [("compute", step_compute), ("verify", step_verify)]
    # Synthesis: run steps and ensure verified
    ok = all(fn(ctx) for _, fn in steps)
    assert ok is True
