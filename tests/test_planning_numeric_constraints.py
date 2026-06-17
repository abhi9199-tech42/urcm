from urcm.core.executive import ExecutiveController

def test_plan_refuses_on_infeasible_numeric_constraints():
    e = ExecutiveController()
    e.engine.brain_data = {
        "relations": [("all","s","t")],
        "numeric_constraints": [
            ({"x":1.0}, "<=", 1.0),
            ({"x":1.0}, ">=", 3.0)
        ]
    }
    path = e.plan_a_star("s","t")
    assert path == []
