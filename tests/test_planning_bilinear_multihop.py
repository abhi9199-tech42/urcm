from urcm.core.executive import ExecutiveController

def test_planning_bilinear_multihop_infeasible():
    e = ExecutiveController()
    e.engine.brain_data = {
        "relations": [("all","a","b"), ("all","b","c")],
        "numeric_constraints": [
            ({"x":1.0}, "<=", 2.0),
            ({"y":1.0}, "<=", 3.0),
            ({"x":1.0}, ">=", 0.0),
            ({"y":1.0}, ">=", 1.0),
            ({"w":1.0}, ">=", 7.0),
        ],
        "bilinear_pairs": [("w","x","y")]
    }
    path = e.plan_a_star("a","c")
    assert path == []

def test_planning_bilinear_multihop_feasible_ingest():
    e = ExecutiveController()
    e.engine.brain_data = {
        "relations": [("all","a","b"), ("all","b","c")],
        "numeric_constraints": [
            ({"x":1.0}, "<=", 5.0),
            ({"y":1.0}, "<=", 5.0),
            ({"x":1.0}, ">=", 0.0),
            ({"y":1.0}, ">=", 0.0),
            ({"w":1.0}, ">=", 0.0),
        ],
        "bilinear_pairs": [("w","x","y")]
    }
    path = e.plan_a_star("a","c")
    assert isinstance(path, list)
    assert len(path) >= 2
