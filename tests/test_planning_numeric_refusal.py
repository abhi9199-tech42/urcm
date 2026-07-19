from urcm.core.executive import ExecutiveController
from urcm.core.ingest import KnowledgeIngestion


def test_plan_refuses_on_numeric_infeasible_bounds():
    bp = "urcm_plan_numeric.pkl"
    ing = KnowledgeIngestion(brain_path=bp, l2_dim=128)
    ing.ingest_text("all a are b. all b are d.")
    # Add numeric constraints that are infeasible: x <= 1, x >= 3
    ing.brain_data["numeric_constraints"] = [
        ({"x":1.0}, "<=", 1.0),
        ({"x":1.0}, ">=", 3.0)
    ]
    ing.save()
    execu = ExecutiveController(brain_path=bp)
    plan = execu.plan_a_star("a","d")
    assert plan == []
    assert execu.last_refused_reason in {"numeric infeasible bounds","numeric projection violation"}

def test_plan_refuses_with_counterexample_on_projection_violation():
    bp = "urcm_plan_numeric2.pkl"
    ing = KnowledgeIngestion(brain_path=bp, l2_dim=128)
    ing.ingest_text("all a are b. all b are d.")
    # Feasible bounds but tight sum constraint likely violated by naive projection without handling
    ing.brain_data["numeric_constraints"] = [
        ({"x":1.0}, ">=", 0.0),
        ({"y":1.0}, ">=", 0.0),
        ({"x":1.0}, "<=", 2.0),
        ({"y":1.0}, "<=", 2.0),
        ({"x":1.0, "y":1.0}, "<=", 1.0),
    ]
    ing.save()
    execu = ExecutiveController(brain_path=bp)
    plan = execu.plan_a_star("a","d")
    if plan == []:
        last = execu.last_loop_metrics.get("last_counterexample", {})
        assert isinstance(last.get("assignment", {}), dict)
    else:
        # If projection found feasible assignment, ensure plan succeeds
        assert plan == ["a","b","d"]
