from urcm.core.executive import ExecutiveController
from urcm.core.ingest import KnowledgeIngestion


def test_planning_execute_with_contingency():
    bp = "urcm_plan_exec.pkl"
    ing = KnowledgeIngestion(brain_path=bp, l2_dim=256)
    ing.ingest_text("all a are b. all b are d.")
    ing.save()
    execu = ExecutiveController(brain_path=bp)
    plan = execu.plan_a_star("a","d")
    assert plan == ["a","b","d"]
    tool = execu.contingency_tool_policy(["kettle","broken"])
    assert tool == "microwave"
