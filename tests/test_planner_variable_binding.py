from urcm.core.executive import ExecutiveController


def test_plan_a_star_and_contingency():
    # Build a small brain with relations
    from urcm.core.ingest import KnowledgeIngestion
    bp = "urcm_planner.pkl"
    ing = KnowledgeIngestion(brain_path=bp, l2_dim=256)
    ing.ingest_text("all a are b. all b are d. some a are c. c implies d.")
    ing.save()
    execu = ExecutiveController(brain_path=bp)
    path = execu.plan_a_star("a","d")
    assert path in (["a","b","d"], ["a","c","d"])
    assert execu.contingency_tool_policy(["kettle","broken"]) == "microwave"
    assert execu.contingency_tool_policy(["drive","nail"]) == "hammer"

def test_variable_binding_unification():
    execu = ExecutiveController()
    bind = execu.bind_variables(("all","x","mortal"), ("all","socrates","mortal"))
    assert bind == {"x":"socrates"}
