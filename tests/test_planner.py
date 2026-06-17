from urcm.core.working_memory import Planner

def test_planner_make_tea_success():
    planner = Planner(max_retries=1)
    steps = planner.decompose("Make tea plan")
    def act(step):
        return True
    ok = planner.execute(steps, act)
    assert ok is True
    assert any("Completed: boil water" in log for log in planner.logs)

def test_planner_retry_and_fail():
    planner = Planner(max_retries=2)
    steps = ["step1", "step2"]
    def act(step):
        return step == "step1"
    ok = planner.execute(steps, act)
    assert ok is False
    assert any("Failed: step2" in log for log in planner.logs)
