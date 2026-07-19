from urcm.core.executive import ExecutiveController


def test_explain_api_chain_and_contradiction():
    exec = ExecutiveController()
    exec.engine.brain_data = {"relations": [
        ("all","glorps","zinks"),
        ("all","zinks","quibbles"),
        ("no","glorps","quibbles")
    ]}
    res = exec.explain("glorps", "quibbles")
    assert res["discovered_chain"] == ["glorps","zinks","quibbles"]
    assert ("no","glorps","quibbles") in res["contradictions"]
