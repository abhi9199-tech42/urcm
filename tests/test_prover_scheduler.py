from urcm.core.logic_gates import FormalLogic, ProverScheduler

def test_prover_scheduler_success():
    ast = FormalLogic.to_cnf("(p implies q) and p")
    clauses = FormalLogic._collect_clauses(ast)
    proof = ProverScheduler.schedule(clauses, "q", ["default"], max_seconds=0.5)
    assert proof.get("success") is True
    assert proof.get("strategy") == "default"
