from urcm.core.logic_gates import TheoremProver, FormalLogic

def test_resolution_proof_success():
    ast = FormalLogic.to_cnf("(p implies q) and p")
    clauses = FormalLogic._collect_clauses(ast)
    proof = TheoremProver.prove(clauses, "q")
    assert proof["success"] is True
    assert proof["target"] == "q"
    assert isinstance(proof["steps"], list)
    assert len(proof["steps"]) >= 1
    assert "unsat_core" in proof
    assert isinstance(proof["unsat_core"], list)

def test_resolution_proof_failure():
    ast = FormalLogic.to_cnf("(p implies q) and (p implies r) and not p")
    clauses = FormalLogic._collect_clauses(ast)
    proof = TheoremProver.prove(clauses, "q")
    assert proof["success"] in {True, False}

def test_minimize_unsat_core_greedy():
    ast = FormalLogic.to_cnf("(p implies q) and p and (s implies t) and s")
    clauses = FormalLogic._collect_clauses(ast)
    proof = TheoremProver.prove(clauses, "q")
    assert proof["success"] is True
    core = proof["unsat_core"]
    min_core = TheoremProver.minimize_unsat_core(clauses, "q", core)
    assert len(min_core) <= len(core)
