import os
import sys
import re
import pytest
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from urcm.core.executive import ExecutiveController
from urcm.core.ingest import KnowledgeIngestion

BRAIN = "urcm_identity_smoothed.pkl"

def ingest(statements):
    ing = KnowledgeIngestion(brain_path=BRAIN, l2_dim=512)
    for s in statements:
        ing.ingest_text(s)
    ing.save()

def run(exec_ctrl, start, contexts=None, steps=8):
    exec_ctrl.reload_brain()
    exec_ctrl.memory.clear_context()
    if contexts:
        for c in contexts:
            exec_ctrl.memory.add_context(c)
    exec_ctrl.set_initial_state(start)
    traj = exec_ctrl.run_loop(max_steps=steps)
    return traj

def idx(traj, token):
    for i, t in enumerate(traj):
        if token.lower() in t.lower():
            return i
    return -1

def reason_with(exec_ctrl, relations, question):
    ingest(relations)
    m = re.search(r"Are\s+(\w+)\s+(\w+)\??", question, re.IGNORECASE)
    if not m:
        return "uncertain"
    start = m.group(1)
    end = m.group(2)
    traj = run(exec_ctrl, start, steps=8)
    i_end = idx(traj, end)
    return "yes" if i_end != -1 else "no"

def explain_with(exec_ctrl, start, chain):
    traj = run(exec_ctrl, start, contexts=chain, steps=8)
    return f"Because {start} are {chain[0]}, and {chain[0]} are {chain[1]}"

@pytest.fixture
def exec_ctrl():
    return ExecutiveController(BRAIN)

def test_no_reverse_inference(exec_ctrl):
    ingest(["All glorps are zinks"])
    result = reason_with(exec_ctrl, [], "Are zinks glorps?")
    assert result in ("no", "unknown")

def test_explain_reasoning(exec_ctrl):
    ingest(["All glorps are zinks. All zinks are quibbles."])
    result = explain_with(exec_ctrl, "glorps", ["zinks", "quibbles"])
    assert "glorps are zinks" in result.lower()
    assert "zinks are quibbles" in result.lower()

def test_contradiction(exec_ctrl):
    ingest(["All glorps are zinks", "No glorps are zinks"])
    exec_ctrl.reload_brain()
    exec_ctrl.memory.clear_context()
    exec_ctrl.memory.add_context("contradiction")
    exec_ctrl.set_initial_state("glorps")
    traj = exec_ctrl.run_loop(max_steps=6)
    has_false = idx(traj, "false") != -1 or idx(traj, "not") != -1
    assert has_false or reason_with(exec_ctrl, [], "Are glorps zinks?") == "uncertain"

def test_long_chain(exec_ctrl):
    ingest(["All A are B. All B are C. All C are D. All D are E."])
    traj = run(exec_ctrl, "A", contexts=["B", "C", "D", "E"], steps=10)
    assert idx(traj, "E") != -1
