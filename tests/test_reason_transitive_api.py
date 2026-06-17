import os
import sys
import re
import pytest
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from urcm.core.executive import ExecutiveController
from urcm.core.ingest import KnowledgeIngestion

BRAIN = "urcm_identity_smoothed.pkl"

def reason(question: str) -> str:
    exec_ctrl = ExecutiveController(BRAIN)
    ing = KnowledgeIngestion(brain_path=BRAIN, l2_dim=512)
    m1 = re.search(r"All\s+(\w+)\s+are\s+(\w+)", question, re.IGNORECASE)
    m2 = re.search(r"All\s+(\w+)\s+are\s+(\w+)", question[m1.end():] if m1 else question, re.IGNORECASE)
    if not m1 or not m2:
        return "uncertain"
    a1, b1 = m1.group(1), m1.group(2)
    a2, b2 = m2.group(1), m2.group(2)
    ing.ingest_text(f"All {a1} are {b1}.")
    ing.ingest_text(f"All {a2} are {b2}.")
    ing.save()
    exec_ctrl.reload_brain()
    exec_ctrl.memory.clear_context()
    exec_ctrl.set_initial_state(a1)
    traj = exec_ctrl.run_loop(max_steps=6)
    def idx(tok):
        for i, t in enumerate(traj):
            if tok.lower() in t.lower():
                return i
        return -1
    i_b1 = idx(b1)
    i_b2 = idx(b2)
    if i_b1 != -1 and i_b2 != -1 and i_b1 < i_b2:
        return "yes"
    return "no"

@pytest.mark.parametrize("question,expected", [
    ("All glorps are zinks. All zinks are quibbles. Are glorps quibbles?", "yes"),
    ("All flargs are mibbles. All mibbles are glorps. Are flargs glorps?", "yes"),
    ("All xargles are yondles. All yondles are zorples. Are xargles zorples?", "yes"),
])
def test_reason_transitive(question, expected):
    result = reason(question)
    assert result.lower() == expected
