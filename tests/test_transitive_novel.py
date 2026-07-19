import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from urcm.core.executive import ExecutiveController
from urcm.core.ingest import KnowledgeIngestion

BRAIN = "urcm_identity_smoothed.pkl"

@pytest.fixture
def exec_ctrl():
    return ExecutiveController(BRAIN)

def ingest_relations(statements):
    ing = KnowledgeIngestion(brain_path=BRAIN, l2_dim=512)
    for s in statements:
        ing.ingest_text(s)
    ing.save()

def run_chain(exec_ctrl, start, steps=6):
    exec_ctrl.reload_brain()
    exec_ctrl.memory.clear_context()
    exec_ctrl.set_initial_state(start)
    traj = exec_ctrl.run_loop(max_steps=steps)
    return traj

def index_of(traj, token):
    for i, t in enumerate(traj):
        if token.lower() in t.lower():
            return i
    return -1

def test_transitivity_glorps_zinks_quibbles(exec_ctrl):
    ingest_relations([
        "All glorps are zinks.",
        "All zinks are quibbles."
    ])
    traj = run_chain(exec_ctrl, "glorps", steps=6)
    i_z = index_of(traj, "zinks")
    i_q = index_of(traj, "quibbles")
    assert i_z != -1 and i_q != -1 and i_z < i_q

def test_transitivity_flargs_mibbles_glorps(exec_ctrl):
    ingest_relations([
        "All flargs are mibbles.",
        "All mibbles are glorps."
    ])
    traj = run_chain(exec_ctrl, "flargs", steps=6)
    i_m = index_of(traj, "mibbles")
    i_g = index_of(traj, "glorps")
    assert i_m != -1 and i_g != -1 and i_m < i_g

def test_transitivity_xargles_yondles_zorples(exec_ctrl):
    ingest_relations([
        "All xargles are yondles.",
        "All yondles are zorples."
    ])
    traj = run_chain(exec_ctrl, "xargles", steps=6)
    i_y = index_of(traj, "yondles")
    i_z = index_of(traj, "zorples")
    assert i_y != -1 and i_z != -1 and i_y < i_z
