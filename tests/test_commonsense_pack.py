from urcm.core.ingest import KnowledgeIngestion
from urcm.core.logic_gates import GeometricLogic
import numpy as np

def test_counterfactual_and_typicality():
    ing = KnowledgeIngestion(l2_dim=512)
    ing.ingest_text("Rain implies wet. Birds are animals. Most birds are able_to_fly.")
    cmap = ing.concept_map
    logic = GeometricLogic(cmap)
    state = cmap["rain"].copy()
    grad = logic.apply_constraint(state, "IMPLIES", ["rain","wet"], weight=1.0)
    moved = state + grad
    dist_to_wet = np.linalg.norm(moved - cmap["wet"])
    assert dist_to_wet < np.linalg.norm(state - cmap["wet"])
    bird = cmap["birds"]
    animal = cmap["animals"]
    target = (bird + animal) / (np.linalg.norm(bird + animal) + 1e-9)
    grad2 = logic.apply_constraint(bird, "AND", ["birds","animals"], weight=1.0)
    moved2 = bird + grad2
    assert np.linalg.norm(moved2 - target) < np.linalg.norm(bird - target)
    
def test_not_gate_repels_and_or_gate_selects():
    ing = KnowledgeIngestion(l2_dim=512)
    ing.ingest_text("Cats are mammals. Dogs are mammals.")
    cmap = ing.concept_map
    logic = GeometricLogic(cmap)
    state = cmap["cats"] + (cmap["dogs"] - cmap["cats"]) * 0.1
    grad_not = logic.apply_constraint(state, "NOT", ["dogs"], weight=1.0)
    moved_not = state + grad_not
    assert np.linalg.norm(moved_not - cmap["dogs"]) > np.linalg.norm(state - cmap["dogs"])
    grad_or = logic.apply_constraint(state, "OR", ["cats","dogs"], weight=1.0)
    moved_or = state + grad_or
    closer_to_either = min(np.linalg.norm(moved_or - cmap["cats"]), np.linalg.norm(moved_or - cmap["dogs"]))
    original_to_either = min(np.linalg.norm(state - cmap["cats"]), np.linalg.norm(state - cmap["dogs"]))
    assert closer_to_either < original_to_either
