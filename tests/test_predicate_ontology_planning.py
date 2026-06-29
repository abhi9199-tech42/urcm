from urcm.core.ingest import KnowledgeIngestion
from urcm.core.executive import ExecutiveController

def test_predicate_ontology_biases_edges():
    bp = "urcm_pred_ont.pkl"
    ing = KnowledgeIngestion(brain_path=bp)
    ing.ingest_text("all cat are animal. all rock are object.")
    # Type map
    ing.brain_data["type_map"] = {
        "cat": ["pet","animal"],
        "animal": ["animal"],
        "rock": ["object"],
        "object": ["object"]
    }
    # Predicate ontology: subclass domain pet, range animal
    ing.brain_data["predicate_ontology"] = {
        "subclass": {
            "domain": ["pet"],
            "range": ["animal"]
        }
    }
    ing.save()
    e = ExecutiveController(brain_path=bp)
    path = e.plan_a_star("cat", "animal")
    assert path == ["cat", "animal"]
