from urcm.core.ingest import KnowledgeIngestion

def test_self_evolve_one_shot_updates_relations_and_types():
    bp = "urcm_self_evolve.pkl"
    ing = KnowledgeIngestion(brain_path=bp)
    obs = [
        ("instance","cat","animal"),
        ("instance","cat","animal"),
        ("instance","dog","animal"),
        ("no","rock","animal")
    ]
    n = ing.self_evolve_one_shot(obs, min_support=1, conf_threshold=0.6, max_updates=10)
    ing.save()
    assert n >= 1
    assert ("all","cat","animal") in ing.brain_data["relations"]
    tmap = ing.brain_data.get("type_map", {})
    assert "animal" in tmap.get("cat", [])
    tel = ing.brain_data.get("telemetry", {})
    assert tel.get("self_evolve_updates", 0) >= 1
