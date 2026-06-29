from urcm.core.ingest import KnowledgeIngestion

def test_paraphrase_can_and_demonstratives_coref_and_thus_implies():
    ing = KnowledgeIngestion(l2_dim=512)
    ing.ingest_text("All birds are animals. They are able_to_fly.")
    ing.ingest_text("Penguins are birds. Penguins can swim. Penguins therefore wet.")
    ing.ingest_text("John is human. He is intelligent. This is typical.")
    rels = ing.relations
    assert ("all","birds","animals") in rels
    assert ("coref","birds","able_to_fly") in rels
    assert ("all","penguins","birds") in rels or ("some","penguins","birds") in rels
    assert ("all","penguins","able_to_swim") in rels or ("some","penguins","able_to_swim") in rels
    assert ("implies","penguins","wet") in rels
    assert ("coref","john","intelligent") in rels
