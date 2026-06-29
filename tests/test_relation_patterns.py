import pytest
from urcm.core.ingest import KnowledgeIngestion

def test_quantifiers_and_implication_and_exception_and_coref():
    ing = KnowledgeIngestion(l2_dim=512)
    text = (
        "Every cats are mammals. "
        "Some cats are pets. "
        "Most birds are animals. "
        "If rain then wet. "
        "truth implies belief. "
        "All dogs are animals except wolves. "
        "They are friendly."
    )
    ing.ingest_text(text)
    rels = ing.relations
    assert ("all", "cats", "mammals") in rels
    assert ("some", "cats", "pets") in rels
    assert ("most", "birds", "animals") in rels
    assert ("implies", "rain", "wet") in rels
    assert ("implies", "truth", "belief") in rels
    assert ("except", "dogs", "animals", "wolves") in rels
    assert ("coref", "dogs", "friendly") in rels or ("coref", "cats", "friendly") in rels
