from urcm.core.ingest import KnowledgeIngestion
from urcm.core.executive import ExecutiveController

def test_executive_prove_structured():
    bp = "urcm_exec_struct.pkl"
    ing = KnowledgeIngestion(brain_path=bp)
    # Provide formal text for structured QE
    text = "forall x exists y: (Rel(x,y) and P(x)) implies S(x,y). Rel(a,sk_a). P(a)."
    ing.brain_data["formal_text"] = text
    # Domain constants from type_map for demo
    ing.brain_data["type_map"] = {"a": ["atom"], "sk_a": ["symbol"]}
    ing.save()
    e = ExecutiveController(brain_path=bp)
    ok = e.prove_structured(("s","a","sk_a"))
    assert ok is True
