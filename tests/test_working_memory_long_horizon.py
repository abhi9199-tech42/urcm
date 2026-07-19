from urcm.core.working_memory import Intent, WorkingMemory


def test_context_capacity_and_interference():
    wm = WorkingMemory()
    for i in range(100):
        wm.add_context("x")
    assert len(wm.context_buffer) <= wm.max_context
    # ensure duplicates are bounded
    assert wm.context_buffer.count("x") <= 8

def test_intent_capacity():
    wm = WorkingMemory()
    for i in range(40):
        wm.add_intent(Intent(description=f"task{i}"))
    assert len(wm.intent_stack) <= wm.max_intents
