from urcm.core.logic_gates import NumericLogic

def test_div_mod_and_inequalities():
    assert NumericLogic.div("10", "2") == 5.0
    assert NumericLogic.mod("10", "3") == 1.0
    assert NumericLogic.gt("10", "3") is True
    assert NumericLogic.lt("3", "10") is True
    assert NumericLogic.eq(NumericLogic.add("2","3"), 5.0) is True
