import pytest

from urcm.core.logic_gates import NumericLogic


def test_numeric_add():
    assert NumericLogic.add("3", "4") == 7.0
    assert NumericLogic.add(2, 5.5) == 7.5
    assert NumericLogic.add("3", "x") is None

def test_numeric_comparisons():
    assert NumericLogic.eq("5", "5.0") is True
    assert NumericLogic.gt("6", "2") is True
    assert NumericLogic.lt("1", "2") is True
    assert NumericLogic.gt("2", "6") is False
    assert NumericLogic.lt("3", "1") is False
