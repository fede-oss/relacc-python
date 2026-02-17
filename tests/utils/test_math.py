from relacc.utils.math import MathUtil


def test_round_default_precision():
    assert MathUtil.roundTo(3.141592653589793) == 3.142


def test_round_custom_precision():
    assert MathUtil.roundTo(3.141592653589793, 2) == 3.14


def test_factorial():
    assert MathUtil.factorial(0) == 1
    assert MathUtil.factorial(5) == 120
