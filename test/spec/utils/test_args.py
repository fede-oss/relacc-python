from relacc.utils.args import Args


def test_args_default_when_missing():
    parser = Args({})
    assert parser.get("missing", 42) == 42


def test_args_cast_third_arg():
    parser = Args({"n": "1.9"})
    assert parser.get("n", None, lambda v: int(float(v))) == 1


def test_args_cast_second_arg_function():
    parser = Args({"n": "1.9"})
    assert parser.get("n", float) == 1.9


def test_args_keep_falsy_values():
    parser = Args({"zero": 0, "bool": False})
    assert parser.get("zero", 10) == 0
    assert parser.get("bool", True) is False
