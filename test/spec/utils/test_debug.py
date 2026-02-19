from relacc.utils.debug import Debug


def test_debug_writes_when_verbose(capsys):
    dbg = Debug({"verbose": True})
    dbg.log("hello")
    captured = capsys.readouterr()
    assert captured.err == "hello\n"


def test_debug_silent_when_not_verbose(capsys):
    dbg = Debug({"verbose": False})
    dbg.log("hello")
    captured = capsys.readouterr()
    assert captured.err == ""


def test_debug_printf_format(capsys):
    dbg = Debug({"verbose": True})
    dbg.fmt("%d is a number", 42)
    captured = capsys.readouterr()
    assert captured.err == "42 is a number\n"


def test_debug_json_format_and_no_msg(capsys):
    dbg = Debug({"verbose": True})
    dbg.fmt("%j", {"a": 1})
    dbg.fmt("plain text")
    captured = capsys.readouterr()
    assert '{"a": 1}\n' in captured.err
    assert "plain text\n" in captured.err
