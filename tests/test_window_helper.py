"""tests/_window.py: окно в тестах создаётся заново, если Tcl не запустился."""
import pytest

from tests import _window

tk = pytest.importorskip("tkinter")


def _flaky(fail_times):
    calls = []

    def factory():
        calls.append(1)
        if len(calls) <= fail_times:
            raise tk.TclError("Can't find a usable init.tcl")
        return "окно"
    return factory, calls


def test_retries_until_window_created(monkeypatch):
    monkeypatch.setattr(_window.time, "sleep", lambda s: None)
    factory, calls = _flaky(2)
    assert _window.new_window(factory) == "окно" and len(calls) == 3


def test_gives_up_after_three_attempts(monkeypatch):
    monkeypatch.setattr(_window.time, "sleep", lambda s: None)
    factory, calls = _flaky(3)
    with pytest.raises(tk.TclError):
        _window.new_window(factory)
    assert len(calls) == 3


def test_other_errors_are_not_retried():
    calls = []

    def factory():
        calls.append(1)
        raise ValueError("ошибка в самом окне")
    with pytest.raises(ValueError):
        _window.new_window(factory)
    assert len(calls) == 1
