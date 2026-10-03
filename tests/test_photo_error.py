"""
Фото, на котором программа упала (2026-10-01, этап 2.1b, поведение — решение
владельца): прогон не останавливается, фото → question/error, в логе ERROR,
при следующем прогоне фото читается заново. Ctrl+C останавливает прогон, как
раньше.
"""
import re
from pathlib import Path

import pytest

import reader
from src.gmr.domain import OUTCOME_FOLDER, Outcome
from tests import _pipeline as pl
from tests._pipeline import fake_models  # noqa: F401 (фикстура)

pytestmark = pytest.mark.usefixtures("fake_models")   # модели — фейки
_run, _log, _reading = pl.run, pl.log_by_name, pl.reading


def _model_fails_on(monkeypatch, name, exc):
    orig = pl.SerialOCR.predict_with_details

    def maybe_fail(self, image_input):
        if pl.ORDER[int(image_input[0, 0, 0]) - 1] == name:
            raise exc
        return orig(self, image_input)

    monkeypatch.setattr(pl.SerialOCR, "predict_with_details", maybe_fail)
    return orig


def test_error_folder_is_in_question():
    assert OUTCOME_FOLDER[Outcome.ERROR] == "question/error"


def test_model_error_on_one_photo_does_not_stop_run(tmp_path, monkeypatch):
    f = pl.setup_month(tmp_path)
    _model_fails_on(monkeypatch, "p0.jpg", RuntimeError("CUDA out of memory"))
    _run(f)
    log = _log(f)
    assert log["p0.jpg"]["outcome"] == "ERROR"
    assert log["p0.jpg"]["notes"] == "ошибка программы: RuntimeError: CUDA out of memory"
    assert log["p0.jpg"]["photo_hash"]                       # фото узнаётся при следующем прогоне
    assert (pl.results_dir(f) / "question" / "error" / "p0.jpg").exists()
    # остальные фото обработаны как обычно
    assert log["p1.jpg"]["outcome"] == "MINUS" and _reading(f, "A-2") == "4000"
    assert log["p3.jpg"]["outcome"] == "PLUS"                # счёт A-1 не был занят p0
    report = (pl.results_dir(f) / "report.txt").read_text(encoding="utf-8")
    assert re.search(r"ERROR\s+1\b", report) and re.search(r"На проверку \(question/\):\s+3\b", report)


def test_error_photo_is_read_again_next_run(tmp_path, monkeypatch):
    f = pl.setup_month(tmp_path)
    orig = _model_fails_on(monkeypatch, "p1.jpg", ValueError("сбой"))
    _run(f)
    assert _log(f)["p1.jpg"]["outcome"] == "ERROR"
    monkeypatch.setattr(pl.SerialOCR, "predict_with_details", orig)   # причину устранили
    _run(f)
    rows = [r for r in pl.log(f) if r["original_filename"] == "p1.jpg"]
    assert [r["outcome"] for r in rows] == ["ERROR", "MINUS"]
    assert _reading(f, "A-2") == "4000"


def test_unreadable_file_goes_to_error(tmp_path, monkeypatch):
    f = pl.setup_month(tmp_path)
    real_fp = reader.photo_fingerprint

    def locked(path):
        if Path(path).name == "p2.jpg":
            raise PermissionError(13, "файл занят", path)
        return real_fp(path)

    monkeypatch.setattr(reader, "photo_fingerprint", locked)
    real_save = reader._save_annotated

    def save(src, *a, **k):
        if Path(src).name == "p2.jpg":
            raise PermissionError(13, "файл занят", src)
        return real_save(src, *a, **k)

    monkeypatch.setattr(reader, "_save_annotated", save)
    _run(f)
    log = _log(f)
    assert log["p2.jpg"]["outcome"] == "ERROR" and log["p2.jpg"]["photo_hash"] == ""
    assert log["p4.jpg"]["outcome"] == "NO_METER"            # прогон дошёл до конца


def test_ctrl_c_still_stops_run(tmp_path, monkeypatch):
    f = pl.setup_month(tmp_path)
    _model_fails_on(monkeypatch, "p1.jpg", KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _run(f)
    assert "p2.jpg" not in _log(f)


def test_save_error_on_normal_photo_still_stops_run(tmp_path, monkeypatch):
    # диск переполнен и т.п. на обычном фото — это не «ошибка на фото», прогон стоит
    f = pl.setup_month(tmp_path)

    def disk_full(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(reader, "_save_annotated", disk_full)
    with pytest.raises(OSError):
        _run(f)
