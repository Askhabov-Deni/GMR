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
from src.gmr.storage import load_log, load_table, log_path_for
from tests import test_shadow_run_pipeline as srp
from tests.test_shadow_run_pipeline import _setup_workspace, fake_models  # noqa: F401 (фикстура)

pytestmark = pytest.mark.usefixtures("fake_models")   # модели — фейки


def _run(root, inp, table):
    reader.run_pipeline(reader.PipelineConfig(
        input_dir=str(inp), output_base_dir=str(root / "out"), table_path=str(table)))


def _log(table):
    return {r["original_filename"]: r for r in load_log(log_path_for(str(table)))}


def _reading(table, account):
    df = load_table(str(table))
    return df.loc[df["Лицевой счет"] == account, "Текущие показания"].iloc[0]


def _model_fails_on(monkeypatch, name, exc):
    orig = srp._SerialOCR.predict_with_details

    def maybe_fail(self, image_input):
        if srp._ORDER[int(image_input[0, 0, 0]) - 1] == name:
            raise exc
        return orig(self, image_input)

    monkeypatch.setattr(srp._SerialOCR, "predict_with_details", maybe_fail)
    return orig


def test_error_folder_is_in_question():
    assert OUTCOME_FOLDER[Outcome.ERROR] == "question/error"


def test_model_error_on_one_photo_does_not_stop_run(tmp_path, monkeypatch):
    inp, table = _setup_workspace(tmp_path)
    _model_fails_on(monkeypatch, "p0.jpg", RuntimeError("CUDA out of memory"))
    _run(tmp_path, inp, table)
    log = _log(table)
    assert log["p0.jpg"]["outcome"] == "ERROR"
    assert log["p0.jpg"]["notes"] == "ошибка программы: RuntimeError: CUDA out of memory"
    assert log["p0.jpg"]["photo_hash"]                       # фото узнаётся при следующем прогоне
    assert (tmp_path / "out" / "question" / "error" / "p0.jpg").exists()
    # остальные фото обработаны как обычно
    assert log["p1.jpg"]["outcome"] == "MINUS" and _reading(table, "A-2") == "4000"
    assert log["p3.jpg"]["outcome"] == "PLUS"                # счёт A-1 не был занят p0
    report = (tmp_path / "out" / "report.txt").read_text(encoding="utf-8")
    assert re.search(r"ERROR\s+1\b", report) and re.search(r"На проверку \(question/\):\s+3\b", report)


def test_error_photo_is_read_again_next_run(tmp_path, monkeypatch):
    inp, table = _setup_workspace(tmp_path)
    orig = _model_fails_on(monkeypatch, "p1.jpg", ValueError("сбой"))
    _run(tmp_path, inp, table)
    assert _log(table)["p1.jpg"]["outcome"] == "ERROR"
    monkeypatch.setattr(srp._SerialOCR, "predict_with_details", orig)   # причину устранили
    _run(tmp_path, inp, table)
    rows = [r for r in load_log(log_path_for(str(table))) if r["original_filename"] == "p1.jpg"]
    assert [r["outcome"] for r in rows] == ["ERROR", "MINUS"]
    assert _reading(table, "A-2") == "4000"


def test_unreadable_file_goes_to_error(tmp_path, monkeypatch):
    inp, table = _setup_workspace(tmp_path)
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
    _run(tmp_path, inp, table)
    log = _log(table)
    assert log["p2.jpg"]["outcome"] == "ERROR" and log["p2.jpg"]["photo_hash"] == ""
    assert log["p4.jpg"]["outcome"] == "NO_METER"            # прогон дошёл до конца


def test_ctrl_c_still_stops_run(tmp_path, monkeypatch):
    inp, table = _setup_workspace(tmp_path)
    _model_fails_on(monkeypatch, "p1.jpg", KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, inp, table)
    assert "p2.jpg" not in _log(table)


def test_save_error_on_normal_photo_still_stops_run(tmp_path, monkeypatch):
    # диск переполнен и т.п. на обычном фото — это не «ошибка на фото», прогон стоит
    inp, table = _setup_workspace(tmp_path)

    def disk_full(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(reader, "_save_annotated", disk_full)
    with pytest.raises(OSError):
        _run(tmp_path, inp, table)
