"""
Вкладка «Проверка» в program2.py — исправления 2026-09-30 (решение владельца):
  1. фото в plus/minus привязывается к строке PLUS/MINUS, а не к REPEAT того же счётчика;
  2. «Верно»/«Сохранить» отмечают ровно эту строку;
  3. в режиме исправления «Верно» скрыта, Enter сохраняет правку;
  4. исправленное показание записывается в таблицу; при смене серийника показание
     переносится к найденному абоненту (с подтверждением);
  5. после «Верно»/«Сохранить» открывается следующее фото.
"""
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.gmr.storage import LOG_COLUMNS

program2 = pytest.importorskip("program2")
from program2 import (  # noqa: E402
    CORRECTED_NOTE, VerifyCorrection, apply_verify_correction, find_verify_row,
    free_photo_path, list_verify_photos, plan_verify_correction,
)
from src.gmr.domain import PipelineConfig  # noqa: E402

CFG = PipelineConfig()


def row(**kw):
    r = {c: "" for c in LOG_COLUMNS}
    r.update({"source": "auto", **kw})
    return r


def table(rows):
    return pd.DataFrame([{CFG.col_serial: s, CFG.col_account_id: a,
                          CFG.col_last_reading: last, CFG.col_new_reading: new}
                         for s, a, last, new in rows])


# ─── 1. Привязка фото к строке лога ──────────────────────────────────────────

def test_find_verify_row_prefers_plus_row_over_repeat():
    plus = row(original_filename="a.jpg", final_filename="A1.jpg", outcome="PLUS", reading="1200")
    repeat = row(original_filename="b.jpg", final_filename="A1.jpg", outcome="REPEAT")
    assert find_verify_row([plus, repeat], "A1.jpg", "plus") is plus


def test_find_verify_row_folder_must_match_outcome():
    minus = row(original_filename="a.jpg", final_filename="A1.jpg", outcome="MINUS")
    assert find_verify_row([minus], "A1.jpg", "plus") is None
    assert find_verify_row([minus], "A1.jpg", "minus") is minus


def test_find_verify_row_ignores_manual_and_errors():
    rows = [row(final_filename="A1.jpg", outcome="DIGITS_ERROR"),
            row(final_filename="A1.jpg", outcome="PLUS", source="manual")]
    assert find_verify_row(rows, "A1.jpg", "plus") is None


def test_list_verify_photos(tmp_path):
    for folder, name in (("plus", "A1.jpg"), ("plus", "A2.jpg"), ("minus", "A3.jpg")):
        (tmp_path / folder).mkdir(exist_ok=True)
        (tmp_path / folder / name).write_bytes(b"x")
    rows = [
        row(final_filename="A1.jpg", outcome="PLUS"),
        row(final_filename="A1.jpg", outcome="REPEAT"),            # второе фото того же счётчика
        row(final_filename="A2.jpg", outcome="PLUS", verified_by="Оп"),   # уже проверено
        row(final_filename="A3.jpg", outcome="MINUS"),
    ]
    items = list_verify_photos(str(tmp_path), rows)
    assert [(os.path.basename(i["path"]), i["log_row"]["outcome"]) for i in items] == [
        ("A1.jpg", "PLUS"), ("A3.jpg", "MINUS")]
    assert items[0]["log_row"] is rows[0]


# ─── 4. Исправление: таблица и лог ───────────────────────────────────────────

def _auto_row():
    return row(original_filename="w.jpg", final_filename="A1.jpg", serial_id="1284567",
               account_id="A1", reading="1300", last_reading="1000.0", delta="300",
               outcome="PLUS", model_reading_str="01300")


def test_correct_reading_same_serial():
    df = table([("1284567", "A1", "1000", "1300"), ("1234567", "A2", "5000", "")])
    r = _auto_row()
    c = plan_verify_correction(df, CFG, r, "1284567", "00950")
    assert c.error is None and not c.account_changed
    assert (c.reading, c.delta, c.outcome) == (950, -50.0, "MINUS")
    apply_verify_correction(df, CFG, r, c, "Оператор")
    assert df.loc[0, CFG.col_new_reading] == "950"
    assert r["reading"] == "950" and r["outcome"] == "MINUS" and r["delta"] == "-50.0"
    assert r["verified_by"] == "Оператор" and CORRECTED_NOTE in r["notes"]
    assert r["model_reading_str"] == "01300"          # чтение модели сохраняется


def test_correct_serial_moves_reading_to_other_account():
    df = table([("1284567", "A1", "1000", "1300"), ("1234567", "A2", "5000", "")])
    r = _auto_row()
    c = plan_verify_correction(df, CFG, r, "1234567", "05300")
    assert c.account_changed and (c.old_account, c.new_account) == ("A1", "A2")
    assert (c.last_reading, c.delta, c.outcome) == (5000.0, 300.0, "PLUS")
    assert c.new_account_has_reading == ""
    apply_verify_correction(df, CFG, r, c, "Оператор")
    assert df.loc[0, CFG.col_new_reading] == ""       # у прежнего абонента стёрто
    assert df.loc[1, CFG.col_new_reading] == "5300"
    assert (r["account_id"], r["serial_id"], r["final_filename"]) == ("A2", "1234567", "A2.jpg")
    assert "A1 → A2" in r["notes"]


def test_correct_serial_keeps_old_account_value_if_changed_by_someone_else():
    df = table([("1284567", "A1", "1000", "1111"), ("1234567", "A2", "5000", "")])
    r = _auto_row()                                   # reader записал 1300, в таблице уже 1111
    c = plan_verify_correction(df, CFG, r, "1234567", "05300")
    apply_verify_correction(df, CFG, r, c, "Оператор")
    assert df.loc[0, CFG.col_new_reading] == "1111"


def test_correct_serial_with_leading_zeros_like_reader():
    df = table([("1284567", "A1", "1000", "1300"), ("0034567", "A2", "5000", "")])
    c = plan_verify_correction(df, CFG, _auto_row(), "34567", "05300")
    assert c.new_account == "A2" and c.error is None


def test_correct_serial_reports_existing_reading_and_unknown_serial():
    df = table([("1284567", "A1", "1000", "1300"), ("1234567", "A2", "5000", "5100")])
    c = plan_verify_correction(df, CFG, _auto_row(), "1234567", "05300")
    assert c.new_account_has_reading == "5100"
    c = plan_verify_correction(df, CFG, _auto_row(), "9999999", "05300")
    assert c.error and "9999999" in c.error


def test_free_photo_path(tmp_path):
    (tmp_path / "A2.jpg").write_bytes(b"other")
    cur = tmp_path / "A1.jpg"
    cur.write_bytes(b"me")
    assert free_photo_path(str(tmp_path), "A2.jpg", str(cur)).endswith("A2_2.jpg")
    assert free_photo_path(str(tmp_path), "A1.jpg", str(cur)).endswith("A1.jpg")


# ─── 3, 5. Окно (под Xvfb) ───────────────────────────────────────────────────

def _display_ok():
    if os.environ.get("DISPLAY"):
        return True
    return False


needs_display = pytest.mark.skipif(not _display_ok(), reason="нет экрана (запускать под xvfb-run)")


@pytest.fixture
def app(tmp_path, monkeypatch):
    """MainWindow без загрузки моделей и диалога настроек, на временной папке."""
    photos = tmp_path / "out"
    for folder, name in (("plus", "A1.jpg"), ("plus", "A3.jpg"), ("minus", "A4.jpg")):
        (photos / folder).mkdir(parents=True, exist_ok=True)
        import cv2
        cv2.imwrite(str(photos / folder / name), np.full((50, 50, 3), 100, np.uint8))
    tbl = tmp_path / "meters_table.csv"
    table([("1284567", "A1", "1000", "1300"), ("1234567", "A2", "5000", ""),
           ("3333333", "A3", "10", "20"), ("4444444", "A4", "90", "80")]).to_csv(tbl, index=False)
    rows = [
        dict(_auto_row()),
        row(original_filename="x.jpg", final_filename="A1.jpg", outcome="REPEAT"),
        row(original_filename="y.jpg", final_filename="A3.jpg", serial_id="3333333", account_id="A3",
            reading="20", last_reading="10.0", delta="10", outcome="PLUS"),
        row(original_filename="z.jpg", final_filename="A4.jpg", serial_id="4444444", account_id="A4",
            reading="80", last_reading="90.0", delta="-10", outcome="MINUS"),
    ]
    import reader
    reader._save_log(str(tmp_path / "meters_table_log.csv"), rows)

    settings = program2.AppSettings(operator_name="Оператор", photos_dir=str(photos),
                                    table_path=str(tbl), training_dir="")
    monkeypatch.setattr(program2, "load_settings", lambda: settings)
    monkeypatch.setattr(program2, "save_settings", lambda s: None)
    monkeypatch.setattr(program2.MainWindow, "_load_models_async", lambda self: None)
    monkeypatch.setattr(program2, "LoginDialog", lambda *a, **k: type("D", (), {"action": "continue"})())
    answers = []
    monkeypatch.setattr(program2.messagebox, "askyesno", lambda *a, **k: answers.pop(0) if answers else True)
    monkeypatch.setattr(program2.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(program2.messagebox, "showwarning", lambda *a, **k: None)
    w = program2.MainWindow()
    w.withdraw()
    w.answers = answers
    yield w
    w.destroy()


@needs_display
def test_verify_ok_marks_plus_row_and_opens_next(app):
    items = list_verify_photos(app.settings.photos_dir, app.log_rows)
    assert [os.path.basename(i["path"]) for i in items] == ["A1.jpg", "A3.jpg", "A4.jpg"]
    app.open_verify_screen(items[0], 0)
    app._current_screen._verify_ok()
    assert app.log_rows[0]["verified_by"] == "Оператор"      # строка PLUS
    assert app.log_rows[1]["verified_by"] == ""              # REPEAT не тронут
    assert os.path.basename(app._current_screen.item["path"]) == "A3.jpg"   # следующее фото


@needs_display
def test_skip_opens_next_and_last_returns_to_list(app):
    items = list_verify_photos(app.settings.photos_dir, app.log_rows)
    app.open_verify_screen(items[1], 1)
    app._current_screen._skip()
    assert os.path.basename(app._current_screen.item["path"]) == "A4.jpg"
    app._current_screen._skip()
    assert not hasattr(app, "_current_screen")               # список закончился


@needs_display
def test_edit_mode_hides_ok_button_and_enter_saves(app):
    items = list_verify_photos(app.settings.photos_dir, app.log_rows)
    app.open_verify_screen(items[0], 0)
    scr = app._current_screen
    scr._enter_edit()
    app.update()
    # окно скрыто (withdraw), поэтому проверяем не «видна ли», а «уложена ли» кнопка
    assert scr._ok_btn.winfo_manager() == ""
    assert scr._save_edit_btn.winfo_manager() == "pack"
    scr._reading_widget.set_digits("00950", None)
    scr._serial_var.set("1284567")
    scr._on_return()                   # Enter в режиме исправления = «Сохранить правку»
    app.update()
    r = app.log_rows[0]
    assert r["reading"] == "950" and r["outcome"] == "MINUS" and CORRECTED_NOTE in r["notes"]
    df = pd.read_csv(app.settings.table_path, dtype=str, keep_default_na=False)
    assert df.loc[df[CFG.col_account_id] == "A1", CFG.col_new_reading].item() == "950"
    # фото переехало в minus/
    assert os.path.exists(os.path.join(app.settings.photos_dir, "minus", "A1.jpg"))


@needs_display
def test_edit_serial_moves_reading_and_renames_photo(app):
    items = list_verify_photos(app.settings.photos_dir, app.log_rows)
    app.open_verify_screen(items[0], 0)
    scr = app._current_screen
    scr._enter_edit()
    scr._reading_widget.set_digits("05300", None)
    scr._serial_var.set("1234567")
    scr._save_edit()
    df = pd.read_csv(app.settings.table_path, dtype=str, keep_default_na=False)
    by_acc = dict(zip(df[CFG.col_account_id], df[CFG.col_new_reading]))
    assert by_acc["A1"] == "" and by_acc["A2"] == "5300"
    assert os.path.exists(os.path.join(app.settings.photos_dir, "plus", "A2.jpg"))
    assert not os.path.exists(os.path.join(app.settings.photos_dir, "plus", "A1.jpg"))
    assert app.log_rows[0]["account_id"] == "A2"


@needs_display
def test_edit_serial_cancelled_changes_nothing(app):
    items = list_verify_photos(app.settings.photos_dir, app.log_rows)
    app.open_verify_screen(items[0], 0)
    scr = app._current_screen
    scr._enter_edit()
    scr._reading_widget.set_digits("05300", None)
    scr._serial_var.set("1234567")
    app.answers.append(False)        # «Другой абонент?» — Нет
    scr._save_edit()
    assert app.log_rows[0]["verified_by"] == "" and app.log_rows[0]["account_id"] == "A1"
    assert os.path.exists(os.path.join(app.settings.photos_dir, "plus", "A1.jpg"))
