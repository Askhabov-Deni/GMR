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
import sys

import numpy as np
import pandas as pd
import pytest

from src.gmr.storage import LOG_COLUMNS

program2 = pytest.importorskip("program2")
from program2 import (  # noqa: E402
    CORRECTED_NOTE, apply_verify_correction, find_verify_row,
    free_photo_path, list_verify_photos, plan_verify_correction,
)
from src.gmr.domain import PipelineConfig  # noqa: E402
from src.gmr.storage import append_log_row, load_log, load_table, save_log, save_table  # noqa: E402

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
    # экран: на Windows есть всегда, в Linux — переменная DISPLAY (xvfb-run)
    return sys.platform == "win32" or bool(os.environ.get("DISPLAY"))


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
    save_log(str(tmp_path / "meters_table_log.csv"), rows)

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


# ─── «Серийник в базе с ошибкой» ─────────────────────────────────────────────

def test_table_serial_for_account():
    df = table([("0045618", "A1", "10", ""), ("1234567", "A2", "5", "")])
    assert program2.table_serial_for_account(df, CFG, "A1") == "0045618"
    assert program2.table_serial_for_account(df, CFG, " A2 ") == "1234567"
    assert program2.table_serial_for_account(df, CFG, "A9") is None
    assert program2.table_serial_for_account(None, CFG, "A1") is None


def test_append_db_serial_fix(tmp_path):
    lst = tmp_path / "db_serial_fix" / "db_serial_fix.csv"
    program2.append_db_serial_fix(str(lst), {"Фото": "A1.jpg", "Контролёр": "Сулиман С",
                                             "Лицевой счёт": "A1", "Серийник в базе": "1284567",
                                             "Серийник на фото": "1234567"})
    program2.append_db_serial_fix(str(lst), {"Фото": "A2.jpg", "Контролёр": "Аюб"})
    raw = lst.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and raw.count(b"\xef\xbb\xbf") == 1   # BOM один раз — для Excel
    lines = raw.decode("utf-8-sig").splitlines()
    assert lines[0] == ";".join(program2.DB_SERIAL_FIX_COLUMNS)
    assert lines[1].startswith("A1.jpg;Сулиман С;A1;1284567;1234567")
    assert len(lines) == 3


@pytest.fixture
def edit_app(tmp_path, monkeypatch):
    """MainWindow + одно фото в question/serial_not_found (без моделей)."""
    import cv2
    photos = tmp_path / "Сулиман С"
    q = photos / "question" / "serial_not_found"
    q.mkdir(parents=True)
    cv2.imwrite(str(q / "w.jpg"), np.full((50, 50, 3), 100, np.uint8))
    tbl = tmp_path / "meters_table.csv"
    table([("1284567", "A1", "1000", ""), ("7654321", "A2", "5", "")]).to_csv(tbl, index=False)
    settings = program2.AppSettings(operator_name="Оператор", photos_dir=str(photos),
                                    table_path=str(tbl), training_dir="")
    monkeypatch.setattr(program2, "load_settings", lambda: settings)
    monkeypatch.setattr(program2, "save_settings", lambda s: None)
    monkeypatch.setattr(program2.MainWindow, "_load_models_async", lambda self: None)
    monkeypatch.setattr(program2, "LoginDialog", lambda *a, **k: type("D", (), {"action": "continue"})())
    for name in ("askyesno", "showinfo", "showwarning", "showerror"):
        monkeypatch.setattr(program2.messagebox, name, lambda *a, **k: True)
    w = program2.MainWindow()
    w.withdraw()
    yield w, photos, q / "w.jpg"
    w.destroy()


@needs_display
def test_db_serial_fix_moves_photo_writes_list_not_table(edit_app, monkeypatch):
    app, photos, photo = edit_app
    app.open_edit_screen(str(photo), "serial_not_found")
    scr = app._current_screen
    scr._account_var.set("A1")
    scr._reading_widget.set_digits("01050", None)
    monkeypatch.setattr(scr, "_ask_photo_serial", lambda suggested: "1234567")
    scr._db_serial_fix()

    fixed = photos / "db_serial_fix" / "A1.jpg"
    assert fixed.exists() and not photo.exists()
    lst = (photos / "db_serial_fix" / "db_serial_fix.csv").read_text(encoding="utf-8-sig").splitlines()
    assert lst[1].split(";")[:6] == ["A1.jpg", "Сулиман С", "A1", "1284567", "1234567", "01050"]
    r = app.log_rows[-1]
    assert (r["outcome"], r["source"], r["serial_id"], r["account_id"], r["reading"]) == (
        "DB_SERIAL_FIX", "manual", "1234567", "A1", "1050")
    assert "в базе 1284567, на фото 1234567" in r["notes"]
    df = pd.read_csv(app.settings.table_path, dtype=str, keep_default_na=False)
    assert df.loc[df[CFG.col_account_id] == "A1", CFG.col_new_reading].item() == ""   # таблица не тронута


@needs_display
def test_db_serial_fix_refuses_same_serial_and_unknown_account(edit_app, monkeypatch):
    app, photos, photo = edit_app
    app.open_edit_screen(str(photo), "serial_not_found")
    scr = app._current_screen
    monkeypatch.setattr(scr, "_ask_photo_serial", lambda suggested: "1284567")
    scr._account_var.set("A9")                 # нет такого лицевого счёта
    scr._db_serial_fix()
    scr._account_var.set("A1")                 # номер как в базе — не опечатка
    scr._db_serial_fix()
    assert photo.exists() and not (photos / "db_serial_fix").exists()
    assert all(r["outcome"] != "DB_SERIAL_FIX" for r in app.log_rows)


# ─── Подсказка «похожие номера в базе» ───────────────────────────────────────

def test_serial_hints_sorted_by_delta():
    df = table([("1234567", "A1", "1000", ""), ("1234568", "A2", "5000", ""),
                ("7654321", "A3", "", ""), ("9999999", "A4", "0", "")])
    idx = program2.build_serial_index(df, CFG)
    hints = program2.serial_hints("1234569", "05100", idx)
    assert [(h.serial, h.account, h.delta) for h in hints] == [
        ("1234568", "A2", 100.0), ("1234567", "A1", 4100.0)]
    # показание не дочитано — расход неизвестен, но номера показываются
    assert {h.serial for h in program2.serial_hints("1234569", "0?100", idx)} == {"1234567", "1234568"}
    # у кандидата нет последнего показания
    (h,) = program2.serial_hints("7654320", "00100", idx)
    assert (h.account, h.last_reading, h.delta) == ("A3", None, None)
    assert program2.serial_hints("5555555", "00100", idx) == []
    assert program2.serial_hints("12", "00100", idx) == []            # слишком короткий
    assert program2.serial_hints("1234569", "00100", None) == []


def test_serial_hints_leading_zeros_like_reader():
    df = table([("0034567", "A1", "10", "")])
    idx = program2.build_serial_index(df, CFG)
    assert [h.serial for h in program2.serial_hints("34568", "", idx)] == ["0034567"]


@needs_display
def test_hint_shown_for_unknown_serial_and_fills_account(edit_app):
    app, photos, photo = edit_app
    app.open_edit_screen(str(photo), "serial_not_found")
    scr = app._current_screen
    scr._reading_widget.set_digits("01050", None)
    scr._lookup_by_serial("1284569")          # в базе 1284567 (A1)
    assert [h.account for h in scr._hints] == ["A1"]
    assert any(isinstance(w, program2.tk.Button) for w in scr._hint_frame.winfo_children())
    scr._use_hint(scr._hints[0])
    assert scr._account_var.get() == "A1" and scr._serial_var.get() == "1284567"
    # номер найден — подсказка исчезает
    scr._lookup_by_serial("1284567")
    assert scr._hint_frame.winfo_children() == []


@needs_display
def test_accept_after_hint_writes_note(edit_app):
    app, photos, photo = edit_app
    app.open_edit_screen(str(photo), "serial_not_found")
    scr = app._current_screen
    scr._reading_widget.set_digits("01050", None)
    scr._lookup_by_serial("1284569")
    scr._use_hint(scr._hints[0])
    scr._accept()
    r = app.log_rows[-1]
    assert r["outcome"] == "PLUS" and r["account_id"] == "A1" and r["serial_id"] == "1284567"
    assert "подсказка: 1284567" in r["notes"]


# ─── Известные ошибки (аудит 2026-10-01; как в tests/test_known_issues.py) ───
# Тест описывает, как ДОЛЖНО быть, и сейчас падает. Исправили — снимите пометку.

def _known_issue(what):
    return pytest.mark.xfail(strict=True, raises=AssertionError,
                             reason=f"известная ошибка, этап 2: {what}")


@needs_display
@_known_issue("«Верно» стирает строки лога, которые reader.py дописал, пока окно открыто")
def test_verify_ok_keeps_log_rows_written_meanwhile(app, tmp_path):
    log_p = str(tmp_path / "meters_table_log.csv")
    append_log_row(log_p, row(original_filename="new.jpg", final_filename="A2.jpg",
                              account_id="A2", reading="5100", outcome="PLUS"))   # reader.py
    items = list_verify_photos(app.settings.photos_dir, app.log_rows)
    app.open_verify_screen(items[0], 0)
    app._current_screen._verify_ok()
    assert "new.jpg" in [r["original_filename"] for r in load_log(log_p)]


@needs_display
@_known_issue("«Принять» стирает показания, которые reader.py записал, пока окно открыто")
def test_accept_keeps_table_values_written_meanwhile(edit_app):
    app, photos, photo = edit_app
    df = load_table(app.settings.table_path)                 # reader.py записал A2
    df.loc[df["Лицевой счет"] == "A2", "Текущие показания"] = "7"
    save_table(df, app.settings.table_path)
    app.open_edit_screen(str(photo), "serial_not_found")
    scr = app._current_screen
    scr._reading_widget.set_digits("01050", None)
    scr._lookup_by_serial("1284567")
    scr._accept()                                            # оператор принял A1
    df = load_table(app.settings.table_path)
    assert df.loc[df["Лицевой счет"] == "A2", "Текущие показания"].iloc[0] == "7"


@needs_display
@_known_issue("«Принять» при занятой таблице: показание остаётся только в логе")
def test_accept_with_locked_table_does_not_lose_reading(edit_app, monkeypatch):
    app, photos, photo = edit_app

    def locked(df, path):     # таблица открыта в Excel
        raise PermissionError(13, "файл занят другим процессом", path)

    monkeypatch.setattr(program2, "_save_table", locked)
    app.open_edit_screen(str(photo), "serial_not_found")
    scr = app._current_screen
    scr._reading_widget.set_digits("01050", None)
    scr._lookup_by_serial("1284567")
    scr._accept()
    manual = [r for r in app.log_rows if r.get("source") == "manual"]
    # показание не потеряно: либо оно в таблице, либо фото осталось в очереди
    # и в логе нет строки «разобрано»
    in_table = load_table(app.settings.table_path).loc[0, "Текущие показания"] == "1050"
    assert in_table or (photo.exists() and not manual)


# ─── Копия таблицы и лога при запуске окна (2026-10-01) ─────────────────────

@needs_display
def test_window_start_backs_up_table_and_log(app, tmp_path):
    copies = list((tmp_path / "gmr_backups" / "meters_table").iterdir())
    assert len(copies) == 1
    assert sorted(p.name for p in copies[0].iterdir()) == ["meters_table.csv", "meters_table_log.csv"]
