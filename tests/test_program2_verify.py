"""
Вкладка «Проверка» в program2.py — исправления 2026-09-30 (решение владельца):
  1. фото в plus/minus привязывается к строке PLUS/MINUS, а не к REPEAT того же счётчика;
  2. «Верно»/«Сохранить» отмечают ровно эту строку;
  3. в режиме исправления «Верно» скрыта, Enter сохраняет правку;
  4. исправленное показание записывается в базу; при смене серийника показание
     переносится к найденному абоненту (с подтверждением);
  5. после «Верно»/«Сохранить» открывается следующее фото.
С этапа 2.3 окно работает с папкой месяца: все контролёры сразу, показания и
лог — в базе месяца (src/gmr/application/operator.py, tests/test_operator.py).
Здесь же «Серийник в базе с ошибкой» и подсказки по номеру.
"""
import os

import pandas as pd
import pytest

from src.gmr.storage import LOG_COLUMNS

program2 = pytest.importorskip("program2")
from program2 import (  # noqa: E402
    find_verify_row, free_photo_path, list_verify_photos, plan_verify_correction,
)
from src.gmr.application.operator import CORRECTED_NOTE  # noqa: E402
from src.gmr.domain import PipelineConfig  # noqa: E402
from src.gmr.storage.month import MonthDB, Reading  # noqa: E402
from tests._month import changes, img, log_rows, make_month, readings, serial_of  # noqa: E402
from tests._window import has_display, new_window  # noqa: E402

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


def test_find_verify_row_prefers_same_controller():
    ayub = row(final_filename="A1.jpg", outcome="PLUS", source_folder="Аюб")
    suliman = row(final_filename="A1.jpg", outcome="PLUS", source_folder="Сулиман С")
    assert find_verify_row([ayub, suliman], "A1.jpg", "plus", controller="Аюб") is ayub
    assert find_verify_row([ayub, suliman], "A1.jpg", "plus", controller="Другой") is suliman


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


def test_list_verify_photos_all_controllers(tmp_path):
    img(tmp_path / "Аюб" / "plus" / "A1.jpg")
    img(tmp_path / "Сулиман С" / "minus" / "A2.jpg")
    (tmp_path / "run_logs").mkdir()
    rows = [row(final_filename="A1.jpg", outcome="PLUS", source_folder="Аюб"),
            row(final_filename="A2.jpg", outcome="MINUS", source_folder="Сулиман С")]
    items = list_verify_photos(str(tmp_path), rows)
    assert [(program2.controller_name(i["path"], str(tmp_path)), i["log_row"]["final_filename"])
            for i in items] == [("Аюб", "A1.jpg"), ("Сулиман С", "A2.jpg")]


# ─── 4. Исправление: что изменится ───────────────────────────────────────────
# (запись в базу — tests/test_operator.py)

def _auto_row():
    return row(original_filename="w.jpg", final_filename="A1.jpg", serial_id="1284567",
               account_id="A1", reading="1300", last_reading="1000.0", delta="300",
               outcome="PLUS", model_reading_str="01300")


def test_correct_reading_same_serial():
    df = table([("1284567", "A1", "1000", "1300"), ("1234567", "A2", "5000", "")])
    c = plan_verify_correction(df, CFG, _auto_row(), "1284567", "00950")
    assert c.error is None and not c.account_changed
    assert (c.reading, c.delta, c.outcome) == (950, -50.0, "MINUS")


def test_correct_serial_moves_reading_to_other_account():
    df = table([("1284567", "A1", "1000", "1300"), ("1234567", "A2", "5000", "")])
    c = plan_verify_correction(df, CFG, _auto_row(), "1234567", "05300")
    assert c.account_changed and (c.old_account, c.new_account) == ("A1", "A2")
    assert (c.last_reading, c.delta, c.outcome) == (5000.0, 300.0, "PLUS")
    assert c.new_account_has_reading == ""


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

needs_display = pytest.mark.skipif(not has_display(), reason="нет экрана (запускать под xvfb-run)")

VERIFY_TABLE = [("1284567", "A1", "1000", ""), ("1234567", "A2", "5000", ""),
                ("3333333", "A3", "10", ""), ("4444444", "A4", "90", "")]
VERIFY_LOG = [
    dict(_auto_row(), source_folder="Аюб"),
    row(original_filename="x.jpg", final_filename="A1.jpg", outcome="REPEAT", source_folder="Аюб"),
    row(original_filename="y.jpg", final_filename="A3.jpg", serial_id="3333333", account_id="A3",
        reading="20", last_reading="10.0", delta="10", outcome="PLUS", source_folder="Аюб"),
    row(original_filename="z.jpg", final_filename="A4.jpg", serial_id="4444444", account_id="A4",
        reading="80", last_reading="90.0", delta="-10", outcome="MINUS", source_folder="Сулиман С"),
]


def _window(monkeypatch, settings):
    monkeypatch.setattr(program2, "load_settings", lambda: settings)
    monkeypatch.setattr(program2, "save_settings", lambda s: None)
    monkeypatch.setattr(program2.MainWindow, "_load_models_async", lambda self: None)
    monkeypatch.setattr(program2, "LoginDialog", lambda *a, **k: type("D", (), {"action": "continue"})())
    answers = []
    monkeypatch.setattr(program2.messagebox, "askyesno", lambda *a, **k: answers.pop(0) if answers else True)
    for name in ("showinfo", "showwarning", "showerror"):
        monkeypatch.setattr(program2.messagebox, name, lambda *a, **k: None)
    w = new_window(program2.MainWindow)
    w.withdraw()
    w.answers = answers
    return w


def _settings(f, tmp_path):
    return program2.AppSettings(operator_name="Оператор", month_dir=str(f.root),
                                training_dir=str(tmp_path / "train"))


@pytest.fixture
def app(tmp_path, monkeypatch):
    """MainWindow на папке месяца: reader.py записал A1 (1300), A3 (20), A4 (80)."""
    f = make_month(tmp_path, VERIFY_TABLE, VERIFY_LOG)
    with MonthDB(f.db) as db, db.transaction():
        for account, value, photo in (("A1", "1300", "w.jpg"), ("A3", "20", "y.jpg"), ("A4", "80", "z.jpg")):
            db.put_reading(Reading(account, value, "01 10 2026", "auto", photo, "t", "auto"))
    img(f.results / "Аюб" / "plus" / "A1.jpg")
    img(f.results / "Аюб" / "plus" / "A3.jpg")
    img(f.results / "Сулиман С" / "minus" / "A4.jpg")
    w = _window(monkeypatch, _settings(f, tmp_path))
    w.month = f
    yield w
    w.destroy()


def _items(app):
    return list_verify_photos(app.results_dir, app.log_rows)


def _saved(app, original):
    return next(r for r in log_rows(app.month) if r["original_filename"] == original)


@needs_display
def test_verify_ok_marks_plus_row_and_opens_next(app):
    items = _items(app)
    assert [os.path.basename(i["path"]) for i in items] == ["A1.jpg", "A3.jpg", "A4.jpg"]
    app.open_verify_screen(items[0], 0)
    app._current_screen._verify_ok()
    assert _saved(app, "w.jpg")["verified_by"] == "Оператор"     # строка PLUS — в базе
    assert _saved(app, "x.jpg")["verified_by"] == ""             # REPEAT не тронут
    assert os.path.basename(app._current_screen.item["path"]) == "A3.jpg"   # следующее фото
    assert len(_items(app)) == 2


@needs_display
def test_skip_opens_next_and_last_returns_to_list(app):
    items = _items(app)
    app.open_verify_screen(items[1], 1)
    app._current_screen._skip()
    assert os.path.basename(app._current_screen.item["path"]) == "A4.jpg"   # другой контролёр
    app._current_screen._skip()
    assert not hasattr(app, "_current_screen")               # список закончился


@needs_display
def test_verify_tab_lists_all_controllers(app):
    tab = app._verify_tab
    tab.refresh()
    assert [tab._tree.item(i)["values"][1] for i in tab._tree.get_children()] == ["Аюб", "Аюб", "Сулиман С"]


@needs_display
def test_edit_mode_hides_ok_button_and_enter_saves(app):
    app.open_verify_screen(_items(app)[0], 0)
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
    r = _saved(app, "w.jpg")
    assert r["reading"] == "950" and r["outcome"] == "MINUS" and CORRECTED_NOTE in r["notes"]
    assert r["verified_by"] == "Оператор" and r["model_reading_str"] == "01300"
    assert readings(app.month)["A1"] == "950"
    ch = changes(app.month)[-1]
    assert (ch["who"], ch["action"], ch["account"], ch["old"], ch["new"]) == (
        "Оператор", "показание исправлено при проверке", "A1", "1300", "950")
    # фото переехало в minus/ своего контролёра
    assert (app.month.results / "Аюб" / "minus" / "A1.jpg").exists()
    assert not (app.month.results / "Аюб" / "plus" / "A1.jpg").exists()
    df = app.df.set_index(CFG.col_account_id)
    assert df.loc["A1", CFG.col_new_reading] == "950"           # таблица окна — как в базе


@needs_display
def test_edit_serial_moves_reading_and_renames_photo(app):
    app.open_verify_screen(_items(app)[0], 0)
    scr = app._current_screen
    scr._enter_edit()
    scr._reading_widget.set_digits("05300", None)
    scr._serial_var.set("1234567")
    scr._save_edit()
    got = readings(app.month)
    assert "A1" not in got and got["A2"] == "5300"
    plus = app.month.results / "Аюб" / "plus"
    assert (plus / "A2.jpg").exists() and not (plus / "A1.jpg").exists()
    r = _saved(app, "w.jpg")
    assert (r["account_id"], r["serial_id"], r["final_filename"]) == ("A2", "1234567", "A2.jpg")
    df = app.df.set_index(CFG.col_account_id)
    assert (df.loc["A1", CFG.col_new_reading], df.loc["A2", CFG.col_new_reading]) == ("", "5300")


@needs_display
def test_edit_serial_keeps_other_value_of_old_account(app):
    with MonthDB(app.month.db) as db, db.transaction():        # A1 уже исправил кто-то другой
        db.put_reading(Reading("A1", "1111", "", "manual", "", "t", "кто-то"))
    app.open_verify_screen(_items(app)[0], 0)
    scr = app._current_screen
    scr._enter_edit()
    scr._reading_widget.set_digits("05300", None)
    scr._serial_var.set("1234567")
    scr._save_edit()
    assert readings(app.month)["A1"] == "1111"
    assert app.df.set_index(CFG.col_account_id).loc["A1", CFG.col_new_reading] == "1111"


@needs_display
def test_edit_serial_name_taken_gets_suffix(app):
    img(app.month.results / "Аюб" / "plus" / "A2.jpg", 50)     # другое фото A2 уже лежит
    app.open_verify_screen(_items(app)[0], 0)
    scr = app._current_screen
    scr._enter_edit()
    scr._reading_widget.set_digits("05300", None)
    scr._serial_var.set("1234567")
    scr._save_edit()
    assert (app.month.results / "Аюб" / "plus" / "A2_2.jpg").exists()
    assert _saved(app, "w.jpg")["final_filename"] == "A2_2.jpg"


@needs_display
def test_edit_serial_cancelled_changes_nothing(app):
    app.open_verify_screen(_items(app)[0], 0)
    scr = app._current_screen
    scr._enter_edit()
    scr._reading_widget.set_digits("05300", None)
    scr._serial_var.set("1234567")
    app.answers.append(False)        # «Другой абонент?» — Нет
    scr._save_edit()
    r = _saved(app, "w.jpg")
    assert r["verified_by"] == "" and r["account_id"] == "A1"
    assert readings(app.month) == {"A1": "1300", "A3": "20", "A4": "80"}
    assert (app.month.results / "Аюб" / "plus" / "A1.jpg").exists()


@needs_display
def test_edit_unknown_serial_refused(app):
    app.open_verify_screen(_items(app)[0], 0)
    scr = app._current_screen
    scr._enter_edit()
    scr._reading_widget.set_digits("05300", None)
    scr._serial_var.set("9999999")
    scr._save_edit()
    assert _saved(app, "w.jpg")["verified_by"] == "" and readings(app.month)["A1"] == "1300"


# ─── «Серийник в базе с ошибкой» ─────────────────────────────────────────────
# С 2026-10-01 (решение владельца): номер в базе месяца исправляется на номер
# с фото, показание записывается сразу, фото — в plus/minus, номер — в список
# для компании db_serial_fix.csv в папке месяца.

def test_table_serial_for_account():
    df = table([("0045618", "A1", "10", ""), ("1234567", "A2", "5", "")])
    assert program2.table_serial_for_account(df, CFG, "A1") == "0045618"
    assert program2.table_serial_for_account(df, CFG, " A2 ") == "1234567"
    assert program2.table_serial_for_account(df, CFG, "A9") is None
    assert program2.table_serial_for_account(None, CFG, "A1") is None


def test_append_db_serial_fix(tmp_path):
    lst = tmp_path / "db_serial_fix.csv"
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


EDIT_AUTO = [{"original_filename": "w.jpg", "outcome": "SERIAL_NOT_FOUND", "source": "auto",
              "processed_by": "auto", "photo_hash": "h-w", "source_folder": "Сулиман С"}]


@pytest.fixture
def edit_app(tmp_path, monkeypatch):
    """MainWindow + одно фото в question/serial_not_found контролёра «Сулиман С»."""
    f = make_month(tmp_path, [("1284567", "A1", "1000", ""), ("7654321", "A2", "5", "")], EDIT_AUTO)
    photos = f.results / "Сулиман С"
    photo = img(photos / "question" / "serial_not_found" / "w.jpg")
    w = _window(monkeypatch, _settings(f, tmp_path))
    w.month = f
    yield w, photos, photo
    w.destroy()


def _manual(app):
    return [r for r in log_rows(app.month) if r["source"] == "manual"]


@needs_display
def test_db_serial_fix_fixes_serial_and_writes_reading(edit_app, monkeypatch):
    app, photos, photo = edit_app
    img(photos / "question" / "no_meter" / "z.jpg")              # в очереди есть ещё фото
    app.open_edit_screen(str(photo), "serial_not_found")
    scr = app._current_screen
    scr._account_var.set("A1")
    scr._lookup_by_account("A1")
    scr._reading_widget.set_digits("01050", None)
    assert app.serial_index().by_serial.get("1284567") == ("A1", 1000.0)
    monkeypatch.setattr(scr, "_ask_photo_serial", lambda suggested: "1234567")
    scr._db_serial_fix()
    assert os.path.basename(app._current_screen.photo_path) == "z.jpg"

    assert serial_of(app.month, "A1") == "1234567"               # номер в базе — как на фото
    assert readings(app.month) == {"A1": "1050"}                 # показание записано сразу
    assert (photos / "plus" / "A1.jpg").exists() and not photo.exists()
    (r,) = _manual(app)
    assert (r["outcome"], r["serial_id"], r["account_id"], r["reading"], r["delta"]) == (
        "PLUS", "1234567", "A1", "1050", "50.0")
    assert "в базе 1284567, на фото 1234567" in r["notes"]
    assert r["photo_hash"] == "h-w" and r["source_folder"] == "Сулиман С"
    acts = [(c["action"], c["account"], c["old"], c["new"]) for c in changes(app.month)]
    assert ("номер исправлен оператором", "A1", "1284567", "1234567") in acts
    assert ("показание записано", "A1", "", "1050") in acts
    lst = (app.month.root / "db_serial_fix.csv").read_text(encoding="utf-8-sig").splitlines()
    assert lst[1].split(";")[:6] == ["A1.jpg", "Сулиман С", "A1", "1284567", "1234567", "01050"]
    df = app.df.set_index(CFG.col_account_id)                    # таблица окна — как в базе
    assert (df.loc["A1", CFG.col_serial], df.loc["A1", CFG.col_new_reading]) == ("1234567", "1050")
    assert app.serial_index().by_serial.get("1234567") == ("A1", 1000.0)   # подсказки — по новому номеру


@needs_display
def test_db_serial_fix_minus_goes_to_minus(edit_app, monkeypatch):
    app, photos, photo = edit_app
    app.open_edit_screen(str(photo), "serial_not_found")
    scr = app._current_screen
    scr._account_var.set("A1")
    scr._lookup_by_account("A1")
    scr._reading_widget.set_digits("00900", None)
    monkeypatch.setattr(scr, "_ask_photo_serial", lambda suggested: "1234567")
    scr._db_serial_fix()
    assert _manual(app)[0]["outcome"] == "MINUS"
    assert (photos / "minus" / "A1.jpg").exists()


@needs_display
def test_db_serial_fix_refuses(edit_app, monkeypatch):
    app, photos, photo = edit_app
    app.open_edit_screen(str(photo), "serial_not_found")
    scr = app._current_screen
    asked = []
    monkeypatch.setattr(scr, "_ask_photo_serial", lambda suggested: asked.append(1) or "1284567")
    scr._reading_widget.set_digits("01050", None)
    scr._account_var.set("A9")                 # нет такого лицевого счёта
    scr._db_serial_fix()
    scr._account_var.set("A1")
    scr._reading_widget.set_digits("01?50", None)   # показание не полное
    scr._db_serial_fix()
    assert asked == []                         # до вопроса о номере не дошло
    scr._reading_widget.set_digits("01050", None)
    scr._db_serial_fix()                       # номер как в базе — не опечатка
    assert asked == [1]
    app.answers.append(False)                  # «Продолжить?» — Нет
    monkeypatch.setattr(scr, "_ask_photo_serial", lambda suggested: "1234567")
    scr._db_serial_fix()
    assert photo.exists() and _manual(app) == [] and readings(app.month) == {}
    assert serial_of(app.month, "A1") == "1284567"
    assert not (app.month.root / "db_serial_fix.csv").exists()


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
    (r,) = _manual(app)
    assert r["outcome"] == "PLUS" and r["account_id"] == "A1" and r["serial_id"] == "1284567"
    assert "подсказка: 1284567" in r["notes"]
    assert readings(app.month) == {"A1": "1050"}


# ─── Окно не затирает то, что reader.py записал, пока оно открыто ───────────
# (в режиме месяца исправлены известные ошибки «Верно стирает строки reader.py»
#  и «Принять стирает показания reader.py»; «таблица занята Excel» больше не
#  бывает — окно пишет в базу, Excel открывает только выгрузку)

def _reader_writes_meanwhile(month):
    with MonthDB(month.db) as db, db.transaction():
        db.append_log_rows([row(original_filename="new.jpg", final_filename="A2.jpg",
                                account_id="A2", reading="5100", outcome="PLUS")])
        db.put_reading(Reading("A2", "5100", "", "auto", "new.jpg", "t", "auto"))


@needs_display
def test_verify_ok_keeps_log_rows_written_meanwhile(app):
    _reader_writes_meanwhile(app.month)
    app.open_verify_screen(_items(app)[0], 0)
    app._current_screen._verify_ok()
    assert "new.jpg" in [r["original_filename"] for r in log_rows(app.month)]
    assert readings(app.month)["A2"] == "5100"


@needs_display
def test_accept_keeps_readings_written_meanwhile(edit_app):
    app, photos, photo = edit_app
    with MonthDB(app.month.db) as db, db.transaction():          # reader.py записал A2
        db.put_reading(Reading("A2", "7", "", "auto", "p.jpg", "t", "auto"))
    app.open_edit_screen(str(photo), "serial_not_found")
    scr = app._current_screen
    scr._reading_widget.set_digits("01050", None)
    scr._lookup_by_serial("1284567")
    scr._accept()                                            # оператор принял A1
    assert readings(app.month) == {"A1": "1050", "A2": "7"}


# ─── Номер у нескольких абонентов (2026-10-01) ───────────────────────────────

def test_serial_choices_one_per_account_in_table_order():
    df = table([("44444", "A4", "10", ""), ("44444", "A5", "20", ""), ("44444", "A4", "10", "")])
    choices = program2.serial_choices(df, CFG, "00025")
    assert [(c.account, c.last_reading, c.delta) for c in choices] == [("A4", 10.0, 15.0), ("A5", 20.0, 5.0)]


@needs_display
def test_ambiguous_serial_operator_chooses_account(tmp_path, monkeypatch):
    f = make_month(tmp_path, [("1284567", "A1", "1000", ""), ("1284567", "A9", "900", "")], EDIT_AUTO)
    photo = img(f.results / "Сулиман С" / "question" / "serial_ambiguous" / "w.jpg")
    app = _window(monkeypatch, _settings(f, tmp_path))
    app.month = f
    try:
        app.open_edit_screen(str(photo), "serial_ambiguous")
        scr = app._current_screen
        scr._reading_widget.set_digits("01050", None)
        scr._lookup_by_serial("1284567")                     # в базе у A1 и A9
        assert scr._account_var.get() == ""                  # сам не выбирает
        assert [h.account for h in scr._hints] == ["A1", "A9"]
        assert "нескольких" in scr._serial_match_var.get()
        scr._use_choice(scr._hints[1])
        assert scr._account_var.get() == "A9"
        scr._accept()
        (r,) = _manual(app)
        assert r["account_id"] == "A9" and f"{program2.CHOICE_NOTE}: л/с A9" in r["notes"]
        assert readings(f) == {"A9": "1050"}
    finally:
        app.destroy()
