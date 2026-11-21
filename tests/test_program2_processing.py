"""
Вкладка «Обработка» program2.py: что пишется в базу месяца, куда уходит фото,
какое фото открывается следующим. С этапа 2.3 окно работает с папкой месяца:
все контролёры сразу, показание и строка лога — одной записью в базу
(src/gmr/application/operator.py).

Оконные тесты — под xvfb-run (без экрана пропускаются).
"""
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

program2 = pytest.importorskip("program2")
from src.gmr.operator import edit_screen as op_edit  # noqa: E402
from src.gmr.operator import settings as op_settings  # noqa: E402
from src.gmr.operator import window as op_window  # noqa: E402
from src.gmr.domain import PipelineConfig, RunSelection, RunSelectionPolicy  # noqa: E402
from src.gmr.storage import LOG_COLUMNS  # noqa: E402
from src.gmr.storage.month import MonthDB, Reading, Recognition  # noqa: E402
from tests._month import changes, img, log_rows, make_month, readings  # noqa: E402
from tests._window import has_display, new_window  # noqa: E402

CFG = PipelineConfig()
needs_display = pytest.mark.skipif(not has_display(), reason="нет экрана (запускать под xvfb-run)")


# ─── Без окон ────────────────────────────────────────────────────────────────

def test_list_question_photos_order_and_filter(tmp_path):
    q = tmp_path / "question"                      # фото без подпапок контролёров
    img(q / "no_meter" / "a.jpg")
    img(q / "digits_error" / "b.jpg")
    img(q / "serial_not_found" / "c.png")
    (q / "digits_error" / "notes.txt").write_text("x")
    img(q / "unknown_reason" / "d.jpg")            # не из QUESTION_SUBFOLDERS
    got = [(r, n) for r, n, _ in program2.list_question_photos(str(tmp_path))]
    assert got == [("digits_error", "b.jpg"), ("serial_not_found", "c.png"), ("no_meter", "a.jpg")]


def test_queue_has_all_controllers_by_reason(tmp_path):
    res = tmp_path / "результат"
    img(res / "Сулиман С" / "question" / "no_meter" / "s1.jpg")
    img(res / "Аюб" / "question" / "no_meter" / "a1.jpg")
    img(res / "Аюб" / "question" / "digits_error" / "A-1.jpg")
    (res / "run_logs").mkdir()
    (res / "report.txt").write_text("x")
    got = [(program2.controller_name(p, str(res)), r, n)
           for r, n, p in program2.list_question_photos(str(res))]
    assert got == [("Аюб", "digits_error", "A-1.jpg"), ("Аюб", "no_meter", "a1.jpg"),
                   ("Сулиман С", "no_meter", "s1.jpg")]


def test_controller_dir_of_photo(tmp_path):
    res = tmp_path / "результат"
    assert program2.controller_dir(str(res / "Аюб" / "question" / "no_meter" / "a.jpg")) == res / "Аюб"
    assert program2.controller_dir(str(res / "Аюб" / "plus" / "A-1.jpg")) == res / "Аюб"
    assert program2.controller_name(str(res / "question" / "no_meter" / "a.jpg"), str(res)) == ""


def test_save_crnn_markup(tmp_path):
    crop = np.full((20, 60, 3), 50, np.uint8)
    # серийник не исправляли — ничего не сохраняется
    program2.save_crnn_markup(tmp_path, "1300000001", crop, "0234567", "0234567")
    program2.save_crnn_markup(tmp_path, "1300000001", None, "0234567", "234567")
    program2.save_crnn_markup(tmp_path, "1300000001", crop, "12-34", "1234")         # не цифры — нет
    assert not (tmp_path / "serials_crnn").exists()
    # исправили — номер в имени (стандарт датасетов), labels нет; тот же счёт ещё раз — «-2»
    program2.save_crnn_markup(tmp_path, "1300000001", crop, "0234567", "234567")
    program2.save_crnn_markup(tmp_path, "1300000001", crop, "0234567", None)
    program2.save_crnn_markup(tmp_path, "h2", crop, "1234567", None)
    new = tmp_path / "serials_crnn"
    assert sorted(p.name for p in new.rglob("*.*")) == [
        "1300000001-2__serial1__0234567.jpg", "1300000001__serial1__0234567.jpg", "h2__serial1__1234567.jpg"]
    assert all(p.parent.name == "images" for p in new.rglob("*.*"))
    (new / "images" / "h3__SERIAL1__1234567.JPG").write_bytes(b"x")   # в Windows регистр имени не важен
    program2.save_crnn_markup(tmp_path, "h3", crop, "1234567", None)
    assert (new / "images" / "h3-2__serial1__1234567.jpg").is_file()


def test_save_cnn_markup_only_changed_digits(tmp_path):
    crops = [np.full((30, 15, 3), v, np.uint8) for v in range(5)]
    crops[1] = None                                            # заглушка — пропускается
    program2.save_cnn_markup(tmp_path, "1300000001", crops, [{}] * 5, "12?45", "12745")
    saved = [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.jpg")]
    assert saved == ["digits_cnn/7/1300000001__digit3.jpg"]   # изменилась только позиция 3
    program2.save_cnn_markup(tmp_path, "1300000001", crops, [{}] * 5, "12?45", "12745")
    assert (tmp_path / "digits_cnn" / "7" / "1300000001-2__digit3.jpg").is_file()
    # длины не совпадают — ничего
    program2.save_cnn_markup(tmp_path / "x", "h1", crops, [{}] * 5, "12345", "1234")
    assert not (tmp_path / "x").exists()


def test_save_meter_markup(tmp_path):
    orig = img(tmp_path / "фото" / "Аюб" / "IMG-1.JPG")
    program2.save_meter_markup(tmp_path / "ds", "1300000001", orig)
    program2.save_meter_markup(tmp_path / "ds", "1300000001", orig)
    program2.save_meter_markup(tmp_path / "ds", "h2", None)
    program2.save_meter_markup(tmp_path / "ds", "h3", tmp_path / "нет.jpg")
    new = tmp_path / "ds" / "meter_yolo"
    assert sorted(p.relative_to(new).as_posix() for p in new.rglob("*.*")) == ["1300000001-2.jpg", "1300000001.jpg"]


def test_default_month_name():
    from datetime import datetime
    assert program2.default_month_name(datetime(2026, 10, 7)) == "Октябрь_2026"
    assert program2.default_month_name(datetime(2027, 1, 1)) == "Январь_2027"


def test_error_folder_in_queue_last(tmp_path):
    for sub in ("error", "digits_error"):
        img(tmp_path / "question" / sub / f"{sub}.jpg")
    items = program2.list_question_photos(str(tmp_path))      # (причина, имя, путь)
    assert [(reason, name) for reason, name, _ in items] == \
        [("digits_error", "digits_error.jpg"), ("error", "error.jpg")]
    assert program2.REASON_LABELS["error"] == "Ошибка программы"


def test_old_settings_ask_for_month_folder(tmp_path, monkeypatch):
    # settings.json до этапа 2.3 (папка контролёра, таблица) и с «папкой для разметки»
    # (до 2026-10-07): имя остаётся, лишние ключи не мешают
    f = tmp_path / "settings.json"
    f.write_text('{"operator_name": "Оператор", "photos_dir": "output/Аюб", '
                 '"table_path": "t.csv", "training_dir": "train"}', encoding="utf-8")
    monkeypatch.setattr(op_settings, "SETTINGS_FILE", f)
    assert program2.load_settings() == program2.AppSettings("Оператор", "")


# ─── Окна: экран обработки ───────────────────────────────────────────────────

AUTO = [  # авто-строки reader.py, создавшие файлы в question/ (отпечаток — в ручную строку)
    {"original_filename": n, "outcome": o, "source": "auto", "processed_by": "auto",
     "photo_hash": f"h-{n}", "source_folder": "Аюб"}
    for n, o in (("IMG-20260915-WA0001.jpg", "SERIAL_NOT_FOUND"), ("p2.jpg", "SERIAL_NOT_FOUND"),
                 ("p3.jpg", "NO_METER"))
]


def _window(monkeypatch, settings, answers=None):
    monkeypatch.setattr(op_window, "load_settings", lambda: settings)
    monkeypatch.setattr(op_window, "save_settings", lambda s: None)
    monkeypatch.setattr(op_window, "ask_login", lambda *a, **k: "continue")
    # настройки распознавания нового месяца — как предложено (окно не открывается)
    monkeypatch.setattr(program2.MainWindow, "ask_preset",
                        staticmethod(lambda parent, preset, title, ok_text="Сохранить": preset))
    answers, shown, asked = list(answers or []), [], []
    monkeypatch.setattr(program2.messagebox, "askyesno",
                        lambda *a, **k: asked.append(a) or (answers.pop(0) if answers else True))
    for name in ("showinfo", "showwarning", "showerror"):
        monkeypatch.setattr(program2.messagebox, name, lambda *a, _n=name, **k: shown.append((_n, a)))
    w = new_window(program2.MainWindow)
    w.withdraw()
    w.answers, w.shown, w.asked = answers, shown, asked
    return w


@pytest.fixture
def app(tmp_path, monkeypatch):
    f = make_month(tmp_path, [("1234567", "A1", "1000", ""), ("7654321", "A2", "500", "")], AUTO)
    photos = f.results / "Аюб"
    q = photos / "question"
    img(q / "serial_not_found" / "IMG-20260915-WA0001.jpg", 10)
    img(q / "serial_not_found" / "p2.jpg", 20)
    img(q / "no_meter" / "p3.jpg", 30)
    w = _window(monkeypatch, program2.AppSettings(operator_name="Оператор", month_dir=str(f.root)))
    w.photos, w.month = photos, f
    yield w
    try:
        w.destroy()
    except program2.tk.TclError:          # окно уже закрыто тестом
        pass


def _open_first(app):
    reason, _, path = program2.list_question_photos(app.results_dir)[0]
    app.open_edit_screen(path, reason)
    return app._current_screen


def _fill(scr, account, reading):
    scr._account_var.set(account)
    scr._lookup_by_account(account)
    scr._reading_widget.set_digits(reading, None)
    scr._update_delta()
    scr._update_accept_state()


def _db_log(app):
    return log_rows(app.month)[len(AUTO):]          # ручные строки


def _reading(app, account):
    with MonthDB(app.month.db) as db:
        return db.reading(account)


@needs_display
def test_accept_writes_db_moves_photo_opens_next(app):
    scr = _open_first(app)
    _fill(scr, "A1", "01200")
    scr._accept()
    assert readings(app.month) == {"A1": "1200"}
    (r,) = _db_log(app)
    assert (r["original_filename"], r["final_filename"], r["serial_id"], r["account_id"]) == (
        "IMG-20260915-WA0001.jpg", "A1.jpg", "1234567", "A1")
    assert (r["reading"], r["last_reading"], r["delta"], r["outcome"], r["source"]) == (
        "1200", "1000.0", "200.0", "PLUS", "manual")
    assert r["notes"] == "serial_not_found"
    assert r["photo_hash"] == "h-IMG-20260915-WA0001.jpg" and r["source_folder"] == "Аюб"   # из авто-строки
    ch = changes(app.month)[-1]
    assert (ch["who"], ch["action"], ch["account"], ch["new"]) == ("Оператор", "показание записано", "A1", "1200")
    assert _reading(app, "A1").date == "15 09 2026"                 # дата из имени фото WhatsApp
    assert (app.photos / "plus" / "A1.jpg").exists()
    assert not (app.photos / "question" / "serial_not_found" / "IMG-20260915-WA0001.jpg").exists()
    # сразу следующее фото очереди
    assert os.path.basename(app._current_screen.photo_path) == "p2.jpg"


@needs_display
def test_accept_minus_goes_to_minus(app):
    scr = _open_first(app)
    _fill(scr, "A1", "00900")
    scr._accept()
    assert _db_log(app)[-1]["outcome"] == "MINUS"
    assert (app.photos / "minus" / "A1.jpg").exists()


@needs_display
def test_accept_suspicious_asks_and_can_be_cancelled(app):
    scr = _open_first(app)
    _fill(scr, "A2", "90000")                      # delta 89500 > 10000
    app.answers.append(False)
    scr._accept()
    assert readings(app.month) == {} and _db_log(app) == []
    app.answers.append(True)
    scr._accept()
    assert readings(app.month) == {"A2": "90000"} and _db_log(app)[-1]["outcome"] == "PLUS"


@needs_display
def test_accept_disabled_without_full_reading(app):
    scr = _open_first(app)
    scr._account_var.set("A1")
    scr._lookup_by_account("A1")
    scr._update_accept_state()
    scr._accept()
    assert _db_log(app) == [] and readings(app.month) == {}


@needs_display
def test_accept_blocked_for_unknown_account(app):
    """Решение владельца 2026-11-20: абонента нет в таблице — «Принять»
    недоступна (раньше — строка лога без показания); для таких фото «Нет в базе»."""
    scr = _open_first(app)
    scr._serial_var.set("5555555")
    scr._account_var.set("A9")                     # нет в таблице
    scr._reading_widget.set_digits("01200", None)
    scr._accept()
    scr._accept()
    assert str(scr._accept_btn["state"]) == "disabled"
    assert _db_log(app) == [] and readings(app.month) == {}


@needs_display
@pytest.mark.parametrize("action,folder,outcome,notes", [
    ("_duplicate", "repeat", "REPEAT", ""),
    ("_unreadable", "unreadable", "UNREADABLE", ""),
    ("_not_in_db", "not_in_db", "NOT_IN_DB", "счётчик не найден в базе"),
])
def test_other_actions(app, action, folder, outcome, notes):
    scr = _open_first(app)
    scr._serial_var.set("5555555")
    getattr(scr, action)()
    (r,) = _db_log(app)
    assert (r["original_filename"], r["outcome"], r["source"], r["notes"]) == (
        "IMG-20260915-WA0001.jpg", outcome, "manual", notes)
    assert r["photo_hash"] == "h-IMG-20260915-WA0001.jpg"
    assert (app.photos / folder / "IMG-20260915-WA0001.jpg").exists()
    assert readings(app.month) == {}                            # показаний нет
    assert os.path.basename(app._current_screen.photo_path) == "p2.jpg"


@needs_display
def test_action_cancelled_changes_nothing(app):
    scr = _open_first(app)
    app.answers.append(False)
    scr._duplicate()
    assert _db_log(app) == []
    assert (app.photos / "question" / "serial_not_found" / "IMG-20260915-WA0001.jpg").exists()


@needs_display
def test_processing_tab_mark_unreadable_from_list(app):
    tab = app._proc_tab
    tab.refresh()
    first = tab._tree.get_children()[0]
    assert tab._tree.item(first)["values"][1] == "Аюб"           # столбец «Контролёр»
    tab._tree.selection_set(first)
    tab._unreadable()
    (r,) = _db_log(app)
    assert (r["outcome"], r["notes"]) == ("UNREADABLE", "нечитаемо (из списка)")
    assert (app.photos / "unreadable" / "IMG-20260915-WA0001.jpg").exists()


@needs_display
def test_last_photo_returns_to_list(app):
    for _ in range(3):
        scr = _open_first(app) if not hasattr(app, "_current_screen") else app._current_screen
        scr._unreadable()
    assert program2.list_question_photos(app.results_dir) == []
    assert app._notebook.winfo_manager() == "pack"                # снова список вкладок


def _seen_wrong_model(app):
    """Модели при обработке ошиблись: номер 1284567, показание 01?00 (кропы — в базе)."""
    _seen(app, serial_read="1284567", serial_conf=0.9, reading_str="01?00",
          digits_ok=[True, True, False, True, True],
          crops={"serial": _jpg(20, 60, 5), **{f"digit{i}": _jpg(30, 15, 1) for i in range(5)}})


@needs_display
def test_accept_saves_silent_markup_when_model_was_wrong(app):
    _seen_wrong_model(app)
    scr = _open_first(app)
    img(app.month.photos / "Аюб" / "IMG-20260915-WA0001.jpg")  # исходное фото есть, но счётчик найден —
    with MonthDB(app.month.db) as db, db.transaction():      # в meter_yolo/new не кладётся
        db.set_meta("created_at", "2026-09-30 23:50:00")       # месяц создан в сентябре
    _fill(scr, "A1", "01200")
    scr._accept()
    ds = app.month.markup                                      # разметка — в папке месяца
    assert not (ds / "meter_yolo").exists()
    # имена — по стандарту, фото — лицевой счёт (месяц — в папке new/<ГГГГ-ММ>/ при --collect)
    assert (ds / "serials_crnn" / "images" / "A1__serial1__1234567.jpg").exists()
    assert [p.relative_to(ds).as_posix() for p in ds.rglob("*.jpg") if "digits_cnn" in p.parts] == [
        "digits_cnn/2/A1__digit3.jpg"]


@needs_display
def test_photos_of_two_controllers_go_to_their_folders(app):
    other = app.month.results / "Сулиман С"
    img(other / "question" / "no_meter" / "s1.jpg", 40)
    with MonthDB(app.month.db) as db, db.transaction():
        db.append_log_rows([{c: "" for c in LOG_COLUMNS} | {
            "original_filename": "s1.jpg", "outcome": "NO_METER", "source": "auto",
            "photo_hash": "h-s1", "source_folder": "Сулиман С"}])
    app.refresh_tabs()
    assert len(app._proc_tab._items) == 4
    app.open_edit_screen(str(other / "question" / "no_meter" / "s1.jpg"), "no_meter")
    app._current_screen._unreadable()
    r = _db_log(app)[-1]
    assert (r["original_filename"], r["photo_hash"], r["source_folder"]) == ("s1.jpg", "h-s1", "Сулиман С")
    assert (other / "unreadable" / "s1.jpg").exists() and not (app.photos / "unreadable").exists()


@needs_display
def test_accept_closes_waiting_photo_of_same_account(app):
    # этап 3: у A1 появилось показание — его фото «цифры не прочитаны» уходит в repeat/
    other = app.month.results / "Сулиман С"
    waiting = img(other / "question" / "digits_error" / "A1.jpg", 40)
    with MonthDB(app.month.db) as db, db.transaction():
        db.append_log_rows([{c: "" for c in LOG_COLUMNS} | {
            "original_filename": "s1.jpg", "final_filename": "A1.jpg", "account_id": "A1",
            "outcome": "DIGITS_ERROR", "source": "auto", "photo_hash": "h-s1", "source_folder": "Сулиман С"}])
    app.refresh_tabs()
    app.open_edit_screen(str(app.photos / "question" / "serial_not_found" / "IMG-20260915-WA0001.jpg"),
                         "serial_not_found")
    _fill(app._current_screen, "A1", "01200")
    app._current_screen._accept()
    assert not waiting.exists() and (other / "repeat" / "A1.jpg").exists()
    assert log_rows(app.month)[-1]["outcome"] == "REPEAT"
    assert os.path.basename(app._current_screen.photo_path) == "p2.jpg"   # следующее — не закрытое


@needs_display
def test_accept_photo_without_hash_in_old_log_goes_to_plus(app):
    # строка старого лога без отпечатка: принимаемое фото не «закрывается» само собой
    q = app.photos / "question" / "digits_error"
    img(q / "A1.jpg", 50)
    with MonthDB(app.month.db) as db, db.transaction():
        db.append_log_rows([{c: "" for c in LOG_COLUMNS} | {
            "original_filename": "old.jpg", "final_filename": "A1.jpg", "account_id": "A1",
            "outcome": "DIGITS_ERROR", "source": "auto", "source_folder": "Аюб"}])
    app.refresh_tabs()
    app.open_edit_screen(str(q / "A1.jpg"), "digits_error")
    _fill(app._current_screen, "A1", "01200")
    app._current_screen._accept()
    assert (app.photos / "plus" / "A1.jpg").exists() and not (app.photos / "repeat").exists()


# ─── Этап 3, пункт 5а: у счёта уже есть показание — «Заменить?» ──────────────

def _put(app, account, value, source="auto", photo="old.jpg", who="auto", date="15 09 2026"):
    with MonthDB(app.month.db) as db, db.transaction():
        db.put_reading(Reading(account, value, date, source, photo, "t", who))


def test_describe_reading():
    r = Reading("A1", "1100", "15 09 2026", "auto", "old.jpg", "t", "auto")
    assert program2.describe_reading(r) == "записала программа по фото old.jpg, дата 15 09 2026"
    r = Reading("A1", "1100", "", "manual", "p.jpg", "t", "Оператор")
    assert program2.describe_reading(r) == "записал оператор Оператор"
    r = Reading("A1", "1100", "", "table", "", "t", "тест")
    assert program2.describe_reading(r) == "было в таблице компании"


@needs_display
def test_accept_no_question_without_reading(app):
    scr = _open_first(app)
    _fill(scr, "A1", "01200")
    scr._accept()
    assert app.asked == [] and readings(app.month) == {"A1": "1200"}


@needs_display
def test_accept_asks_before_replacing_reading(app):
    _put(app, "A1", "1100")
    scr = _open_first(app)
    _fill(scr, "A1", "01200")
    app.answers.append(False)                       # «Заменить?» — Нет
    scr._accept()
    (title, text), = [a[:2] for a in app.asked]
    assert "уже есть показание" in title
    assert "1100" in text and "1200" in text and "программа по фото old.jpg, дата 15 09 2026" in text
    assert readings(app.month) == {"A1": "1100"} and _db_log(app) == []          # ничего не изменилось
    assert (app.photos / "question" / "serial_not_found" / "IMG-20260915-WA0001.jpg").exists()
    assert app._current_screen is scr                                              # то же фото
    app.answers.append(True)                        # «Да»
    scr._accept()
    assert readings(app.month) == {"A1": "1200"}
    ch = changes(app.month)[-1]
    assert (ch["action"], ch["old"], ch["new"]) == ("показание записано", "1100", "1200")


@needs_display
def test_accept_does_not_overwrite_photo_in_plus(app):
    _put(app, "A1", "1100")
    first = img(app.photos / "plus" / "A1.jpg", 77)  # фото, по которому программа записала 1100
    scr = _open_first(app)
    _fill(scr, "A1", "01200")
    scr._accept()                                   # «Заменить?» — Да
    assert (app.photos / "plus" / "A1_2.jpg").exists()
    assert program2.read_image(str(first))[0, 0, 0] == 77                          # не затёрто
    assert _db_log(app)[-1]["final_filename"] == "A1_2.jpg"


@needs_display
def test_no_question_for_account_not_in_table(app):
    _put(app, "A9", "100")                          # показание абонента, которого нет в таблице
    scr = _open_first(app)
    scr._serial_var.set("5555555")
    scr._account_var.set("A9")
    scr._reading_widget.set_digits("01200", None)
    scr._accept()
    scr._accept()
    assert app.asked == [] and _db_log(app) == []            # «Принять» недоступна
    assert readings(app.month) == {"A9": "100"}


@needs_display
def test_db_serial_fix_asks_before_replacing(app, monkeypatch):
    _put(app, "A1", "1100", source="manual", who="Оператор")
    scr = _open_first(app)
    _fill(scr, "A1", "01200")
    monkeypatch.setattr(scr, "_ask_photo_serial", lambda suggested: "1234999")
    app.answers.append(False)
    scr._db_serial_fix()
    assert "записал оператор Оператор" in app.asked[0][1]
    assert readings(app.month) == {"A1": "1100"} and _db_log(app) == []


@needs_display
@pytest.mark.parametrize("action,folder", [("_duplicate", "repeat"), ("_unreadable", "unreadable"),
                                           ("_not_in_db", "not_in_db")])
def test_decision_does_not_overwrite_photo_with_same_name(app, action, folder):
    taken = img(app.photos / folder / "IMG-20260915-WA0001.jpg", 77)               # другое фото
    scr = _open_first(app)
    getattr(scr, action)()
    assert (app.photos / folder / "IMG-20260915-WA0001_2.jpg").exists()
    assert program2.read_image(str(taken))[0, 0, 0] == 77
    assert _db_log(app)[-1]["final_filename"] == "IMG-20260915-WA0001_2.jpg"


# ─── «Нет в базе»: номер для следующего прогона (решение владельца 2026-11-21) ─
# Прогон читает фото заново, если номер из строки «Нет в базе» появился в
# таблице. Окно читает копию с подписью и может прочитать другой номер, чем
# прогон с исходного фото; номер, который есть в таблице, возвращал фото в
# очередь при первом же прогоне (тест месяца 2026-11-20).

def test_not_in_db_serial_rule():
    nid = program2.not_in_db_serial
    assert nid("88888", "88888", "99999") == "99999"     # номер окна не меняли — номер прогона
    assert nid("", None, "99999") == "99999"             # модели окна не ответили, поле пустое
    assert nid("", "88888", "99999") == "99999"          # номер стёрли — пустое поле ничего не говорит
    assert nid("77777", "88888", "99999") == "77777"     # оператор вписал свой номер
    assert nid(" 77777 ", None, "") == "77777"
    assert nid("88888", "88888", "") == "88888"          # авто-строки нет — номер окна
    assert nid("", "88888", "") == ""


def test_accounts_with_serial_looks_up_as_run():
    from src.gmr.pipeline.run import serial_lookup
    df = pd.DataFrame({CFG.col_serial: ["1234567", "0056", "0056", "nan", ""],
                       CFG.col_account_id: ["A1", "A2", "A3", "A4", "A5"]})
    assert program2.accounts_with_serial(df, CFG, " 1234567 ") == ["A1"]
    assert program2.accounts_with_serial(df, CFG, "56") == ["A2", "A3"]          # '00'+номер
    assert program2.accounts_with_serial(df, CFG, "") == []
    assert program2.accounts_with_serial(None, CFG, "1234567") == []
    in_table = serial_lookup(df, CFG)                                            # правило прогона
    for s in ("1234567", "234567", "56", "056", "0056", "00056", "nan", "99999"):
        assert bool(program2.accounts_with_serial(df, CFG, s)) == in_table(s), s


def _run_read(app, serial, outcome="SERIAL_NOT_FOUND", name="IMG-20260915-WA0001.jpg", **fields):
    """Обработка прочитала с исходного фото номер serial (её строка лога, создавшая файл в question/)."""
    with MonthDB(app.month.db) as db, db.transaction():
        db.append_log_rows([{c: "" for c in LOG_COLUMNS} | {
            "original_filename": name, "serial_id": serial, "outcome": outcome, "source": "auto",
            "processed_by": "auto", "photo_hash": f"h-{name}", "source_folder": "Аюб", **fields}])
    app.reload()


def _ask_serial(scr, monkeypatch, answer):
    """Окно спрашивает номер с фото — ответ answer (None — Cancel)."""
    asked = []
    monkeypatch.setattr(scr, "_ask_serial_not_in_table",
                        lambda prompt, suggested: asked.append((prompt, suggested)) or answer)
    return asked


def _manual(app):
    return [r for r in log_rows(app.month) if r["source"] == "manual"]


def _next_run(app, name="IMG-20260915-WA0001.jpg"):
    """Что сделает с фото следующий прогон (правило и поиск номера — как у прогона)."""
    from src.gmr.pipeline.run import serial_lookup
    rows = [r for r in log_rows(app.month) if r["photo_hash"] == f"h-{name}"]
    return RunSelectionPolicy().decide(rows, False, serial_lookup(app.df, CFG))


@needs_display
def test_not_in_db_when_abonent_found_asks_photo_serial(app, monkeypatch):
    # обработка прочитала номер абонента A1 неуверенно — окно его показывает и находит A1
    _run_read(app, "1234567", outcome="SERIAL_LOW_CONF")
    scr = _open_first(app)
    assert scr._account_var.get() == "A1"
    asked = _ask_serial(scr, monkeypatch, "99999")         # на фото другой номер
    scr._not_in_db()
    ((prompt, suggested),) = asked
    assert "л/с A1" in prompt and "1234567" in prompt and suggested == ""   # номер из таблицы не предлагается
    assert app.asked == []                                 # второго вопроса нет
    (r,) = _manual(app)
    assert (r["outcome"], r["serial_id"], r["account_id"]) == ("NOT_IN_DB", "99999", "")
    assert r["photo_hash"] == "h-IMG-20260915-WA0001.jpg"
    assert (app.photos / "not_in_db" / "IMG-20260915-WA0001.jpg").exists()
    assert _next_run(app) == RunSelection.SKIP_DONE        # с номером A1 фото вернулось бы в очередь


@needs_display
def test_not_in_db_found_abonent_cancel_or_table_serial_changes_nothing(app, monkeypatch):
    _run_read(app, "1234567", outcome="SERIAL_LOW_CONF")
    scr = _open_first(app)
    _ask_serial(scr, monkeypatch, None)                     # Cancel
    scr._not_in_db()
    _ask_serial(scr, monkeypatch, "  ")                     # пусто
    scr._not_in_db()
    _ask_serial(scr, monkeypatch, "7654321")                # этот номер в таблице — у A2
    scr._not_in_db()
    assert _manual(app) == []
    ((kind, (title, text)),) = [(k, a[:2]) for k, a in app.shown]
    assert kind == "showwarning" and "7654321" in text and "л/с A2" in text and "«Принять»" in text
    assert (app.photos / "question" / "serial_not_found" / "IMG-20260915-WA0001.jpg").exists()
    assert app._current_screen is scr


@needs_display
def test_not_in_db_writes_run_serial(app):
    _run_read(app, "99999")                                 # номера нет в таблице
    scr = _open_first(app)
    scr._not_in_db()
    assert "Номер счётчика: 99999\n" in app.asked[0][1]
    (r,) = _manual(app)
    assert (r["serial_id"], r["account_id"]) == ("99999", "")


@needs_display
def test_not_in_db_empty_field_writes_run_serial(app):
    _run_read(app, "99999")
    scr = _open_first(app)
    scr._serial_var.set("")                                 # оператор стёр номер
    scr._not_in_db()
    assert "Номер счётчика: 99999 (прочитан при обработке фото)" in app.asked[0][1]
    assert _manual(app)[0]["serial_id"] == "99999"


@needs_display
def test_not_in_db_writes_operator_serial_without_previous_abonent(app):
    _run_read(app, "1234567", outcome="SERIAL_LOW_CONF")    # окно нашло A1
    scr = _open_first(app)
    scr._serial_var.set("55555")                            # оператор: на фото другой номер
    scr._not_in_db()                                        # поиск по нему — сразу, до решения
    text = app.asked[0][1]
    assert "Номер счётчика: 55555\n" in text and "Лицевой счёт: —" in text
    (r,) = _manual(app)
    assert (r["serial_id"], r["account_id"]) == ("55555", "")    # счёт A1 — от прежнего номера


# ─── Этап Е: окно показывает, что прочитала обработка (решение владельца 2026-11-21) ─

def _jpg(h, w, value):
    return cv2.imencode(".jpg", np.full((h, w, 3), value, np.uint8))[1].tobytes()


def _seen(app, name="IMG-20260915-WA0001.jpg", **kw):
    """Обработка записала в базу, что прочитали модели на фото name."""
    with MonthDB(app.month.db) as db, db.transaction():
        db.put_recognition(f"h-{name}", Recognition(**kw))


@needs_display
def test_window_shows_what_processing_read(app):
    _run_read(app, "1234567", outcome="DIGITS_ERROR", model_reading_str="01?50")
    _seen(app, serial_read="1234567", serial_conf=0.97, reading_str="01?50",
          digits_ok=[True, True, False, True, False])      # последняя цифра подставлена настройкой
    scr = _open_first(app)
    assert scr._serial_var.get() == "1234567" and scr._account_var.get() == "A1"
    assert scr._reading_widget.get_string() == "01?5?"   # неуверенные цифры — «?», как раньше в окне
    assert scr._status_var.get() == "✓ Прочитано при обработке фото"
    assert "Номер: обработка (0.97)" in scr._source_var.get()
    _fill(scr, "A1", "01253")
    scr._accept()
    r = _manual(app)[-1]
    assert (r["model_serial_conf"], r["model_reading_str"]) == ("0.9700", "01?50")


@needs_display
def test_window_photo_processed_before_stage_e(app):
    # в базе нет записи моделей (фото обработано раньше) — номер и показание из строки лога
    _run_read(app, "7654321", outcome="DIGITS_ERROR", model_reading_str="00?00", model_serial_conf="0.8800")
    scr = _open_first(app)
    assert (scr._serial_var.get(), scr._account_var.get()) == ("7654321", "A2")
    assert scr._reading_widget.get_string() == "00?00"
    assert "Номер: обработка (0.88)" in scr._source_var.get()


@needs_display
def test_window_serial_read_without_meter(app):
    # счётчик не найден: у строки обработки номера нет, модель номер прочитала
    _seen(app, serial_read="7654321", serial_conf=0.9)
    scr = _open_first(app)
    assert (scr._serial_var.get(), scr._account_var.get()) == ("7654321", "A2")


@needs_display
def test_window_nothing_read(app):
    scr = _open_first(app)                                   # строка обработки без номера и показания
    assert scr._status_var.get() == "⚠ Обработка ничего не прочитала — введите вручную"
    assert scr._serial_var.get() == "" and scr._reading_widget.get_string() == "?????"


@needs_display
def test_window_does_not_start_models(app):
    assert not hasattr(app, "models") and not hasattr(app, "_load_models_async")


def test_window_imports_no_models():
    # окно оператора не загружает torch и модели: запуск быстрее, памяти меньше
    import subprocess
    import sys
    code = ("import sys, program2; "
            "print(sorted(m for m in ('torch', 'ultralytics', 'src.gmr.ml.loader', 'src.gmr.pipeline') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=str(Path(__file__).resolve().parent.parent),
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


# ─── Выгрузка и закрытие окна ────────────────────────────────────────────────

@needs_display
def test_export_button(app):
    scr = _open_first(app)
    _fill(scr, "A1", "01200")
    scr._accept()
    assert app.export_as("xlsx").path == str(app.month.export_xlsx)
    assert app.month.export_xlsx.is_file() and app.month.export_log.is_file()


@needs_display
def test_export_dialog_formats_and_remembers_choice(app, monkeypatch):
    # 1.3.0: «Выгрузить» спрашивает формат; выбранный запоминается в настройках
    from src.gmr.operator.dialogs import ExportDialog
    saved, opened = [], []
    monkeypatch.setattr(op_window, "save_settings", saved.append)
    monkeypatch.setattr(app, "_open_export", lambda path, folder=False: opened.append((path, folder)))

    def choose_and_export():
        dlg = next(w for w in app.winfo_children() if isinstance(w, ExportDialog))
        assert dlg._fmt.get() == "xlsx"                  # по умолчанию — обычный Excel
        dlg._fmt.set("csv")
        dlg._go()                                       # «Выгрузить»
        dlg._go()                                       # Enter ещё раз — закрыть
    app.after(50, choose_and_export)
    assert app.export() is True
    assert (app.month.root / "показания.csv").is_file() and app.month.export_xlsx.is_file()
    assert app.settings.export_format == "csv" and saved[-1].export_format == "csv"
    app.after(50, lambda: next(w for w in app.winfo_children() if isinstance(w, ExportDialog)).destroy())
    assert app.export() is False                        # «Отмена» — ничего


@needs_display
def test_close_exports(app):
    app._on_close()
    assert app.month.export_xlsx.is_file()


@needs_display
def test_close_with_export_open_in_excel(app, monkeypatch):
    def locked():
        raise program2.ExportLocked("Файл открыт в Excel: показания.xlsx. Закройте его")
    monkeypatch.setattr(app.session, "export", locked)
    asked = []
    monkeypatch.setattr(program2.messagebox, "askretrycancel", lambda *a, **k: asked.append(a) or False)
    app._on_close()                                               # «Отмена» — выйти без выгрузки
    assert len(asked) == 1 and not app.month.export_xlsx.exists()


@needs_display
def test_not_a_month_folder_asks_settings_again(tmp_path, monkeypatch):
    f = make_month(tmp_path, [("1234567", "A1", "1000", "")])
    asked = []

    def settings_dialog(parent, settings, **kw):
        asked.append(settings.month_dir)
        assert kw["new_month"] == parent.ask_new_month         # в настройках есть «Новый месяц…»
        return SimpleNamespace(result=program2.AppSettings("Оператор", str(f.root)))

    monkeypatch.setattr(op_window, "SettingsDialog", settings_dialog)
    # «Создать в этой папке новый месяц?» — Нет: окно снова спрашивает папку
    w = _window(monkeypatch, program2.AppSettings("Оператор", str(tmp_path / "нет")), answers=[False])
    try:
        assert asked == [str(tmp_path / "нет")]
        assert "Месяц не создан" in w.asked[0][1] and "Создать в этой папке" in w.asked[0][1]
        assert w.session.folder.root == f.root
    finally:
        w.destroy()


@needs_display
def test_change_user_cancelled_keeps_settings(tmp_path, monkeypatch):
    f = make_month(tmp_path, [("1234567", "A1", "1000", "")])
    monkeypatch.setattr(op_window, "SettingsDialog", lambda parent, settings, **kw: SimpleNamespace(result=None))
    w = _window(monkeypatch, program2.AppSettings("Оператор", str(f.root)))
    monkeypatch.setattr(op_window, "ask_login", lambda *a, **k: "change")
    w.destroy()
    w = new_window(program2.MainWindow)            # «Сменить пользователя» → «Отмена»
    try:
        w.withdraw()
        assert w.session.folder.root == f.root and w.settings.operator_name == "Оператор"
    finally:
        w.destroy()


@needs_display
def test_window_start_backs_up_month_db(app):
    assert list(app.month.backups.glob("*/gmr.sqlite"))


# ─── Этап 4: «Обработать новые» из окна ──────────────────────────────────────
# Настоящий прогон — отдельный процесс `gmr.py process` с моделями; здесь вместо
# него скрипт, который печатает те же строки (формат строк — общий с reader.py,
# см. tests/test_month_cycle.py::test_progress_lines_parse_in_window).

import sys  # noqa: E402

from src.gmr.ui import run_dialog  # noqa: E402

FAKE_RUN = r"""
import sys, time
from pathlib import Path
stop, results = Path(sys.argv[1]), Path(sys.argv[2])
print("Загружаем модели...", flush=True)
print("Новых фото к чтению: 3", flush=True)
for k, name in enumerate(["a.jpg", "b.jpg", "c.jpg"], 1):
    for _ in range(int(sys.argv[3]) * 50):            # «долгое» фото: ждём «Остановить»
        if stop.exists():
            break
        time.sleep(0.02)
    if stop.exists():
        print(f"WARNING  ⏹ Остановлено оператором: прочитано {k - 1} из 3.", flush=True)
        stop.unlink()
        break
    print(f"INFO  Фото {k} из 3: {name}", flush=True)
results.mkdir(parents=True, exist_ok=True)
if len(sys.argv) > 5:                                  # прогон положил фото в очередь
    Path(sys.argv[6]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[6]).write_bytes(Path(sys.argv[5]).read_bytes())
(results / "report.txt").write_text("ОБЩИЙ ОТЧЁТ\nНовых фото в этом прогоне: 3\n", encoding="utf-8")
sys.exit(int(sys.argv[4]))
"""


def _fake_run(monkeypatch, app, slow=0, code=0, calls=None, new_photo=()):
    def cmd(project, month, reread=False):
        if calls is not None:
            calls.append((month, reread))
        return [sys.executable, "-c", FAKE_RUN, str(app.month.stop_file), str(app.month.results),
                str(slow), str(code), *map(str, new_photo)]
    monkeypatch.setattr(op_window, "process_command", cmd)
    shown = []
    monkeypatch.setattr(op_window, "show_text", lambda parent, title, text: shown.append((title, text)))
    return shown


def test_parse_progress():
    assert run_dialog.parse_progress("12:00  INFO  Новых фото к чтению: 40") == ("total", 40)
    assert run_dialog.parse_progress("INFO  Фото 3 из 40: IMG 1.jpg  (читается заново)") == (
        "photo", 3, 40, "IMG 1.jpg  (читается заново)")
    assert run_dialog.parse_progress("WARNING ⏹ Остановлено оператором: прочитано 1 из 3") == ("stopped",)
    assert run_dialog.parse_progress("INFO  Загружаем модели...") == ("models",)
    assert run_dialog.parse_progress("INFO  Фото в папке: 12") is None


def test_process_command(tmp_path, monkeypatch):
    cmd = run_dialog.process_command(tmp_path, "D:/Май")
    assert cmd[1:] == [str(tmp_path / "gmr.py"), "process", "D:/Май"]
    assert run_dialog.process_command(tmp_path, "D:/Май", reread=True)[-1] == "--reread"
    # ярлык запускает окно через pythonw.exe — прогону нужен python.exe рядом
    (tmp_path / "pythonw.exe").write_bytes(b"")
    (tmp_path / "python.exe").write_bytes(b"")
    monkeypatch.setattr(run_dialog.sys, "executable", str(tmp_path / "pythonw.exe"))
    assert run_dialog.process_command(tmp_path, "D:/Май")[0] == str(tmp_path / "python.exe")


@needs_display
def test_process_new_shows_report_and_refreshes(app, monkeypatch):
    calls = []
    img(app.month.root.parent / "new.jpg")
    shown = _fake_run(monkeypatch, app, calls=calls,       # «прогон» добавит фото в очередь
                      new_photo=(app.month.root.parent / "new.jpg", app.photos / "question" / "no_meter" / "new.jpg"))
    _open_first(app)                                       # экран разбора открыт
    app.process_new()
    assert calls == [(str(app.month.root), False)]
    assert not hasattr(app, "_current_screen") or not app._current_screen.winfo_exists()
    assert shown == [("Итог обработки", "ОБЩИЙ ОТЧЁТ\nНовых фото в этом прогоне: 3\n")]
    assert len(app._proc_tab._items) == 4                  # очередь перечитана после прогона
    lines = app._process_dialog.lines
    assert "INFO  Фото 3 из 3: c.jpg" in lines and not app._process_dialog.winfo_exists()
    app._process_dialog.stop()                             # «Остановить» уже после конца — ничего
    assert not app.month.stop_file.exists()


@needs_display
def test_reread_runs_with_reread(app, monkeypatch):
    calls, titles = [], []
    _fake_run(monkeypatch, app, calls=calls)
    real = program2.ProcessDialog
    monkeypatch.setattr(op_window, "ProcessDialog", lambda *a, **k: titles.append(k["title"]) or real(*a, **k))
    app.process_new(reread=True)
    assert calls == [(str(app.month.root), True)] and titles == ["Обработка фото с ошибками"]


@needs_display
def test_process_new_stop(app, monkeypatch):
    shown = _fake_run(monkeypatch, app, slow=1)
    app.after(400, lambda: app._process_dialog.stop())     # оператор нажал «Остановить»
    app.process_new()
    assert any("Остановлено оператором: прочитано 0 из 3" in x for x in app._process_dialog.lines)
    assert not app.month.stop_file.exists() and shown[0][0] == "Итог обработки"


@needs_display
def test_process_new_failure_shows_output(app, monkeypatch):
    shown = _fake_run(monkeypatch, app, code=1)
    app.process_new()
    (title, text), = shown
    assert title == "Обработка не удалась" and "Код завершения: 1" in text and "Фото 3 из 3" in text


# ─── Этап 4: меню «Месяц» ────────────────────────────────────────────────────

from src.gmr.application import month as month_app  # noqa: E402
from tests._month import TABLE_HEAD  # noqa: E402


def _table_csv(path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(TABLE_HEAD)
        w.writerows(rows)
    return path


def _dialogs(monkeypatch, files=(), dirs=(), saves=()):
    files, dirs, saves = list(files), list(dirs), list(saves)
    monkeypatch.setattr(program2.filedialog, "askopenfilename", lambda **k: str(files.pop(0)) if files else "")
    monkeypatch.setattr(program2.filedialog, "askdirectory", lambda **k: str(dirs.pop(0)) if dirs else "")
    monkeypatch.setattr(program2.filedialog, "asksaveasfilename", lambda **k: str(saves.pop(0)) if saves else "")
    shown = []
    monkeypatch.setattr(op_window, "show_text", lambda parent, title, text: shown.append((title, text)))
    return shown


@needs_display
def test_first_run_creates_month_from_window(tmp_path, monkeypatch):
    table = _table_csv(tmp_path / "компания" / "Май.csv", [("1234567", "A1", "1000", "")])
    shown = _dialogs(monkeypatch, files=[table])
    folder = tmp_path / "Май_2026"
    w = _window(monkeypatch, program2.AppSettings("Оператор", str(folder)), answers=[True])
    try:
        assert w.session.folder.root == folder and (folder / "gmr.sqlite").is_file()
        assert shown[0][0] == "Месяц создан" and "МЕСЯЦ СОЗДАН" in shown[0][1]
        assert (folder / "показания.xlsx").is_file() and (folder / "фото").is_dir()
    finally:
        w.destroy()


@needs_display
def test_new_month_from_menu_switches_window(app, tmp_path, monkeypatch):
    table = _table_csv(tmp_path / "компания" / "Июнь.csv", [("7777777", "B1", "10", "")])
    shown = _dialogs(monkeypatch, files=[table], saves=[tmp_path / "Июнь_2026"])
    saved = []
    monkeypatch.setattr(op_window, "save_settings", lambda s: saved.append(s.month_dir))
    app.new_month()
    assert app.session.folder.root == tmp_path / "Июнь_2026"
    assert saved == [str(tmp_path / "Июнь_2026")] and "Июнь_2026" in app._month_var.get()
    assert list(app.df["Лицевой счет"]) == ["B1"] and app._proc_tab._items == []
    assert shown[0][0] == "Месяц создан"


@needs_display
def test_new_month_refuses_existing_month(app, tmp_path, monkeypatch):
    table = _table_csv(tmp_path / "компания" / "Июнь.csv", [("7777777", "B1", "10", "")])
    shown = _dialogs(monkeypatch, files=[table], saves=[app.month.root])
    app.new_month()
    assert shown == []
    assert any(t == "showerror" and "месяц уже есть" in a[1] for t, a in app.shown)
    assert list(app.df["Лицевой счет"]) == ["A1", "A2"]


@needs_display
@pytest.mark.parametrize("answer,rows", [(False, 0), (True, 1)])
def test_new_month_asks_about_old_log(app, tmp_path, monkeypatch, answer, rows):
    table = _table_csv(tmp_path / "компания" / "Июнь.csv", [("7777777", "B1", "10", "")])
    from src.gmr.storage import LOG_COLUMNS as cols, save_log
    save_log(str(tmp_path / "компания" / "Июнь_log.csv"),
             [{c: "" for c in cols} | {"original_filename": "old.jpg", "outcome": "PLUS"}])
    _dialogs(monkeypatch, files=[table], saves=[tmp_path / "Июнь_2026"])
    app.answers.append(answer)
    app.new_month()
    assert "старой версии программы" in app.asked[0][1]
    assert len(log_rows(MonthFolderOf(tmp_path / "Июнь_2026"))) == rows


def MonthFolderOf(path):
    from src.gmr.storage.month import MonthFolder
    return MonthFolder(path)


@needs_display
def test_update_table_from_menu(app, tmp_path, monkeypatch):
    table = _table_csv(tmp_path / "компания" / "новая.csv",
                       [("1234567", "A1", "1000", ""), ("7654321", "A2", "500", ""), ("5555555", "A3", "1", "")])
    shown = _dialogs(monkeypatch, files=[table, table])
    app.answers.append(False)                              # «Загрузить таблицу?» — Нет
    app.update_table()
    assert list(app.df["Лицевой счет"]) == ["A1", "A2"] and shown == []
    _open_first(app)                                       # экран разбора закрывается
    app.update_table()
    assert not app.screen_open()
    assert list(app.df["Лицевой счет"]) == ["A1", "A2", "A3"]
    assert shown[0][0] == "Таблица обновлена" and "Выгрузка" in shown[0][1]
    assert list(app.month.backups.glob("*/gmr.sqlite"))


def _all_children(w):
    for c in w.winfo_children():
        yield c
        yield from _all_children(c)


@needs_display
def test_summary_and_choose_month(app, tmp_path, monkeypatch):
    other = make_month(tmp_path / "другой", [("9999999", "C1", "5", "")], name="Июль")
    _dialogs(monkeypatch, dirs=[other.root, tmp_path / "пусто"])
    app.show_summary()
    # 1.3.0: итог — экраном с полосами в окне; текст — в итог_месяца.txt
    from src.gmr.operator.summary_screen import SummaryScreen
    scr = app._current_screen
    assert isinstance(scr, SummaryScreen) and scr.st.abonents == 2
    txt = (app.month.root / month_app.SUMMARY_FILE).read_text(encoding="utf-8-sig")
    assert month_app.month_summary(str(app.month.root)) in txt
    scr.update()
    from src.gmr.operator.summary_screen import Chart
    charts = [w for w in _all_children(scr) if isinstance(w, Chart)]
    assert len(charts) == 4
    for chart in charts:
        chart.redraw()                                     # рисунки не падают
    assert app.bind_all("<Escape>")                         # Esc — к спискам
    scr._back()
    assert not app.screen_open() and not app.bind_all("<Escape>")
    app.choose_month()
    assert app.session.folder.root == other.root
    (tmp_path / "пусто").mkdir()
    session = app.session
    app.answers.append(False)                              # «Создать месяц здесь?» — Нет
    app.choose_month()
    assert app.session is session                          # открытый месяц не трогается
    asked = len(app.asked)
    _dialogs(monkeypatch, dirs=[tmp_path / "пусто"])       # «Да», но таблицу не выбрали
    app.choose_month()
    assert app.session.folder.root == other.root and len(app.asked) == asked + 1
    assert app.settings.month_dir == str(other.root)


@needs_display
def test_month_that_does_not_open_keeps_window_in_old_month(app, tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr(op_window, "save_settings", lambda s: saved.append(s.month_dir))
    app.answers.append(False)                              # не месяц и создавать не нужно
    app._switch_month(str(tmp_path / "пусто"))
    assert app.session.folder.root == app.month.root and saved == [str(app.month.root)]
    assert app.settings.month_dir == str(app.month.root)


@needs_display
def test_reread_errors_from_menu(app, monkeypatch):
    calls = []
    monkeypatch.setattr(app, "process_new", lambda reread=False: calls.append(reread))
    app.answers.append(False)
    app.reread_errors()
    app.reread_errors()
    assert calls == [True]


def test_window_log_goes_to_file_without_console():
    # окно с ярлыка (pythonw.exe): консоли нет — журнал в program2.log папки программы
    assert program2._log_handlers_for(sys.stderr) is None
    (h,) = program2._log_handlers_for(None)
    try:
        assert Path(h.baseFilename) == program2._BASE / "program2.log"
        assert h.maxBytes == 1_000_000 and h.backupCount == 3
    finally:
        h.close()


def test_open_in_explorer(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(os, "startfile", opened.append, raising=False)
    program2.open_in_explorer(tmp_path)
    assert opened == [str(tmp_path)]

    def broken(path):
        raise OSError("нет программы для папок")
    monkeypatch.setattr(os, "startfile", broken, raising=False)
    program2.open_in_explorer(tmp_path)                    # ошибка не роняет окно


@needs_display
def test_error_in_button_shown_to_operator(app):
    app.report_callback_exception(ValueError, ValueError("сломалось"), None)
    assert any(t == "showerror" and "ValueError: сломалось" in a[1] and "Подробности" in a[1]
               for t, a in app.shown)


# ─── 2026-10-07: настройки без «папки для разметки», новый месяц из окна ────

@needs_display
def test_markup_of_etalon_photo_not_saved(app, monkeypatch):
    monkeypatch.setattr(op_edit, "to_etalon", lambda h: h == "h-IMG-20260915-WA0001.jpg")
    _seen_wrong_model(app)
    scr = _open_first(app)
    _fill(scr, "A1", "01200")
    scr._accept()
    assert not app.month.markup.exists()                      # фото эталона — не в обучение


@needs_display
def test_no_meter_accept_copies_original_for_labeling(app):
    orig = img(app.month.photos / "Аюб" / "p3.jpg", 77)          # исходное фото контролёра
    app.open_edit_screen(str(app.photos / "question" / "no_meter" / "p3.jpg"), "no_meter")
    scr = app._current_screen
    _fill(scr, "A2", "00600")
    scr._accept()
    new = app.month.markup / "meter_yolo"
    assert [p.name for p in new.iterdir()] == ["A2.jpg"]                # фото — лицевой счёт
    assert [p.read_bytes() for p in new.iterdir()] == [orig.read_bytes()]


@needs_display
def test_settings_dialog_name_and_month(app, tmp_path):
    calls = []

    def new_month(parent, who):
        calls.append(who)
        return str(tmp_path / "Ноябрь_2026")

    def fill():
        dlg = next(w for w in app.winfo_children() if isinstance(w, program2.SettingsDialog))
        dlg._vars["name"].set("Мадина")
        dlg._save()                                            # папки месяца нет — не закрывается
        assert dlg.winfo_exists() and dlg.result is None
        dlg._create()
        dlg._vars["month"].set(f"  {dlg._vars['month'].get()}  ")   # лишние пробелы не мешают
        dlg._save()
    app.after(100, fill)
    dlg = program2.SettingsDialog(app, program2.AppSettings("", ""), new_month=new_month)
    assert calls == ["Мадина"]
    assert dlg.result == program2.AppSettings("Мадина", str(tmp_path / "Ноябрь_2026"))
    assert any(t == "showwarning" and "Новый месяц" in a[1] for t, a in app.shown)


@needs_display
def test_ask_new_month_like_save_as(app, tmp_path, monkeypatch):
    table = _table_csv(tmp_path / "компания" / "Ноябрь.csv", [("7777777", "B1", "10", "")])
    asked = []
    files, saves = [table, "", table], [str(tmp_path / "Ноябрь_2026"), ""]
    monkeypatch.setattr(program2.filedialog, "askopenfilename", lambda **k: str(files.pop(0)))
    monkeypatch.setattr(program2.filedialog, "asksaveasfilename",
                        lambda **k: asked.append(k) or saves.pop(0))
    monkeypatch.setattr(op_window, "show_text", lambda *a: None)
    assert app.ask_new_month(who="Мадина") == str(tmp_path / "Ноябрь_2026")
    assert asked[0]["initialfile"] == program2.default_month_name()
    assert asked[0]["initialdir"] == str(app.month.root.parent)    # рядом с текущим месяцем
    assert (tmp_path / "Ноябрь_2026" / "gmr.sqlite").is_file()
    assert app.ask_new_month() is None                         # таблицу не выбрали
    assert app.ask_new_month() is None and len(asked) == 2     # папку не выбрали
    files.append(table)
    saves.append(str(app.month.root))                          # там уже месяц — не создаётся
    assert app.ask_new_month() is None


@needs_display
@pytest.mark.parametrize("clear,readings", [(True, 0), (False, 1)])
def test_new_month_asks_to_clear_table_readings(app, tmp_path, monkeypatch, clear, readings):
    table = _table_csv(tmp_path / "компания" / "Ноябрь.csv",
                       [("7777777", "B1", "10", "15"), ("8888888", "B2", "20", "")])
    _dialogs(monkeypatch, files=[table], saves=[tmp_path / "Ноябрь_2026"])
    app.answers.append(clear)
    app.new_month()
    assert "В таблице уже есть показания у 1 абонентов" in app.asked[0][1]
    assert len(readings_of(tmp_path / "Ноябрь_2026")) == readings
    assert app.session.folder.root == tmp_path / "Ноябрь_2026"


def readings_of(path):
    from tests._month import readings
    return readings(MonthFolderOf(path))


# ─── 2026-10-11: настройки распознавания месяца (решение 2а) ─────────────────

from src.gmr.domain.preset import RecognitionPreset  # noqa: E402


@needs_display
def test_preset_dialog(app):
    def fill():
        dlg = next(w for w in app.winfo_children() if isinstance(w, program2.PresetDialog))
        dlg._vars["missing_digit"].set("operator")
        dlg._vars["forgive_last"].set(1)
        dlg._vars["serial_pad"].set(0.05)
        dlg._vars["account_marker"].set(True)               # модели нет — галочка не сработает
        dlg._vars["drum_rule"].set(True)
        dlg._vars["serial_by_table"].set(True)
        dlg._vars["turn_photo"].set(True)
        dlg._vars["overlay_notes"].set(True)
        dlg._save()
    app.after(100, fill)
    dlg = program2.PresetDialog(app, RecognitionPreset(forgive_with="model"))
    assert dlg.result == RecognitionPreset("operator", 1, "model", 0.05, False, drum_rule=True,
                                           serial_by_table=True, turn_photo=True, overlay_notes=True)
    app.after(100, lambda: next(w for w in app.winfo_children()
                                if isinstance(w, program2.PresetDialog))._save())
    with_model = program2.PresetDialog(app, RecognitionPreset(account_marker=True, first_from_last=True,
                                                              turn_photo=True), marker_model=True)
    assert with_model.result == RecognitionPreset(account_marker=True, first_from_last=True, turn_photo=True)
    app.after(100, lambda: next(w for w in app.winfo_children()
                                if isinstance(w, program2.PresetDialog)).destroy())     # «Отмена»
    assert program2.PresetDialog(app, RecognitionPreset()).result is None


@needs_display
def test_preset_dialog_tabs_confidence_and_standard(app):
    # 2026-11-21: «Обычные» и «Расширенные», уверенность каждой модели, «Вернуть стандартные»
    def run(action):
        app.after(100, lambda: action(next(w for w in app.winfo_children() if isinstance(w, program2.PresetDialog))))

    def change(dlg):
        assert [dlg._tabs.tab(i, "text") for i in dlg._tabs.tabs()] == ["Обычные", "Расширенные"]
        dlg._vars["serial_conf"].set("0,55")                      # запятая — тоже число
        dlg._vars["account_conf"].set("0.95")
        dlg._save()
    run(change)
    dlg = program2.PresetDialog(app, RecognitionPreset(), marker_model=True)
    assert (dlg.result.serial_conf, dlg.result.account_conf, dlg.result.meter_conf) == (0.55, 0.95, 0.7)

    def bad(dlg):
        dlg._vars["digit_conf"].set("2")
        dlg._save()                                               # не сохраняет: вне 0,05–0,99
        assert dlg.winfo_exists() and dlg._tabs.index("current") == 1 and app.shown[-1][0] == "showwarning"
        dlg.destroy()
    run(bad)
    assert program2.PresetDialog(app, RecognitionPreset()).result is None

    def standard(dlg):
        dlg._fill(RecognitionPreset.new_month())                  # «Вернуть стандартные настройки»
        dlg._save()
    run(standard)
    old = RecognitionPreset(missing_digit="operator", serial_conf=0.4, overlay_notes=False)
    assert program2.PresetDialog(app, old, marker_model=True).result == RecognitionPreset.new_month()


def test_ask_preset_marker_only_with_model(monkeypatch):
    seen = []
    monkeypatch.setattr(op_window, "PresetDialog", lambda *a, **k: seen.append(k) or SimpleNamespace(result="r"))
    monkeypatch.setattr(op_window, "PipelineConfig", lambda: SimpleNamespace(account_ocr_model=""))
    assert program2.MainWindow.ask_preset(None, RecognitionPreset(), "t") == "r"
    monkeypatch.setattr(op_window, "PipelineConfig", lambda: SimpleNamespace(account_ocr_model="acc.pt"))
    program2.MainWindow.ask_preset(None, RecognitionPreset(), "t", ok_text="Дальше")
    assert [(k["marker_model"], k["ok_text"]) for k in seen] == [(False, "Сохранить"), (True, "Дальше")]


@needs_display
def test_new_month_gets_preset_of_open_month(app, tmp_path, monkeypatch):
    month_app.set_month_preset(app.month.root, RecognitionPreset(forgive_last=0))
    offered = []
    monkeypatch.setattr(program2.MainWindow, "ask_preset", staticmethod(
        lambda parent, preset, title, ok_text="Сохранить": offered.append((preset, ok_text))
        or RecognitionPreset(missing_digit="model")))
    table = _table_csv(tmp_path / "компания" / "Декабрь.csv", [("7777777", "B1", "10", "")])
    _dialogs(monkeypatch, files=[table], saves=[tmp_path / "Декабрь_2026"])
    app.new_month()
    assert offered == [(RecognitionPreset(forgive_last=0), "Дальше")]   # как в прошлом месяце
    assert month_app.month_preset(tmp_path / "Декабрь_2026") == RecognitionPreset(missing_digit="model")
    assert app.config.missing_digit_mode == "model"           # окно читает фото с настройками месяца


@needs_display
def test_new_month_cancelled_on_preset(app, tmp_path, monkeypatch):
    monkeypatch.setattr(program2.MainWindow, "ask_preset",
                        staticmethod(lambda parent, preset, title, ok_text="Сохранить": None))
    table = _table_csv(tmp_path / "компания" / "Декабрь.csv", [("7777777", "B1", "10", "")])
    _dialogs(monkeypatch, files=[table], saves=[tmp_path / "Декабрь_2026"])
    app.new_month()
    assert not (tmp_path / "Декабрь_2026" / "gmr.sqlite").exists() and app.session.folder.root == app.month.root


@needs_display
def test_edit_preset_from_menu(app, monkeypatch):
    answers = [None, RecognitionPreset(), RecognitionPreset(forgive_last=1, forgive_with="model")]
    offered = []
    monkeypatch.setattr(program2.MainWindow, "ask_preset", staticmethod(
        lambda parent, preset, title, ok_text="Сохранить": offered.append(preset) or answers.pop(0)))
    app.edit_preset()                                          # «Отмена»
    app.edit_preset()                                          # без изменений
    assert month_app.month_preset(app.month.root) is None
    app.edit_preset()
    assert offered == [RecognitionPreset()] * 3
    assert month_app.month_preset(app.month.root) == RecognitionPreset(forgive_last=1, forgive_with="model")
    assert (app.config.ignore_last_digits, app.config.forgiven_digit_mode) == (1, "model")
    assert [c["who"] for c in changes(app.month) if c["action"] == month_app.PRESET_ACTION] == ["Оператор"]
    assert "Настройки распознавания…" in [app._month_menu.entrycget(i, "label")
                                         for i in range(app._month_menu.index("end") + 1)
                                         if app._month_menu.type(i) == "command"]


# ─── Код участка контролёра (area_code.py) ───────────────────────────────────

@needs_display
def test_serial_of_other_area_is_marked(tmp_path, monkeypatch):
    # у «Аюб» найдено 10 счетов кода 1300 — номер, ведущий к счёту 0429…, оранжевый с пояснением
    table = [(f"10000{i:02d}", f"13000000{i:02d}", "100", "") for i in range(1, 11)]
    table.append(("7654321", "0429000001", "500", ""))
    done = [{"original_filename": f"d{i}.jpg", "outcome": "PLUS", "source": "auto", "processed_by": "auto",
             "account_id": f"13000000{i:02d}", "source_folder": "Аюб"} for i in range(1, 11)]
    f = make_month(tmp_path, table, AUTO + done)
    img(f.results / "Аюб" / "question" / "serial_not_found" / "IMG-20260915-WA0001.jpg", 10)
    app = _window(monkeypatch, program2.AppSettings(operator_name="Оператор", month_dir=str(f.root)))
    try:
        assert app.controller_code("Аюб") == "1300" and app.controller_code("Фатима Я") is None
        scr = _open_first(app)
        scr._lookup_by_serial("7654321")
        assert scr._serial_match_var.get() == ("⚠ есть в таблице: л/с 0429000001, но счёт не с участка "
                                               "контролёра: код 0429, у контролёра 1300")
        scr._lookup_by_serial("1000001")
        assert scr._serial_match_var.get() == "✓ есть в таблице: л/с 1300000001"
    finally:
        app.destroy()


@needs_display
def test_queue_next_photo_keeps_hotkeys_and_one_screen(app):
    # после «Нечитаемо» следующее фото открывалось без горячих клавиш (их снимал
    # прежний экран), а Enter в списке под экраном открывал второй экран
    app.answers.extend([True, True])
    scr = _open_first(app)
    app._proc_tab._open()                                    # Enter, дошедший до списка
    assert app._current_screen is scr
    scr._unreadable()
    nxt = app._current_screen
    assert nxt is not scr and app._keys_owner is nxt and app.bind_all("<Delete>")
    from src.gmr.operator.edit_screen import EditScreen
    assert [c for c in app.winfo_children() if isinstance(c, EditScreen) and c.winfo_manager()] == [nxt]


@needs_display
def test_login_enter_continues(monkeypatch):
    # вход при каждом запуске: Enter = «Продолжить» (раньше — только мышью);
    # с 1.3.0 — экраном в главном окне, а не отдельным окошком «Вход»
    import tkinter as tk
    from src.gmr.operator.welcome import WelcomeScreen
    root = new_window(tk.Tk)
    root.withdraw()
    scr = WelcomeScreen(root)
    seen = []

    def press_enter():
        seen.append((bool(root.bind("<Return>")), root.winfo_toplevel() is root, len(root.winfo_children())))
        scr._on_return()
    root.after(50, press_enter)
    action = scr.ask(program2.AppSettings(operator_name="Оператор", month_dir="D:/GMR/Май_2026"))
    assert action == "continue" and seen == [(True, True, 1)]      # одно окно, Enter привязан
    assert not root.bind("<Return>")                                # после входа Enter — спискам
    root.after(50, scr._quit)                                       # крестик окна на экране входа
    assert scr.ask(program2.AppSettings(operator_name="Оператор", month_dir="")) == "quit"
    root.destroy()
