"""
Вкладка «Обработка» program2.py: что пишется в базу месяца, куда уходит фото,
какое фото открывается следующим. С этапа 2.3 окно работает с папкой месяца:
все контролёры сразу, показание и строка лога — одной записью в базу
(src/gmr/application/operator.py).

Оконные тесты — под xvfb-run (без экрана пропускаются).
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

program2 = pytest.importorskip("program2")
from src.gmr.domain import PipelineConfig  # noqa: E402
from src.gmr.storage import LOG_COLUMNS  # noqa: E402
from src.gmr.storage.month import MonthDB  # noqa: E402
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
    program2.save_crnn_markup(str(tmp_path), "p.jpg", crop, "1234567", "1234567")
    program2.save_crnn_markup(str(tmp_path), "p.jpg", None, "1234567", "1284567")
    assert not (tmp_path / "crnn").exists()
    # исправили — кроп + .txt с правильным текстом; повтор — уникальное имя
    program2.save_crnn_markup(str(tmp_path), "p.jpg", crop, "1234567", "1284567")
    program2.save_crnn_markup(str(tmp_path), "p.jpg", crop, "1234567", "1284567")
    imgs = sorted((tmp_path / "crnn" / "images").glob("*.jpeg"))
    assert len(imgs) == 2 and imgs[0].name == "1234567.jpeg"
    assert all(i.with_suffix(".txt").read_text(encoding="utf-8") == "1234567" for i in imgs)


def test_save_cnn_markup_only_changed_digits(tmp_path):
    crops = [np.full((30, 15, 3), v, np.uint8) for v in range(5)]
    crops[1] = None                                            # заглушка — пропускается
    program2.save_cnn_markup(str(tmp_path), crops, [{}] * 5, "12?45", "12745")
    saved = {p.parent.name for p in (tmp_path / "cnn").rglob("*.jpg")}
    assert saved == {"7"}                                      # изменилась только позиция 2
    # длины не совпадают — ничего
    program2.save_cnn_markup(str(tmp_path / "x"), crops, [{}] * 5, "12345", "1234")
    assert not (tmp_path / "x").exists()


def test_error_folder_in_queue_last(tmp_path):
    for sub in ("error", "digits_error"):
        img(tmp_path / "question" / sub / f"{sub}.jpg")
    items = program2.list_question_photos(str(tmp_path))      # (причина, имя, путь)
    assert [(reason, name) for reason, name, _ in items] == \
        [("digits_error", "digits_error.jpg"), ("error", "error.jpg")]
    assert program2.REASON_LABELS["error"] == "Ошибка программы"


def test_old_settings_ask_for_month_folder(tmp_path, monkeypatch):
    # settings.json до этапа 2.3: папка контролёра и таблица — имя и папка разметки остаются
    f = tmp_path / "settings.json"
    f.write_text('{"operator_name": "Оператор", "photos_dir": "output/Аюб", '
                 '"table_path": "t.csv", "training_dir": "train"}', encoding="utf-8")
    monkeypatch.setattr(program2, "SETTINGS_FILE", f)
    s = program2.load_settings()
    assert (s.operator_name, s.month_dir, s.training_dir) == ("Оператор", "", "train")


# ─── Окна: экран обработки ───────────────────────────────────────────────────

AUTO = [  # авто-строки reader.py, создавшие файлы в question/ (отпечаток — в ручную строку)
    {"original_filename": n, "outcome": o, "source": "auto", "processed_by": "auto",
     "photo_hash": f"h-{n}", "source_folder": "Аюб"}
    for n, o in (("IMG-20260915-WA0001.jpg", "SERIAL_NOT_FOUND"), ("p2.jpg", "SERIAL_NOT_FOUND"),
                 ("p3.jpg", "NO_METER"))
]


def _window(monkeypatch, settings):
    monkeypatch.setattr(program2, "load_settings", lambda: settings)
    monkeypatch.setattr(program2, "save_settings", lambda s: None)
    monkeypatch.setattr(program2.MainWindow, "_load_models_async", lambda self: None)
    monkeypatch.setattr(program2, "LoginDialog", lambda *a, **k: type("D", (), {"action": "continue"})())
    answers, shown = [], []
    monkeypatch.setattr(program2.messagebox, "askyesno", lambda *a, **k: answers.pop(0) if answers else True)
    for name in ("showinfo", "showwarning", "showerror"):
        monkeypatch.setattr(program2.messagebox, name, lambda *a, _n=name, **k: shown.append((_n, a)))
    w = new_window(program2.MainWindow)
    w.withdraw()
    w.answers, w.shown = answers, shown
    return w


@pytest.fixture
def app(tmp_path, monkeypatch):
    f = make_month(tmp_path, [("1234567", "A1", "1000", ""), ("7654321", "A2", "500", "")], AUTO)
    photos = f.results / "Аюб"
    q = photos / "question"
    img(q / "serial_not_found" / "IMG-20260915-WA0001.jpg", 10)
    img(q / "serial_not_found" / "p2.jpg", 20)
    img(q / "no_meter" / "p3.jpg", 30)
    w = _window(monkeypatch, program2.AppSettings(
        operator_name="Оператор", month_dir=str(f.root), training_dir=str(tmp_path / "train")))
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
def test_accept_unknown_account_writes_log_only(app):
    scr = _open_first(app)
    scr._serial_var.set("5555555")
    scr._account_var.set("A9")                     # нет в таблице
    scr._reading_widget.set_digits("01200", None)
    scr._update_accept_state()
    scr._accept()
    assert _db_log(app)[-1]["account_id"] == "A9" and readings(app.month) == {}


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


@needs_display
def test_accept_saves_silent_markup_when_model_was_wrong(app):
    scr = _open_first(app)
    scr._model_result = {"serial_text": "1284567", "serial_crop": np.full((20, 60, 3), 5, np.uint8),
                         "reading_str": "01?00", "digit_crops": [np.full((30, 15, 3), 1, np.uint8)] * 5,
                         "digit_preds": [{}] * 5}
    _fill(scr, "A1", "01200")
    scr._accept()
    train = app.month.root.parent / "train"
    assert (train / "crnn" / "images" / "1234567.jpeg").exists()
    assert {p.parent.name for p in (train / "cnn").rglob("*.jpg")} == {"2"}


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


# ─── Выгрузка и закрытие окна ────────────────────────────────────────────────

@needs_display
def test_export_button(app):
    scr = _open_first(app)
    _fill(scr, "A1", "01200")
    scr._accept()
    assert app.export() is True
    assert app.month.export_xlsx.is_file() and app.month.export_log.is_file()


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

    def settings_dialog(parent, settings):
        asked.append(settings.month_dir)
        return SimpleNamespace(result=program2.AppSettings("Оператор", str(f.root), ""))

    monkeypatch.setattr(program2, "SettingsDialog", settings_dialog)
    w = _window(monkeypatch, program2.AppSettings("Оператор", str(tmp_path / "нет"), ""))
    try:
        assert asked == [str(tmp_path / "нет")]
        assert w.shown[0][0] == "showerror" and "Месяц не создан" in w.shown[0][1][1]
        assert w.session.folder.root == f.root
    finally:
        w.destroy()


@needs_display
def test_change_user_cancelled_keeps_settings(tmp_path, monkeypatch):
    f = make_month(tmp_path, [("1234567", "A1", "1000", "")])
    monkeypatch.setattr(program2, "SettingsDialog", lambda parent, settings: SimpleNamespace(result=None))
    w = _window(monkeypatch, program2.AppSettings("Оператор", str(f.root), ""))
    monkeypatch.setattr(program2, "LoginDialog", lambda *a, **k: SimpleNamespace(action="change"))
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
