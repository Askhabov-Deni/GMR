"""
Фаза 6 (docs/MIGRATION_TZ.md): characterization tests вкладки «Обработка»
program2.py — фиксируют ТЕКУЩЕЕ поведение (что пишется в таблицу, лог и куда
уходит фото) до переноса функций из reader.py в src/gmr. Ожидания здесь —
описание того, как program2 работает сейчас, а не пожелания.

Оконные тесты — под xvfb-run (без экрана пропускаются).
"""
import os

import cv2
import numpy as np
import pandas as pd
import pytest

from src.gmr.storage import LOG_COLUMNS

program2 = pytest.importorskip("program2")
from src.gmr.domain import PipelineConfig  # noqa: E402
from src.gmr.storage import load_log, save_log  # noqa: E402

CFG = PipelineConfig()
needs_display = pytest.mark.skipif(not os.environ.get("DISPLAY"),
                                   reason="нет экрана (запускать под xvfb-run)")


def _img(path, value=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((40, 40, 3), value, np.uint8))


# ─── Без окон ────────────────────────────────────────────────────────────────

def test_list_question_photos_order_and_filter(tmp_path):
    q = tmp_path / "question"
    _img(q / "no_meter" / "a.jpg")
    _img(q / "digits_error" / "b.jpg")
    _img(q / "serial_not_found" / "c.png")
    (q / "digits_error" / "notes.txt").write_text("x")
    _img(q / "unknown_reason" / "d.jpg")          # не из QUESTION_SUBFOLDERS
    got = [(r, n) for r, n, _ in program2.list_question_photos(str(tmp_path))]
    assert got == [("digits_error", "b.jpg"), ("serial_not_found", "c.png"), ("no_meter", "a.jpg")]


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


# ─── Окна: экран обработки ───────────────────────────────────────────────────

@pytest.fixture
def app(tmp_path, monkeypatch):
    photos = tmp_path / "Аюб"
    q = photos / "question"
    _img(q / "serial_not_found" / "p1.jpg", 10)
    _img(q / "serial_not_found" / "p2.jpg", 20)
    _img(q / "no_meter" / "p3.jpg", 30)
    tbl = tmp_path / "meters_table.csv"
    pd.DataFrame([{CFG.col_serial: s, CFG.col_account_id: a, CFG.col_last_reading: l, CFG.col_new_reading: ""}
                  for s, a, l in [("1234567", "A1", "1000"), ("7654321", "A2", "500")]]).to_csv(tbl, index=False)
    # авто-строки, создавшие файлы в question/ (для переноса отпечатка в ручные строки)
    auto = []
    for name, outcome in (("p1.jpg", "SERIAL_NOT_FOUND"), ("p2.jpg", "SERIAL_NOT_FOUND"), ("p3.jpg", "NO_METER")):
        r = {c: "" for c in LOG_COLUMNS}
        r.update(original_filename=name, outcome=outcome, source="auto", processed_by="auto",
                 photo_hash=f"h-{name}", source_folder="Аюб")
        auto.append(r)
    save_log(str(tmp_path / "meters_table_log.csv"), auto)

    settings = program2.AppSettings(operator_name="Оператор", photos_dir=str(photos),
                                    table_path=str(tbl), training_dir=str(tmp_path / "train"))
    monkeypatch.setattr(program2, "load_settings", lambda: settings)
    monkeypatch.setattr(program2, "save_settings", lambda s: None)
    monkeypatch.setattr(program2.MainWindow, "_load_models_async", lambda self: None)
    monkeypatch.setattr(program2, "LoginDialog", lambda *a, **k: type("D", (), {"action": "continue"})())
    answers = []
    monkeypatch.setattr(program2.messagebox, "askyesno", lambda *a, **k: answers.pop(0) if answers else True)
    for name in ("showinfo", "showwarning", "showerror"):
        monkeypatch.setattr(program2.messagebox, name, lambda *a, **k: None)
    w = program2.MainWindow()
    w.withdraw()
    w.answers = answers
    w.photos = photos
    yield w
    w.destroy()


def _open_first(app):
    reason, _, path = program2.list_question_photos(app.settings.photos_dir)[0]
    app.open_edit_screen(path, reason)
    return app._current_screen


def _fill(scr, account, reading):
    scr._account_var.set(account)
    scr._lookup_by_account(account)
    scr._reading_widget.set_digits(reading, None)
    scr._update_delta()
    scr._update_accept_state()


def _table(app):
    df = pd.read_csv(app.settings.table_path, dtype=str, keep_default_na=False)
    return dict(zip(df[CFG.col_account_id], df[CFG.col_new_reading]))


@needs_display
def test_accept_writes_table_log_moves_photo_opens_next(app):
    scr = _open_first(app)
    _fill(scr, "A1", "01200")
    scr._accept()
    assert _table(app)["A1"] == "1200"
    r = app.log_rows[-1]
    assert (r["original_filename"], r["final_filename"], r["serial_id"], r["account_id"]) == (
        "p1.jpg", "A1.jpg", "1234567", "A1")
    assert (r["reading"], r["last_reading"], r["delta"], r["outcome"], r["source"]) == (
        "1200", "1000.0", "200.0", "PLUS", "manual")
    assert r["notes"] == "serial_not_found"
    assert r["photo_hash"] == "h-p1.jpg" and r["source_folder"] == "Аюб"   # из авто-строки
    assert (app.photos / "plus" / "A1.jpg").exists()
    assert not (app.photos / "question" / "serial_not_found" / "p1.jpg").exists()
    # сразу следующее фото очереди
    assert os.path.basename(app._current_screen.photo_path) == "p2.jpg"
    # лог на диске = в памяти
    assert len(load_log(str(app.photos.parent / "meters_table_log.csv"))) == 4


@needs_display
def test_accept_minus_goes_to_minus(app):
    scr = _open_first(app)
    _fill(scr, "A1", "00900")
    scr._accept()
    assert app.log_rows[-1]["outcome"] == "MINUS"
    assert (app.photos / "minus" / "A1.jpg").exists()


@needs_display
def test_accept_suspicious_asks_and_can_be_cancelled(app):
    scr = _open_first(app)
    _fill(scr, "A2", "90000")                      # delta 89500 > 10000
    app.answers.append(False)
    scr._accept()
    assert _table(app)["A2"] == "" and len(app.log_rows) == 3
    app.answers.append(True)
    scr._accept()
    assert _table(app)["A2"] == "90000" and app.log_rows[-1]["outcome"] == "PLUS"


@needs_display
def test_accept_disabled_without_full_reading(app):
    scr = _open_first(app)
    scr._account_var.set("A1")
    scr._lookup_by_account("A1")
    scr._update_accept_state()
    scr._accept()
    assert len(app.log_rows) == 3 and _table(app)["A1"] == ""


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
    r = app.log_rows[-1]
    assert (r["original_filename"], r["outcome"], r["source"], r["notes"]) == ("p1.jpg", outcome, "manual", notes)
    assert r["photo_hash"] == "h-p1.jpg"
    assert (app.photos / folder / "p1.jpg").exists()
    assert all(v == "" for v in _table(app).values())          # таблица не тронута
    assert os.path.basename(app._current_screen.photo_path) == "p2.jpg"


@needs_display
def test_action_cancelled_changes_nothing(app):
    scr = _open_first(app)
    app.answers.append(False)
    scr._duplicate()
    assert len(app.log_rows) == 3
    assert (app.photos / "question" / "serial_not_found" / "p1.jpg").exists()


@needs_display
def test_processing_tab_mark_unreadable_from_list(app):
    tab = app._proc_tab
    tab.refresh()
    first = tab._tree.get_children()[0]
    tab._tree.selection_set(first)
    tab._unreadable()
    r = app.log_rows[-1]
    assert (r["outcome"], r["notes"]) == ("UNREADABLE", "нечитаемо (из списка)")
    assert (app.photos / "unreadable" / "p1.jpg").exists()


@needs_display
def test_last_photo_returns_to_list(app):
    for _ in range(3):
        scr = _open_first(app) if not hasattr(app, "_current_screen") else app._current_screen
        scr._unreadable()
    assert program2.list_question_photos(app.settings.photos_dir) == []
    assert app._notebook.winfo_manager() == "pack"                # снова список вкладок


@needs_display
def test_accept_saves_silent_markup_when_model_was_wrong(app):
    scr = _open_first(app)
    scr._model_result = {"serial_text": "1284567", "serial_crop": np.full((20, 60, 3), 5, np.uint8),
                         "reading_str": "01?00", "digit_crops": [np.full((30, 15, 3), 1, np.uint8)] * 5,
                         "digit_preds": [{}] * 5}
    _fill(scr, "A1", "01200")
    scr._accept()
    train = app.photos.parent / "train"
    assert (train / "crnn" / "images" / "1234567.jpeg").exists()
    assert {p.parent.name for p in (train / "cnn").rglob("*.jpg")} == {"2"}


# ─── «0 из 0»: выбрана общая папка output/ (вариант а, 2026-10-01) ─────────────

def test_controller_folders(tmp_path):
    out = tmp_path / "output"
    (out / "Аюб" / "question").mkdir(parents=True)
    (out / "Сулиман С" / "plus").mkdir(parents=True)
    (out / "пустая").mkdir()
    (out / "report.txt").write_text("x")
    assert program2.controller_folders(str(out)) == ["Аюб", "Сулиман С"]
    assert "Аюб, Сулиман С" in program2.wrong_folder_hint(str(out))
    # папка контролёра выбрана правильно — подсказки нет
    assert program2.controller_folders(str(out / "Аюб")) == []
    assert program2.wrong_folder_hint(str(out / "Аюб")) == ""
    assert program2.wrong_folder_hint("") == "" and program2.wrong_folder_hint(str(tmp_path / "нет")) == ""


@needs_display
def test_tabs_show_hint_for_output_folder(app):
    out = app.photos.parent
    app.settings.photos_dir = str(out)          # выбрали общую папку
    app.refresh_tabs()
    for tab in (app._proc_tab, app._verify_tab):
        assert tab._counter_var.get().startswith("⚠ Это общая папка")
        assert "Аюб" in tab._counter_var.get()
    app.settings.photos_dir = str(app.photos)
    app.refresh_tabs()
    assert app._proc_tab._counter_var.get().startswith("Всего:")
