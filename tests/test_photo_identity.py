"""
Идентичность фото в логе — отпечаток содержимого + подпапка
(решение владельца 2026-09-29, docs/MIGRATION_STATUS.md).

  1. photo_fingerprint зависит только от байтов файла.
  2. Старая SQLite-база дополняется столбцами (дозапись в CSV-лог убрана
     вместе со старым режимом на этапе 2.3b).
  3. run_pipeline: два разных фото с одинаковым именем в разных подпапках
     обрабатываются оба; то же фото под другим именем в новой пачке узнаётся
     без запуска моделей и (с этапа 3) пропускается молча.
  4. program2.py переносит отпечаток в ручную строку из автоматической
     строки своей подпапки.
"""
import sqlite3
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import reader
from src.gmr.ml import loader
from src.gmr.storage import LOG_COLUMNS, SqliteLogStore, photo_fingerprint
from src.gmr.storage.month import MonthDB
from tests._month import make_month

OLD_COLUMNS = LOG_COLUMNS[:16]   # формат лога до 2026-09-29


# ─── 1. Отпечаток ────────────────────────────────────────────────────────────

def test_fingerprint_depends_only_on_content(tmp_path):
    a, b, c = tmp_path / "a.jpg", tmp_path / "другое имя.jpg", tmp_path / "c.jpg"
    a.write_bytes(b"photo-1"); b.write_bytes(b"photo-1"); c.write_bytes(b"photo-2")
    assert photo_fingerprint(str(a)) == photo_fingerprint(str(b))
    assert photo_fingerprint(str(a)) != photo_fingerprint(str(c))
    assert len(photo_fingerprint(str(a))) == 16


# ─── 2. Старая SQLite-база ───────────────────────────────────────────────────

def test_old_sqlite_schema_gets_new_columns(tmp_path):
    db = tmp_path / "t_log.sqlite"
    cols = ", ".join(f'"{c}" TEXT NOT NULL DEFAULT \'\'' for c in OLD_COLUMNS)
    with sqlite3.connect(db) as conn:
        conn.execute(f"CREATE TABLE processing_log (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})")
        conn.execute("INSERT INTO processing_log (original_filename) VALUES ('old.jpg')")
    store = SqliteLogStore(str(db))
    store.append({"original_filename": "new.jpg", "photo_hash": "abc", "source_folder": "Аюб"})
    rows = store.load()
    assert [r["original_filename"] for r in rows] == ["old.jpg", "new.jpg"]
    assert rows[1]["source_folder"] == "Аюб"


# ─── 3. run_pipeline с подпапками ────────────────────────────────────────────
# Фейковые модели узнают фото по цвету заливки файла (не по имени):
# 1 -> счётчик 11111, 2 -> 22222.

_BY_COLOR = {1: ("11111", "01200"), 2: ("22222", "04000")}
_TABLE = [("11111", "A-1", "1000", ""), ("22222", "A-2", "5000", "")]


def _img(color_id):
    return np.full((100, 200, 3), 50 * color_id, dtype=np.uint8)


def _color_of(img):
    return int(round(float(img[0, 0, 0]) / 50))


class _Meter:
    calls = 0

    def process_image(self, img_path, save_crops=False, max_per_class=1, straighten=None):
        _Meter.calls += 1
        k = _color_of(cv2.imread(img_path))
        if k not in _BY_COLOR:          # цвет 3 — счётчика на фото нет
            return []
        return [
            {"class": "gas_meter", "conf": 0.95, "angle": 0.0, "bbox": (0, 0, 200, 40),
             "crop": np.full((40, 200, 3), k, dtype=np.uint8), "path": None},
            {"class": "serial_number", "conf": 0.95, "angle": 0.0, "bbox": (0, 40, 100, 60),
             "crop": np.full((20, 100, 3), k, dtype=np.uint8), "path": None},
        ]


class _Digits:
    def process_array(self, img, save_crops=False, max_per_class=5, straighten=None):
        digits = _BY_COLOR[int(img[0, 0, 0])][1]
        return [{"class": "digit", "conf": 0.95, "angle": 0.0,
                 "bbox": (i * 40, 0, i * 40 + 20, 30),
                 "crop": np.full((30, 20, 3), int(d), dtype=np.uint8), "path": None}
                for i, d in enumerate(digits)]


class _DigitOCR:
    def predict(self, image_input):
        return {"digit": int(image_input[0, 0, 0]), "confidence": 0.95}


class _SerialOCR:
    def predict_with_details(self, image_input):
        return {"text": _BY_COLOR[int(image_input[0, 0, 0])][0], "avg_confidence": 0.95, "details": []}


@pytest.fixture
def models(monkeypatch):
    _Meter.calls = 0
    monkeypatch.setattr(loader, "YOLOInferer",
                        lambda model, conf_thresh=0.8, straighten=True, output_dir=None:
                        _Digits() if straighten is False else _Meter())
    monkeypatch.setattr(loader, "CNNInferer", lambda *a, **k: _DigitOCR())
    monkeypatch.setattr(loader, "CRNNInferer", lambda *a, **k: _SerialOCR())


def _run(f):
    reader.run_pipeline(reader.PipelineConfig(month_dir=str(f.root)))
    with MonthDB(f.db) as db:
        return db.log_rows()


def test_same_filename_in_two_subfolders_both_processed(tmp_path, models):
    f = make_month(tmp_path, _TABLE)
    inp = f.photos
    for folder, color in (("Аюб", 1), ("Сулиман", 2)):
        (inp / folder).mkdir(parents=True)
        cv2.imwrite(str(inp / folder / "IMG_0001.jpg"), _img(color))

    rows = _run(f)

    got = {(r["source_folder"], r["original_filename"]): (r["outcome"], r["account_id"]) for r in rows}
    assert got == {("Аюб", "IMG_0001.jpg"): ("PLUS", "A-1"),
                   ("Сулиман", "IMG_0001.jpg"): ("MINUS", "A-2")}
    assert rows[0]["photo_hash"] != rows[1]["photo_hash"] != ""


def test_same_photo_renamed_in_new_batch_is_skipped_without_models(tmp_path, models):
    f = make_month(tmp_path, _TABLE)
    inp = f.photos
    (inp / "пачка1").mkdir(parents=True)
    cv2.imwrite(str(inp / "пачка1" / "IMG_0001.jpg"), _img(1))
    _run(f)

    # вторая пачка: тот же файл под другим именем в другой папке
    (inp / "пачка2").mkdir()
    (inp / "пачка2" / "переслано.jpg").write_bytes((inp / "пачка1" / "IMG_0001.jpg").read_bytes())
    (inp / "пачка1" / "IMG_0001.jpg").unlink()
    (inp / "пачка1").rmdir()
    calls_before = _Meter.calls

    rows = _run(f)

    # этап 3: уже разобранное фото пропускается молча — строки нет
    assert [r["original_filename"] for r in rows] == ["IMG_0001.jpg"]
    assert _Meter.calls == calls_before     # модели не запускались
    assert not list((f.results / "пачка2").rglob("*.jpg"))


# ─── 4. program2.py: отпечаток в ручной строке ───────────────────────────────

def test_program2_manual_row_gets_hash_from_its_subfolder(tmp_path):
    import program2
    auto = [
        {c: "" for c in LOG_COLUMNS} | {"original_filename": "IMG_0001.jpg", "outcome": "NO_METER",
                                          "source": "auto", "photo_hash": "hash_suliman", "source_folder": "Сулиман"},
        {c: "" for c in LOG_COLUMNS} | {"original_filename": "IMG_0001.jpg", "outcome": "NO_METER",
                                          "source": "auto", "photo_hash": "hash_ayub", "source_folder": "Аюб"},
    ]
    res = tmp_path / "Октябрь" / "результат"
    app = SimpleNamespace(log_rows=list(auto), results_dir=str(res))
    manual = {c: "" for c in LOG_COLUMNS} | {"original_filename": "IMG_0001.jpg",
                                               "outcome": "UNREADABLE", "source": "manual"}
    photo = res / "Сулиман" / "question" / "no_meter" / "IMG_0001.jpg"
    assert program2.MainWindow._with_photo_identity(app, manual, str(photo)) is auto[0]
    assert manual["photo_hash"] == "hash_suliman"
    assert manual["source_folder"] == "Сулиман"


def test_program2_digits_error_file_renamed_to_account(tmp_path):
    # DIGITS_ERROR лежит в question/ под именем лицевого счёта (final_filename)
    import program2
    auto = [{c: "" for c in LOG_COLUMNS} | {"original_filename": "WhatsApp 1.jpeg",
             "final_filename": "A-1.jpeg", "outcome": "DIGITS_ERROR", "source": "auto",
             "photo_hash": "h1", "source_folder": ""}]
    res = tmp_path / "результат"                      # фото без подпапок контролёров
    app = SimpleNamespace(log_rows=list(auto), results_dir=str(res))
    manual = {c: "" for c in LOG_COLUMNS} | {"original_filename": "A-1.jpeg", "outcome": "PLUS",
                                               "source": "manual"}
    program2.MainWindow._with_photo_identity(app, manual, str(res / "question" / "digits_error" / "A-1.jpeg"))
    assert manual["photo_hash"] == "h1"


# ─── 5. Побайтные копии в одном прогоне ──────────────────────────────────────
# В реальных данных владельца 26 групп копий (одно фото переслано 2-4 раза);
# 4 лишние копии с ошибкой ложились оператору в question/ второй раз.

def test_copies_in_one_run_reach_question_once(tmp_path, models):
    f = make_month(tmp_path, _TABLE)
    inp = f.photos
    (inp / "Аюб").mkdir(parents=True)
    (inp / "Сулиман").mkdir(parents=True)
    cv2.imwrite(str(inp / "Аюб" / "a.jpg"), _img(3))                 # NO_METER
    data = (inp / "Аюб" / "a.jpg").read_bytes()
    (inp / "Аюб" / "b (1).jpg").write_bytes(data)                    # копия в той же папке
    (inp / "Сулиман" / "c.jpg").write_bytes(data)                    # копия в другой подпапке

    rows = _run(f)

    # с этапа 3 копии пропускаются молча (строки нет), в отчёте — «копия»
    assert [(r["source_folder"], r["original_filename"], r["outcome"]) for r in rows] == [
        ("Аюб", "a.jpg", "NO_METER")]
    assert _Meter.calls == 1
    in_question = [p for p in f.results.rglob("*.jpg") if "question" in p.parts]
    assert len(in_question) == 1

    # следующий прогон: фото ждёт оператора — не перечитывается; с --reread — одна копия
    _run(f)
    assert _Meter.calls == 1
    reader.run_pipeline(reader.PipelineConfig(month_dir=str(f.root), reread_errors=True))
    assert _Meter.calls == 2
    assert len([p for p in f.results.rglob("*.jpg") if "question" in p.parts]) == 1
