"""
Идентичность фото в логе — отпечаток содержимого + подпапка
(решение владельца 2026-09-29, docs/MIGRATION_STATUS.md).

  1. photo_fingerprint зависит только от байтов файла.
  2. Лог старого формата (16 столбцов) при дозаписи переписывается с новой
     шапкой без потери данных; старая SQLite-база дополняется столбцами.
  3. run_pipeline: два разных фото с одинаковым именем в разных подпапках
     обрабатываются оба; то же фото под другим именем в новой пачке узнаётся
     без запуска моделей.
  4. program2.py переносит отпечаток в ручную строку из автоматической
     строки своей подпапки.
"""
import csv
import sqlite3
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import reader
from src.gmr.ml import loader
from src.gmr.storage import LOG_COLUMNS, CsvLogStore, SqliteLogStore, photo_fingerprint
from src.gmr.storage import append_log_row, load_log, log_path_for  # noqa: E402

OLD_COLUMNS = LOG_COLUMNS[:16]   # формат лога до 2026-09-29


# ─── 1. Отпечаток ────────────────────────────────────────────────────────────

def test_fingerprint_depends_only_on_content(tmp_path):
    a, b, c = tmp_path / "a.jpg", tmp_path / "другое имя.jpg", tmp_path / "c.jpg"
    a.write_bytes(b"photo-1"); b.write_bytes(b"photo-1"); c.write_bytes(b"photo-2")
    assert photo_fingerprint(str(a)) == photo_fingerprint(str(b))
    assert photo_fingerprint(str(a)) != photo_fingerprint(str(c))
    assert len(photo_fingerprint(str(a))) == 16


# ─── 2. Переход старого лога ─────────────────────────────────────────────────

def _write_old_log(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OLD_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def test_old_log_upgraded_on_append_without_losing_data(tmp_path):
    log = tmp_path / "t_log.csv"
    old_rows = [{c: f"{c}-{i}" for c in OLD_COLUMNS} for i in range(3)]
    _write_old_log(log, old_rows)

    new_row = {c: "" for c in LOG_COLUMNS} | {"original_filename": "new.jpg", "photo_hash": "abc"}
    append_log_row(str(log), new_row)   # так же дописывает и program2.py

    store = CsvLogStore(str(log))
    assert store.header() == LOG_COLUMNS
    rows = store.load()
    assert len(rows) == 4
    for old, got in zip(old_rows, rows):
        assert {c: got[c] for c in OLD_COLUMNS} == old
        assert got["photo_hash"] == "" and got["source_folder"] == ""
    assert rows[3]["original_filename"] == "new.jpg" and rows[3]["photo_hash"] == "abc"


def test_upgrade_is_noop_for_current_format(tmp_path):
    log = tmp_path / "t_log.csv"
    CsvLogStore(str(log)).save([{c: "x" for c in LOG_COLUMNS}])
    before = log.read_bytes()
    assert CsvLogStore(str(log)).upgrade_if_needed() is False
    assert log.read_bytes() == before


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
_TABLE = [
    {"Номер счетчика": "11111", "Лицевой счет": "A-1", "Последние показания": "1000", "Текущие показания": ""},
    {"Номер счетчика": "22222", "Лицевой счет": "A-2", "Последние показания": "5000", "Текущие показания": ""},
]


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


def _table(root):
    path = root / "table.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_TABLE[0]))
        w.writeheader()
        w.writerows(_TABLE)
    return path


def _run(root, inp, table):
    reader.run_pipeline(reader.PipelineConfig(
        input_dir=str(inp), output_base_dir=str(root / "out"),
        table_path=str(table), draw_boxes=False,
    ))
    return load_log(log_path_for(str(table)))


def test_same_filename_in_two_subfolders_both_processed(tmp_path, models):
    inp = tmp_path / "input"
    for folder, color in (("Аюб", 1), ("Сулиман", 2)):
        (inp / folder).mkdir(parents=True)
        cv2.imwrite(str(inp / folder / "IMG_0001.jpg"), _img(color))

    rows = _run(tmp_path, inp, _table(tmp_path))

    got = {(r["source_folder"], r["original_filename"]): (r["outcome"], r["account_id"]) for r in rows}
    assert got == {("Аюб", "IMG_0001.jpg"): ("PLUS", "A-1"),
                   ("Сулиман", "IMG_0001.jpg"): ("MINUS", "A-2")}
    assert rows[0]["photo_hash"] != rows[1]["photo_hash"] != ""


def test_same_photo_renamed_in_new_batch_is_repeat_without_models(tmp_path, models):
    inp = tmp_path / "input"
    (inp / "пачка1").mkdir(parents=True)
    cv2.imwrite(str(inp / "пачка1" / "IMG_0001.jpg"), _img(1))
    table = _table(tmp_path)
    _run(tmp_path, inp, table)

    # вторая пачка: тот же файл под другим именем в другой папке
    (inp / "пачка2").mkdir()
    (inp / "пачка2" / "переслано.jpg").write_bytes((inp / "пачка1" / "IMG_0001.jpg").read_bytes())
    (inp / "пачка1" / "IMG_0001.jpg").unlink()
    (inp / "пачка1").rmdir()
    calls_before = _Meter.calls

    rows = _run(tmp_path, inp, table)

    assert rows[-1]["original_filename"] == "переслано.jpg"
    assert rows[-1]["outcome"] == "REPEAT"
    assert "already in log" in rows[-1]["notes"]
    assert _Meter.calls == calls_before     # модели не запускались


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
    inp = tmp_path / "input"
    (inp / "Аюб").mkdir(parents=True)
    (inp / "Сулиман").mkdir(parents=True)
    cv2.imwrite(str(inp / "Аюб" / "a.jpg"), _img(3))                 # NO_METER
    data = (inp / "Аюб" / "a.jpg").read_bytes()
    (inp / "Аюб" / "b (1).jpg").write_bytes(data)                    # копия в той же папке
    (inp / "Сулиман" / "c.jpg").write_bytes(data)                    # копия в другой подпапке
    table = _table(tmp_path)

    rows = _run(tmp_path, inp, table)

    assert [(r["source_folder"], r["original_filename"], r["outcome"]) for r in rows] == [
        ("Аюб", "a.jpg", "NO_METER"),
        ("Аюб", "b (1).jpg", "REPEAT"),
        ("Сулиман", "c.jpg", "REPEAT"),
    ]
    assert all("duplicate file in current run" in r["notes"] for r in rows[1:])
    assert _Meter.calls == 1
    in_question = [p for p in (tmp_path / "out").rglob("*.jpg") if "question" in p.parts]
    assert len(in_question) == 1

    # следующий прогон: первая копия перечитывается (правило Г), остальные — нет
    import shutil
    shutil.rmtree(tmp_path / "out")
    rows = _run(tmp_path, inp, table)[3:]
    assert [r["outcome"] for r in rows] == ["NO_METER", "REPEAT", "REPEAT"]
    assert _Meter.calls == 2
