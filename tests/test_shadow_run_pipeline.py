"""
Интеграционный shadow-run (Фаза 2b): run_pipeline целиком, на настоящих
файлах (фото, таблица, лог, output-папки), модели заменены фейками.

Проверяем три вещи:
  1. shadow_sqlite_log=True: SQLite-лог совпадает с CSV построчно, отчёт
     сверки говорит "совпадение".
  2. shadow-режим не меняет наблюдаемый результат: CSV-лог, таблица и
     разложенные по папкам фото — те же, что при shadow_sqlite_log=False.
  3. если CSV-лог уже существовал до прогона (повторный запуск), SQLite
     засевается из него, и сверка всё равно чистая.

Фейковые модели определяют, какое фото обрабатывается, по значению
пикселя в кропе (см. _PHOTOS) — иначе детектор цифр не знает, из какого
фото пришёл кроп счётчика.
"""
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

import reader
from src.gmr.ml import loader
from src.gmr.storage import LOG_COLUMNS, append_log_row, load_log, load_table, log_path_for  # noqa: E402

# имя фото -> (серийник, 5 цифр или None, есть ли счётчик на фото)
_PHOTOS = {
    "p0.jpg": ("11111", "01200", True),   # PLUS (1000 -> 1200)
    "p1.jpg": ("22222", "04000", True),   # MINUS (5000 -> 4000)
    "p2.jpg": ("99999", "01000", True),   # SERIAL_NOT_FOUND
    "p3.jpg": ("11111", "01300", True),   # REPEAT (тот же счётчик в прогоне)
    "p4.jpg": (None,    None,    False),  # NO_METER
}
_ORDER = sorted(_PHOTOS)
_TABLE = [
    {"Номер счетчика": "11111", "Лицевой счет": "A-1", "Последние показания": "1000", "Текущие показания": ""},
    {"Номер счетчика": "22222", "Лицевой счет": "A-2", "Последние показания": "5000", "Текущие показания": ""},
    {"Номер счетчика": "33333", "Лицевой счет": "A-3", "Последние показания": "100",  "Текущие показания": ""},
]


def _tag(name):
    return _ORDER.index(name) + 1  # значение пикселя кропа = номер фото


class _MeterDetector:
    def process_image(self, img_path, save_crops=False, max_per_class=1, straighten=None):
        name = Path(img_path).name
        serial, _, has_meter = _PHOTOS[name]
        if not has_meter:
            return []
        t = _tag(name)
        return [
            {"class": "gas_meter", "conf": 0.95, "angle": 0.0, "bbox": (0, 0, 200, 40),
             "crop": np.full((40, 200, 3), t, dtype=np.uint8), "path": None},
            {"class": "serial_number", "conf": 0.95, "angle": 0.0, "bbox": (0, 40, 100, 60),
             "crop": np.full((20, 100, 3), t, dtype=np.uint8), "path": None},
        ]


class _DigitDetector:
    def process_array(self, img, save_crops=False, max_per_class=5, straighten=None):
        name = _ORDER[int(img[0, 0, 0]) - 1]
        digits = _PHOTOS[name][1]
        crops = []
        for i, d in enumerate(digits):
            arr = np.full((30, 20, 3), int(d), dtype=np.uint8)  # цифра зашита в пиксель
            crops.append({"class": "digit", "conf": 0.95, "angle": 0.0,
                          "bbox": (i * 40, 0, i * 40 + 20, 30), "crop": arr, "path": None})
        return crops


class _DigitOCR:
    def predict(self, image_input):
        return {"digit": int(image_input[0, 0, 0]), "confidence": 0.95}


class _SerialOCR:
    def predict_with_details(self, image_input):
        name = _ORDER[int(image_input[0, 0, 0]) - 1]
        return {"text": _PHOTOS[name][0], "avg_confidence": 0.95, "details": []}


@pytest.fixture
def fake_models(monkeypatch):
    def yolo(model, conf_thresh=0.8, straighten=True, output_dir=None):
        return _DigitDetector() if straighten is False else _MeterDetector()
    monkeypatch.setattr(loader, "YOLOInferer", yolo)
    monkeypatch.setattr(loader, "CNNInferer", lambda *a, **k: _DigitOCR())
    monkeypatch.setattr(loader, "CRNNInferer", lambda *a, **k: _SerialOCR())


def _image(name):
    # у каждого фото своё содержимое: с 2026-09-29 фото узнаётся в логе по
    # отпечатку байтов, и одинаковые картинки были бы одним и тем же фото
    return np.full((100, 200, 3), 20 + 40 * _ORDER.index(name), dtype=np.uint8)


def _setup_workspace(root: Path):
    inp = root / "input"
    inp.mkdir(parents=True)
    for name in _ORDER:
        cv2.imwrite(str(inp / name), _image(name))
    table = root / "table.csv"
    with open(table, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_TABLE[0]))
        w.writeheader()
        w.writerows(_TABLE)
    return inp, table


def _run(root: Path, shadow: bool):
    inp, table = _setup_workspace(root)
    cfg = reader.PipelineConfig(
        input_dir=str(inp), output_base_dir=str(root / "out"), table_path=str(table),
        draw_boxes=False, shadow_sqlite_log=shadow,
    )
    reader.run_pipeline(cfg)
    return table


def _log_without_timestamps(log_path):
    rows = load_log(str(log_path))
    for r in rows:
        r["processed_at"] = ""
    return rows


def _output_tree(root: Path):
    return sorted(str(p.relative_to(root / "out")) for p in (root / "out").rglob("*.jpg"))


def test_shadow_run_sqlite_matches_csv(tmp_path, fake_models):
    table = _run(tmp_path, shadow=True)
    log_csv = Path(log_path_for(str(table)))
    sqlite = reader.SqliteLogStore(str(log_csv.with_suffix(".sqlite")))

    assert sqlite.load() == load_log(str(log_csv))
    assert len(sqlite.load()) == len(_PHOTOS)

    report = log_csv.with_name(log_csv.stem + "_shadow_report.txt").read_text(encoding="utf-8")
    assert "совпадение построчно" in report


def test_shadow_mode_does_not_change_results(tmp_path, fake_models):
    table_off = _run(tmp_path / "off", shadow=False)
    table_on = _run(tmp_path / "on", shadow=True)

    assert _log_without_timestamps(log_path_for(str(table_off))) == \
           _log_without_timestamps(log_path_for(str(table_on)))
    assert Path(table_off).read_bytes() == Path(table_on).read_bytes()
    assert _output_tree(tmp_path / "off") == _output_tree(tmp_path / "on")

    # самопроверка фикстуры: прогон дал именно те исходы, что задуманы
    outcomes = [r["outcome"] for r in load_log(log_path_for(str(table_on)))]
    assert outcomes == ["PLUS", "MINUS", "SERIAL_NOT_FOUND", "REPEAT", "NO_METER"]
    # и без shadow-режима SQLite-файла не появляется
    assert not Path(log_path_for(str(table_off))).with_suffix(".sqlite").exists()


def test_shadow_run_seeds_sqlite_from_existing_csv_log(tmp_path, fake_models):
    table = _run(tmp_path, shadow=False)          # первый прогон: только CSV
    log_csv = Path(log_path_for(str(table)))
    shutil.rmtree(tmp_path / "input")
    shutil.rmtree(tmp_path / "out")

    _setup_workspace(tmp_path)                    # те же фото ещё раз (лог остаётся)
    cfg = reader.PipelineConfig(
        input_dir=str(tmp_path / "input"), output_base_dir=str(tmp_path / "out"),
        table_path=str(table), draw_boxes=False, shadow_sqlite_log=True,
    )
    reader.run_pipeline(cfg)

    rows = load_log(str(log_csv))
    assert len(rows) == 2 * len(_PHOTOS)
    # Второй прогон (вариант Г): успешные и REPEAT -> REPEAT без моделей,
    # авто-ошибки (SERIAL_NOT_FOUND, NO_METER) перечитываются — модели те же,
    # поэтому исходы те же.
    assert [r["outcome"] for r in rows[len(_PHOTOS):]] == \
           ["REPEAT", "REPEAT", "SERIAL_NOT_FOUND", "REPEAT", "NO_METER"]

    sqlite = reader.SqliteLogStore(str(log_csv.with_suffix(".sqlite")))
    assert sqlite.load() == rows
    report = log_csv.with_name(log_csv.stem + "_shadow_report.txt").read_text(encoding="utf-8")
    assert "совпадение построчно" in report



def _rerun(tmp_path, table, shadow=False):
    shutil.rmtree(tmp_path / "input")
    shutil.rmtree(tmp_path / "out")
    inp = tmp_path / "input"
    inp.mkdir()
    for name in _ORDER:
        cv2.imwrite(str(inp / name), _image(name))
    cfg = reader.PipelineConfig(
        input_dir=str(inp), output_base_dir=str(tmp_path / "out"),
        table_path=str(table), draw_boxes=False, shadow_sqlite_log=shadow,
    )
    reader.run_pipeline(cfg)


def test_variant_g_manually_handled_photo_not_returned_to_question(tmp_path, fake_models):
    table = _run(tmp_path, shadow=False)
    log_csv = log_path_for(str(table))
    # оператор в program2.py пометил p2 (SERIAL_NOT_FOUND) как "нет в базе"
    append_log_row(log_csv, {c: "" for c in LOG_COLUMNS} | {
        "original_filename": "p2.jpg", "outcome": "NOT_IN_DB", "source": "manual",
        "processed_by": "Оператор",
    })

    _rerun(tmp_path, table)

    second = load_log(log_csv)[len(_PHOTOS) + 1:]
    by_name = {r["original_filename"]: r["outcome"] for r in second}
    assert by_name == {"p0.jpg": "REPEAT", "p1.jpg": "REPEAT", "p2.jpg": "REPEAT",
                       "p3.jpg": "REPEAT", "p4.jpg": "NO_METER"}
    tree = _output_tree(tmp_path)
    assert not any("serial_not_found" in p for p in tree)   # p2 не вернулся оператору
    assert any("no_meter" in p for p in tree)               # p4 не разобран — снова в question


def test_variant_g_improved_model_reads_old_failure(tmp_path, fake_models, monkeypatch):
    table = _run(tmp_path, shadow=False)                    # p4: NO_METER
    # "обучили модель получше": теперь на p4 находится счётчик 33333
    monkeypatch.setitem(_PHOTOS, "p4.jpg", ("33333", "00150", True))

    _rerun(tmp_path, table, shadow=True)

    log_csv = log_path_for(str(table))
    last = load_log(log_csv)[-1]
    assert (last["original_filename"], last["outcome"], last["reading"]) == ("p4.jpg", "PLUS", "150")
    df = load_table(str(table))
    assert df.loc[df["Лицевой счет"] == "A-3", "Текущие показания"].item() == "150"
    report = Path(log_csv).with_name(Path(log_csv).stem + "_shadow_report.txt").read_text(encoding="utf-8")
    assert "совпадение построчно" in report
