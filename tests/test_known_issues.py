"""
Известные ошибки (аудит 2026-10-01, docs/BACKLOG.md → «Дорожная карта»).

Каждый тест описывает, как ДОЛЖНО быть, и сейчас падает — он помечен
`известная ошибка` (xfail, strict). Когда ошибку исправят, тест пройдёт, и
pytest сообщит об этом как об ошибке (XPASS strict): снимите пометку — тест
станет обычным. Ошибки окна оператора — в конце tests/test_program2_verify.py.

Модели — фейки: фото — PNG-байты, пиксель [0,0] — номер записи в PHOTOS.
"""
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import reader
from src.gmr.ml import loader
from src.gmr.render import read_image
from src.gmr.storage import load_log, load_table, log_path_for, save_table


def known_issue(stage: str, what: str):
    return pytest.mark.xfail(strict=True, raises=AssertionError,
                             reason=f"известная ошибка, {stage}: {what}")


# номер -> (серийник, 5 цифр или None = цифры не прочитаны)
PHOTOS = {
    1: ("11111", "01200"),
    2: ("22222", "04000"),
    3: ("33333", None),
    4: ("33333", "00150"),
}
TABLE = [("11111", "A-1", "1000"), ("22222", "A-2", "5000"), ("33333", "A-3", "100")]


class _Meter:
    def process_image(self, img_path, **kwargs):
        t = int(read_image(img_path)[0, 0, 0])
        return [
            {"class": "gas_meter", "conf": .95, "angle": 0., "bbox": (0, 0, 200, 40),
             "crop": np.full((40, 200, 3), t, np.uint8), "path": None},
            {"class": "serial_number", "conf": .95, "angle": 0., "bbox": (0, 40, 100, 60),
             "crop": np.full((20, 100, 3), t, np.uint8), "path": None},
        ]


class _Digits:
    def process_array(self, img, **kwargs):
        digits = PHOTOS[int(img[0, 0, 0])][1]
        if digits is None:
            return []
        return [{"class": "digit", "conf": .95, "angle": 0., "bbox": (i * 40, 0, i * 40 + 20, 30),
                 "crop": np.full((30, 20, 3), int(d), np.uint8), "path": None}
                for i, d in enumerate(digits)]


class _DigitOCR:
    def predict(self, img):
        return {"digit": int(img[0, 0, 0]), "confidence": .95}


class _SerialOCR:
    crash_on = None    # номер фото, на котором «нажали Ctrl+C»

    def predict_with_details(self, img):
        t = int(img[0, 0, 0])
        if t == self.crash_on:
            raise KeyboardInterrupt
        return {"text": PHOTOS[t][0], "avg_confidence": .95, "details": []}


@pytest.fixture
def serial_ocr(monkeypatch):
    ocr = _SerialOCR()
    monkeypatch.setattr(loader, "YOLOInferer",
                        lambda m, conf_thresh=.8, straighten=True, output_dir=None:
                        _Digits() if straighten is False else _Meter())
    monkeypatch.setattr(loader, "CNNInferer", lambda *a, **k: _DigitOCR())
    monkeypatch.setattr(loader, "CRNNInferer", lambda *a, **k: ocr)
    return ocr


def _photo(path: Path, n: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(".png", np.full((100, 200, 3), n, np.uint8))[1].tobytes())


def _table(path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Номер счетчика", "Лицевой счет", "Последние показания", "Текущие показания"])
        w.writerows([(s, a, last, "") for s, a, last in TABLE])


def _run(root: Path):
    reader.run_pipeline(reader.PipelineConfig(
        input_dir=str(root / "input"), output_base_dir=str(root / "out"),
        table_path=str(root / "table.csv")))


def _reading(root: Path, account: str) -> str:
    df = load_table(str(root / "table.csv"))
    v = df.loc[df["Лицевой счет"] == account, "Текущие показания"].iloc[0]
    return "" if pd.isna(v) else str(v)


# ─── Этап 2: не терять показания ─────────────────────────────────────────────

@known_issue("этап 2", "после сбоя посреди прогона показание теряется навсегда")
def test_reading_survives_crash_and_rerun(tmp_path, serial_ocr):
    _table(tmp_path / "table.csv")
    _photo(tmp_path / "input" / "Аюб" / "p1.jpg", 1)     # A-1 прочитан
    _photo(tmp_path / "input" / "Аюб" / "p2.jpg", 2)     # на нём — Ctrl+C
    serial_ocr.crash_on = 2
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path)
    serial_ocr.crash_on = None
    _run(tmp_path)                                       # перезапуск
    # сейчас: p1 в логе PLUS -> при перезапуске REPEAT, в таблицу 1200 так и не попадает
    assert _reading(tmp_path, "A-1") == "1200"
    assert _reading(tmp_path, "A-2") == "4000"


@known_issue("этап 2", "таблица открыта в Excel — показания папки теряются")
def test_readings_survive_table_locked_by_excel(tmp_path, serial_ocr, monkeypatch):
    _table(tmp_path / "table.csv")
    _photo(tmp_path / "input" / "Аюб" / "p1.jpg", 1)
    _photo(tmp_path / "input" / "Аюб" / "p2.jpg", 2)

    def locked(df, path):     # так Windows отвечает, пока файл открыт в Excel
        raise PermissionError(13, "файл занят другим процессом", path)

    monkeypatch.setattr(reader, "save_table", locked)
    try:
        _run(tmp_path)
    except PermissionError:
        pass                                             # сейчас прогон падает
    monkeypatch.setattr(reader, "save_table", save_table)   # Excel закрыли
    _run(tmp_path)
    assert _reading(tmp_path, "A-1") == "1200"
    assert _reading(tmp_path, "A-2") == "4000"


@known_issue("этап 2", "сохранение .xlsx стирает другие листы книги")
def test_xlsx_save_keeps_other_sheets(tmp_path):
    p = tmp_path / "table.xlsx"
    with pd.ExcelWriter(p) as w:
        pd.DataFrame([{"Номер счетчика": "11111", "Текущие показания": ""}]).to_excel(
            w, sheet_name="Абоненты", index=False)
        pd.DataFrame([{"Итого": 1}]).to_excel(w, sheet_name="Итоги", index=False)
    df = load_table(str(p))
    df.loc[0, "Текущие показания"] = "1200"
    save_table(df, str(p))
    assert pd.ExcelFile(p).sheet_names == ["Абоненты", "Итоги"]


# ─── Этап 3: месячный цикл ───────────────────────────────────────────────────

@known_issue("этап 3", "неудачное фото счётчика блокирует хорошее фото того же счётчика")
def test_good_photo_read_after_failed_photo_of_same_meter(tmp_path, serial_ocr):
    _table(tmp_path / "table.csv")
    _photo(tmp_path / "input" / "Аюб" / "y1.jpg", 3)     # A-3, цифры не прочитаны
    _photo(tmp_path / "input" / "Аюб" / "y2.jpg", 4)     # A-3, хорошее фото
    _run(tmp_path)
    _run(tmp_path)                                       # и на следующий день
    # сейчас: y2 -> REPEAT «duplicate in current run», потом «already in log» навсегда
    assert _reading(tmp_path, "A-3") == "150"


@known_issue("этап 3", "одно фото в корне папки с фото — папки контролёров не обрабатываются")
def test_photo_in_root_does_not_hide_controller_folders(tmp_path, serial_ocr):
    _table(tmp_path / "table.csv")
    _photo(tmp_path / "input" / "Аюб" / "p1.jpg", 1)
    _photo(tmp_path / "input" / "Сулиман С" / "p2.jpg", 2)
    _photo(tmp_path / "input" / "случайно.jpg", 4)
    _run(tmp_path)
    names = {r["original_filename"] for r in load_log(log_path_for(str(tmp_path / "table.csv")))}
    assert {"p1.jpg", "p2.jpg"} <= names
