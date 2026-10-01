"""
Известные ошибки (аудит 2026-10-01, docs/BACKLOG.md → «Дорожная карта»).

Каждый тест описывает, как ДОЛЖНО быть, и сейчас падает — он помечен
`известная ошибка` (xfail, strict). Когда ошибку исправят, тест пройдёт, и
pytest сообщит об этом как об ошибке (XPASS strict): снимите пометку — тест
станет обычным.

Ошибки этапа 2 исправлены в режиме папки месяца (этапы 2.2b, 2.3a), их тесты
стали обычными: tests/test_month_run.py, tests/test_operator.py,
tests/test_program2_verify.py. Старый режим (таблица и CSV-лог) с его
ошибками убран на этапе 2.3b.

Модели — фейки: фото — PNG-байты, пиксель [0,0] — номер записи в PHOTOS.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

import reader
from src.gmr.ml import loader
from src.gmr.render import read_image
from src.gmr.storage.month import MonthDB, MonthFolder
from tests._month import make_month


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
TABLE = [("11111", "A-1", "1000", ""), ("22222", "A-2", "5000", ""), ("33333", "A-3", "100", "")]


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
    def predict_with_details(self, img):
        return {"text": PHOTOS[int(img[0, 0, 0])][0], "avg_confidence": .95, "details": []}


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


def _month(root: Path) -> MonthFolder:
    return make_month(root, TABLE)


def _run(f: MonthFolder):
    reader.run_pipeline(reader.PipelineConfig(month_dir=str(f.root)))


def _reading(f: MonthFolder, account: str) -> str:
    with MonthDB(f.db) as db:
        r = db.reading(account)
    return r.value if r else ""


# ─── Этап 3: месячный цикл ───────────────────────────────────────────────────

@known_issue("этап 3", "неудачное фото счётчика блокирует хорошее фото того же счётчика")
def test_good_photo_read_after_failed_photo_of_same_meter(tmp_path, serial_ocr):
    f = _month(tmp_path)
    _photo(f.photos / "Аюб" / "y1.jpg", 3)               # A-3, цифры не прочитаны
    _photo(f.photos / "Аюб" / "y2.jpg", 4)               # A-3, хорошее фото
    _run(f)
    _run(f)                                              # и на следующий день
    # сейчас: y2 -> REPEAT «duplicate in current run», потом «already in log» навсегда
    assert _reading(f, "A-3") == "150"


@known_issue("этап 3", "одно фото в корне папки с фото — папки контролёров не обрабатываются")
def test_photo_in_root_does_not_hide_controller_folders(tmp_path, serial_ocr):
    f = _month(tmp_path)
    _photo(f.photos / "Аюб" / "p1.jpg", 1)
    _photo(f.photos / "Сулиман С" / "p2.jpg", 2)
    _photo(f.photos / "случайно.jpg", 4)
    _run(f)
    with MonthDB(f.db) as db:
        names = {r["original_filename"] for r in db.log_rows()}
    assert {"p1.jpg", "p2.jpg"} <= names
