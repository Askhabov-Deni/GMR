"""
Прогон reader.py по папке месяца в тестах: фейковые модели и пять фото.

Фейковые модели узнают фото по имени файла (детектор счётчика) и по значению
пикселя в кропе (остальные модели) — иначе детектор цифр не знает, из какого
фото пришёл кроп счётчика. Фото лежат прямо в <месяц>/фото (без папок
контролёров), результат — в <месяц>/результат.

До этапа 2.3b это жило в tests/test_shadow_run_pipeline.py (прогон по
таблице и CSV-логу).
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

import reader
from src.gmr.ml import loader
from src.gmr.storage.month import MonthDB, MonthFolder
from tests._month import make_month

# имя фото -> (серийник, 5 цифр или None, есть ли счётчик на фото)
PHOTOS = {
    "p0.jpg": ("11111", "01200", True),   # PLUS (1000 -> 1200)
    "p1.jpg": ("22222", "04000", True),   # MINUS (5000 -> 4000)
    "p2.jpg": ("99999", "01000", True),   # SERIAL_NOT_FOUND
    "p3.jpg": ("11111", "01300", True),   # REPEAT (тот же счётчик в прогоне)
    "p4.jpg": (None,    None,    False),  # NO_METER
}
ORDER = sorted(PHOTOS)
OUTCOMES = ["PLUS", "MINUS", "SERIAL_NOT_FOUND", "REPEAT", "NO_METER"]
TABLE = [("11111", "A-1", "1000", ""), ("22222", "A-2", "5000", ""), ("33333", "A-3", "100", "")]


def _tag(name):
    return ORDER.index(name) + 1  # значение пикселя кропа = номер фото


class MeterDetector:
    calls = 0

    def process_image(self, img_path, save_crops=False, max_per_class=1, straighten=None):
        MeterDetector.calls += 1
        name = Path(img_path).name
        serial, _, has_meter = PHOTOS[name]
        if not has_meter:
            return []
        t = _tag(name)
        return [
            {"class": "gas_meter", "conf": 0.95, "angle": 0.0, "bbox": (0, 0, 200, 40),
             "crop": np.full((40, 200, 3), t, dtype=np.uint8), "path": None},
            {"class": "serial_number", "conf": 0.95, "angle": 0.0, "bbox": (0, 40, 100, 60),
             "crop": np.full((20, 100, 3), t, dtype=np.uint8), "path": None},
        ]


class DigitDetector:
    def process_array(self, img, save_crops=False, max_per_class=5, straighten=None):
        name = ORDER[int(img[0, 0, 0]) - 1]
        digits = PHOTOS[name][1]
        return [{"class": "digit", "conf": 0.95, "angle": 0.0,
                 "bbox": (i * 40, 0, i * 40 + 20, 30),
                 "crop": np.full((30, 20, 3), int(d), dtype=np.uint8), "path": None}   # цифра — в пикселе
                for i, d in enumerate(digits)]


class DigitOCR:
    def predict(self, image_input):
        return {"digit": int(image_input[0, 0, 0]), "confidence": 0.95}


class SerialOCR:
    def predict_with_details(self, image_input):
        name = ORDER[int(image_input[0, 0, 0]) - 1]
        return {"text": PHOTOS[name][0], "avg_confidence": 0.95, "details": []}


@pytest.fixture
def fake_models(monkeypatch):
    MeterDetector.calls = 0

    def yolo(model, conf_thresh=0.8, straighten=True, output_dir=None):
        return DigitDetector() if straighten is False else MeterDetector()
    monkeypatch.setattr(loader, "YOLOInferer", yolo)
    monkeypatch.setattr(loader, "CNNInferer", lambda *a, **k: DigitOCR())
    monkeypatch.setattr(loader, "CRNNInferer", lambda *a, **k: SerialOCR())


def image(name):
    # у каждого фото своё содержимое: фото узнаётся в логе по отпечатку байтов,
    # и одинаковые картинки были бы одним и тем же фото
    return np.full((100, 200, 3), 20 + 40 * ORDER.index(name), dtype=np.uint8)


def put_photos(folder: MonthFolder) -> None:
    folder.photos.mkdir(parents=True, exist_ok=True)
    for name in ORDER:
        cv2.imwrite(str(folder.photos / name), image(name))


def setup_month(root: Path) -> MonthFolder:
    """Месяц из TABLE и пять фото в <месяц>/фото."""
    folder = make_month(Path(root), TABLE)
    put_photos(folder)
    return folder


def run(folder: MonthFolder, **kw) -> None:
    reader.run_pipeline(reader.PipelineConfig(month_dir=str(folder.root), **kw))


def log(folder: MonthFolder) -> list[dict]:
    with MonthDB(folder.db) as db:
        return db.log_rows()


def log_by_name(folder: MonthFolder) -> dict:
    return {r["original_filename"]: r for r in log(folder)}


def log_without_timestamps(folder: MonthFolder) -> list[dict]:
    return [dict(r, processed_at="") for r in log(folder)]


def reading(folder: MonthFolder, account: str) -> str:
    with MonthDB(folder.db) as db:
        r = db.reading(account)
    return r.value if r else ""


def output_tree(folder: MonthFolder) -> list[str]:
    # через «/» и на Windows — тесты сравнивают с путями, записанными текстом
    return sorted(p.relative_to(folder.results).as_posix() for p in folder.results.rglob("*.jpg"))
