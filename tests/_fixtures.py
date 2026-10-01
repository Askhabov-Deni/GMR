"""
tests/_fixtures.py — fake-модели и билдеры данных для тестов reader.py
(golden tests — docs/MIGRATION_TZ.md, раздел 2).

Тесты не грузят веса и не создают настоящие YOLOInferer/CRNNInferer/
CNNInferer. Вместо них — лёгкие классы с тем же интерфейсом
(process_image/process_array/predict/predict_with_details), обёрнутые в те же
адаптеры, что и настоящие модели (src/gmr/ml/adapters.py) — см. process_photo
ниже. torch/ultralytics при этом всё равно должны быть установлены: reader.py
импортирует загрузчик моделей.
"""
import numpy as np
import pandas as pd

import reader  # noqa: E402  (репозиторий уже в sys.path благодаря корневому conftest.py)
from src.gmr.ml import (
    CnnDigitRecognizer, CrnnSerialRecognizer, YoloDigitDetector, YoloMeterDetector,
)


# ─── Fake-детекторы / OCR (реализуют интерфейс реальных Inferer-классов) ─────

class FakeMeterDetector:
    """Имитирует YOLOInferer(meter_detect_model). process_image() -> list[dict]."""

    def __init__(self, crops):
        self._crops = crops

    def process_image(self, img_path, save_crops=False, max_per_class=1, straighten=None):
        return self._crops


class FakeDigitDetector:
    """Имитирует YOLOInferer(digit_detect_model, straighten=False). process_array() -> list[dict]."""

    def __init__(self, crops):
        self._crops = crops

    def process_array(self, img, save_crops=False, max_per_class=5, straighten=None):
        return self._crops


class FakeDigitOCR:
    """
    Имитирует CNNInferer.predict(crop) -> {"digit": int, "confidence": float}.

    predictions — dict {id(crop_array): (digit_str, confidence)}, ключуется
    по id() самого np.ndarray-кропа, который тест положил в digit-crops.
    """

    def __init__(self, predictions):
        self._predictions = predictions

    def predict(self, image_input):
        digit, conf = self._predictions[id(image_input)]
        return {"digit": int(digit), "confidence": float(conf)}


class FakeSerialOCR:
    """Имитирует CRNNInferer.predict_with_details(crop) -> {"text","avg_confidence","details"}."""

    def __init__(self, text, avg_confidence):
        self._text = text
        self._avg_confidence = avg_confidence

    def predict_with_details(self, image_input):
        return {"text": self._text, "avg_confidence": self._avg_confidence, "details": []}


def process_photo(photo_path, df, config, meter_detector, digit_detector,
                  digit_ocr, serial_ocr, **kwargs):
    """
    reader.process_photo с fake-моделями в интерфейсе YOLOInferer/CNNInferer/
    CRNNInferer. С Фазы 3 process_photo принимает модели в виде контрактов
    (src/gmr/domain/ml.py) — fake оборачиваются теми же адаптерами, что и
    реальные модели в src/gmr/ml/loader.py, так что адаптеры тоже под тестом.
    """
    return reader.process_photo(
        photo_path, df, config,
        YoloMeterDetector(meter_detector), YoloDigitDetector(digit_detector),
        CnnDigitRecognizer(digit_ocr), CrnnSerialRecognizer(serial_ocr),
        **kwargs,
    )


# ─── Вспомогательные билдеры данных ──────────────────────────────────────────

def make_crop(h=20, w=20):
    """Уникальный np.ndarray-кроп (нужна разная identity для digit_ocr predictions dict)."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_meter_crops(meter_bbox=(0, 0, 300, 100), serial_bbox=(0, 100, 150, 130)):
    """Стандартный результат meter_detector.process_image(): gas_meter + serial_number."""
    return [
        {"class": "gas_meter", "conf": 0.95, "angle": 0.0,
         "bbox": meter_bbox, "crop": make_crop(100, 300), "path": None},
        {"class": "serial_number", "conf": 0.95, "angle": 0.0,
         "bbox": serial_bbox, "crop": make_crop(30, 150), "path": None},
    ]


def make_digit_crops_with_centers(x_starts, width=20, height=30):
    """
    Строит список digit-кропов (класс YOLO output dict) с заданными
    x1-координатами (bbox[0] используется для сортировки в _read_meter_digits,
    а center = (x1+x2)//2 — для поиска gap в алгоритме восстановления).
    Возвращает (crops, real_crop_arrays) — real_crop_arrays в исходном
    порядке x_starts, для последующей привязки в FakeDigitOCR.predictions.
    """
    crops = []
    real_arrays = []
    for x1 in x_starts:
        arr = make_crop(height, width)
        real_arrays.append(arr)
        crops.append({
            "class": "digit", "conf": 0.95, "angle": 0.0,
            "bbox": (x1, 0, x1 + width, height), "crop": arr, "path": None,
        })
    return crops, real_arrays


def make_df(rows, col_serial="Номер счетчика", col_account_id="Лицевой счет",
            col_last_reading="Последние показания", col_new_reading="Текущие показания"):
    """rows — list[dict] с ключами serial/account_id/last_reading/new_reading (может быть '')."""
    return pd.DataFrame([
        {
            col_serial: r.get("serial", ""),
            col_account_id: r.get("account_id", ""),
            col_last_reading: r.get("last_reading", ""),
            col_new_reading: r.get("new_reading", ""),
        }
        for r in rows
    ])
