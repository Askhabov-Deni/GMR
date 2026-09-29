"""
src/gmr/ml/adapters.py — обёртки реальных моделей в контракты
src/gmr/domain/ml.py (Фаза 3).

Адаптер только переводит вызов контракта в вызов модели и ответ модели —
в тип контракта. Внутреннее поведение моделей не меняется: аргументы те же,
что передавал reader.py до Фазы 3.

  YoloMeterDetector   → YOLOInferer.process_image(path, save_crops=False)
  YoloDigitDetector   → YOLOInferer.process_array(crop, save_crops=False, max_per_class=5)
  CrnnSerialRecognizer→ CRNNInferer.predict_with_details(crop)
  CnnDigitRecognizer  → CNNInferer.predict(crop)

Выпрямление кропов (straighten) — настройка самого YOLOInferer при создании
(включено для детектора счётчика, выключено для детектора цифр, см.
src/gmr/ml/loader.py). Адаптеры его не переопределяют.

Модуль не импортирует torch/ultralytics: адаптер принимает уже созданный
объект модели (или fake в тестах).
"""
from typing import Any, Optional

from src.gmr.domain.ml import Detection, DigitPrediction, SerialPrediction


class YoloMeterDetector:
    """MeterDetector поверх YOLOInferer(meter_detect_model)."""

    def __init__(self, inferer: Any):
        self.inferer = inferer

    def detect(self, photo_path: str) -> Optional[list[Detection]]:
        return self.inferer.process_image(photo_path, save_crops=False)


class YoloDigitDetector:
    """DigitDetector поверх YOLOInferer(digit_detect_model, straighten=False)."""

    MAX_DIGITS = 5   # max_per_class, как в reader.py до Фазы 3

    def __init__(self, inferer: Any):
        self.inferer = inferer

    def detect(self, meter_crop: Any) -> Optional[list[Detection]]:
        return self.inferer.process_array(
            meter_crop, save_crops=False, max_per_class=self.MAX_DIGITS,
        )


class CrnnSerialRecognizer:
    """SerialRecognizer поверх CRNNInferer."""

    def __init__(self, inferer: Any):
        self.inferer = inferer

    def recognize(self, serial_crop: Any) -> SerialPrediction:
        res = self.inferer.predict_with_details(serial_crop)
        return SerialPrediction(
            text=res["text"],
            confidence=res["avg_confidence"],
            details=res.get("details", []),
        )


class CnnDigitRecognizer:
    """DigitRecognizer поверх CNNInferer."""

    def __init__(self, inferer: Any):
        self.inferer = inferer

    def recognize(self, digit_crop: Any) -> DigitPrediction:
        res = self.inferer.predict(digit_crop)
        return DigitPrediction(digit=str(res["digit"]), confidence=res["confidence"])
