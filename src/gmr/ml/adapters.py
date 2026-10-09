"""
src/gmr/ml/adapters.py — обёртки реальных моделей в контракты
src/gmr/domain/ml.py (Фаза 3).

Адаптер только переводит вызов контракта в вызов модели и ответ модели —
в тип контракта. Внутреннее поведение моделей не меняется: аргументы те же,
что передавал reader.py до Фазы 3.

  YoloMeterDetector   → YOLOInferer.process_image(path, save_crops=False)
                        (повёрнутое фото — process_array, этапы 7a/7b)
  YoloDigitDetector   → YOLOInferer.process_array(crop, save_crops=False, max_per_class=5)
  CrnnSerialRecognizer→ CRNNInferer.predict_with_details(crop)
  CnnDigitRecognizer  → CNNInferer.predict(crop)
  CrnnAccountRecognizer → AccountInferer.predict(crop, accounts)

Выпрямление кропов (straighten) — настройка самого YOLOInferer при создании
(включено для детектора счётчика, выключено для детектора цифр, см.
src/gmr/ml/loader.py). Адаптеры его не переопределяют.

Модуль не импортирует torch/ultralytics: адаптер принимает уже созданный
объект модели (или fake в тестах).
"""
from typing import Any, Optional

from src.gmr.domain.ml import AccountPrediction, Detection, DigitPrediction, SerialPrediction, SerialTableMatch
from src.gmr.render.image_io import TURNS, read_image, turn_image


class YoloMeterDetector:
    """MeterDetector поверх YOLOInferer(meter_detect_model)."""

    def __init__(self, inferer: Any):
        self.inferer = inferer

    def detect(self, photo_path: str) -> Optional[list[Detection]]:
        return self.inferer.process_image(photo_path, save_crops=False)

    def detect_array(self, img: Any) -> Optional[list[Detection]]:
        """То же для картинки в памяти (замер наклона: фото, повёрнутое на 90°…)."""
        return self.inferer.process_array(img, save_crops=False)

    def detect_turned(self, photo_path: str) -> tuple[Optional[list[Detection]], int]:
        """Фото, повёрнутое на 90, 180, 270° по часовой (этап 7b): первый поворот,
        на котором найдены показания (gas_meter) — (детекции, градусы);
        ни на одном — (None, 0). Рамки — в координатах повёрнутого фото."""
        img = read_image(photo_path)
        if img is None:
            return None, 0
        for degrees in TURNS:
            found = self.detect_array(turn_image(img, degrees))
            if any(d["class"] == "gas_meter" for d in found or ()):
                return found, degrees
        return None, 0


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

    def match_table(self, serial_crop: Any, serials: dict) -> SerialTableMatch:
        """Серийник по таблице (этап 6b): номер таблицы, к которому картинка
        подходит лучше всего, и его доля (models/ctc_lexicon.py)."""
        res = self.inferer.predict_in_table(serial_crop, serials)
        return SerialTableMatch(text=res["text"], serial=res["serial"], confidence=res["confidence"])


class CnnDigitRecognizer:
    """DigitRecognizer поверх CNNInferer."""

    def __init__(self, inferer: Any):
        self.inferer = inferer

    def recognize(self, digit_crop: Any) -> DigitPrediction:
        res = self.inferer.predict(digit_crop)
        return DigitPrediction(digit=str(res["digit"]), confidence=res["confidence"],
                               top=[(str(d), float(p)) for d, p in res.get("top", [])])


class CrnnAccountRecognizer:
    """AccountRecognizer поверх AccountInferer (models/account/infer_account.py)."""

    def __init__(self, inferer: Any):
        self.inferer = inferer

    def recognize(self, account_crop: Any, accounts: Optional[dict] = None) -> AccountPrediction:
        res = self.inferer.predict(account_crop, accounts)
        return AccountPrediction(
            text=res["text"], text_conf=res["text_conf"], account=res["group"],
            confidence=res["confidence"], top=res.get("top", []),
        )
