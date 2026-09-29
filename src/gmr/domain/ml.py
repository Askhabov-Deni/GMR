"""
src/gmr/domain/ml.py — контракты ML-моделей (Фаза 3, docs/MIGRATION_TZ.md).

Четыре роли, которые модели играют в пайплайне:

  MeterDetector    — фото → детекции gas_meter / serial_number (YOLO)
  SerialRecognizer — кроп серийника → текст + уверенность (CRNN)
  DigitDetector    — кроп счётчика → детекции цифр (YOLO)
  DigitRecognizer  — кроп одной цифры → цифра + уверенность (CNN)

Бизнес-логика (reader.py, src/gmr/application/) работает только через эти
контракты и не знает, какая модель стоит за ними. Реальные YOLOInferer /
CRNNInferer / CNNInferer оборачиваются адаптерами из src/gmr/ml/adapters.py.

Здесь нет torch/cv2/numpy: кропы — это просто объекты, которые модель
отдала и которые другая модель примет (на практике — np.ndarray BGR).

Детекция — dict в том виде, как её отдаёт YOLOInferer (формат не менялся):
  {"class": str, "conf": float, "angle": float,
   "bbox": (x1, y1, x2, y2), "crop": <изображение>, "path": str | None}
bbox — в координатах изображения, поданного на вход детектору.
"""
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

Detection = dict


@dataclass
class SerialPrediction:
    text: str
    confidence: float
    details: list = field(default_factory=list)


@dataclass
class DigitPrediction:
    digit: str          # "0".."9"
    confidence: float


@runtime_checkable
class MeterDetector(Protocol):
    def detect(self, photo_path: str) -> Optional[list[Detection]]:
        """Детекции на фото; пусто/None — ничего не найдено."""
        ...


@runtime_checkable
class SerialRecognizer(Protocol):
    def recognize(self, serial_crop: Any) -> SerialPrediction:
        ...


@runtime_checkable
class DigitDetector(Protocol):
    def detect(self, meter_crop: Any) -> Optional[list[Detection]]:
        """Детекции цифр на кропе счётчика (bbox — в координатах кропа)."""
        ...


@runtime_checkable
class DigitRecognizer(Protocol):
    def recognize(self, digit_crop: Any) -> DigitPrediction:
        ...
