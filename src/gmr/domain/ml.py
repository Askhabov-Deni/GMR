"""
src/gmr/domain/ml.py — контракты ML-моделей (Фаза 3, docs/MIGRATION_TZ.md).

Роли, которые модели играют в пайплайне:

  MeterDetector     — фото → детекции gas_meter / serial_number / надписи
                      маркером (YOLO)
  SerialRecognizer  — кроп серийника → текст + уверенность (CRNN)
  DigitDetector     — кроп счётчика → детекции цифр (YOLO)
  DigitRecognizer   — кроп одной цифры → цифра + уверенность (CNN)
  AccountRecognizer — кроп надписи маркером → лицевой счёт из таблицы +
                      уверенность (CRNN, models/account). Необязательная:
                      без весов (PipelineConfig.account_ocr_model = "") её нет.

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


@dataclass
class AccountPrediction:
    """Надпись маркером.

    text / text_conf — что написано (жадное чтение) и средняя уверенность
    символов; account / confidence — лицевой счёт из таблицы, к которому
    надпись подходит лучше всего, и его доля среди всех счетов таблицы
    (models/ctc_lexicon.py); top — [(счёт, доля), ...] по убыванию.
    Без таблицы account=None, confidence=0."""
    text: str
    text_conf: float
    account: Optional[str] = None
    confidence: float = 0.0
    top: list = field(default_factory=list)


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


@runtime_checkable
class AccountRecognizer(Protocol):
    def recognize(self, account_crop: Any, accounts: Optional[dict] = None) -> AccountPrediction:
        """accounts — {лицевой счёт: (варианты записи, ...)}
        (src/gmr/domain/account_match.account_groups); один и тот же объект
        на весь прогон — модель готовит его один раз."""
        ...
