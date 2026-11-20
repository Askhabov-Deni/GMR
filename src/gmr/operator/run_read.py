"""
Что прочитала обработка на фото из очереди (этап Е, решение владельца
2026-11-21): окно показывает это сразу и само моделей не запускает.

Номер — тот, с которым обработка искала абонента (serial_id её строки лога).
Показание, уверенность цифр и кропы — из базы месяца (MonthDB.recognition,
пишет src/gmr/pipeline/seen.py). Фото, обработанные до этапа Е, этой записи не
имеют: номер и показание тогда — из строки лога, кропов нет (тихая разметка
по такому фото не сохраняется).
"""
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.gmr.storage.month import Recognition

from .photos import _to_float

DIGITS = 5


@dataclass
class RunRead:
    found:       bool = False             # есть строка обработки этого фото
    serial:      str = ""                 # номер в поле окна
    serial_read: str = ""                 # номер, как прочитала модель (разметка: что исправил оператор)
    serial_conf: Optional[float] = None
    reading_str: str = ""                 # «?» — неуверенная цифра
    digits_ok:   list = field(default_factory=list)
    crops:       dict = field(default_factory=dict)   # JPEG: serial, digit0 … digit4

    def digit_preds(self) -> Optional[list]:
        """По позициям {"ok": …} — как ждут поле показаний и разметка; None — не известно."""
        return [{"ok": ok} for ok in self.digits_ok] or None

    def serial_crop(self) -> Optional[np.ndarray]:
        return _decode(self.crops.get("serial"))

    def digit_crops(self) -> Optional[list]:
        """Кропы цифр по позициям (None — заглушка или нет кропа); None — кропов нет совсем."""
        if not any(f"digit{i}" in self.crops for i in range(DIGITS)):
            return None
        return [_decode(self.crops.get(f"digit{i}")) for i in range(DIGITS)]


def _decode(jpeg: Optional[bytes]) -> Optional[np.ndarray]:
    if not jpeg:
        return None
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def run_read_of(auto_row: Optional[dict], rec: Optional[Recognition]) -> RunRead:
    """auto_row — строка обработки, создавшая файл в question/ (None — её нет);
    rec — что прочитали модели (None — не записано)."""
    if auto_row is None:
        return RunRead()
    if rec is None:
        serial = (auto_row.get("serial_id") or "").strip()
        return RunRead(True, serial, serial, _to_float(auto_row.get("model_serial_conf")),
                       auto_row.get("model_reading_str") or "")
    return RunRead(
        found=True,
        # счётчик не найден — у строки номера нет, а модель его прочитала
        serial=(auto_row.get("serial_id") or "").strip() or rec.serial_read,
        serial_read=rec.serial_read,
        serial_conf=rec.serial_conf,
        reading_str=rec.reading_str,
        digits_ok=list(rec.digits_ok),
        crops=dict(rec.crops),
    )
