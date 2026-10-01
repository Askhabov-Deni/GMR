"""
src/gmr/domain/models.py — типы результата обработки фото.

Перенесено из reader.py в рамках Фазы 2a (docs/MIGRATION_TZ.md) БЕЗ изменения
полей, порядка значений enum или дефолтов — это прямой перенос, не рефакторинг
данных. Поведение подтверждается golden tests (tests/test_golden_reader.py,
19 кейсов, покрывают все 9 значений Outcome; ERROR добавлен 2026-10-01 —
tests/test_photo_error.py).

reader.py импортирует эти имена и реэкспортирует их (from .domain import ...),
так что `reader.Outcome`, `reader.PhotoResult`, `reader.OUTCOME_FOLDER`
продолжают работать для внешнего кода. program2.py с Фазы 6 берёт их отсюда.
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class Outcome(Enum):
    PLUS             = auto()
    MINUS            = auto()
    REPEAT           = auto()
    NO_METER         = auto()
    NO_SERIAL        = auto()
    SERIAL_LOW_CONF  = auto()
    SERIAL_NOT_FOUND = auto()
    DIGITS_ERROR     = auto()
    SUSPICIOUS       = auto()
    # программа упала на этом фото (битый файл, ошибка модели) — прогон идёт
    # дальше, фото ждёт оператора (решение владельца 2026-10-01)
    ERROR            = auto()


OUTCOME_FOLDER = {
    Outcome.PLUS:             "plus",
    Outcome.MINUS:            "minus",
    Outcome.REPEAT:           "repeat",
    Outcome.NO_METER:         "question/no_meter",
    Outcome.NO_SERIAL:        "question/no_serial",
    Outcome.SERIAL_LOW_CONF:  "question/serial_low_conf",
    Outcome.SERIAL_NOT_FOUND: "question/serial_not_found",
    Outcome.DIGITS_ERROR:     "question/digits_error",
    Outcome.SUSPICIOUS:       "question/suspicious",
    Outcome.ERROR:            "question/error",
}


@dataclass
class PhotoResult:
    photo_path:     str
    outcome:        Outcome
    serial_text:    Optional[str]   = None
    serial_conf:    Optional[float] = None
    account_id:     Optional[str]   = None
    reading:        Optional[int]   = None
    # ↓ Частично распознанные показания, например "5?3?1"
    #   заполняется даже при DIGITS_ERROR — для аннотации на фото
    reading_str:    Optional[str]   = None
    last_reading:   Optional[float] = None
    delta:          Optional[float] = None
    new_photo_name: Optional[str]   = None
    error_detail:   Optional[str]   = None
    # ↓ Данные для отрисовки боксов (заполняются в process_photo)
    # meter_crops_raw — список dict из meter_detector (gas_meter, serial_number)
    meter_crops_raw: Optional[list] = None
    # digit_bboxes_in_orig — боксы цифр в координатах оригинала [(x1,y1,x2,y2), ...]
    digit_bboxes_in_orig: Optional[list] = None
    # ↓ Какие цифры в reading_str не прочитаны, а подставлены (заглушка на месте
    #   пропущенной цифры, прощённые хвостовые) — пишется в notes лога.
    #   None — подстановок не было. Решение владельца 2026-09-30 (вариант А).
    digit_notes:    Optional[str]   = None
