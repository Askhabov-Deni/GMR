"""Подсказки оператору: похожие номера в базе, выбор абонента при одном номере у нескольких."""
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.gmr.domain import PipelineConfig
from src.gmr.domain.serial_match import alphabet_for, one_char_matches
from src.gmr.domain.serial_match import normalize_serial as _normalize_serial

from .photos import _to_float

HINT_NOTE = "подсказка"
CHOICE_NOTE = "номер у нескольких абонентов, выбран"   # notes: выбор оператора (2026-10-01)


@dataclass
class SerialIndex:
    """Серийники таблицы для поиска похожих: номер → (лицевой счёт, последнее показание)."""
    by_serial: dict
    alphabet:  str


@dataclass
class SerialHint:
    serial:       str               # номер в таблице
    account:      str
    last_reading: Optional[float]
    delta:        Optional[float]   # показание − последнее показание


def serial_choices(matches: pd.DataFrame, cfg: PipelineConfig, reading_str: str) -> list[SerialHint]:
    """Абоненты, у которых в таблице записан этот номер (по одному на лицевой
    счёт, в порядке таблицы). Больше одного — у одной записи в базе номер с
    ошибкой, выбирает оператор (решение владельца 2026-10-01)."""
    reading = int(reading_str) if reading_str and reading_str.isdigit() else None
    out, seen = [], set()
    for serial, account, last in zip(matches[cfg.col_serial], matches[cfg.col_account_id],
                                     matches[cfg.col_last_reading]):
        account = str(account).strip()
        if account in seen:
            continue
        seen.add(account)
        last = _to_float(last)
        delta = (reading - last) if reading is not None and last is not None else None
        out.append(SerialHint(str(serial).strip(), account, last, delta))
    return out


def build_serial_index(df: Optional[pd.DataFrame], cfg: PipelineConfig) -> Optional[SerialIndex]:
    if df is None:
        return None
    by_serial = {}
    for serial, account, last in zip(df[cfg.col_serial], df[cfg.col_account_id], df[cfg.col_last_reading]):
        key = _normalize_serial(str(serial))
        if key and key != "nan" and key not in by_serial:
            by_serial[key] = (str(account).strip(), _to_float(last))
    return SerialIndex(by_serial, alphabet_for(set(by_serial)))


def serial_hints(serial: str, reading_str: str, index: Optional[SerialIndex], limit: int = 3) -> list[SerialHint]:
    """
    Номера из таблицы «в одном символе» от serial. Первыми — с наименьшим
    расходом (|показание − последнее|), если показание прочитано полностью.
    """
    serial = (serial or "").strip()
    if index is None or len(serial) < 3:
        return []
    reading = int(reading_str) if reading_str and reading_str.isdigit() else None
    hints = []
    for t in sorted({t for _, t in one_char_matches(serial, set(index.by_serial), index.alphabet)}):
        account, last = index.by_serial[t]
        delta = (reading - last) if reading is not None and last is not None else None
        hints.append(SerialHint(t, account, last, delta))
    hints.sort(key=lambda h: (h.delta is None, abs(h.delta) if h.delta is not None else 0))
    return hints[:limit]
