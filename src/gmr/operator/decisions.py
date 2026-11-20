"""Решения оператора без окна: исправление на «Проверке», номер для «Нет в базе»."""
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.gmr.domain import PipelineConfig
from src.gmr.domain.serial_match import lookup_variants
from src.gmr.domain.serial_match import normalize_serial as _normalize_serial

from .photos import _to_float


@dataclass
class VerifyCorrection:
    """Что изменится, если сохранить исправление на вкладке «Проверка»."""
    error:           Optional[str] = None     # не сохранять, показать оператору
    serial:          str = ""
    reading:         int = 0
    old_account:     str = ""
    new_account:     str = ""
    last_reading:    Optional[float] = None   # последнее показание (нового) абонента
    delta:           Optional[float] = None
    outcome:         str = "PLUS"
    new_account_has_reading: str = ""         # уже записанное показание у нового абонента

    @property
    def account_changed(self) -> bool:
        return self.new_account != self.old_account


def plan_verify_correction(df: Optional[pd.DataFrame], cfg: PipelineConfig, row: dict,
                           new_serial: str, new_reading_str: str) -> VerifyCorrection:
    """
    Исправление при проверке. Серийник тот же — меняется только показание
    этого абонента. Серийник другой — ищем его в таблице (как reader.py:
    номер, '0'+номер, '00'+номер); фото принадлежит найденному абоненту,
    показание переносится к нему.
    """
    serial = new_serial.strip()
    c = VerifyCorrection(serial=serial, reading=int(new_reading_str),
                         old_account=row.get("account_id", ""),
                         new_account=row.get("account_id", ""),
                         last_reading=_to_float(row.get("last_reading")))
    if _normalize_serial(serial) != _normalize_serial(row.get("serial_id", "")):
        if df is None:
            return VerifyCorrection(error="Таблица не загружена — сменить абонента нельзя.")
        match = None
        for cand in lookup_variants(serial):
            m = df[df[cfg.col_serial].apply(lambda x: _normalize_serial(str(x)) == _normalize_serial(cand))]
            if not m.empty:
                match = m.iloc[0]
                break
        if match is None:
            return VerifyCorrection(error=f"Серийного номера {serial} нет в таблице.")
        c.new_account = str(match[cfg.col_account_id]).strip()
        c.last_reading = _to_float(match[cfg.col_last_reading])
        if c.account_changed:
            existing = _to_float(match[cfg.col_new_reading])
            c.new_account_has_reading = str(match[cfg.col_new_reading]).strip() if existing is not None else ""
    c.delta = (c.reading - c.last_reading) if c.last_reading is not None else None
    c.outcome = "MINUS" if c.delta is not None and c.delta < 0 else "PLUS"
    return c


# ─── «Нет в базе»: какой номер записать (решение владельца 2026-11-21) ───────
# Номер в строке «Нет в базе» нужен прогону: если номер потом появится в
# таблице, фото прочитается заново (RunSelectionPolicy). Окно читает копию
# фото с подписью из question/ и может прочитать другой номер, чем прогон с
# исходного фото (тест месяца 2026-11-20: одна цифра другая, и такой номер у
# другого абонента есть в таблице). По такому номеру окно находило абонента,
# номер из таблицы попадал в строку, и следующий прогон сразу возвращал фото
# в очередь.

def accounts_with_serial(df: Optional[pd.DataFrame], cfg: PipelineConfig, serial: str) -> list[str]:
    """Лицевые счета таблицы с этим номером — так же, как ищет прогон: номер,
    '0'+номер, '00'+номер. Пустой номер — ни одного."""
    serial = _normalize_serial(serial or "")
    if df is None or not serial:
        return []
    variants = set(lookup_variants(serial))
    mask = df[cfg.col_serial].apply(lambda x: _normalize_serial(str(x)) in variants)
    return list(dict.fromkeys(str(a).strip() for a in df.loc[mask, cfg.col_account_id]))


def not_in_db_serial(field: str, model_serial: Optional[str], run_serial: str) -> str:
    """Номер для строки «Нет в базе». Оператор вписал свой номер — его номер.
    Иначе (в поле номер, прочитанный окном, или пусто) — номер, который
    прочитал прогон с исходного фото: его же прочитает и следующий прогон."""
    field = (field or "").strip()
    if field and field != (model_serial or "").strip():
        return field
    return (run_serial or "").strip() or field
