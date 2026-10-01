"""
src/gmr/storage/table.py — таблица счётчиков (CSV/XLSX) — Фаза 6.

Перенесено из reader.py (_load_table/_save_table) без изменений поведения,
чтобы program2.py не импортировал приватные функции reader.py. В reader.py
старые имена остались ссылками на эти функции.
"""
import csv
from pathlib import Path

import pandas as pd


def load_table(path: str) -> pd.DataFrame:
    """
    Загружает таблицу счётчиков.

    ВАЖНО — ведущие нули:

    CSV:  dtype=str достаточно — pandas читает ячейки как текст.

    Excel: dtype=str НЕ спасает числовые ячейки. Если ячейка хранится
    как число (openpyxl отдаёт int), dtype=str превратит 42306 в "42306",
    а не в "0042306". Используем converters={col: str} — это заставляет
    pandas вызывать str() на сыром значении из openpyxl. Текстовые ячейки
    вернут строку как есть, числовые — str(int), т.е. без ведущих нулей.
    Полная защита от потери нулей обеспечивается fallback-ом с добавлением
    нулей в process_photo (шаг 3).
    """
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        _tmp = pd.read_excel(path, nrows=0)
        converters = {col: str for col in _tmp.columns}
        return pd.read_excel(path, converters=converters)
    return pd.read_csv(path, dtype=str)


def save_table(df: pd.DataFrame, path: str) -> None:
    """
    Сохраняет таблицу обратно на диск.

    Для CSV принудительно используем quoting=QUOTE_ALL, чтобы ведущие нули
    в строковых столбцах не потерялись при следующей загрузке сторонними
    инструментами (Excel, pandas без dtype=str и т.д.).
    """
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
