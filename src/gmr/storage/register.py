"""
src/gmr/storage/register.py — чтение таблицы абонентов (реестра) месяца
(этап 2.2a, 2026-10-01).

Реестр приходит от компании в Excel: обычно старый формат .xls (сохранён в
WPS), бывает .xlsx; старые таблицы владельца — .csv. Читается первый лист,
заголовки — в первой строке. Все значения приводятся к тексту так, как их
видит человек в Excel:
  - число 1234567890 → "1234567890" (а не "1234567890.0": в реестре 4134
    лицевых счёта из 9002 хранятся числами);
  - текст — без пробелов по краям, ведущие нули сохраняются ("0012345");
  - дата → "дд мм гггг" (так в реестре пишут столбец «Дата»);
  - пустая ячейка и ошибка формулы → "".

Только чтение: исходный файл программа не меняет.
"""
import csv
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

DATE_FORMAT = "%d %m %Y"

# Лишние знаки по краям номера счётчика — опечатки ручного ввода
# («1234567.», «123456,», «.123456»): в реестре 100 таких номеров из 9002.
_SERIAL_EDGE_JUNK = re.compile(r"^[\s.,/\\]+|[\s.,/\\]+$")


@dataclass
class Register:
    columns: list[str]           # заголовки в порядке файла
    rows: list[dict[str, str]]   # строки без пустых, значения — текст
    sheet: str = "Лист1"         # имя листа (выгрузка называет лист так же)
    # столбцы, где в исходнике числа в большинстве: выгрузка пишет их числами,
    # остальные — текстом (телефон «89001234567» остаётся текстом, как был)
    numeric_columns: list = None


def clean_serial(text: str) -> str:
    """Номер счётчика без пробелов и знаков . , / \\ по краям. Внутри и ведущие
    нули не трогаются."""
    return _SERIAL_EDGE_JUNK.sub("", text)


def number_text(value: float) -> str:
    """Число из ячейки → текст: целое без «.0», дробное — как есть."""
    if value == int(value):
        return str(int(value))
    return repr(value)


def _cell_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return number_text(float(value)) if isinstance(value, float) else str(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.strftime(DATE_FORMAT)
    return str(value).strip()


_NUMBER_TEXT = re.compile(r"-?\d+(\.\d+)?")


def _read_xls(path: Path) -> tuple[str, list[list[str]], list[list[bool]]]:
    import xlrd
    book = xlrd.open_workbook(str(path))
    sheet = book.sheet_by_index(0)
    out, is_num = [], []
    for r in range(sheet.nrows):
        row = []
        is_num.append([sheet.cell_type(r, c) == xlrd.XL_CELL_NUMBER for c in range(sheet.ncols)])
        for c in range(sheet.ncols):
            cell = sheet.cell(r, c)
            if cell.ctype == xlrd.XL_CELL_DATE:
                row.append(xlrd.xldate_as_datetime(cell.value, book.datemode).strftime(DATE_FORMAT))
            elif cell.ctype == xlrd.XL_CELL_NUMBER:
                row.append(number_text(cell.value))
            elif cell.ctype == xlrd.XL_CELL_TEXT:
                row.append(cell.value.strip())
            elif cell.ctype == xlrd.XL_CELL_BOOLEAN:
                row.append("1" if cell.value else "0")
            else:                                    # пусто, ошибка формулы
                row.append("")
        out.append(row)
    return sheet.name, out, is_num


def _read_xlsx(path: Path) -> tuple[str, list[list[str]], list[list[bool]]]:
    import openpyxl
    book = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    try:
        sheet = book.worksheets[0]
        rows = list(sheet.iter_rows(values_only=True))
        is_num = [[isinstance(v, (int, float)) and not isinstance(v, bool) for v in row] for row in rows]
        return sheet.title, [[_cell_text(v) for v in row] for row in rows], is_num
    finally:
        book.close()


def _read_csv(path: Path) -> tuple[str, list[list[str]], list[list[bool]]]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1251")          # выгрузка из 1С / Excel по-русски
    first = text.splitlines()[0] if text else ""
    delimiter = ";" if first.count(";") > first.count(",") else ","
    grid = [[c.strip() for c in row] for row in csv.reader(text.splitlines(), delimiter=delimiter)]
    # в CSV типов нет: число — то, что выглядит как число
    return "Лист1", grid, [[bool(_NUMBER_TEXT.fullmatch(v)) for v in row] for row in grid]


def read_register(path: str) -> Register:
    """Читает реестр (.xls, .xlsx, .csv). Ошибка ValueError — если формат не тот
    или в файле нет строки заголовков."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".xls":
        sheet, grid, is_num = _read_xls(p)
    elif ext in (".xlsx", ".xlsm"):
        sheet, grid, is_num = _read_xlsx(p)
    elif ext == ".csv":
        sheet, grid, is_num = _read_csv(p)
    else:
        raise ValueError(f"Таблица должна быть .xls, .xlsx или .csv, а не «{p.suffix}»: {p}")
    if not grid or not any(grid[0]):
        raise ValueError(f"В таблице нет строки заголовков: {p}")

    columns = [h.strip() for h in grid[0]]
    while columns and not columns[-1]:           # пустые столбцы справа
        columns.pop()
    rows = []
    filled = [0] * len(columns)
    numbers = [0] * len(columns)
    for values, nums in zip(grid[1:], is_num[1:]):
        values = list(values) + [""] * (len(columns) - len(values))
        row = {col: values[i] for i, col in enumerate(columns) if col}
        if any(row.values()):
            rows.append(row)
            for i in range(len(columns)):
                if values[i]:
                    filled[i] += 1
                    numbers[i] += i < len(nums) and nums[i]
    numeric = [c for i, c in enumerate(columns) if c and filled[i] and numbers[i] * 2 > filled[i]]
    return Register(columns=[c for c in columns if c], rows=rows, sheet=sheet or "Лист1",
                    numeric_columns=numeric)
