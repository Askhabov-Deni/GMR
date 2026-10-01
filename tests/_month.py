"""
Папка месяца для тестов окна оператора (этап 2.3): таблица → база месяца,
строки лога в базе, фото в <месяц>/результат/<контролёр>/…
"""
import csv
from pathlib import Path

import cv2
import numpy as np

from src.gmr.application import month
from src.gmr.storage import LOG_COLUMNS
from src.gmr.storage.month import MonthDB, MonthFolder

TABLE_HEAD = ["Номер счетчика", "Лицевой счет", "Последние показания", "Текущие показания"]


def make_month(root: Path, table: list, log_rows: list = (), name: str = "Октябрь") -> MonthFolder:
    """table — строки (номер, л/с, последние, текущие); log_rows — строки лога
    (недостающие столбцы — пустые). Текущие показания из таблицы попадают в
    базу как «было в таблице» (source=table)."""
    src = Path(root) / f"{name}_таблица.csv"
    with open(src, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(TABLE_HEAD)
        w.writerows(table)
    month.load_table(str(Path(root) / name), str(src), who="тест")
    folder = MonthFolder(Path(root) / name)
    if log_rows:
        with MonthDB(folder.db) as db, db.transaction():
            db.append_log_rows([{c: "" for c in LOG_COLUMNS} | dict(r) for r in log_rows])
    return folder


def img(path: Path, value: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((40, 40, 3), value, np.uint8))
    return path


def readings(folder: MonthFolder) -> dict:
    with MonthDB(folder.db) as db:
        return {a: r.value for a, r in db.readings().items()}


def log_rows(folder: MonthFolder) -> list:
    with MonthDB(folder.db) as db:
        return db.log_rows(with_id=True)


def changes(folder: MonthFolder) -> list:
    with MonthDB(folder.db) as db:
        return db.changes()


def serial_of(folder: MonthFolder, account: str) -> str:
    with MonthDB(folder.db) as db:
        return db.abonent(account).serial
