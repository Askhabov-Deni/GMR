"""Файлы решений: список «Серийник в базе с ошибкой», раскладка и подпись фото."""
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.gmr.domain import Outcome, PhotoResult, PipelineConfig
from src.gmr.domain.serial_match import normalize_serial as _normalize_serial
from src.gmr.render import draw_annotation as _draw_annotation
from src.gmr.render import read_image, write_image

from .common import log

DB_SERIAL_FIX_LIST    = "db_serial_fix.csv"   # в папке месяца — список для компании
DB_SERIAL_FIX_COLUMNS = ["Фото", "Контролёр", "Лицевой счёт", "Серийник в базе",
                         "Серийник на фото", "Показание", "Дата", "Оператор"]


def table_serial_for_account(df: Optional[pd.DataFrame], cfg: PipelineConfig, account: str) -> Optional[str]:
    """Серийник абонента в таблице (None — лицевого счёта нет)."""
    if df is None or not account:
        return None
    m = df[df[cfg.col_account_id].apply(lambda x: str(x).strip() == account.strip())]
    return None if m.empty else _normalize_serial(str(m.iloc[0][cfg.col_serial]))


def append_db_serial_fix(list_path: str, entry: dict) -> None:
    """Дописывает строку в список для исправления базы (CSV «;», открывается в Excel)."""
    import csv
    p = Path(list_path)
    new = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DB_SERIAL_FIX_COLUMNS, delimiter=";")
        if new:
            w.writeheader()
        w.writerow({k: entry.get(k, "") for k in DB_SERIAL_FIX_COLUMNS})


def free_photo_path(dst_dir: str, name: str, current: str) -> str:
    """Путь для фото в dst_dir; если имя занято другим файлом — добавляется _2, _3…"""
    p = Path(dst_dir) / name
    k = 2
    while p.exists() and p.resolve() != Path(current).resolve():
        p = Path(dst_dir) / f"{Path(name).stem}_{k}{Path(name).suffix}"
        k += 1
    return str(p)


def open_in_explorer(path) -> None:
    """Открыть папку в проводнике (Windows) или файловом менеджере."""
    import os
    import subprocess
    try:
        if hasattr(os, "startfile"):
            os.startfile(str(path))                       # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError as e:
        log.warning(f"Не удалось открыть {path}: {e}")


def describe_reading(r) -> str:
    """Чьё показание и когда — для вопроса «Заменить?»."""
    who = {"auto": f"записала программа по фото {r.photo}" if r.photo else "записала программа",
           "manual": f"записал оператор {r.updated_by}",
           "table": "было в таблице компании"}.get(r.source, f"записал {r.updated_by}")
    return f"{who}, дата {r.date}" if r.date else who


def place_decision(row: dict, photo_path: str, dst_dir: str) -> str:
    """Имя для фото в dst_dir («Дубль», «Нечитаемо», «Нет в базе»): если имя
    занято другим фото — <имя>_2… (этап 3: файлы не затирают друг друга),
    тогда оно же — final_filename строки лога. Возвращает имя."""
    name = Path(free_photo_path(dst_dir, Path(photo_path).name, photo_path)).name
    if name != Path(photo_path).name:
        row["final_filename"] = name
    return name


def move_photo(src: str, dst_dir: str, new_name: Optional[str] = None) -> str:
    """Перемещает фото в dst_dir, возвращает новый путь."""
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    name = new_name or Path(src).name
    dst = str(Path(dst_dir) / name)
    shutil.move(src, dst)
    return dst

def redraw_annotation(path: str, serial: str, reading_str: str) -> None:
    """
    Перерисовывает аннотацию в левом верхнем углу уже сохранённого фото.
    Использует финальные значения оператора вместо модельных.
    """
    try:
        img = read_image(path)          # пути с кириллицей — см. src/gmr/render/image_io.py
        if img is None:
            log.warning(f"Не удалось открыть фото для перерисовки подписи: {path}")
            return
            
        result = PhotoResult(photo_path=path, outcome=Outcome.MINUS)
        result.serial_text = serial or None
        
        # reading_str → reading если все цифры известны
        if reading_str and reading_str.isdigit() and len(reading_str) == 5:
            result.reading     = int(reading_str)
            result.reading_str = None
        else:
            result.reading     = None
            result.reading_str = reading_str or None
            
        _draw_annotation(img, result)
        if write_image(path, img):
            log.info(f"Аннотация перерисована: {Path(path).name}")
        else:
            log.warning(f"Не удалось сохранить перерисованное фото: {path}")
    except Exception as e:
        log.warning(f"Не удалось перерисовать аннотацию: {e}")

def now_iso() -> str:
    return datetime.now().strftime("%d.%m.%Y %H:%M")
