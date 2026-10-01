"""
src/gmr/storage/backup.py — копии таблицы и лога (2026-10-01).

Перед каждым прогоном reader.py и при запуске окна оператора таблица и лог
копируются в `<папка таблицы>/gmr_backups/<имя таблицы>/<дата_время>/`.
Хранятся последние KEEP копий, старые удаляются. Если что-то испортилось —
можно вернуться к состоянию «до прогона». База месяца (gmr.sqlite)
копируется средствами SQLite — backup_sqlite.
"""
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

KEEP = 30


def backup_dir_for(table_path: str) -> Path:
    t = Path(table_path)
    return t.parent / "gmr_backups" / t.stem


def backup_files(paths: Iterable[str], backup_root: Path, keep: int = KEEP) -> Optional[Path]:
    """Копирует существующие файлы из paths в новую папку с датой и временем.
    Возвращает эту папку (None — копировать было нечего)."""
    existing = [Path(p) for p in paths if p and Path(p).is_file()]
    if not existing:
        return None
    dest = _new_backup_dir(Path(backup_root))
    for p in existing:
        shutil.copy2(p, dest / p.name)
    remove_old(Path(backup_root), keep)
    return dest


def backup_sqlite(db_path: Path, backup_root: Path, keep: int = KEEP) -> Optional[Path]:
    """Копия базы SQLite средствами самой SQLite (верная, даже если базу в это
    время кто-то пишет). Возвращает папку копии; None — базы ещё нет."""
    if not Path(db_path).is_file():
        return None
    dest = _new_backup_dir(Path(backup_root))
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(dest / Path(db_path).name))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    remove_old(Path(backup_root), keep)
    return dest


def _new_backup_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dest, n = root / stamp, 2
    while dest.exists():                     # две копии в одну секунду — не затирать первую
        dest, n = root / f"{stamp}_{n}", n + 1
    dest.mkdir(parents=True)
    return dest


def remove_old(folder: Path, keep: int) -> None:
    """Оставляет в folder последние keep элементов (имена начинаются с даты и времени)."""
    items = sorted(folder.iterdir(), key=lambda p: p.name)
    for old in items[:-keep] if keep > 0 else items:
        if old.is_dir():
            shutil.rmtree(old)
        else:
            old.unlink()
