"""
src/gmr/storage/backup.py — копии таблицы и лога (2026-10-01).

Перед каждым прогоном reader.py и при запуске окна оператора таблица и лог
копируются в `<папка таблицы>/gmr_backups/<имя таблицы>/<дата_время>/`.
Хранятся последние KEEP копий, старые удаляются. Если что-то испортилось —
можно вернуться к состоянию «до прогона».
"""
import shutil
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
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dest, n = Path(backup_root) / stamp, 2
    while dest.exists():                     # две копии в одну секунду — не затирать первую
        dest, n = Path(backup_root) / f"{stamp}_{n}", n + 1
    dest.mkdir(parents=True)
    for p in existing:
        shutil.copy2(p, dest / p.name)
    remove_old(Path(backup_root), keep)
    return dest


def remove_old(folder: Path, keep: int) -> None:
    """Оставляет в folder последние keep элементов (имена начинаются с даты и времени)."""
    items = sorted(folder.iterdir(), key=lambda p: p.name)
    for old in items[:-keep] if keep > 0 else items:
        if old.is_dir():
            shutil.rmtree(old)
        else:
            old.unlink()
