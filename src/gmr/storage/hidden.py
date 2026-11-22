"""
src/gmr/storage/hidden.py — служебное в папке месяца — скрытое (1.5.0,
просьба владельца 2026-11-22: «в папке месяца слишком много всего, это может
напугать контролёра»). В проводнике видно только фото/ и показания.xlsx
(и другие выгрузки), остальное — скрытое (атрибут Windows «Скрытый»).

Почему не отдельная папка: месяц остаётся тем же — его открывает и прошлая
версия программы, пути в базе и в инструкциях не меняются. Скрытое видно,
если в проводнике включить «Показывать скрытые элементы».

Подвох Windows: файл со «Скрытым» нельзя перезаписать открытием на запись
(open(…, "w") — «Отказано в доступе»); дописывать (append), открывать базу
SQLite и заменять файл (os.replace) — можно. Поэтому перед перезаписью —
unhide(путь), после — снова hide.
"""
import sys
from pathlib import Path

FILE_ATTRIBUTE_HIDDEN = 0x2
_INVALID = 0xFFFFFFFF


def _set(path: Path, hidden: bool) -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.GetFileAttributesW.restype = ctypes.c_uint32
        attrs = k.GetFileAttributesW(str(path))
        if attrs == _INVALID:
            return False
        new = attrs | FILE_ATTRIBUTE_HIDDEN if hidden else attrs & ~FILE_ATTRIBUTE_HIDDEN
        return new == attrs or bool(k.SetFileAttributesW(str(path), new))
    except (AttributeError, OSError):
        return False


def hide(path) -> bool:
    """Скрыть файл или папку (нет такого или не Windows — False)."""
    return _set(Path(path), True)


def unhide(path) -> bool:
    """Снять «Скрытый» — перед перезаписью файла."""
    return _set(Path(path), False)


def is_hidden(path) -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.GetFileAttributesW.restype = ctypes.c_uint32
        attrs = k.GetFileAttributesW(str(path))
        return attrs != _INVALID and bool(attrs & FILE_ATTRIBUTE_HIDDEN)
    except (AttributeError, OSError):
        return False
