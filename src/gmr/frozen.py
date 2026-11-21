"""
src/gmr/frozen.py — запуск из собранной программы (PyInstaller, docs/BUILD.md).

Собранная программа — папка с «GMR Оператор.exe» (окно) и gmr.exe (обработка,
команды). Файлы программы — веса моделей, образцы символов — лежат внутри
сборки (`_internal\\`), а настройки окна и его журнал — рядом с exe.
Без сборки обе папки — папка проекта, и всё как раньше.
"""
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))
_PROJECT = Path(__file__).resolve().parents[2]


def program_dir() -> Path:
    """Папка программы: settings.json, program2.log (в сборке — рядом с exe)."""
    return Path(sys.executable).resolve().parent if FROZEN else _PROJECT


def bundle_dir() -> Path:
    """Где лежат файлы программы: веса моделей, models/overlay (в сборке — _internal)."""
    return Path(getattr(sys, "_MEIPASS", _PROJECT)) if FROZEN else _PROJECT


def resolve(path) -> str:
    """Путь к файлу программы из конфига. Без сборки — как есть (от текущей папки,
    как всегда); в сборке относительный путь — от bundle_dir()."""
    p = Path(path)
    if FROZEN and str(path) and not p.is_absolute():
        return str(bundle_dir() / p)
    return str(path)
