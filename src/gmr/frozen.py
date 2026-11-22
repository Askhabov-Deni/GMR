"""
src/gmr/frozen.py — запуск из собранной программы (PyInstaller, docs/BUILD.md).

Собранная программа — папка с «GMR.exe»: без параметров — окно, с
параметрами — команды gmr.py (так окно запускает обработку фото). Файлы
программы — веса моделей, образцы символов — лежат внутри сборки
(`_internal\\`), а настройки окна и его журнал — рядом с exe. Без сборки обе
папки — папка проекта, и всё как раньше.
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


def attach_std_streams() -> None:
    """Собранное окно (без консоли) запущено с параметрами — командой gmr.py,
    например обработкой из окна. У программы без консоли Python не открывает
    sys.stdout/sys.stderr — подключаем их к тому, куда их передал запустивший
    (труба окна обработки или консоль). Без сборки и с консолью — ничего.

    Кодировка: собранная программа не читает переменные PYTHON* (PyInstaller),
    и если Windows дала ей трубу окна, Python открывает её в кодировке Windows
    (cp1251) — окно не узнавало строки «Фото N из M» и ход обработки стоял на
    «Запуске» (1.5.1). Поэтому, если запустивший просит PYTHONIOENCODING
    (окно обработки — utf-8), вывод переключается на неё, построчно."""
    import os
    want = os.environ.get("PYTHONIOENCODING", "").split(":")[0]
    if FROZEN and want:
        for name in ("stdout", "stderr"):
            stream = getattr(sys, name)
            if stream is not None and hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding=want, errors="replace", line_buffering=True)
                except (ValueError, OSError, LookupError):
                    pass
    if sys.stdout is not None and sys.stderr is not None:
        return
    import io
    try:
        import ctypes
        import msvcrt
    except ImportError:            # не Windows
        return
    enc = os.environ.get("PYTHONIOENCODING", "").split(":")[0] or "utf-8"
    for name, std in (("stdout", -11), ("stderr", -12)):
        if getattr(sys, name) is not None:
            continue
        handle = ctypes.windll.kernel32.GetStdHandle(std)
        if not handle or handle == ctypes.c_void_p(-1).value:
            continue
        try:
            fd = msvcrt.open_osfhandle(handle, os.O_WRONLY)
            stream = io.TextIOWrapper(io.FileIO(fd, "w", closefd=False), encoding=enc,
                                      errors="replace", line_buffering=True)
        except OSError:
            continue
        setattr(sys, name, stream)
        setattr(sys, f"__{name}__", stream)
