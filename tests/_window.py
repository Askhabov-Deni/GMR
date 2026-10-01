"""
Окно program2.py в тестах.

has_display — есть ли экран: на Windows есть всегда, в Linux — переменная
DISPLAY (xvfb-run).

new_window — создать окно с повтором. На Windows Tcl иногда не может прочитать
свой init.tcl, когда в одном процессе подряд создаётся много окон (тесты создают
по окну на тест): «Can't find a usable init.tcl … No error», затем «invalid
command name "tcl_findLibrary"». У владельца — 2 окна из ~30 за прогон тестов
(2026-10-01). В работе program2.py создаёт одно окно за запуск, ошибка не
встречается. Окно создаётся заново до 3 раз; если не вышло — ошибка как есть.
"""
import gc
import os
import sys
import time


def has_display() -> bool:
    return sys.platform == "win32" or bool(os.environ.get("DISPLAY"))


def new_window(factory, attempts: int = 3, pause: float = 0.5):
    import tkinter as tk
    for attempt in range(1, attempts + 1):
        try:
            return factory()
        except tk.TclError:
            if attempt == attempts:
                raise
            gc.collect()           # освободить интерпретаторы Tcl прошлых окон
            time.sleep(pause)
