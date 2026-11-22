"""
src/gmr/operator/dropfiles.py — перетаскивание файлов и папок из проводника в
окно (1.5.0, просьба владельца 2026-11-22: «таблицу и папку с фото — просто
перетащить с рабочего стола»).

Без сторонних библиотек: окно объявляет Windows, что принимает файлы
(DragAcceptFiles), а сообщение WM_DROPFILES перехватывается подменой оконной
процедуры (SetWindowLongPtrW). Файлы с рабочего стола падают в любое место
окна; что с ними делать, решает окно (MainWindow._on_drop). Не Windows —
перетаскивания нет, остаётся выбор через проводник.
"""
import ctypes
import sys
from typing import Callable

WM_DROPFILES = 0x0233
GWLP_WNDPROC = -4


def enable_drop(window, on_drop: Callable[[list], None]) -> bool:
    """Окно window (tk.Tk / Toplevel) принимает файлы; on_drop(список путей)
    вызывается в потоке окна после отпускания мыши. True — включено."""
    if sys.platform != "win32":
        return False
    try:
        from ctypes import wintypes
        user32, shell32 = ctypes.windll.user32, ctypes.windll.shell32
        window.update_idletasks()
        hwnd = int(window.wm_frame(), 16)
        lresult = ctypes.c_ssize_t
        proc_type = ctypes.WINFUNCTYPE(lresult, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
        user32.SetWindowLongPtrW.restype = ctypes.c_void_p
        user32.SetWindowLongPtrW.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_void_p)
        user32.CallWindowProcW.restype = lresult
        user32.CallWindowProcW.argtypes = (ctypes.c_void_p, wintypes.HWND, wintypes.UINT,
                                           wintypes.WPARAM, wintypes.LPARAM)
        shell32.DragQueryFileW.restype = wintypes.UINT
        shell32.DragQueryFileW.argtypes = (ctypes.c_void_p, wintypes.UINT, ctypes.c_wchar_p, wintypes.UINT)
        shell32.DragFinish.argtypes = (ctypes.c_void_p,)
        shell32.DragAcceptFiles.argtypes = (wintypes.HWND, wintypes.BOOL)
        old = None

        def proc(h, msg, wparam, lparam):
            if msg == WM_DROPFILES:
                paths = []
                for i in range(shell32.DragQueryFileW(wparam, 0xFFFFFFFF, None, 0)):
                    size = shell32.DragQueryFileW(wparam, i, None, 0) + 1
                    buf = ctypes.create_unicode_buffer(size)
                    shell32.DragQueryFileW(wparam, i, buf, size)
                    paths.append(buf.value)
                shell32.DragFinish(wparam)
                if paths:
                    window.after_idle(lambda: on_drop(paths))    # Tk — не внутри оконной процедуры
                return 0
            return user32.CallWindowProcW(old, h, msg, wparam, lparam)

        callback = proc_type(proc)
        old = user32.SetWindowLongPtrW(hwnd, GWLP_WNDPROC, ctypes.cast(callback, ctypes.c_void_p))
        if not old:
            return False
        window._drop_callback = callback          # иначе сборщик мусора уберёт процедуру
        shell32.DragAcceptFiles(hwnd, True)
        return True
    except (AttributeError, OSError, ValueError, TypeError):
        return False
