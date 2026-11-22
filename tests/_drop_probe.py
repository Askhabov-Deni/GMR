"""Перетаскивание как от проводника — отдельным процессом (tests/test_ui15.py).

WM_DROPFILES кладётся в очередь окна (PostMessage), пока идёт его цикл, — так
делает проводник. 1.5.0 в офисе: на этом программа закрывалась («Fatal Python
error: PyEval_RestoreThread» — вызов Tk из оконной процедуры). Печатает
«GOT …» — что дошло до окна; падение — код возврата не 0.

  python tests/_drop_probe.py <папка проекта>
"""
import ctypes
import sys
import tkinter as tk
from ctypes import wintypes

sys.path.insert(0, sys.argv[1])
from src.gmr.operator.dropfiles import enable_drop  # noqa: E402


def post_drop(hwnd, paths):
    k32, u32 = ctypes.windll.kernel32, ctypes.windll.user32
    k32.GlobalAlloc.restype = ctypes.c_void_p
    k32.GlobalLock.restype = ctypes.c_void_p
    k32.GlobalLock.argtypes = (ctypes.c_void_p,)
    k32.GlobalUnlock.argtypes = (ctypes.c_void_p,)
    u32.PostMessageW.argtypes = (wintypes.HWND, wintypes.UINT, ctypes.c_void_p, ctypes.c_void_p)
    text = ("\0".join(paths) + "\0\0").encode("utf-16-le")
    data = (20).to_bytes(4, "little") + b"\0" * 12 + (1).to_bytes(4, "little") + text   # DROPFILES, fWide
    h = k32.GlobalAlloc(0x0042, len(data))                                                # GMEM_MOVEABLE|ZEROINIT
    ctypes.memmove(k32.GlobalLock(h), data, len(data))
    k32.GlobalUnlock(h)
    u32.PostMessageW(hwnd, 0x0233, h, None)


root = tk.Tk()
root.geometry("300x200+-3000+-3000")
root.update()
hwnd = int(root.wm_frame(), 16)
got = []
done = tk.BooleanVar(root)


def on_drop(paths):
    got.append(paths)
    if len(got) == 1:              # как поле «Добавить фото»: ждёт ответа внутри обработки
        root.after(100, lambda: post_drop(hwnd, ["второе"]))
        root.wait_variable(done)
    else:
        done.set(True)
        root.after(50, root.destroy)


assert enable_drop(root, on_drop)
root.after(200, lambda: post_drop(hwnd, ["C:/Рабочий стол/реестр.xlsx", "папка"]))
root.after(8000, root.destroy)
root.mainloop()
print("GOT", got, flush=True)
