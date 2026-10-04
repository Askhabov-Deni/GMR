"""
src/gmr/ui/run_dialog.py — «Обработать новые» из окна оператора (этап 4,
решения владельца 2026-10-03: 2а — полоса прогресса и «Остановить», разбирать
фото во время обработки нельзя; 3а — только кнопкой).

Прогон — отдельный процесс `python gmr.py process <месяц>`: окно не
импортирует reader.py (граница с Фазы 6), сбой моделей не роняет окно,
модели прогона не мешают моделям окна. Прогресс окно берёт из строк вывода
прогона («Новых фото к чтению: N», «Фото k из N: имя»). «Остановить» —
файл <месяц>/.gmr_stop: прогон дочитывает текущее фото и заканчивает
аккуратно (база цела, выгрузка сделана).
"""
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable, Optional

_TOTAL = re.compile(r"Новых фото к чтению: (\d+)")
_PHOTO = re.compile(r"Фото (\d+) из (\d+): (.+)$")       # конец строки уже отрезан
_STOPPED = "Остановлено оператором"
_MODELS = "Загружаем модели"


def process_command(project: Path, month_dir: str, reread: bool = False) -> list[str]:
    """Команда прогона. Под pythonw.exe (ярлык без консоли) вывод нужен
    процессу прогона, поэтому он запускается обычным python.exe."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").is_file():
        exe = exe.with_name("python.exe")
    cmd = [str(exe), str(Path(project) / "gmr.py"), "process", str(month_dir)]
    return cmd + (["--reread"] if reread else [])


def parse_progress(line: str) -> Optional[tuple]:
    """Строка вывода прогона → ("total", N) | ("photo", k, N, имя) |
    ("models",) | ("stopped",) | None."""
    if m := _PHOTO.search(line):
        return "photo", int(m.group(1)), int(m.group(2)), m.group(3)
    if m := _TOTAL.search(line):
        return "total", int(m.group(1))
    if _STOPPED in line:
        return ("stopped",)
    if _MODELS in line:
        return ("models",)
    return None


class ProcessDialog(tk.Toplevel):
    """Окно прогона: полоса, что сейчас читается, «Остановить». Пока оно
    открыто, главное окно недоступно (grab). Закрывается само, когда прогон
    закончился; on_done(код выхода, все строки вывода)."""

    def __init__(self, parent, cmd: list[str], stop_file: Path,
                 on_done: Callable[[int, list[str]], None], title: str = "Обработка новых фото",
                 cwd: Optional[Path] = None):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.transient(parent)
        self._stop_file, self._on_done = Path(stop_file), on_done
        self.lines: list[str] = []
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._status = tk.StringVar(value="Запуск…")
        self._count = tk.StringVar(value="")
        f = ttk.Frame(self, padding=16)
        f.pack(fill=tk.BOTH, expand=True)
        ttk.Label(f, textvariable=self._count, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.bar = ttk.Progressbar(f, length=420, mode="indeterminate")
        self.bar.pack(fill=tk.X, pady=8)
        self.bar.start(15)
        ttk.Label(f, textvariable=self._status, width=60).pack(anchor="w")
        self.stop_btn = ttk.Button(f, text="⏹  Остановить", command=self.stop)
        self.stop_btn.pack(pady=(12, 0))
        self.protocol("WM_DELETE_WINDOW", self.stop)
        self.grab_set()

        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)     # без чёрного окна на Windows
        self.proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, creationflags=flags)
        threading.Thread(target=self._read, daemon=True).start()
        self.after(100, self._poll)

    def _read(self):
        for raw in self.proc.stdout:
            self._queue.put(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
        self._queue.put(None)

    def _poll(self):
        finished = False
        try:
            while True:
                line = self._queue.get_nowait()
                if line is None:
                    finished = True
                    break
                self.lines.append(line)
                self._show(parse_progress(line))
        except queue.Empty:
            pass
        if finished:
            code = self.proc.wait()
            self.grab_release()
            self.destroy()
            self._on_done(code, self.lines)
        else:
            self.after(100, self._poll)

    def _show(self, p: Optional[tuple]):
        if p is None:
            return
        if p[0] == "total":
            self.bar.stop()
            self.bar.configure(mode="determinate", maximum=max(p[1], 1), value=0)
            self._count.set(f"Новых фото: {p[1]}" if p[1] else "Новых фото нет")
        elif p[0] == "photo":
            self.bar.configure(mode="determinate", maximum=max(p[2], 1), value=p[1])
            self._count.set(f"Обработано {p[1]} из {p[2]} новых фото")
            self._status.set(p[3])
        elif p[0] == "models":
            self._status.set("Загружаем модели…")
        elif p[0] == "stopped":
            self._status.set("Остановлено")

    def stop(self):
        """«Остановить»: прогон дочитает текущее фото и закончит сам."""
        if self.proc.poll() is not None:
            return
        self._stop_file.write_text("", encoding="utf-8")
        self.stop_btn.configure(state=tk.DISABLED)
        self._status.set("Останавливаем после текущего фото…")


def show_text(parent, title: str, text: str) -> None:
    """Окно с текстом (отчёт прогона, итог месяца) — можно прокрутить и
    скопировать."""
    win = tk.Toplevel(parent)
    win.title(title)
    win.transient(parent)
    box = tk.Text(win, width=90, height=min(30, max(8, text.count("\n") + 2)), wrap="word",
                  font=("Consolas", 10))
    sb = ttk.Scrollbar(win, orient="vertical", command=box.yview)
    box.configure(yscrollcommand=sb.set)
    box.insert("1.0", text)
    box.configure(state=tk.DISABLED)
    ttk.Button(win, text="Закрыть", command=win.destroy).pack(side=tk.BOTTOM, pady=8)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    box.pack(fill=tk.BOTH, expand=True, padx=(8, 0), pady=(8, 0))
