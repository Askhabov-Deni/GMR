"""
src/gmr/ui/run_dialog.py — «Обработать новые» из окна оператора (этап 4,
решения владельца 2026-10-03: 2а — полоса прогресса и «Остановить», разбирать
фото во время обработки нельзя; 3а — только кнопкой). Экран хода и итога —
src/gmr/operator/process_screen.py (с 1.4.0 — в главном окне, не окном поверх).

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

from src.gmr.operator.theme import px

_TOTAL = re.compile(r"Новых фото к чтению: (\d+)")
_PHOTO = re.compile(r"Фото (\d+) из (\d+): (.+)$")       # конец строки уже отрезан
_STOPPED = "Остановлено оператором"
_MODELS = "Загружаем модели"


def process_command(project: Path, month_dir: str, reread: bool = False) -> list[str]:
    """Команда прогона. Под pythonw.exe (ярлык без консоли) вывод нужен
    процессу прогона, поэтому он запускается обычным python.exe. В сборке
    (docs/BUILD.md) прогон — та же программа с параметрами: «GMR.exe
    process <месяц>»."""
    from src.gmr.frozen import FROZEN
    if FROZEN:
        cmd = [str(Path(sys.executable)), "process", str(month_dir)]
        return cmd + (["--reread"] if reread else [])
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").is_file():
        exe = exe.with_name("python.exe")
    cmd = [str(exe), str(Path(project) / "gmr.py"), "process", str(month_dir)]
    return cmd + (["--reread"] if reread else [])


def center_on(win: tk.Toplevel, parent) -> None:
    """Окно — посередине над родительским (иначе Windows ставит его в угол экрана)."""
    win.update_idletasks()
    w, h = win.winfo_reqwidth(), win.winfo_reqheight()
    x = parent.winfo_rootx() + max(0, (parent.winfo_width() - w) // 2)
    y = parent.winfo_rooty() + max(0, (parent.winfo_height() - h) // 3)
    win.geometry(f"+{x}+{y}")


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


class ProcessRunner:
    """Прогон отдельным процессом и его ход — без окна (1.4.0: ход показывает
    экран в главном окне, src/gmr/operator/process_screen.py; раньше — окно
    поверх). Вывод читается в потоке, разбирается в потоке окна:
    on_progress(parse_progress(строка)) на каждую понятную строку,
    on_done(код выхода, все строки) — когда прогон закончился."""

    def __init__(self, widget: tk.Misc, cmd: list[str], stop_file: Path,
                 on_progress: Callable[[tuple], None], on_done: Callable[[int, list[str]], None],
                 cwd: Optional[Path] = None):
        self.widget, self._stop_file = widget, Path(stop_file)
        self._on_progress, self._on_done = on_progress, on_done
        self.lines: list[str] = []
        self.stopping = False
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)     # без чёрного окна на Windows
        self.proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, creationflags=flags)
        threading.Thread(target=self._read, daemon=True).start()
        widget.after(100, self._poll)

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
                p = parse_progress(line)
                if p is not None:
                    self._on_progress(p)
        except queue.Empty:
            pass
        if finished:
            self._on_done(self.proc.wait(), self.lines)
        else:
            self.widget.after(100, self._poll)

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    def stop(self):
        """«Остановить»: прогон дочитает текущее фото и закончит сам."""
        if not self.running or self.stopping:
            return
        self._stop_file.write_text("", encoding="utf-8")
        self.stopping = True


_REPORT_NUMBERS = {
    "new": r"Новых фото в этом прогоне:\s*(\d+)",
    "copies": r"Тот же файл ещё раз \(копия\):\s*(\d+)",
    "written": r"Успешно \(PLUS\+MINUS\):\s*(\d+)",
    "queue": r"На проверку \(question/\):\s*(\d+)",
    "repeat": r"Дубли \(REPEAT\):\s*(\d+)",
}


def report_numbers(text: str) -> dict:
    """Главные числа итога обработки (report.txt, общий отчёт — первый в файле):
    new, copies, written, queue, repeat; чего нет — нет в словаре."""
    out = {}
    for key, pattern in _REPORT_NUMBERS.items():
        if m := re.search(pattern, text):
            out[key] = int(m.group(1))
    return out


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
    ttk.Button(win, text="Закрыть", command=win.destroy).pack(side=tk.BOTTOM, pady=px(8))
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    box.pack(fill=tk.BOTH, expand=True, padx=(px(8), 0), pady=(px(8), 0))
    center_on(win, parent)
