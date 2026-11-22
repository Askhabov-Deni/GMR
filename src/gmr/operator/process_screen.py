"""
Экран «Обработка новых фото» — в главном окне (1.4.0, просьба владельца:
меньше окон поверх). Было: окно с полосой поверх главного, потом окно с
текстом отчёта. Стало: на месте списков —

  ход:   «Обработано 12 из 40 новых фото», полоса, какое фото читается,
         сколько примерно осталось, «Остановить»;
  итог:  «Готово» и главные числа (новых фото, записала сама, в очередь,
         повторы), «Подробный отчёт» — тот же report.txt по кнопке;
         «Разобрать очередь (N)» [Enter] или «К спискам» [Esc].

Пока идёт обработка, шапка окна выключена (разбирать фото нельзя —
решение 2а), закрыть окно — значит остановить обработку.
Сам прогон — src/gmr/ui/run_dialog.py (ProcessRunner).
"""
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable, Optional

from src.gmr.ui.run_dialog import ProcessRunner, report_numbers

from .theme import (
    BORDER,
    DANGER,
    FONT,
    MUTED,
    SUCCESS,
    SURFACE,
    TEXT,
    WARNING,
    action_button,
    px,
    set_enabled,
)


def minutes_text(sec: float) -> str:
    if sec < 60:
        return "меньше минуты"
    m = round(sec / 60)
    return f"≈ {m} мин" if m < 60 else f"≈ {m // 60} ч {m % 60:02d} мин"


class ProcessScreen(ttk.Frame):
    """Ход обработки и её итог. on_done(код, строки) — окно обновляет списки;
    после него экран показывает итог (show_result). done — True, когда всё."""

    def __init__(self, app, cmd: list[str], stop_file: Path, title: str, cwd: Optional[Path],
                 on_done: Callable[[int, list[str]], None]):
        super().__init__(app)
        self.app, self._on_done = app, on_done
        self.done = tk.BooleanVar(value=False)
        self.report = ""                      # текст report.txt после обработки
        self._t0: Optional[float] = None
        self._first = 0
        self._build(title)
        self._runner = ProcessRunner(self, cmd, stop_file, self._progress, self._finished, cwd=cwd)
        self.lines = self._runner.lines

    # ── экран ──
    def _build(self, title: str):
        self._card = tk.Frame(self, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER,
                              padx=px(36), pady=px(28))
        self._card.place(relx=0.5, rely=0.42, anchor="center", width=px(680))
        self._title = tk.Label(self._card, text=title, font=(FONT, 15, "bold"), fg=TEXT, bg=SURFACE)
        self._title.pack(anchor="w")
        self._count = tk.StringVar(value="Запуск…")
        tk.Label(self._card, textvariable=self._count, font=(FONT, 11), fg=TEXT, bg=SURFACE).pack(
            anchor="w", pady=(px(14), px(8)))
        self.bar = ttk.Progressbar(self._card, mode="indeterminate")
        self.bar.pack(fill=tk.X)
        self.bar.start(15)
        row = tk.Frame(self._card, bg=SURFACE)
        row.pack(fill=tk.X, pady=(px(8), 0))
        self._status = tk.StringVar()
        self._eta = tk.StringVar()
        tk.Label(row, textvariable=self._status, font=(FONT, 9), fg=MUTED, bg=SURFACE, anchor="w").pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(row, textvariable=self._eta, font=(FONT, 9), fg=MUTED, bg=SURFACE).pack(side=tk.RIGHT)
        tk.Label(self._card, text="Пока идёт обработка, разбирать фото нельзя. Можно остановить — программа "
                 "дочитает текущее фото, остальные прочитает в следующий раз.", font=(FONT, 9), fg=MUTED,
                 bg=SURFACE, justify="left", wraplength=px(600)).pack(anchor="w", pady=(px(18), px(12)))
        self.stop_btn = action_button(self._card, "Остановить", self.stop, kind="danger")
        self.stop_btn.pack(anchor="w")

    def _progress(self, p: tuple):
        if p[0] == "total":
            self.bar.stop()
            self.bar.configure(mode="determinate", maximum=max(p[1], 1), value=0)
            self._count.set(f"Новых фото: {p[1]}" if p[1] else "Новых фото нет")
        elif p[0] == "photo":
            k, n = p[1], p[2]
            if self._t0 is None:
                self._t0, self._first = time.monotonic(), k
            self.bar.stop()
            self.bar.configure(mode="determinate", maximum=max(n, 1), value=k)
            self._count.set(f"Обработано {k} из {n} новых фото")
            self._status.set(p[3])
            done = k - self._first
            if done >= 3:
                left = (time.monotonic() - self._t0) / done * (n - k)
                self._eta.set(f"осталось {minutes_text(left)}")
        elif p[0] == "models":
            self._status.set("Загружаем модели…")
        elif p[0] == "stopped":
            self._status.set("Остановлено")

    def stop(self):
        """«Остановить» (и закрыть окно во время обработки)."""
        if self._runner.running and not self._runner.stopping:
            self._runner.stop()
            set_enabled(self.stop_btn, False)
            self._status.set("Останавливаем после текущего фото…")

    @property
    def running(self) -> bool:
        return not self.done.get()

    # ── итог ──
    def _finished(self, code: int, lines: list[str]):
        self.bar.stop()
        try:
            self._on_done(code, lines)
        finally:
            self.done.set(True)

    def show_result(self, code: int, report: str, lines: list[str], queue_len: int, open_queue: Callable,
                    back: Callable):
        """Итог на месте хода: числа, подробный отчёт по кнопке, куда дальше."""
        self.report = report
        self._open_queue, self._back_cb = open_queue, back
        for w in self._card.winfo_children():
            w.destroy()
        stopped = any("Остановлено оператором" in x for x in lines)
        ok = code == 0 and bool(report)
        if not ok:
            head, color = f"⚠  Обработка не удалась (код {code})", DANGER
        elif stopped:
            head, color = "⏹  Обработка остановлена", WARNING
        else:
            head, color = "✓  Обработка закончена", SUCCESS
        tk.Label(self._card, text=head, font=(FONT, 15, "bold"), fg=color, bg=SURFACE).pack(anchor="w")
        if ok:
            nums = report_numbers(report)
            tiles = tk.Frame(self._card, bg=SURFACE)
            tiles.pack(fill=tk.X, pady=(px(16), px(4)))
            for i, (label, key) in enumerate((("Новых фото", "new"), ("Записала сама", "written"),
                                              ("В очередь", "queue"), ("Повторы", "repeat"))):
                t = tk.Frame(tiles, bg="#f6f8fb", padx=px(14), pady=px(10))
                t.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else px(8), 0))
                tiles.columnconfigure(i, weight=1, uniform="t")
                tk.Label(t, text=label, font=(FONT, 9), fg=MUTED, bg="#f6f8fb").pack(anchor="w")
                tk.Label(t, text=str(nums.get(key, "—")), font=(FONT, 18, "bold"), fg=TEXT, bg="#f6f8fb").pack(
                    anchor="w")
            if "⚠" in report:
                tk.Label(self._card, text="⚠  В отчёте есть предупреждения — откройте «Подробный отчёт».",
                         font=(FONT, 9), fg=WARNING, bg=SURFACE).pack(anchor="w", pady=(px(6), 0))
            detail = report
        else:
            detail = "\n".join(lines[-25:]) or "(нет вывода)"
        self._detail = tk.Text(self._card, height=12, wrap="none", font=("Consolas", 9), bg="#f6f8fb", bd=0,
                               padx=px(10), pady=px(8), fg=TEXT)
        self._detail.insert("1.0", detail)
        self._detail.configure(state=tk.DISABLED)
        buttons = tk.Frame(self._card, bg=SURFACE)
        buttons.pack(side=tk.BOTTOM, fill=tk.X, pady=(px(16), 0))
        if queue_len:
            action_button(buttons, f"Разобрать очередь ({queue_len})    Enter", self._go_queue,
                          kind="accent").pack(side=tk.LEFT)
            action_button(buttons, "К спискам    Esc", self._go_back, kind="soft").pack(side=tk.LEFT, padx=(px(8), 0))
        else:
            action_button(buttons, "К спискам    Enter", self._go_back, kind="accent").pack(side=tk.LEFT)
        self._toggle = action_button(buttons, "Подробный отчёт  ▾", self.toggle_detail, kind="soft")
        self._toggle.pack(side=tk.RIGHT)
        if not ok:
            self.toggle_detail()               # не удалось — вывод сразу
        self._bind_keys(queue_len)

    def toggle_detail(self):
        if self._detail.winfo_manager():
            self._detail.pack_forget()
            self._toggle.config(text="Подробный отчёт  ▾")
        else:
            self._detail.pack(fill=tk.BOTH, expand=True, pady=(px(14), 0))
            self._toggle.config(text="Подробный отчёт  ▴")

    def _go_queue(self):
        self._unbind_keys()
        self._open_queue()

    def _go_back(self):
        self._unbind_keys()
        self._back_cb()

    # ── клавиши итога (общие для окна, как у экранов разбора) ──
    def _key_bindings(self):
        enter = self._go_queue if getattr(self, "_queue_len", 0) else self._go_back
        return [("<Return>", lambda e: enter()), ("<KP_Enter>", lambda e: enter()),
                ("<Escape>", lambda e: self._go_back())]

    def _bind_keys(self, queue_len: int):
        self._queue_len = queue_len
        for seq, cb in self._key_bindings():
            self.app.bind_all(seq, cb)
        self.app._keys_owner = self

    def _unbind_keys(self):
        if getattr(self.app, "_keys_owner", None) is not self:
            return
        self.app._keys_owner = None
        for seq, _ in self._key_bindings():
            try:
                self.app.unbind_all(seq)
            except tk.TclError:
                pass

    def _back(self):
        """Закрыть экран (окно переходит к другому экрану или месяцу)."""
        if self.running:
            return
        self._go_back()
