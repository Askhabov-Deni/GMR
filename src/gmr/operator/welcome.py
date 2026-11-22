"""
Экран входа — внутри главного окна.

1.3.0: вместо окошка «Вход», которое закрывало большое окно, — экран во всё
окно. 1.4.0 (просьба владельца: «при запуске всё равно два окна — основное и
настройки»): и первая настройка, и «Сменить пользователя» — здесь же, на
этой карточке, без окна «Настройки» поверх.

Карточка посередине окна:
  - вход: значок, «GMR», «Вы вошли как …», месяц; «Продолжить» (Enter) и
    «Сменить пользователя или месяц»;
  - настройка: «Ваше имя», «Папка месяца» с «Выбрать папку…» и «Новый
    месяц…» (из таблицы компании); ошибка — красной строкой здесь же;
  - «Открываю месяц…», пока открывается месяц.
Потом на месте этого экрана появляются списки.
"""
import tkinter as tk
from dataclasses import replace
from tkinter import filedialog, ttk
from typing import Callable, Optional

from src.gmr.version import __version__

from .settings import AppSettings
from .theme import APP_NAME, BG, BORDER, DANGER, FONT, MUTED, SURFACE, TEXT, action_button, logo_image, px

TAGLINE = "Показания газовых счётчиков по фото контролёров"


def month_name(folder: str) -> str:
    return folder.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


class WelcomeScreen(tk.Frame):
    """Заставка во всё окно.
    ask(settings) — «Продолжить» / «Сменить…» / закрыть окно: "continue" | "change" | "quit".
    ask_setup(...) — имя и папка месяца: AppSettings или None (отмена или закрыть
    окно — тогда self.quit)."""

    def __init__(self, window: tk.Tk):
        super().__init__(window, bg=BG)
        self.window = window
        self.quit = False
        self.pack(fill=tk.BOTH, expand=True)
        self._card = tk.Frame(self, bg=SURFACE, padx=px(56), pady=px(36), highlightthickness=1,
                              highlightbackground=BORDER)
        self._card.place(relx=0.5, rely=0.45, anchor="center")
        self._logo = logo_image(window, 112)
        if self._logo is not None:
            tk.Label(self._card, image=self._logo, bg=SURFACE).pack()
        tk.Label(self._card, text=APP_NAME, font=(FONT, 26, "bold"), fg=TEXT, bg=SURFACE).pack(pady=(px(8), 0))
        tk.Label(self._card, text=TAGLINE, font=(FONT, 10), fg=MUTED, bg=SURFACE).pack()
        tk.Label(self._card, text=f"версия {__version__}", font=(FONT, 9), fg=MUTED, bg=SURFACE).pack(pady=(px(2), 0))
        self._body = tk.Frame(self._card, bg=SURFACE)
        self._body.pack(fill=tk.X, pady=(px(24), 0))
        self._status = tk.StringVar(master=self)
        self._status_lbl = tk.Label(self._card, textvariable=self._status, font=(FONT, 10), fg=MUTED, bg=SURFACE)
        self._action = tk.StringVar(master=self)
        self._on_enter: Optional[Callable[[], None]] = None

    # ── общее: ждать ответа, Enter, крестик окна ──
    def _wait(self, first_focus) -> str:
        for seq in ("<Return>", "<KP_Enter>"):
            self.window.bind(seq, self._on_return)
        self.window.protocol("WM_DELETE_WINDOW", self._quit)
        self.window.after_idle(lambda: self._take_focus(first_focus))
        self._action.set("")
        self.window.wait_variable(self._action)
        for seq in ("<Return>", "<KP_Enter>"):
            self.window.unbind(seq)
        self.window.protocol("WM_DELETE_WINDOW", lambda: None)   # месяц открывается — не закрывать
        return self._action.get()

    def _clear(self):
        for w in self._body.winfo_children():
            w.destroy()

    def _take_focus(self, widget):
        """Окно — на передний план, клавиши — ему: Enter сразу работает."""
        try:
            self.window.lift()
            self.window.focus_force()
            widget.focus_set()
        except tk.TclError:
            pass

    def _on_return(self, event=None):
        if self._on_enter is not None:
            self._on_enter()
        return "break"

    def _quit(self):
        self.quit = True
        self._action.set("quit")

    # ── вход ──
    def ask(self, settings: AppSettings) -> str:
        self._clear()
        b = self._body
        tk.Label(b, text="Вы вошли как", font=(FONT, 10), fg=MUTED, bg=SURFACE).pack()
        tk.Label(b, text=settings.operator_name, font=(FONT, 16, "bold"), fg=TEXT, bg=SURFACE).pack()
        if month_name(settings.month_dir):
            tk.Label(b, text=f"месяц: {month_name(settings.month_dir)}", font=(FONT, 10), fg=MUTED,
                     bg=SURFACE).pack(pady=(px(2), 0))
        self._go = action_button(b, "Продолжить    Enter", self._continue, kind="accent")
        self._go.pack(fill=tk.X, pady=(px(20), px(8)))
        link = tk.Label(b, text="Сменить пользователя или месяц", font=(FONT, 10, "underline"), fg="#1858bd",
                        bg=SURFACE, cursor="hand2")
        link.pack()
        link.bind("<Button-1>", lambda e: self._change())
        self._on_enter = self._continue
        action = self._wait(self._go)
        self._clear()
        return action

    def _continue(self):
        self._action.set("continue")

    def _change(self):
        self._action.set("change")

    # ── настройка: имя и папка месяца ──
    def ask_setup(self, settings: AppSettings, new_month: Optional[Callable] = None,
                  error: Optional[str] = None, can_cancel: bool = False) -> Optional[AppSettings]:
        """Первый запуск, «Сменить пользователя или месяц» или папка — не месяц.
        new_month(родитель, имя) → папка созданного месяца или None."""
        self._clear()
        b = self._body
        first = not settings.operator_name
        tk.Label(b, text="Добро пожаловать!" if first else "Пользователь и месяц", font=(FONT, 13, "bold"),
                 fg=TEXT, bg=SURFACE).pack(anchor="w")
        tk.Label(b, text="Один раз: ваше имя и папка месяца. Дальше программа их помнит.", font=(FONT, 9),
                 fg=MUTED, bg=SURFACE).pack(anchor="w", pady=(0, px(12)))

        self._base = settings
        self._name_var = tk.StringVar(master=self, value=settings.operator_name)
        self._month_var = tk.StringVar(master=self, value=settings.month_dir)
        tk.Label(b, text="ВАШЕ ИМЯ", font=(FONT, 9, "bold"), fg=MUTED, bg=SURFACE).pack(anchor="w")
        name_entry = ttk.Entry(b, textvariable=self._name_var, font=(FONT, 12), width=34)
        name_entry.pack(fill=tk.X, pady=(px(2), px(12)))
        tk.Label(b, text="ПАПКА МЕСЯЦА", font=(FONT, 9, "bold"), fg=MUTED, bg=SURFACE).pack(anchor="w")
        ttk.Entry(b, textvariable=self._month_var, font=(FONT, 10), width=34).pack(fill=tk.X, pady=(px(2), px(6)))
        row = tk.Frame(b, bg=SURFACE)
        row.pack(fill=tk.X)
        action_button(row, "Выбрать папку…", self._browse, kind="soft").pack(side=tk.LEFT)
        if new_month is not None:
            action_button(row, "Новый месяц…", lambda: self._create(new_month), kind="soft").pack(
                side=tk.LEFT, padx=(px(8), 0))
        self._error = tk.StringVar(master=self, value=error or "")
        tk.Label(b, textvariable=self._error, font=(FONT, 9), fg=DANGER, bg=SURFACE, justify="left",
                 wraplength=px(340)).pack(anchor="w", pady=(px(8), 0))
        buttons = tk.Frame(b, bg=SURFACE)
        buttons.pack(fill=tk.X, pady=(px(10), 0))
        go = action_button(buttons, "Продолжить    Enter", self._save, kind="accent")
        go.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if can_cancel:
            action_button(buttons, "Отмена", self._cancel, kind="soft").pack(side=tk.LEFT, padx=(px(8), 0))
        self._on_enter = self._save
        self._result: Optional[AppSettings] = None
        action = self._wait(name_entry if first else go)
        self._clear()
        return self._result if action == "save" else None

    def _browse(self):
        p = filedialog.askdirectory(parent=self.window, title="Папка месяца")
        if p:
            self._month_var.set(p)
            self._error.set("")

    def _create(self, new_month: Callable):
        folder = new_month(self.window, self._name_var.get().strip())
        if folder:
            self._month_var.set(folder)
            self._error.set("")

    def _save(self):
        name, month = self._name_var.get().strip(), self._month_var.get().strip()
        if not name:
            self._error.set("Введите ваше имя — оно пишется в журнал рядом с каждым решением.")
            return
        if not month:
            self._error.set("Выберите папку месяца («Выбрать папку…») или создайте новый месяц из таблицы "
                            "компании («Новый месяц…»).")
            return
        self._result = replace(self._base, operator_name=name, month_dir=month)   # формат выгрузки — прежний
        self._action.set("save")

    def _cancel(self):
        self._action.set("cancel")

    def status(self, text: Optional[str]) -> None:
        """Строка под карточкой: «Открываю месяц…» (None — убрать)."""
        if text:
            self._status.set(text)
            self._status_lbl.pack(pady=(px(18), 0))
        else:
            self._status_lbl.pack_forget()
        self.window.update_idletasks()
