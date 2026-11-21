"""
Экран входа — внутри главного окна (2026-11-22, просьба владельца: раньше
открывалось маленькое окно «Вход», а большое окно его закрывало — приходилось
искать маленькое на панели задач).

Окно программы открывается сразу на весь экран: посередине значок, «GMR»,
имя оператора и месяц, «Продолжить» (Enter) и «Сменить пользователя». Пока
открывается месяц — там же «Открываю месяц…». Потом на месте этого экрана
появляются списки.
"""
import tkinter as tk
from typing import Optional

from src.gmr.version import __version__

from .settings import AppSettings
from .theme import APP_NAME, BG, FONT, MUTED, SURFACE, TEXT, action_button, logo_image

TAGLINE = "Показания газовых счётчиков по фото контролёров"


class WelcomeScreen(tk.Frame):
    """Заставка во всё окно. ask(settings) — ждёт «Продолжить» / «Сменить
    пользователя» / закрытие окна и возвращает "continue" | "change" | "quit"."""

    def __init__(self, window: tk.Tk):
        super().__init__(window, bg=BG)
        self.window = window
        self.pack(fill=tk.BOTH, expand=True)
        self._card = tk.Frame(self, bg=SURFACE, padx=56, pady=40, highlightthickness=1,
                              highlightbackground="#d5dbe3")
        self._card.place(relx=0.5, rely=0.42, anchor="center")
        self._logo = logo_image(window, 128)
        if self._logo is not None:
            tk.Label(self._card, image=self._logo, bg=SURFACE).pack()
        tk.Label(self._card, text=APP_NAME, font=(FONT, 28, "bold"), fg=TEXT, bg=SURFACE).pack(pady=(10, 0))
        tk.Label(self._card, text=TAGLINE, font=(FONT, 10), fg=MUTED, bg=SURFACE).pack()
        tk.Label(self._card, text=f"версия {__version__}", font=(FONT, 9), fg=MUTED, bg=SURFACE).pack(pady=(2, 0))
        self._body = tk.Frame(self._card, bg=SURFACE)
        self._body.pack(fill=tk.X, pady=(26, 0))
        self._status = tk.StringVar()
        self._status_lbl = tk.Label(self._card, textvariable=self._status, font=(FONT, 10), fg=MUTED, bg=SURFACE)
        self._action = tk.StringVar()

    def ask(self, settings: AppSettings) -> str:
        b = self._body
        tk.Label(b, text="Вы вошли как", font=(FONT, 10), fg=MUTED, bg=SURFACE).pack()
        tk.Label(b, text=settings.operator_name, font=(FONT, 16, "bold"), fg=TEXT, bg=SURFACE).pack()
        month = settings.month_dir.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        if month:
            tk.Label(b, text=f"месяц: {month}", font=(FONT, 10), fg=MUTED, bg=SURFACE).pack(pady=(2, 0))
        self._go = action_button(b, "Продолжить    Enter", self._continue, kind="accent")
        self._go.pack(fill=tk.X, pady=(20, 8))
        action_button(b, "Сменить пользователя", self._change).pack(fill=tk.X)
        for seq in ("<Return>", "<KP_Enter>"):
            self.window.bind(seq, self._on_return)
        self.window.protocol("WM_DELETE_WINDOW", self._quit)
        self.window.after_idle(self._take_focus)
        self.window.wait_variable(self._action)
        for seq in ("<Return>", "<KP_Enter>"):
            self.window.unbind(seq)
        self.window.protocol("WM_DELETE_WINDOW", lambda: None)   # месяц открывается — не закрывать
        for w in b.winfo_children():
            w.destroy()
        return self._action.get()

    def _take_focus(self):
        """Окно — на передний план, клавиши — ему: Enter сразу «Продолжить»."""
        try:
            self.window.lift()
            self.window.focus_force()
            self._go.focus_set()
        except tk.TclError:
            pass

    def _on_return(self, event=None):
        self._continue()
        return "break"

    def _continue(self):
        self._action.set("continue")

    def _change(self):
        self._action.set("change")

    def _quit(self):
        self._action.set("quit")

    def status(self, text: Optional[str]) -> None:
        """Строка под карточкой: «Открываю месяц…» (None — убрать)."""
        if text:
            self._status.set(text)
            self._status_lbl.pack(pady=(18, 0))
        else:
            self._status_lbl.pack_forget()
        self.window.update_idletasks()
