"""
src/gmr/operator/drop_panel.py — «перетащите сюда» (1.5.0, просьба владельца
2026-11-22): поле посреди окна, куда перетаскивают файл или папку из
проводника или с рабочего стола; щелчок по полю — тот же выбор через
проводник.

  TableDropPanel — таблица компании для нового месяца и «Обновить таблицу…»;
  PhotoDropPanel — «Добавить фото»: папка (или фото) → чьи это фото →
                   «Добавить» (копирование — application/add_photos.py).

Панель лежит поверх всего окна (place), окно ждёт ответа (wait). Куда падают
файлы, решает окно: пока панель открыта — в неё (MainWindow._on_drop).
"""
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable, Optional

from src.gmr.application.add_photos import PhotoGroup, check_name, find_groups, match_controller

from .theme import (
    ACCENT,
    BG,
    BORDER,
    DANGER,
    FONT,
    MUTED,
    SELECT,
    SURFACE,
    TEXT,
    action_button,
    px,
)

TABLE_EXTS = (".xlsx", ".xls", ".csv")


def photos_word(n: int) -> str:
    return f"{n} фото"


class DropZone(tk.Canvas):
    """Прямоугольник с пунктирной рамкой: «перетащите сюда…», щелчок — проводник."""

    def __init__(self, parent, icon: str, text: str, hint: str, on_click: Callable[[], None]):
        super().__init__(parent, bg=SURFACE, highlightthickness=0, height=px(190), cursor="hand2")
        self._icon, self._text, self._hint = icon, text, hint
        self._hover = False
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<Button-1>", lambda e: on_click())

    def _set_hover(self, on: bool):
        self._hover = on
        self.draw()

    def flash(self):
        """Файлы упали — короткая подсветка."""
        self._set_hover(True)
        self.after(400, lambda: self._set_hover(False))

    def set_text(self, text: str, hint: Optional[str] = None):
        self._text = text
        if hint is not None:
            self._hint = hint
        self.draw()

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10:
            return
        m = px(4)
        self.create_rectangle(m, m, w - m, h - m, outline=ACCENT if self._hover else BORDER,
                              width=px(2), dash=(px(7), px(5)), fill=SELECT if self._hover else "#f7f9fc")
        self.create_text(w / 2, h * 0.30, text=self._icon, font=(FONT, 30), fill=ACCENT)
        self.create_text(w / 2, h * 0.58, text=self._text, font=(FONT, 12, "bold"), fill=TEXT,
                         width=w - px(40), justify="center")
        self.create_text(w / 2, h * 0.78, text=self._hint, font=(FONT, 10), fill=MUTED,
                         width=w - px(40), justify="center")


class DropPanel(tk.Frame):
    """Общее: заголовок, поле, красная строка ошибки, кнопки. result — ответ
    (None — «Отмена» / Esc)."""

    def __init__(self, window, title: str, subtitle: str, icon: str, text: str, hint: str):
        super().__init__(window, bg=BG)
        self.window = window
        self.result = None
        self._done = tk.BooleanVar(master=self, value=False)
        self.card = tk.Frame(self, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER,
                             padx=px(36), pady=px(28))
        self.card.place(relx=0.5, rely=0.45, anchor="center", width=px(640))
        tk.Label(self.card, text=title, font=(FONT, 15, "bold"), fg=TEXT, bg=SURFACE).pack(anchor="w")
        tk.Label(self.card, text=subtitle, font=(FONT, 10), fg=MUTED, bg=SURFACE, justify="left",
                 wraplength=px(560)).pack(anchor="w", pady=(px(4), px(16)))
        self.zone = DropZone(self.card, icon, text, hint, self._browse)
        self.zone.pack(fill=tk.X)
        self.body = tk.Frame(self.card, bg=SURFACE)          # второй шаг (PhotoDropPanel)
        self.body.pack(fill=tk.X)
        self._error = tk.Label(self.card, text="", font=(FONT, 10), fg=DANGER, bg=SURFACE, justify="left",
                               wraplength=px(560))
        self._error.pack(anchor="w", pady=(px(8), 0))
        self.buttons = tk.Frame(self.card, bg=SURFACE)
        self.buttons.pack(fill=tk.X, pady=(px(14), 0))
        action_button(self.buttons, "Отмена   Esc", self.cancel, kind="soft").pack(side=tk.RIGHT)

    # ── ожидание ──
    def wait(self):
        self.place(x=0, y=0, relwidth=1, relheight=1)
        self.lift()
        self.window._drop_target = self
        self._old_keys = {k: self.window.bind_all(k) for k in ("<Escape>", "<Return>", "<KP_Enter>")}
        self.window.bind_all("<Escape>", lambda e: self.cancel())
        self.window.bind_all("<Return>", lambda e: self.confirm())
        self.window.bind_all("<KP_Enter>", lambda e: self.confirm())
        self.focus_set()
        busy = getattr(self.window, "_set_busy", None) if hasattr(self.window, "_menubar") else None
        if busy:
            busy(True)                 # меню «Месяц» и шапка — выключены, пока поле открыто
        try:
            self.window.wait_variable(self._done)
        except tk.TclError:            # окно закрыли, пока поле было открыто
            self.result = None
        finally:
            try:
                if getattr(self.window, "_drop_target", None) is self:
                    self.window._drop_target = None
                for k, v in self._old_keys.items():
                    self.window.bind_all(k, v or "")
                if busy:
                    busy(False)
                if self.winfo_exists():
                    self.destroy()
            except tk.TclError:
                pass
        return self.result

    def finish(self, result):
        self.result = result
        self._done.set(True)

    def cancel(self):
        self.finish(None)

    def confirm(self):
        """Enter — у таблицы ничего; у фото — «Добавить»."""

    def error(self, text: str):
        self._error.configure(text=text)

    # ── файлы ──
    def dropped(self, paths: list):
        self.zone.flash()
        self.take(paths)

    def take(self, paths: list):
        raise NotImplementedError

    def _browse(self):
        raise NotImplementedError


class TableDropPanel(DropPanel):
    """Таблица компании: .xlsx, .xls или .csv. result — путь к файлу."""

    def __init__(self, window, title: str, subtitle: str):
        super().__init__(window, title, subtitle, "⇩",
                         "Перетащите сюда таблицу компании",
                         "из проводника или с рабочего стола (.xlsx, .xls, .csv)\nили щёлкните здесь, чтобы выбрать")

    def take(self, paths: list):
        tables = [p for p in paths if Path(p).is_file() and Path(p).suffix.lower() in TABLE_EXTS]
        if len(tables) != 1:
            names = ", ".join(Path(p).name for p in paths[:3])
            self.error("Нужна одна таблица (.xlsx, .xls или .csv)" + (f", а это: {names}" if names else "") + ".")
            return
        self.finish(tables[0])

    def _browse(self):
        p = filedialog.askopenfilename(parent=self.window, title="Таблица компании",
                                       filetypes=[("Таблицы", "*.xls *.xlsx *.csv"), ("Все файлы", "*.*")])
        if p:
            self.take([p])


class PhotoDropPanel(DropPanel):
    """«Добавить фото»: что перетащили → группы по контролёрам → «Добавить».
    result — список PhotoGroup с выбранными именами."""

    def __init__(self, window, existing: list, paths: Optional[list] = None):
        super().__init__(window, "Добавить фото",
                         "Фото копируются в папку месяца «фото» — оригиналы остаются на месте. "
                         "Если у контролёра уже есть папка, новые фото добавятся к ней; "
                         "те же самые фото второй раз не копируются.",
                         "⇩", "Перетащите сюда папку с фото контролёра",
                         "или папку с папками контролёров, или сами фото\nили щёлкните здесь, чтобы выбрать папку")
        self._existing = existing
        self._groups: list[PhotoGroup] = []
        self._name_var = tk.StringVar(master=self)
        self._add_btn = action_button(self.buttons, "Добавить   Enter", self.confirm, kind="primary")
        if paths:
            self.after_idle(lambda: self.take(paths))

    def take(self, paths: list):
        groups = find_groups(paths)
        if not groups:
            self.error("Здесь нет фото (.jpg, .jpeg, .png).")
            return
        self.error("")
        self._groups = groups
        for g in groups:
            if g.name:
                g.name = match_controller(g.name, self._existing)
        self._show_groups()

    def _show_groups(self):
        for c in self.body.winfo_children():
            c.destroy()
        total = sum(len(g.files) for g in self._groups)
        self.zone.set_text(f"Найдено: {photos_word(total)}", "можно перетащить другую папку — тогда возьмётся она")
        box = tk.Frame(self.body, bg=SURFACE)
        box.pack(fill=tk.X, pady=(px(14), 0))
        if len(self._groups) == 1:
            g = self._groups[0]
            row = tk.Frame(box, bg=SURFACE)
            row.pack(fill=tk.X)
            tk.Label(row, text="Чьи это фото (контролёр):", font=(FONT, 10), fg=TEXT, bg=SURFACE).pack(side=tk.LEFT)
            self._name_var.set(g.name)
            combo = ttk.Combobox(row, textvariable=self._name_var, values=self._existing, width=28)
            combo.pack(side=tk.LEFT, padx=(px(8), 0))
            self._name_var.trace_add("write", lambda *a: self._update_note())
            self._note = tk.Label(box, text="", font=(FONT, 9), fg=MUTED, bg=SURFACE, justify="left",
                                  wraplength=px(560))
            self._note.pack(anchor="w", pady=(px(6), 0))
            self._update_note()
            if not g.name:
                combo.focus_set()
        else:
            tk.Label(box, text="По контролёрам:", font=(FONT, 10), fg=TEXT, bg=SURFACE).pack(anchor="w")
            for g in self._groups:
                known = g.name in self._existing
                tk.Label(box, text=f"   • {g.name} — {photos_word(len(g.files))}"
                         + ("  (добавятся к его папке)" if known else "  (новая папка)"),
                         font=(FONT, 10), fg=TEXT, bg=SURFACE).pack(anchor="w")
        self._add_btn.pack(side=tk.RIGHT, padx=(0, px(8)))

    def _update_note(self):
        name = match_controller(self._name_var.get(), self._existing)
        if not name.strip():
            text = "Выберите контролёра из списка или впишите имя — будет новая папка."
        elif name in self._existing:
            text = f"У «{name}» папка уже есть — фото добавятся к ней."
        else:
            text = f"Будет новая папка контролёра «{name}»."
        self._note.configure(text=text)

    def confirm(self):
        if not self._groups:
            return
        if len(self._groups) == 1:
            name = match_controller(self._name_var.get(), self._existing)
            problem = check_name(name)
            if problem:
                self.error(problem)
                return
            self._groups[0].name = name
        self.finish(self._groups)

    def _browse(self):
        p = filedialog.askdirectory(parent=self.window, title="Папка с фото")
        if p:
            self.take([p])
