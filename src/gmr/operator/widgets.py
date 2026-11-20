"""Виджеты: пять полей показаний."""
import tkinter as tk
from tkinter import ttk
from typing import Optional

from .common import CLR_WHITE


class ReadingWidget(ttk.Frame):
    """
    5 однозначных полей для показаний счётчика.
    Подсвечивает оранжевым неуверенные позиции.
    """
    def __init__(self, parent, on_change=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._on_change = on_change
        self._entries: list[tk.Entry] = []
        self._vars:  list[tk.StringVar] = []
        self._build()

    def _build(self):
        for i in range(5):
            var = tk.StringVar()
            var.trace_add("write", lambda *a, i=i: self._on_var_change(i))
            e = tk.Entry(
                self, textvariable=var, width=3,
                font=("Segoe UI", 18, "bold"),
                justify="center",
                relief="solid",
                bd=1,
            )
            e.grid(row=0, column=i, padx=3)
            e.bind("<Key>", lambda ev, i=i: self._on_key(ev, i))
            e.bind("<FocusIn>", lambda ev, i=i: e.select_range(0, tk.END))
            self._entries.append(e)
            self._vars.append(var)

    def _on_var_change(self, pos: int):
        val = self._vars[pos].get()
        if len(val) > 1:
            self._vars[pos].set(val[-1])
            val = val[-1]
        if val == "?" or val == "":
            self._entries[pos].config(bg="#FFE0B2")
        else:
            self._entries[pos].config(bg=CLR_WHITE)
        if self._on_change:
            self._on_change()

    def _on_key(self, ev, pos: int):
        key = ev.keysym
        if key in ("BackSpace",):
            if self._vars[pos].get() == "":
                if pos > 0:
                    self._entries[pos - 1].focus_set()
            else:
                self._vars[pos].set("")
            return "break"
        if key == "Left" and pos > 0:
            self._entries[pos - 1].focus_set()
            return "break"
        if key == "Right" and pos < 4:
            self._entries[pos + 1].focus_set()
            return "break"
        if key.isdigit():
            self._vars[pos].set(key)
            if pos < 4:
                self._entries[pos + 1].focus_set()
            return "break"

        passthrough = {
            "Return", "KP_Enter", "Escape", "Delete",
            "d", "D", "n", "N", "Right",
            "Tab", "Shift_L", "Shift_R",
            "Control_L", "Control_R", "Alt_L", "Alt_R",
            "F1", "F2", "F3", "F4", "F5",
        }
        if key not in passthrough:
            return "break"

    def get_entry_widgets(self) -> list[tk.Entry]:
        return self._entries

    def set_digits(self, reading_str: Optional[str], digit_preds: Optional[list]):
        chars = list(reading_str or "?????")
        if len(chars) != 5:
            chars = ["?"] * 5

        for i, (ch, entry, var) in enumerate(zip(chars, self._entries, self._vars)):
            conf_ok = True
            if digit_preds and i < len(digit_preds):
                conf_ok = digit_preds[i].get("ok", True)

            var.set(ch if ch.isdigit() and conf_ok else "?")
            color = CLR_WHITE if (ch.isdigit() and conf_ok) else "#FFE0B2"
            entry.config(bg=color)

    def get_string(self) -> str:
        return "".join(v.get() or "?" for v in self._vars)

    def is_complete(self) -> bool:
        s = self.get_string()
        return len(s) == 5 and all(c.isdigit() for c in s)

    def clear(self):
        for v in self._vars:
            v.set("")
        for e in self._entries:
            e.config(bg=CLR_WHITE)



# ─── Вспомогательный миксин ───────────────────────────────────────────────────
def bind_tree(widget, seq, callback):
    widget.bind(seq, callback)
    for child in widget.winfo_children():
        bind_tree(child, seq, callback)

def unbind_tree(widget, seq):
    try:
        widget.unbind(seq)
    except Exception:
        pass
    for child in widget.winfo_children():
        unbind_tree(child, seq)
