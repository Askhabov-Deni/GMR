"""Виджеты: пять полей показаний."""
import tkinter as tk
from tkinter import ttk
from typing import Optional

from .common import CLR_WHITE
from .theme import px


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
            e.grid(row=0, column=i, padx=px(3))
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


# ─── Фото с масштабом ────────────────────────────────────────────────────────
def typing_in_field(window) -> bool:
    """Курсор в поле ввода — буквы и цифры идут в поле, а не в горячие клавиши."""
    try:
        w = window.focus_get()
    except (KeyError, tk.TclError):
        return False
    return isinstance(w, (tk.Entry, ttk.Entry, ttk.Spinbox, tk.Text))


class PhotoView(tk.Canvas):
    """
    Фото на экранах разбора и проверки (1.4.0 — общий для обоих экранов).

    - сначала фото целиком;
    - колёсико — масштаб к точке под мышью (раньше — к середине: нужное место
      уезжало), клавиши + и − — к середине, 0 или двойной щелчок — снова целиком;
    - перетащить мышью — сдвинуть;
    - при большом увеличении рисуется только видимая часть фото — не тормозит
      (раньше 10× пересчитывало всё фото 4000×3000 в 40000×30000).
    """
    MIN_ZOOM, MAX_ZOOM, STEP = 1.0, 12.0, 1.25
    FULL_LIMIT = 6_000_000          # пикселей: до стольких фото целиком в памяти — сдвиг мгновенный

    def __init__(self, parent, path: str, bg: str = "#2b2f36", **kw):
        super().__init__(parent, bg=bg, highlightthickness=0, **kw)
        self.path = path
        self._img = None                         # PIL.Image — открывается при первом показе
        self.zoom = 1.0                          # 1 — фото целиком
        self._center: Optional[list] = None      # точка фото в середине окна (в пикселях фото)
        self._drag = (0, 0)
        self._cache = None                       # (масштаб, PhotoImage целиком) — пока помещается в память
        self._tk_img = None
        self._pending = False
        self.bind("<Configure>", lambda e: self.render())
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Button-4>", self._on_wheel)
        self.bind("<Button-5>", self._on_wheel)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<Double-Button-1>", lambda e: self.reset())

    # ── управление ──
    def zoom_by(self, factor: float, at: Optional[tuple] = None) -> None:
        """Масштаб в factor раз; at — точка окна, которая остаётся на месте (None — середина)."""
        if self._image() is None:
            return
        w, h = self.winfo_width(), self.winfo_height()
        new = max(self.MIN_ZOOM, min(self.zoom * factor, self.MAX_ZOOM))
        if new == self.zoom:
            return
        ax, ay = at if at is not None else (w / 2, h / 2)
        s0, s1 = self._scale(w, h), self._scale(w, h, new)
        cx, cy = self._center
        px_, py_ = cx + (ax - w / 2) / s0, cy + (ay - h / 2) / s0          # точка фото под мышью
        self.zoom = new
        self._center = [px_ - (ax - w / 2) / s1, py_ - (ay - h / 2) / s1]
        self.render()

    def zoom_in(self) -> None:
        self.zoom_by(self.STEP)

    def zoom_out(self) -> None:
        self.zoom_by(1 / self.STEP)

    def reset(self) -> None:
        img = self._image()
        if img is None:
            return
        self.zoom = 1.0
        self._center = [img.width / 2, img.height / 2]
        self.render()

    def _on_wheel(self, e):
        up = e.num == 4 or getattr(e, "delta", 0) > 0
        self.zoom_by(self.STEP if up else 1 / self.STEP, at=(e.x, e.y))

    def _on_press(self, e):
        self._drag = (e.x, e.y)
        self.focus_set()                    # клавиши — экрану, а не полю ввода

    def _on_drag(self, e):
        if self._center is None:
            return
        s = self._scale(self.winfo_width(), self.winfo_height())
        self._center[0] -= (e.x - self._drag[0]) / s
        self._center[1] -= (e.y - self._drag[1]) / s
        self._drag = (e.x, e.y)
        self._schedule()

    # ── рисование ──
    def _image(self):
        if self._img is None:
            try:
                from PIL import Image
                self._img = Image.open(self.path)
                self._img.load()
                self._center = [self._img.width / 2, self._img.height / 2]
            except Exception:                # нет файла — серый холст, без ошибки
                return None
        return self._img

    def _scale(self, w, h, zoom=None) -> float:
        img = self._img
        return min(w / img.width, h / img.height) * (self.zoom if zoom is None else zoom)

    def _schedule(self):
        """Сдвиг мышью — не чаще кадра: события копятся, рисуется последнее."""
        if not self._pending:
            self._pending = True
            self.after_idle(self._flush)

    def _flush(self):
        self._pending = False
        self.render()

    def render(self) -> None:
        w, h = self.winfo_width(), self.winfo_height()
        img = self._image()
        if img is None or w < 10 or h < 10:
            return
        from PIL import Image, ImageTk
        s = self._scale(w, h)
        full_w, full_h = max(1, round(img.width * s)), max(1, round(img.height * s))
        # середина — не дальше края фото
        half_w, half_h = w / 2 / s, h / 2 / s
        cx = min(max(self._center[0], min(half_w, img.width / 2)), max(img.width - half_w, img.width / 2))
        cy = min(max(self._center[1], min(half_h, img.height / 2)), max(img.height - half_h, img.height / 2))
        self._center = [cx, cy]
        self.delete("all")
        if full_w * full_h <= self.FULL_LIMIT:
            if self._cache is None or self._cache[0] != (full_w, full_h):
                self._cache = ((full_w, full_h), ImageTk.PhotoImage(
                    img.resize((full_w, full_h), Image.LANCZOS, reducing_gap=3.0), master=self))
            self._tk_img = self._cache[1]
            self.create_image(w / 2 - cx * s, h / 2 - cy * s, anchor="nw", image=self._tk_img)
        else:                                       # крупно: только то, что видно
            left, top = max(0, cx - half_w), max(0, cy - half_h)
            right, bottom = min(img.width, cx + half_w), min(img.height, cy + half_h)
            part = img.crop((int(left), int(top), int(right) + 1, int(bottom) + 1))
            pw, ph = max(1, round(part.width * s)), max(1, round(part.height * s))
            self._tk_img = ImageTk.PhotoImage(part.resize((pw, ph), Image.BILINEAR), master=self)
            self.create_image(w / 2 + (int(left) - cx) * s, h / 2 + (int(top) - cy) * s, anchor="nw",
                              image=self._tk_img)
        if self.zoom > 1.01:
            from .theme import FONT, px
            self.create_text(px(10), px(8), anchor="nw", text=f"×{self.zoom:.1f}   0 — целиком",
                             font=(FONT, 9, "bold"), fill="#ffffff")


PHOTO_HINT = "колёсико или + −  — масштаб   ·   перетащить — сдвиг   ·   0 — целиком"


def photo_keys(view: "PhotoView", window) -> list:
    """Клавиши масштаба фото для экрана: + − 0 (и на цифровом блоке). Пока
    курсор в поле ввода — не мешают печатать."""
    def act(fn):
        return lambda e: None if typing_in_field(window) else (fn(), "break")[1]
    return ([(s, act(view.zoom_in)) for s in ("<plus>", "<equal>", "<KP_Add>")]
            + [(s, act(view.zoom_out)) for s in ("<minus>", "<KP_Subtract>")]
            + [(s, act(view.reset)) for s in ("<Key-0>", "<KP_0>", "<KP_Insert>")])
