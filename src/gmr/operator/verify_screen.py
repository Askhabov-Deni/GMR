"""Экран проверки одного фото из plus/ и minus/: Верно, Исправить, Пропустить."""
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Optional

from PIL import Image, ImageTk

from .decisions import plan_verify_correction
from .files import free_photo_path, move_photo, now_iso, redraw_annotation
from .photos import controller_dir
from .theme import (
    FONT,
    MUTED,
    PHOTO_BG,
    SUCCESS,
    SURFACE,
    TEXT,
    action_button,
    card,
    key_sequences,
)
from .widgets import ReadingWidget

OUTCOME_WORDS = {"PLUS": "▲ больше прошлого", "MINUS": "▼ меньше прошлого"}

if TYPE_CHECKING:                      # только для подсказки типов: окно импортирует экраны
    from .window import MainWindow


class VerifyScreen(ttk.Frame):
    """Проверка фото из plus/ minus/ которые обработал reader.py автоматически."""
    def __init__(self, parent: "MainWindow", item: dict, on_back, index: int = 0):
        super().__init__(parent)
        self.app      = parent
        self.item     = item
        self.index    = index        # номер фото в списке «Проверки» — для перехода к следующему
        self._on_back = on_back
        self._tk_img: Optional[ImageTk.PhotoImage] = None
        self._edit_mode = False

        self._zoom_scale = 1.0
        self._pan_start  = (0, 0)
        self._pan_offset = [0, 0]
        self._orig_image: Optional[Image.Image] = None

        self._build()
        self._load_photo()
        self._bind_keys()

    def _build(self):
        r = self.item["log_row"]

        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=16, pady=(12, 8))
        ttk.Button(top, text="←  К списку", command=self._back).pack(side=tk.LEFT)
        tk.Label(top, text="Проверка", bg="#e3edfd", fg="#1858bd", font=(FONT, 10, "bold"),
                 padx=10, pady=3).pack(side=tk.LEFT, padx=(16, 10))
        ttk.Label(top, text=f"Фото {self.index + 1}   ·   {Path(self.item['path']).name}",
                  style="Muted.TLabel").pack(side=tk.LEFT)
        ttk.Label(top, text="колёсико — увеличить  ·  тащить мышью  ·  двойной щелчок — сброс",
                  style="Muted.TLabel", font=(FONT, 9)).pack(side=tk.RIGHT)

        center = ttk.Frame(self)
        center.pack(fill=tk.BOTH, expand=True, padx=16, pady=4)

        self._canvas = tk.Canvas(center, bg=PHOTO_BG, width=560, highlightthickness=0)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._canvas.bind("<Configure>",       lambda e: self._load_photo())
        self._canvas.bind("<MouseWheel>",      self._on_mousewheel)
        self._canvas.bind("<Button-4>",        self._on_mousewheel)
        self._canvas.bind("<Button-5>",        self._on_mousewheel)
        self._canvas.bind("<ButtonPress-1>",   self._on_pan_start)
        self._canvas.bind("<B1-Motion>",       self._on_pan_move)
        self._canvas.bind("<Double-Button-1>", lambda e: self._reset_zoom())

        right = card(center)
        right.configure(width=360)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(16, 0))
        right.pack_propagate(False)
        self._build_right(right, r)

        self._build_buttons()

    def _build_right(self, parent, r: dict):
        ttk.Label(parent, text="Программа записала сама — верно?", style="CardMuted.TLabel").pack(
            anchor="w", pady=(0, 12))

        def row(label, value, big=False, color=None):
            ttk.Label(parent, text=label, style="CardTitle.TLabel").pack(anchor="w")
            lbl = tk.Label(parent, text=value, font=(FONT, 18 if big else 13, "bold" if big else "normal"),
                           fg=color or TEXT, bg=SURFACE, anchor="w")
            lbl.pack(anchor="w", pady=(0, 10))
            return lbl

        row("НОМЕР СЧЁТЧИКА", r.get("serial_id") or "—")
        row("ЛИЦЕВОЙ СЧЁТ", r.get("account_id") or "—")
        row("ПОКАЗАНИЕ", r.get("reading") or "—", big=True)
        row("ПРОШЛОЕ ПОКАЗАНИЕ", r.get("last_reading") or "—")

        try:
            delta_val = float(r.get("delta", ""))
            color, delta_label = (SUCCESS if delta_val >= 0 else "#c62828"), f"{delta_val:+.0f}"
        except (ValueError, TypeError):
            color, delta_label = MUTED, "—"
        row("РАСХОД", delta_label, color=color)
        ttk.Label(parent, text=f"{OUTCOME_WORDS.get(r.get('outcome', ''), r.get('outcome', ''))}   ·   "
                               f"прочитано {r.get('processed_at', '')[:16].replace('T', ' ')}",
                  style="CardMuted.TLabel").pack(anchor="w")

        self._edit_frame = ttk.LabelFrame(parent, text="ИСПРАВЛЕНИЕ", padding=10)
        ttk.Label(self._edit_frame, text="Номер счётчика", style="Card.TLabel").pack(anchor="w")
        self._serial_var = tk.StringVar(value=r.get("serial_id", ""))
        ttk.Entry(self._edit_frame, textvariable=self._serial_var, font=(FONT, 13)).pack(fill=tk.X)
        ttk.Label(self._edit_frame, text="Показание", style="Card.TLabel").pack(anchor="w", pady=(8, 0))
        self._reading_widget = ReadingWidget(self._edit_frame, style="Card.TFrame")
        self._reading_widget.pack(anchor="w")
        self._reading_widget.set_digits(r.get("reading", ""), None)

    def _build_buttons(self):
        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=16, pady=(8, 14))

        self._ok_btn = action_button(bottom, "✓   Верно     Enter", self._verify_ok, kind="primary")
        self._ok_btn.pack(side=tk.LEFT, padx=(0, 12))
        self._edit_btn = action_button(bottom, "Исправить     E", self._enter_edit)
        self._edit_btn.pack(side=tk.LEFT, padx=4)
        self._save_edit_btn = action_button(bottom, "✓   Сохранить исправление     Enter", self._save_edit,
                                            kind="accent")
        action_button(bottom, "Пропустить     →", self._skip).pack(side=tk.RIGHT)

    def _key_bindings(self) -> list:
        edit = lambda e: None if self._edit_mode else self._enter_edit()   # noqa: E731
        return ([("<Return>", lambda e: self._on_return() if self.app.take_return(e) else None),
                 ("<Escape>", lambda e: self._back()),
                 ("<Right>", lambda e: None if self._edit_mode else self._skip())]
                + [(seq, edit) for seq in key_sequences("e")])

    def _bind_keys(self):
        # Enter: «Верно», а в режиме исправления — «Сохранить исправление»
        for seq, cb in self._key_bindings():
            self.app.bind_all(seq, cb)

    def _unbind_keys(self):
        for seq, _ in self._key_bindings():
            try:
                self.app.unbind_all(seq)
            except Exception:
                pass

    def _on_mousewheel(self, event):
        if event.num == 4 or event.delta > 0:
            factor = 1.15
        elif event.num == 5 or event.delta < 0:
            factor = 1 / 1.15
        else:
            return
        self._zoom_scale = max(0.2, min(self._zoom_scale * factor, 10.0))
        self._load_photo()

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y)

    def _on_pan_move(self, event):
        self._pan_offset[0] += event.x - self._pan_start[0]
        self._pan_offset[1] += event.y - self._pan_start[1]
        self._pan_start = (event.x, event.y)
        self._load_photo()

    def _reset_zoom(self):
        self._zoom_scale = 1.0
        self._pan_offset = [0, 0]
        self._load_photo()

    def _load_photo(self):
        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        if w < 10 or h < 10:
            self.after(100, self._load_photo)
            return
        try:
            if self._orig_image is None:
                self._orig_image = Image.open(self.item["path"])
            orig_w, orig_h = self._orig_image.size
            base_scale = min(w / orig_w, h / orig_h)
            disp_w = max(1, int(orig_w * base_scale * self._zoom_scale))
            disp_h = max(1, int(orig_h * base_scale * self._zoom_scale))
            # Кеш, как на экране «Обработка»: перетаскивание не пересчитывает
            # фото заново (было 0,2 с на каждый шаг мыши, замер 2026-11-20)
            cached = getattr(self, "_cached_render", None)
            if cached is None or cached[0] != (disp_w, disp_h):
                img = self._orig_image.resize((disp_w, disp_h), Image.LANCZOS, reducing_gap=3.0)
                self._tk_img = ImageTk.PhotoImage(img)
                self._cached_render = ((disp_w, disp_h), self._tk_img)
            else:
                self._tk_img = cached[1]
            self._canvas.delete("all")
            cx = w // 2 + self._pan_offset[0]
            cy = h // 2 + self._pan_offset[1]
            self._canvas.create_image(cx, cy, anchor="center", image=self._tk_img)
            if abs(self._zoom_scale - 1.0) > 0.01:
                self._canvas.create_text(
                    8, 8, anchor="nw",
                    text=f"×{self._zoom_scale:.1f}",
                    font=("Segoe UI", 10, "bold"),
                    fill="#ffffff",
                )
        except Exception:
            pass

    def _back(self):
        try:
            self._on_back()
        finally:
            self._unbind_keys()

    def _on_return(self):
        if self._edit_mode:
            self._save_edit()
        else:
            self._verify_ok()

    def _skip(self):
        self._unbind_keys()
        self.app.open_verify_at(self.index + 1)

    def _verify_ok(self):
        # Отметка — ровно в ту строку, по которой фото попало в список
        # (раньше искалась первая строка с тем же original_filename).
        r = self.item["log_row"]
        self.app.session.mark_verified(r, self.app.settings.operator_name, now_iso())
        self._unbind_keys()
        self.app.open_verify_at(self.index)

    def _enter_edit(self):
        if self._edit_mode:
            return
        self._edit_mode = True
        self._edit_frame.pack(fill=tk.X, pady=(12, 0))
        # «Верно» в режиме исправления прячем: её нажатие выбрасывало правку
        self._ok_btn.pack_forget()
        self._edit_btn.pack_forget()
        self._save_edit_btn.pack(side=tk.LEFT, padx=4)

    def _save_edit(self):
        r = self.item["log_row"]
        if not self._reading_widget.is_complete():
            messagebox.showwarning("Ошибка", "Введите все 5 цифр показаний.", parent=self.app)
            return
        cfg = self.app.config
        c = plan_verify_correction(self.app.df, cfg, r,
                                   self._serial_var.get(), self._reading_widget.get_string())
        if c.error:
            messagebox.showwarning("Нельзя сохранить", c.error, parent=self.app)
            return
        if c.account_changed:
            if not messagebox.askyesno(
                "Другой абонент",
                f"Серийный номер {c.serial} принадлежит абоненту {c.new_account}.\n\n"
                f"Показание {c.reading} будет записано ему, а у абонента {c.old_account} "
                f"показание, записанное автоматически, будет стёрто.\n\nПродолжить?",
                parent=self.app,
            ):
                return
            if c.new_account_has_reading and not messagebox.askyesno(
                "У абонента уже есть показание",
                f"У абонента {c.new_account} уже записано показание {c.new_account_has_reading}.\n\n"
                f"Заменить его на {c.reading}?",
                parent=self.app,
            ):
                return

        old_path = self.item["path"]
        self.app.session.apply_correction(r, c, self.app.settings.operator_name, now_iso())
        for account in {c.old_account, c.new_account}:
            self.app._sync_df_reading(account)

        # Фото — в папку по новому исходу и под именем нового абонента
        folder = "minus" if c.outcome == "MINUS" else "plus"
        dst_dir = str(controller_dir(old_path) / folder)
        name = r.get("final_filename") or Path(old_path).name
        try:
            dst = free_photo_path(dst_dir, name, old_path)
            if Path(dst).resolve() != Path(old_path).resolve():
                dst = move_photo(old_path, dst_dir, Path(dst).name)
            if Path(dst).name != r.get("final_filename"):
                self.app.session.rename_photo_in_log(r, Path(dst).name)
            redraw_annotation(dst, c.serial, f"{c.reading:05d}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось переместить фото:\n{e}", parent=self.app)

        self._unbind_keys()
        self.app.open_verify_at(self.index)
