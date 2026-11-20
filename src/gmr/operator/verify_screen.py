"""Экран проверки одного фото из plus/ и minus/: Верно, Исправить, Пропустить."""
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Optional

from PIL import Image, ImageTk

from .common import CLR_BG, CLR_GRAY, CLR_GREEN, CLR_RED
from .decisions import plan_verify_correction
from .files import free_photo_path, move_photo, now_iso, redraw_annotation
from .photos import controller_dir
from .widgets import ReadingWidget

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
        top.pack(fill=tk.X, padx=12, pady=(10, 6))
        ttk.Button(top, text="← Назад", command=self._back).pack(side=tk.LEFT)
        ttk.Label(
            top,
            text=f"Проверка  •  {Path(self.item['path']).name}",
            font=("Segoe UI", 10, "bold"),
        ).pack(side=tk.LEFT, padx=16)
        ttk.Label(top, text="🔍 колёсико — зум  |  ЛКМ тащить  |  2× клик — сброс",
                  font=("Segoe UI", 8), foreground=CLR_GRAY).pack(side=tk.RIGHT, padx=8)

        center = ttk.Frame(self)
        center.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        self._canvas = tk.Canvas(center, bg="#222222", width=560)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self._canvas.bind("<Configure>",       lambda e: self._load_photo())
        self._canvas.bind("<MouseWheel>",      self._on_mousewheel)
        self._canvas.bind("<Button-4>",        self._on_mousewheel)
        self._canvas.bind("<Button-5>",        self._on_mousewheel)
        self._canvas.bind("<ButtonPress-1>",   self._on_pan_start)
        self._canvas.bind("<B1-Motion>",       self._on_pan_move)
        self._canvas.bind("<Double-Button-1>", lambda e: self._reset_zoom())

        right = ttk.Frame(center, width=320)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(16, 0))
        right.pack_propagate(False)
        self._build_right(right, r)

        self._build_buttons()

    def _build_right(self, parent, r: dict):
        def row(label, value, bold=False, color=None):
            ttk.Label(parent, text=label, font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 0))
            lbl = tk.Label(parent, text=value,
                           font=("Segoe UI", 12, "bold" if bold else "normal"),
                           fg=color or "black", bg=CLR_BG, anchor="w")
            lbl.pack(anchor="w")
            return lbl

        row("Serial ID",           r.get("serial_id", "—"))
        row("Account ID",          r.get("account_id", "—"))
        row("Показания",           r.get("reading", "—"),   bold=True)
        row("Последние показания", r.get("last_reading", "—"))

        delta_str = r.get("delta", "")
        try:
            delta_val = float(delta_str)
            color = CLR_GREEN if delta_val >= 0 else CLR_RED
            delta_label = f"Δ {delta_val:+.0f}"
        except (ValueError, TypeError):
            color = CLR_GRAY
            delta_label = "—"
        row("Дельта", delta_label, bold=True, color=color)

        row("Статус",      r.get("outcome", "—"), bold=True)
        row("Обработано",  f"auto | {r.get('processed_at', '')[:16]}")

        self._edit_frame = ttk.LabelFrame(parent, text="Исправление", padding=6)
        ttk.Label(self._edit_frame, text="Серийный номер").pack(anchor="w")
        self._serial_var = tk.StringVar(value=r.get("serial_id", ""))
        ttk.Entry(self._edit_frame, textvariable=self._serial_var, width=22).pack(anchor="w")
        ttk.Label(self._edit_frame, text="Показания").pack(anchor="w", pady=(6, 0))
        self._reading_widget = ReadingWidget(self._edit_frame)
        self._reading_widget.pack(anchor="w")
        self._reading_widget.set_digits(r.get("reading", ""), None)

    def _build_buttons(self):
        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=12, pady=(6, 10))

        self._ok_btn = tk.Button(
            bottom, text="✓  Верно  [Enter]",
            bg=CLR_GREEN, fg="white", font=("Segoe UI", 11, "bold"),
            relief="flat", padx=16, pady=8,
            command=self._verify_ok,
        )
        self._ok_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._edit_btn = tk.Button(
            bottom, text="✎  Исправить  [E]",
            bg="#1565C0", fg="white", font=("Segoe UI", 10),
            relief="flat", padx=12, pady=8,
            command=self._enter_edit,
        )
        self._edit_btn.pack(side=tk.LEFT, padx=4)

        self._save_edit_btn = tk.Button(
            bottom, text="💾  Сохранить правку  [Enter]",
            bg="#6A1B9A", fg="white", font=("Segoe UI", 10),
            relief="flat", padx=12, pady=8,
            command=self._save_edit,
        )

        tk.Button(
            bottom, text="→  Пропустить  [→]",
            bg=CLR_GRAY, fg="white", font=("Segoe UI", 10),
            relief="flat", padx=12, pady=8,
            command=self._skip,
        ).pack(side=tk.RIGHT, padx=4)

    def _bind_keys(self):
        # Enter: «Верно», а в режиме исправления — «Сохранить правку»
        self.app.bind_all("<Return>", lambda e: self._on_return() if self.app.take_return(e) else None)
        self.app.bind_all("<Escape>", lambda e: self._back())
        self.app.bind_all("<e>",      lambda e: None if self._edit_mode else self._enter_edit())
        self.app.bind_all("<E>",      lambda e: None if self._edit_mode else self._enter_edit())
        self.app.bind_all("<Right>",  lambda e: None if self._edit_mode else self._skip())

    def _unbind_keys(self):
        for seq in ("<Return>", "<Escape>", "<e>", "<E>", "<Right>"):
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
