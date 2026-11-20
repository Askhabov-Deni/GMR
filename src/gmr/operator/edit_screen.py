"""Экран разбора одного фото из question/: Принять, Дубль, Нечитаемо, Нет в базе, Серийник в базе с ошибкой."""
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import TYPE_CHECKING, Optional

import pandas as pd
from PIL import Image, ImageTk

from src.gmr.domain import find_auto_row_for_output_file
from src.gmr.domain.datasets import to_etalon
from src.gmr.domain.serial_match import lookup_variants
from src.gmr.domain.serial_match import normalize_serial as _normalize_serial
from src.gmr.storage import LOG_COLUMNS as _LOG_COLUMNS

from .common import (
    CLR_BG,
    CLR_BROWN,
    CLR_GRAY,
    CLR_GREEN,
    CLR_ORANGE,
    CLR_RED,
    REASON_LABELS,
    log,
)
from .decisions import accounts_with_serial, not_in_db_serial
from .files import (
    DB_SERIAL_FIX_LIST,
    append_db_serial_fix,
    describe_reading,
    free_photo_path,
    move_photo,
    now_iso,
    place_decision,
    redraw_annotation,
    table_serial_for_account,
)
from .hints import CHOICE_NOTE, HINT_NOTE, SerialHint, serial_choices, serial_hints
from .markup import save_cnn_markup, save_crnn_markup, save_meter_markup
from .photos import _to_float, controller_dir, controller_name
from .widgets import ReadingWidget

if TYPE_CHECKING:                      # только для подсказки типов: окно импортирует экраны
    from .window import MainWindow


class EditScreen(ttk.Frame):
    """
    Главный рабочий экран оператора.
    Слева — фото, справа — поля ввода и кнопки.
    """
    def __init__(self, parent: "MainWindow", photo_path: str, reason: str, on_back):
        super().__init__(parent)
        self.app        = parent
        self.photo_path = photo_path
        self.folder     = controller_dir(photo_path)   # папка контролёра этого фото
        self.reason     = reason
        self._on_back   = on_back
        self._model_result: Optional[dict] = None
        self._tk_img: Optional[ImageTk.PhotoImage] = None
        self._confirmed_suspicious  = False
        self._filling_other: bool = False
        self._model_serial:  Optional[str] = None
        self._model_reading: Optional[str] = None
        self._reading_manually_changed: bool = False
        self._abonent_found: bool = False      # абонент найден в таблице по номеру или счёту

        self._zoom_scale = 1.0
        self._pan_start  = (0, 0)
        self._pan_offset = [0, 0]
        self._orig_image: Optional[Image.Image] = None

        self._build()
        self._start_inference()

    def _build(self):
        self.configure(padding=0)

        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=12, pady=(10, 6))

        ttk.Button(top, text="← Назад", command=self._back).pack(side=tk.LEFT)
        ttk.Label(
            top,
            text=f"{REASON_LABELS.get(self.reason, self.reason)}  •  {Path(self.photo_path).name}",
            font=("Segoe UI", 10, "bold"),
        ).pack(side=tk.LEFT, padx=16)

        ttk.Label(top, text="🔍 колёсико — зум  |  ЛКМ тащить  |  2× клик — сброс",
                  font=("Segoe UI", 8), foreground=CLR_GRAY).pack(side=tk.RIGHT, padx=8)

        center = ttk.Frame(self)
        center.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        self._canvas = tk.Canvas(center, bg="#222222", width=560)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self._canvas.bind("<Configure>",       lambda e: self._display_photo())
        self._canvas.bind("<MouseWheel>",      self._on_mousewheel)
        self._canvas.bind("<Button-4>",        self._on_mousewheel)
        self._canvas.bind("<Button-5>",        self._on_mousewheel)
        self._canvas.bind("<ButtonPress-1>",   self._on_pan_start)
        self._canvas.bind("<B1-Motion>",       self._on_pan_move)
        self._canvas.bind("<Double-Button-1>", lambda e: self._reset_zoom())

        right = ttk.Frame(center, width=320)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(16, 0))
        right.pack_propagate(False)
        self._build_fields(right)

        self._build_buttons()
        self._bind_keys()

    def _build_fields(self, parent):
        self._status_var = tk.StringVar(value="⏳ Загрузка…")
        ttk.Label(parent, textvariable=self._status_var,
                  font=("Segoe UI", 9), foreground=CLR_GRAY).pack(anchor="w", pady=(0, 8))

        ttk.Label(parent, text="Серийный номер", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self._serial_var = tk.StringVar()
        self._serial_var.trace_add("write", lambda *a: self._on_serial_change())
        self._serial_entry = ttk.Entry(parent, textvariable=self._serial_var, width=24,
                                       font=("Segoe UI", 13))
        self._serial_entry.pack(anchor="w", pady=(2, 2))

        self._serial_match_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self._serial_match_var,
                  font=("Segoe UI", 8), foreground=CLR_GREEN).pack(anchor="w", pady=(0, 6))

        # Подсказка: похожие номера в базе (заполняется, если номера нет в таблице)
        self._hint_frame = ttk.Frame(parent)
        self._hint_frame.pack(anchor="w", fill=tk.X)
        self._hints: list[SerialHint] = []
        self._hint_used: Optional[SerialHint] = None

        ttk.Label(parent, text="Account ID", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self._account_var = tk.StringVar()
        self._account_var.trace_add("write", lambda *a: self._on_account_change())
        self._account_entry = ttk.Entry(parent, textvariable=self._account_var, width=24,
                                        font=("Segoe UI", 13))
        self._account_entry.pack(anchor="w", pady=(2, 2))

        self._account_match_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self._account_match_var,
                  font=("Segoe UI", 8), foreground=CLR_GREEN).pack(anchor="w", pady=(0, 6))

        ttk.Label(parent, text="Показания", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self._reading_widget = ReadingWidget(parent, on_change=self._on_reading_change)
        self._reading_widget.pack(anchor="w", pady=(2, 8))

        ttk.Label(parent, text="Последние показания", font=("Segoe UI", 9)).pack(anchor="w")
        self._last_reading_var = tk.StringVar(value="—")
        ttk.Label(parent, textvariable=self._last_reading_var,
                  font=("Segoe UI", 11)).pack(anchor="w", pady=(2, 4))

        self._delta_var = tk.StringVar(value="—")
        self._delta_lbl = tk.Label(parent, textvariable=self._delta_var,
                                   font=("Segoe UI", 14, "bold"), bg=CLR_BG)
        self._delta_lbl.pack(anchor="w", pady=4)

        self._outcome_var = tk.StringVar(value="—")
        ttk.Label(parent, textvariable=self._outcome_var,
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")

        self._source_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self._source_var,
                  font=("Segoe UI", 8), foreground=CLR_GRAY,
                  wraplength=280, justify="left").pack(anchor="w", pady=(12, 0))

    def _build_buttons(self):
        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=12, pady=(6, 10))

        self._accept_btn = tk.Button(
            bottom, text="✓  Принять  [Enter]",
            bg=CLR_GREEN, fg="white", font=("Segoe UI", 11, "bold"),
            relief="flat", padx=16, pady=8,
            command=self._accept, state=tk.DISABLED,
        )
        self._accept_btn.pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            bottom, text="↩  Дубль  [D]",
            bg="#1565C0", fg="white", font=("Segoe UI", 10),
            relief="flat", padx=12, pady=8,
            command=self._duplicate,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            bottom, text="✗  Нечитаемо  [Del]",
            bg=CLR_RED, fg="white", font=("Segoe UI", 10),
            relief="flat", padx=12, pady=8,
            command=self._unreadable,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            bottom, text="⊘  Нет в базе  [N]",
            bg=CLR_BROWN, fg="white", font=("Segoe UI", 10),
            relief="flat", padx=12, pady=8,
            command=self._not_in_db,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            bottom, text="≠  Серийник в базе с ошибкой  [B]",
            bg="#8D6E63", fg="white", font=("Segoe UI", 10),
            relief="flat", padx=12, pady=8,
            command=self._db_serial_fix,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            bottom, text="→  Пропустить  [Esc]",
            bg=CLR_GRAY, fg="white", font=("Segoe UI", 10),
            relief="flat", padx=12, pady=8,
            command=self._back,
        ).pack(side=tk.RIGHT, padx=4)

    def _bind_keys(self):
        for seq, cb in [
            ("<Return>",   self._hotkey_accept),
            ("<KP_Enter>", self._hotkey_accept),
            ("<Escape>",   self._hotkey_back),
            ("<Delete>",   self._hotkey_unreadable),
            ("<d>",        self._hotkey_duplicate),
            ("<D>",        self._hotkey_duplicate),
            ("<n>",        self._hotkey_not_in_db),
            ("<N>",        self._hotkey_not_in_db),
            ("<b>",        self._hotkey_db_serial_fix),
            ("<B>",        self._hotkey_db_serial_fix),
        ]:
            self.app.bind_all(seq, cb)

    def _unbind_keys(self):
        for seq in ("<Return>", "<KP_Enter>", "<Escape>", "<Delete>",
                    "<d>", "<D>", "<n>", "<N>", "<b>", "<B>"):
            try:
                self.app.unbind_all(seq)
            except Exception:
                pass

    def _focused_widget(self):
        try:
            return self.app.focus_get()
        except Exception:
            return None

    def _is_in_text_entry(self) -> bool:
        return self._focused_widget() in (self._serial_entry, self._account_entry)

    def _is_in_reading_entry(self) -> bool:
        return self._focused_widget() in self._reading_widget.get_entry_widgets()

    def _hotkey_accept(self, event):
        if event is not None and not self.app.take_return(event):
            return "break"
        self._accept()
        return "break"

    def _hotkey_back(self, event):
        w = self._focused_widget()
        if isinstance(w, (tk.Entry, ttk.Entry)):
            self._canvas.focus_set()
            return "break"
        self._back()
        return "break"

    def _hotkey_unreadable(self, event):
        if self._is_in_text_entry():
            return
        if self._is_in_reading_entry():
            return
        self._unreadable()
        return "break"

    def _hotkey_duplicate(self, event):
        if self._is_in_text_entry() or self._is_in_reading_entry():
            return
        self._duplicate()
        return "break"

    def _hotkey_not_in_db(self, event):
        if self._is_in_text_entry() or self._is_in_reading_entry():
            return
        self._not_in_db()
        return "break"

    def _hotkey_db_serial_fix(self, event):
        if self._is_in_text_entry() or self._is_in_reading_entry():
            return
        self._db_serial_fix()
        return "break"

    def _on_mousewheel(self, event):
        if event.num == 4 or event.delta > 0:
            factor = 1.15
        elif event.num == 5 or event.delta < 0:
            factor = 1 / 1.15
        else:
            return
        self._zoom_scale = max(0.2, min(self._zoom_scale * factor, 10.0))
        self._display_photo()

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y)

    def _on_pan_move(self, event):
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self._pan_offset[0] += dx
        self._pan_offset[1] += dy
        self._pan_start = (event.x, event.y)
        self._display_photo()

    def _reset_zoom(self):
        self._zoom_scale = 1.0
        self._pan_offset = [0, 0]
        self._display_photo()

    def _display_photo(self):
        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        if w < 10 or h < 10:
            return

        try:
            if self._orig_image is None:
                self._orig_image = Image.open(self.photo_path)

            orig_w, orig_h = self._orig_image.size
            base_scale = min(w / orig_w, h / orig_h)
            disp_w = max(1, int(orig_w * base_scale * self._zoom_scale))
            disp_h = max(1, int(orig_h * base_scale * self._zoom_scale))

            # Кеш: не пересчитываем LANCZOS если размер не изменился
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

    def _start_inference(self):
        self._display_photo()
        if getattr(self.app, "_model_load_error", None):
            self._status_var.set("⚠ Модели не загружены — введите вручную")
            return
        if self.app.models is None:
            self.after(500, self._start_inference)
            return
        self._status_var.set("⏳ Распознавание…")
        q = queue.Queue()
        def _worker():
            r = self.app.models.run_on_photo(self.photo_path)
            q.put(r)
        threading.Thread(target=_worker, daemon=True).start()
        self._poll_inference(q)

    def _poll_inference(self, q: queue.Queue):
        # Прекращаем поллинг если виджет уже уничтожен
        try:
            self.winfo_exists()
        except Exception:
            return
        if not self.winfo_exists():
            return
        try:
            result = q.get_nowait()
            self._model_result = result
            self._apply_model_result(result)
        except queue.Empty:
            self.after(100, lambda: self._poll_inference(q))

    def _apply_model_result(self, r: dict):
        if r.get("error"):
            self._status_var.set(f"⚠ Ошибка модели: {r['error']}")
        else:
            self._status_var.set("✓ Модель отработала")

        serial = r.get("serial_text") or ""
        conf   = r.get("serial_conf")

        self._model_serial  = serial
        self._model_reading = r.get("reading_str") or ""

        self._filling_other = True
        self._serial_var.set(serial)
        self._filling_other = False

        self._serial_source  = f"Серийник: модель ({conf:.2f})" if conf else "Серийник: нет"
        self._reading_source = "Показания: модель"

        reading_str = r.get("reading_str")
        digit_preds = r.get("digit_preds")
        self._reading_widget.set_digits(reading_str, digit_preds)
        self._reading_manually_changed = False

        self._update_source_label()
        self._lookup_by_serial(serial)
        self._update_accept_state()

    def _update_source_label(self):
        parts = []
        if hasattr(self, "_serial_source"):
            parts.append(self._serial_source)
        if hasattr(self, "_reading_source"):
            parts.append(self._reading_source)
        self._source_var.set("  |  ".join(parts))

    def _lookup_by_serial(self, serial: str):
        self._abonent_found = False
        self._serial_match_var.set("")
        for w in self._hint_frame.winfo_children():   # подсказка — только для ненайденного номера
            w.destroy()

        if self.app.df is None or not serial:
            self._update_accept_state()
            return

        cfg = self.app.config
        for candidate in lookup_variants(serial):
            mask = self.app.df[cfg.col_serial].apply(
                lambda x: _normalize_serial(str(x)) == _normalize_serial(candidate)
            )
            matches = self.app.df[mask]
            if not matches.empty:
                choices = serial_choices(matches, cfg, self._current_reading_str())
                if len(choices) > 1:
                    self._serial_match_var.set(f"⚠ номер у нескольких абонентов ({len(choices)})")
                    self._show_choices(choices)
                    self._update_delta()
                    self._update_accept_state()
                    return
                row     = matches.iloc[0]
                account = str(row.get(cfg.col_account_id, ""))
                last    = row.get(cfg.col_last_reading)

                if candidate != serial:
                    self._model_serial = candidate
                    self._serial_source = "Серийник: модель (авто-нуль)"
                    self._update_source_label()
                    
                    self._filling_other = True
                    self._serial_var.set(candidate)
                    self._filling_other = False

                self._filling_other = True
                self._account_var.set(account)
                self._filling_other = False
                self._account_match_var.set("")

                self._last_reading_var.set(str(last) if pd.notna(last) else "—")
                self._abonent_found = True
                self._serial_match_var.set(f"✓ в базе → account: {account}")
                self._update_delta()
                self._update_accept_state()
                return

        self._serial_match_var.set("⚠ не найден в базе")
        self._show_hints(serial)
        self._update_delta()
        self._update_accept_state()

    def _current_reading_str(self) -> str:
        if self._reading_widget.is_complete():
            return self._reading_widget.get_string()
        return getattr(self, "_model_reading", "") or ""

    def _show_hints(self, serial: str):
        for w in self._hint_frame.winfo_children():
            w.destroy()
        self._hints = serial_hints(serial, self._current_reading_str(), self.app.serial_index())
        if not self._hints:
            return
        ttk.Label(self._hint_frame, text="Похожие номера в базе:",
                  font=("Segoe UI", 8, "bold")).pack(anchor="w")
        for i, h in enumerate(self._hints):
            last = f"{h.last_reading:.0f}" if h.last_reading is not None else "—"
            delta = f"{h.delta:+.0f}" if h.delta is not None else "—"
            tk.Button(
                self._hint_frame,
                text=f"{h.serial}   л/с {h.account}\nпосл. {last}   расход {delta}",
                font=("Segoe UI", 9), relief="groove", anchor="w", justify="left",
                command=lambda h=h: self._use_hint(h),
            ).pack(anchor="w", fill=tk.X, pady=1)
        ttk.Label(self._hint_frame,
                  text="Клик — подставить лицевой счёт. Дальше: «Принять», если это он "
                       "(модель ошиблась), или «Серийник в базе с ошибкой».",
                  font=("Segoe UI", 8), foreground=CLR_GRAY,
                  wraplength=280, justify="left").pack(anchor="w", pady=(0, 6))

    def _show_choices(self, choices: list[SerialHint]):
        """Номер в таблице у нескольких абонентов — кнопки выбора."""
        for w in self._hint_frame.winfo_children():
            w.destroy()
        self._hints = choices
        ttk.Label(self._hint_frame, text="Этот номер в базе у нескольких абонентов:",
                  font=("Segoe UI", 8, "bold")).pack(anchor="w")
        for h in choices:
            last = f"{h.last_reading:.0f}" if h.last_reading is not None else "—"
            delta = f"{h.delta:+.0f}" if h.delta is not None else "—"
            tk.Button(
                self._hint_frame,
                text=f"л/с {h.account}\nпосл. {last}   расход {delta}",
                font=("Segoe UI", 9), relief="groove", anchor="w", justify="left",
                command=lambda h=h: self._use_choice(h),
            ).pack(anchor="w", fill=tk.X, pady=1)
        ttk.Label(self._hint_frame,
                  text="Клик — выбрать абонента, потом «Принять». У другой записи в базе "
                       "номер с ошибкой.",
                  font=("Segoe UI", 8), foreground=CLR_GRAY,
                  wraplength=280, justify="left").pack(anchor="w", pady=(0, 6))

    def _use_choice(self, h: SerialHint):
        """Выбран абонент из нескольких с одним номером (как подсказка: лицевой
        счёт подставляется, остальное подтягивается из таблицы)."""
        self._choice_used = h
        self._account_var.set(h.account)
        self._lookup_by_account(h.account)

    def _use_hint(self, h: SerialHint):
        """Подставляет лицевой счёт кандидата (серийник подтянется из таблицы)."""
        self._hint_used = h
        self._account_var.set(h.account)
        self._lookup_by_account(h.account)

    def _hint_note(self) -> str:
        if getattr(self, "_choice_used", None):
            return f" | {CHOICE_NOTE}: л/с {self._choice_used.account}"
        return f" | {HINT_NOTE}: {self._hint_used.serial}" if self._hint_used else ""

    def _lookup_by_account(self, account: str):
        self._abonent_found = False
        self._account_match_var.set("")

        if self.app.df is None or not account:
            self._update_accept_state()
            return

        cfg = self.app.config
        mask = self.app.df[cfg.col_account_id].apply(
            lambda x: str(x).strip() == account.strip()
        )
        matches = self.app.df[mask]
        if not matches.empty:
            row    = matches.iloc[0]
            serial = str(row.get(cfg.col_serial, ""))
            last   = row.get(cfg.col_last_reading)

            self._filling_other = True
            self._serial_var.set(serial)
            self._filling_other = False
            self._serial_match_var.set("")

            self._last_reading_var.set(str(last) if pd.notna(last) else "—")
            self._abonent_found = True
            self._account_match_var.set(f"✓ в базе → serial: {serial}")
            self._update_delta()
            self._update_accept_state()
        else:
            self._account_match_var.set("⚠ не найден в базе")
            self._update_delta()
            self._update_accept_state()

    def _on_serial_change(self):
        if self._filling_other:
            return
        model_serial = getattr(self, "_model_serial", None)
        if model_serial is not None and self._serial_var.get() != model_serial:
            self._serial_source = "Серийник: вручную"
        elif model_serial is not None and self._serial_var.get() == model_serial:
            self._serial_source = "Серийник: модель"
        else:
            self._serial_source = "Серийник: вручную"
        self._serial_match_var.set("")
        self._update_source_label()
        self._abonent_found = False             # до поиска по новому номеру
        self._update_accept_state()
        if getattr(self, "_serial_lookup_job", None):
            self.after_cancel(self._serial_lookup_job)
        self._serial_lookup_job = self.after(
            350, lambda: self._lookup_by_serial(self._serial_var.get())
        )

    def _on_account_change(self):
        if self._filling_other:
            return
        self._account_match_var.set("")
        self._abonent_found = False             # до поиска по новому счёту
        self._update_accept_state()
        if getattr(self, "_account_lookup_job", None):
            self.after_cancel(self._account_lookup_job)
        self._account_lookup_job = self.after(
            350, lambda: self._lookup_by_account(self._account_var.get())
        )

    def _on_reading_change(self):
        current = self._reading_widget.get_string()
        model_r = getattr(self, "_model_reading", None)
        manually = getattr(self, "_reading_manually_changed", False)
        if not manually:
            if model_r is not None and current != model_r:
                self._reading_manually_changed = True
                self._reading_source = "Показания: вручную"
            elif model_r is not None and current == model_r:
                self._reading_manually_changed = False
                self._reading_source = "Показания: модель"
        self._update_source_label()
        self._update_delta()
        self._update_accept_state()

    def _update_delta(self):
        last_str = self._last_reading_var.get()
        if not self._reading_widget.is_complete():
            self._delta_var.set("—")
            self._outcome_var.set("—")
            self._delta_lbl.config(fg=CLR_GRAY)
            return
        try:
            last = float(last_str.replace(",", ".")) if last_str and last_str != "—" else None
        except ValueError:
            last = None

        reading = int(self._reading_widget.get_string())
        if last is not None:
            delta = reading - last
            cfg   = self.app.config
            if abs(delta) > cfg.delta_threshold:
                self._delta_var.set(f"Δ {delta:+.0f}")
                self._outcome_var.set("⚠ SUSPICIOUS")
                self._delta_lbl.config(fg=CLR_ORANGE)
            elif delta >= 0:
                self._delta_var.set(f"Δ {delta:+.0f}")
                self._outcome_var.set("▲ PLUS")
                self._delta_lbl.config(fg=CLR_GREEN)
            else:
                self._delta_var.set(f"Δ {delta:+.0f}")
                self._outcome_var.set("▼ MINUS")
                self._delta_lbl.config(fg=CLR_RED)
        else:
            self._delta_var.set("—")
            self._outcome_var.set("PLUS (нет истории)")
            self._delta_lbl.config(fg=CLR_GREEN)

    def _update_accept_state(self):
        # Решение владельца 2026-11-20: принять можно, только если абонент
        # найден в таблице по номеру или счёту. Иначе в поле счёта мог
        # остаться прежний абонент, и показание уходило ему.
        if not hasattr(self, "_accept_btn"):
            return
        ok = (
            self._reading_widget.is_complete()
            and (self._serial_var.get().strip() or self._account_var.get().strip())
            and (self.app.df is None or getattr(self, "_abonent_found", False))
        )
        self._accept_btn.config(state=tk.NORMAL if ok else tk.DISABLED)

    def _back(self):
        try:
            self._on_back()
        finally:
            self._unbind_keys()

    def _get_outcome_str(self) -> str:
        outcome = self._outcome_var.get()
        if "SUSPICIOUS" in outcome:
            return "SUSPICIOUS"
        if "PLUS" in outcome:
            return "PLUS"
        if "MINUS" in outcome:
            return "MINUS"
        return "PLUS"

    def _flush_lookups(self) -> bool:
        """Поиск по таблице, отложенный на 0,35 с после ввода номера или счёта,
        выполнить сейчас. True — поля абонента от этого изменились."""
        before = (self._serial_var.get(), self._account_var.get(), self._last_reading_var.get())
        for attr, lookup, var in (("_serial_lookup_job", self._lookup_by_serial, self._serial_var),
                                  ("_account_lookup_job", self._lookup_by_account, self._account_var)):
            job = getattr(self, attr, None)
            if not job:
                continue
            setattr(self, attr, None)
            try:
                self.tk.call("after", "info", job)      # ошибка — поиск уже выполнен
            except tk.TclError:
                continue
            self.after_cancel(job)
            lookup(var.get())
        return before != (self._serial_var.get(), self._account_var.get(), self._last_reading_var.get())

    def _accept(self):
        # Номер исправили и сразу «Принять»: поиск абонента ещё не прошёл, и
        # показание уходило прежнему счёту (замер 2026-11-20). Сначала поиск;
        # если абонент от этого сменился — показать, второй Enter примет.
        if self._flush_lookups():
            return
        if self._accept_btn["state"] == tk.DISABLED:
            return

        outcome_str = self._get_outcome_str()

        if outcome_str == "SUSPICIOUS" and not self._confirmed_suspicious:
            delta_text = self._delta_var.get()
            if not messagebox.askyesno(
                "Подозрительные показания",
                f"Дельта: {delta_text}\n\nЭто большое отклонение. Всё верно?",
                parent=self.app,
            ):
                return
            self._confirmed_suspicious = True
            try:
                delta_val = float(self._delta_var.get().replace("Δ", "").replace(",", ".").strip())
                outcome_str = "PLUS" if delta_val >= 0 else "MINUS"
            except Exception:
                outcome_str = "PLUS"

        serial    = self._serial_var.get().strip()
        account   = self._account_var.get().strip()
        reading_s = self._reading_widget.get_string()
        reading   = int(reading_s)
        if not self._confirm_replace(account, reading):
            return

        folder   = "minus" if outcome_str == "MINUS" else "plus"
        dst_dir  = str(self.folder / folder)
        ext      = Path(self.photo_path).suffix
        new_name = f"{account}{ext}" if account else Path(self.photo_path).name
        new_name = Path(free_photo_path(dst_dir, new_name, self.photo_path)).name   # не затирать другое фото

        last_str = self._last_reading_var.get()
        try:
            last = float(last_str.replace(",", ".")) if last_str and last_str != "—" else None
        except ValueError:
            last = None

        delta = (reading - last) if last is not None else None
        mr = self._model_result or {}

        row = {k: "" for k in _LOG_COLUMNS}
        row.update({
            "original_filename": Path(self.photo_path).name,
            "final_filename":    new_name,
            "serial_id":         serial,
            "account_id":        account,
            "reading":           str(reading),
            "last_reading":      str(last) if last is not None else "",
            "delta":             str(delta) if delta is not None else "",
            "outcome":           outcome_str,
            "source":            "manual",
            "processed_by":      self.app.settings.operator_name,
            "processed_at":      now_iso(),
            "model_serial_conf": f"{mr.get('serial_conf', ''):.4f}" if mr.get("serial_conf") else "",
            "model_reading_str": mr.get("reading_str") or "",
            "notes":             self.reason + self._hint_note(),
        })
        self.app.accept_reading(row, self.photo_path)    # показание и строка лога — одной записью

        self._save_markup_silent(serial, reading_s, mr, row)

        try:
            new_path = move_photo(self.photo_path, dst_dir, new_name)
            redraw_annotation(new_path, serial, reading_s)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось переместить фото:\n{e}")
            return

        try:
            self.app.open_next_or_back(self.photo_path)
        finally:
            self._unbind_keys()

    def _confirm_replace(self, account: str, reading: int) -> bool:
        """Этап 3, пункт 5а (решение владельца 2026-10-03): у счёта уже есть
        показание — показать, чьё и когда, и спросить «Заменить?»."""
        old = self.app.existing_reading(account)
        if old is None:
            return True
        return messagebox.askyesno(
            "У счёта уже есть показание",
            f"У л/с {account} уже есть показание {old.value}\n"
            f"({describe_reading(old)}).\n\n"
            f"Заменить на {reading}?\n\n"
            f"«Нет» — ничего не менять: фото останется в очереди "
            f"(если это повтор того же счётчика — кнопка «Дубль»).",
            parent=self.app,
        )

    def _save_markup_silent(self, final_serial: str, final_reading_str: str, mr: dict, row: dict):
        """Исправления оператора — в <месяц>/разметка. Фото эталона не
        сохраняются: эталон и обучение не пересекаются (to_etalon)."""
        h = row.get("photo_hash") or ""
        if h and to_etalon(h):
            return
        photo = row.get("account_id") or h or Path(self.photo_path).stem    # фото кропов — лицевой счёт
        try:
            root = self.app.session.folder.markup
            save_crnn_markup(root, photo, mr.get("serial_crop"), final_serial, mr.get("serial_text"))
            save_cnn_markup(root, photo, mr.get("digit_crops"), mr.get("digit_preds"),
                            mr.get("reading_str"), final_reading_str)
            if self.reason == "no_meter":
                save_meter_markup(root, photo, self._original_photo(row))
        except Exception as e:
            log.warning(f"Ошибка сохранения разметки: {e}")

    def _original_photo(self, row: dict) -> Optional[Path]:
        """Исходное фото в <месяц>/фото/<контролёр>/ (файл в question/ — с подписью)."""
        src = find_auto_row_for_output_file(
            self.app.log_rows, row.get("original_filename", ""),
            controller_name(self.photo_path, self.app.results_dir))
        name = (src or row).get("original_filename", "")
        p = Path(self.app.session.folder.photos) / (row.get("source_folder") or "") / name
        return p if name and p.is_file() else None

    def _duplicate(self):
        if not messagebox.askyesno("Дубль", "Пометить как дубль?", parent=self.app):
            return
        dst_dir = str(self.folder / "repeat")
        row = {k: "" for k in _LOG_COLUMNS}
        row.update({
            "original_filename": Path(self.photo_path).name,
            "outcome":            "REPEAT",
            "source":            "manual",
            "processed_by":      self.app.settings.operator_name,
            "processed_at":      now_iso(),
        })
        name = place_decision(row, self.photo_path, dst_dir)
        self.app.append_log(row, self.photo_path)
        try:
            move_photo(self.photo_path, dst_dir, name)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return
        try:
            self.app.open_next_or_back(self.photo_path)
        finally:
            self._unbind_keys()

    def _unreadable(self):
        if not messagebox.askyesno("Нечитаемо", "Пометить как нечитаемо?", parent=self.app):
            return
        dst_dir = str(self.folder / "unreadable")
        row = {k: "" for k in _LOG_COLUMNS}
        row.update({
            "original_filename": Path(self.photo_path).name,
            "outcome":            "UNREADABLE",
            "source":            "manual",
            "processed_by":      self.app.settings.operator_name,
            "processed_at":      now_iso(),
        })
        name = place_decision(row, self.photo_path, dst_dir)
        self.app.append_log(row, self.photo_path)
        try:
            move_photo(self.photo_path, dst_dir, name)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return
        try:
            self.app.open_next_or_back(self.photo_path)
        finally:
            self._unbind_keys()

    def _run_serial(self) -> str:
        """Номер, который прочитал прогон с исходного фото (авто-строка лога,
        создавшая этот файл в question/)."""
        src = find_auto_row_for_output_file(
            self.app.log_rows, Path(self.photo_path).name,
            controller_name(self.photo_path, self.app.results_dir))
        return ((src or {}).get("serial_id") or "").strip()

    def _ask_serial_not_in_table(self, prompt: str, suggested: str) -> Optional[str]:
        return simpledialog.askstring("Нет в базе", prompt, initialvalue=suggested, parent=self.app)

    def _not_in_db(self):
        """
        «Нет в базе»: счётчика нет в таблице компании. По номеру в строке лога
        прогон решает, читать ли фото заново: появился номер в таблице — читает.
        Решение владельца 2026-11-21 (decisions.not_in_db_serial):
          - оператор номер не менял — пишется номер, прочитанный прогоном;
          - окно нашло абонента или номер есть в таблице — окно спрашивает
            номер с фото: с номером из таблицы следующий прогон сразу вернул
            бы фото в очередь;
          - лицевой счёт из таблицы не пишется: это другой абонент (в поле
            мог остаться счёт, найденный по прежнему номеру).
        """
        self._flush_lookups()
        cfg, df = self.app.config, self.app.df
        field   = self._serial_var.get().strip()
        account = self._account_var.get().strip()
        run_serial = self._run_serial()
        serial = not_in_db_serial(field, self._model_serial, run_serial)
        in_table = accounts_with_serial(df, cfg, serial)

        if self._abonent_found or in_table:
            if self._abonent_found:
                what = f"В таблице найден абонент: л/с {account}" + (f", номер {field}" if field else "") + "."
            else:
                what = f"Номер {serial} есть в таблице: л/с {', '.join(in_table)}."
            suggested = next((s for s in (run_serial, field)
                              if s and not accounts_with_serial(df, cfg, s)), "")
            typed = self._ask_serial_not_in_table(
                f"{what}\n\n"
                f"Если на фото этот счётчик — Cancel, затем «Принять».\n"
                f"Если на фото другой номер — впишите его, как на фото:",
                suggested)
            serial = (typed or "").strip()
            if not serial:
                return
            again = accounts_with_serial(df, cfg, serial)
            if again:
                messagebox.showwarning(
                    "Номер есть в таблице",
                    f"Номер {serial} есть в таблице: л/с {', '.join(again)}.\n\n"
                    f"Если на фото этот счётчик — впишите номер в поле "
                    f"«Серийный номер» и нажмите «Принять».",
                    parent=self.app)
                return
            account = ""
        else:
            if table_serial_for_account(df, cfg, account) is not None:
                account = ""            # счёт из таблицы, найденный по прежнему номеру
            by_run = " (прочитан при обработке фото)" if serial and serial != field else ""
            if not messagebox.askyesno(
                "Нет в базе",
                f"Счётчик не найден в базе абонентов.\n\n"
                f"Серийник: {serial or '—'}{by_run}\n"
                f"Account:  {account or '—'}\n\n"
                f"Переместить в папку «not_in_db»?",
                parent=self.app,
            ):
                return

        dst_dir = str(self.folder / "not_in_db")
        mr = self._model_result or {}
        row = {k: "" for k in _LOG_COLUMNS}
        row.update({
            "original_filename": Path(self.photo_path).name,
            "serial_id":         serial,
            "account_id":        account,
            "outcome":            "NOT_IN_DB",
            "source":            "manual",
            "processed_by":      self.app.settings.operator_name,
            "processed_at":      now_iso(),
            "model_serial_conf": f"{mr.get('serial_conf', ''):.4f}" if mr.get("serial_conf") else "",
            "model_reading_str": mr.get("reading_str") or "",
            "notes":             "счётчик не найден в базе",
        })
        name = place_decision(row, self.photo_path, dst_dir)
        self.app.append_log(row, self.photo_path)
        try:
            move_photo(self.photo_path, dst_dir, name)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return
        try:
            self.app.open_next_or_back(self.photo_path)
        finally:
            self._unbind_keys()

    def _ask_photo_serial(self, suggested: str) -> Optional[str]:
        return simpledialog.askstring(
            "Серийник на фото",
            "Серийный номер — как на фото (не как в базе):",
            initialvalue=suggested, parent=self.app,
        )

    def _db_serial_fix(self):
        """
        «Серийник в базе с ошибкой»: счётчик найден по лицевому счёту, номер на
        фото верный, в базе — с ошибкой. С 2026-10-01 (решение владельца) номер
        в базе месяца исправляется на номер с фото и показание записывается
        сразу; номер попадает в список для компании (db_serial_fix.csv в папке
        месяца).
        """
        cfg = self.app.config
        account = self._account_var.get().strip()
        db_serial = table_serial_for_account(self.app.df, cfg, account)
        if db_serial is None:
            messagebox.showwarning(
                "Нужен лицевой счёт",
                "Укажите лицевой счёт абонента (поле Account ID) — он должен быть в таблице.",
                parent=self.app)
            return
        if not self._reading_widget.is_complete():
            messagebox.showwarning(
                "Нужно показание",
                "Введите все 5 цифр показания — оно запишется вместе с исправлением номера.",
                parent=self.app)
            return
        mr = self._model_result or {}
        photo_serial = self._ask_photo_serial(mr.get("serial_text") or "")
        if photo_serial is None:
            return
        photo_serial = photo_serial.strip()
        if not photo_serial:
            return
        if _normalize_serial(photo_serial) == db_serial:
            messagebox.showwarning(
                "Серийник совпадает с базой",
                "Номер на фото совпадает с номером в базе — используйте «Принять».",
                parent=self.app)
            return
        reading_s = self._reading_widget.get_string()
        reading = int(reading_s)
        if not self._confirm_replace(account, reading):
            return
        if not messagebox.askyesno(
            "Серийник в базе с ошибкой",
            f"Лицевой счёт: {account}\n"
            f"Номер в базе: {db_serial} → будет исправлен на {photo_serial}\n"
            f"Показание: {reading} — запишется сразу.\n\n"
            f"Номер попадёт в список для исправления базы компании "
            f"({DB_SERIAL_FIX_LIST}). Продолжить?",
            parent=self.app,
        ):
            return

        last = _to_float(self._last_reading_var.get())
        delta = (reading - last) if last is not None else None
        outcome = "MINUS" if delta is not None and delta < 0 else "PLUS"
        ext = Path(self.photo_path).suffix
        dst_dir = str(self.folder / outcome.lower())
        dst = free_photo_path(dst_dir, f"{account}{ext}", self.photo_path)
        row = {k: "" for k in _LOG_COLUMNS}
        row.update({
            "original_filename": Path(self.photo_path).name,
            "final_filename":    Path(dst).name,
            "serial_id":         photo_serial,
            "account_id":        account,
            "reading":           str(reading),
            "last_reading":      str(last) if last is not None else "",
            "delta":             str(delta) if delta is not None else "",
            "outcome":           outcome,
            "source":            "manual",
            "processed_by":      self.app.settings.operator_name,
            "processed_at":      now_iso(),
            "model_serial_conf": f"{mr.get('serial_conf', ''):.4f}" if mr.get("serial_conf") else "",
            "model_reading_str": mr.get("reading_str") or "",
            "notes":             f"серийник в базе с ошибкой: в базе {db_serial}, на фото {photo_serial}"
                                 + self._hint_note(),
        })
        self.app.fix_serial_and_accept(row, self.photo_path, photo_serial)
        append_db_serial_fix(str(Path(self.app.session.folder.root) / DB_SERIAL_FIX_LIST), {
            "Фото":             Path(dst).name,
            "Контролёр":        controller_name(self.photo_path, self.app.results_dir),
            "Лицевой счёт":     account,
            "Серийник в базе":  db_serial,
            "Серийник на фото": photo_serial,
            "Показание":        reading_s,
            "Дата":             now_iso()[:10],
            "Оператор":         self.app.settings.operator_name,
        })
        # номер на фото верный — годится в разметку для дообучения CRNN
        self._save_markup_silent(photo_serial, reading_s, mr, row)
        try:
            new_path = move_photo(self.photo_path, dst_dir, Path(dst).name)
            redraw_annotation(new_path, photo_serial, reading_s)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось переместить фото:\n{e}", parent=self.app)
            return
        try:
            self.app.open_next_or_back(self.photo_path)
        finally:
            self._unbind_keys()
