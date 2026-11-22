"""Экран проверки одного фото из plus/ и minus/: Верно, Исправить, Пропустить."""
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Optional


from .decisions import plan_verify_correction
from .files import free_photo_path, move_photo, now_iso, redraw_annotation
from .photos import controller_dir
from .theme import (
    action_button,
    card,
    FONT,
    key_sequences,
    MUTED,
    PHOTO_BG,
    px,
    SUCCESS,
    SURFACE,
    TEXT,
)
from .widgets import PHOTO_HINT, PhotoView, ReadingWidget, photo_keys

OUTCOME_WORDS = {"PLUS": "▲ больше прошлого", "MINUS": "▼ меньше прошлого"}

if TYPE_CHECKING:                      # только для подсказки типов: окно импортирует экраны
    from .window import MainWindow


class VerifyScreen(ttk.Frame):
    """Проверка фото из plus/ minus/ которые обработал reader.py автоматически."""
    def __init__(self, parent: "MainWindow", item: dict, on_back, index: int = 0, total: Optional[int] = None):
        super().__init__(parent)
        self.app      = parent
        self.item     = item
        self.index    = index        # номер фото в списке «Проверки» — для перехода к следующему
        self.total    = total        # сколько фото сейчас на проверке (после «Верно» — на одно меньше)
        self._on_back = on_back
        self._edit_mode = False

        self._build()
        self._bind_keys()

    def _build(self):
        r = self.item["log_row"]

        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=px(16), pady=(px(12), px(8)))
        ttk.Button(top, text="←  К списку   Esc", command=self._back).pack(side=tk.LEFT)
        tk.Label(top, text="Проверка", bg="#e3edfd", fg="#1858bd", font=(FONT, 10, "bold"),
                 padx=px(10), pady=px(3)).pack(side=tk.LEFT, padx=(px(16), px(10)))
        if self.total is not None:           # сколько осталось (просьба владельца 2026-11-21)
            tk.Label(top, text=f"На проверке: {self.total}", bg="#e6f4ea", fg="#1e7b34",
                     font=(FONT, 10, "bold"), padx=px(10), pady=px(3)).pack(side=tk.LEFT, padx=(0, px(10)))
        where = f"Фото {self.index + 1} из {self.total}" if self.total else f"Фото {self.index + 1}"
        ttk.Label(top, text=f"{where}   ·   {Path(self.item['path']).name}",
                  style="Muted.TLabel").pack(side=tk.LEFT)
        ttk.Label(top, text=PHOTO_HINT, style="Muted.TLabel", font=(FONT, 9)).pack(side=tk.RIGHT)

        center = ttk.Frame(self)
        center.pack(fill=tk.BOTH, expand=True, padx=px(16), pady=px(4))

        self._canvas = PhotoView(center, self.item["path"], bg=PHOTO_BG, width=px(560))
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = card(center)
        right.configure(width=px(380))
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(px(16), 0))
        right.pack_propagate(False)
        self._build_right(right, r)

        self._build_buttons()

    def _build_right(self, parent, r: dict):
        ttk.Label(parent, text="Программа записала сама — верно?", style="CardMuted.TLabel").pack(
            anchor="w", pady=(0, px(12)))

        def row(label, value, big=False, color=None):
            ttk.Label(parent, text=label, style="CardTitle.TLabel").pack(anchor="w")
            lbl = tk.Label(parent, text=value, font=(FONT, 18 if big else 13, "bold" if big else "normal"),
                           fg=color or TEXT, bg=SURFACE, anchor="w")
            lbl.pack(anchor="w", pady=(0, px(10)))
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
        outcome = r.get("outcome", "")      # стрелка: зелёная — больше прошлого, красная — меньше
        tk.Label(parent, text=OUTCOME_WORDS.get(outcome, outcome), font=(FONT, 10, "bold"), bg=SURFACE,
                 fg={"PLUS": SUCCESS, "MINUS": "#c62828"}.get(outcome, MUTED)).pack(anchor="w")
        ttk.Label(parent, text=f"прочитано {r.get('processed_at', '')[:16].replace('T', ' ')}",
                  style="CardMuted.TLabel").pack(anchor="w")

        self._edit_frame = ttk.LabelFrame(parent, text="ИСПРАВЛЕНИЕ", padding=px(10))
        ttk.Label(self._edit_frame, text="Номер счётчика", style="Card.TLabel").pack(anchor="w")
        self._serial_var = tk.StringVar(value=r.get("serial_id", ""))
        ttk.Entry(self._edit_frame, textvariable=self._serial_var, font=(FONT, 13)).pack(fill=tk.X)
        ttk.Label(self._edit_frame, text="Показание", style="Card.TLabel").pack(anchor="w", pady=(px(8), 0))
        self._reading_widget = ReadingWidget(self._edit_frame, style="Card.TFrame")
        self._reading_widget.pack(anchor="w")
        self._reading_widget.set_digits(r.get("reading", ""), None)

    def _build_buttons(self):
        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=px(16), pady=(px(8), px(14)))

        self._ok_btn = action_button(bottom, "✓   Верно     Enter", self._verify_ok, kind="primary")
        self._ok_btn.pack(side=tk.LEFT, padx=(0, px(12)))
        self._edit_btn = action_button(bottom, "Исправить     E", self._enter_edit)
        self._edit_btn.pack(side=tk.LEFT, padx=px(4))
        self._save_edit_btn = action_button(bottom, "✓   Сохранить исправление     Enter", self._save_edit,
                                            kind="accent")
        action_button(bottom, "Пропустить     →", self._skip).pack(side=tk.RIGHT)
        if self.index > 0:
            action_button(bottom, "←     Предыдущее", self._previous).pack(side=tk.RIGHT, padx=(0, px(8)))

    def _key_bindings(self) -> list:
        edit = lambda e: None if self._edit_mode else self._enter_edit()   # noqa: E731
        return ([("<Return>", lambda e: self._on_return() if self.app.take_return(e) else None),
                 ("<Escape>", lambda e: self._exit_edit() if self._edit_mode else self._back()),
                 ("<Right>", lambda e: None if self._edit_mode else self._skip()),
                 ("<Left>", lambda e: None if self._edit_mode else self._previous())]
                + [(seq, edit) for seq in key_sequences("e")]
                + photo_keys(self._canvas, self.app))

    def _bind_keys(self):
        # Enter: «Верно», а в режиме исправления — «Сохранить исправление»
        for seq, cb in self._key_bindings():
            self.app.bind_all(seq, cb)
        self.app._keys_owner = self

    def _unbind_keys(self):
        # Клавиши общие для окна (bind_all): снимает их только экран, который
        # их поставил, — иначе закрывающийся экран снимал клавиши следующего
        # (2026-11-21: со второго фото очереди Enter, D, N, B, Del не работали)
        if getattr(self.app, "_keys_owner", None) is not self:
            return
        self.app._keys_owner = None
        for seq, _ in self._key_bindings():
            try:
                self.app.unbind_all(seq)
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

    def _previous(self):
        """← — предыдущее фото списка (вернуться к пропущенному)."""
        if self.index > 0:
            self._unbind_keys()
            self.app.open_verify_at(self.index - 1)

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
        self._edit_frame.pack(fill=tk.X, pady=(px(12), 0))
        # «Верно» в режиме исправления прячем: её нажатие выбрасывало правку
        self._ok_btn.pack_forget()
        self._edit_btn.pack_forget()
        self._save_edit_btn.pack(side=tk.LEFT, padx=px(4))

    def _exit_edit(self):
        """Esc в режиме исправления — отменить исправление (поля — как прочитано)."""
        r = self.item["log_row"]
        self._edit_mode = False
        self._edit_frame.pack_forget()
        self._save_edit_btn.pack_forget()
        self._ok_btn.pack(side=tk.LEFT, padx=(0, px(12)))
        self._edit_btn.pack(side=tk.LEFT, padx=px(4))
        self._serial_var.set(r.get("serial_id", ""))
        self._reading_widget.set_digits(r.get("reading", ""), None)
        self._canvas.focus_set()

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
