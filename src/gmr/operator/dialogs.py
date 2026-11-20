"""Диалоги: первый запуск и смена пользователя, вход, настройки распознавания месяца."""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from src.gmr.domain.preset import (
    FORGIVE_MODES,
    MISSING_MODES,
    SERIAL_PADS,
    RecognitionPreset,
)

from .settings import AppSettings


class SettingsDialog(tk.Toplevel):
    """Первый запуск или смена пользователя: имя и папка месяца. «Новый…» —
    создать месяц из таблицы компании (new_month(диалог, имя) → папка | None)."""
    def __init__(self, parent, settings: AppSettings, new_month=None):
        super().__init__(parent)
        self.title("Настройки")
        self.resizable(False, False)
        self.grab_set()
        self.result: Optional[AppSettings] = None
        self._new_month = new_month
        self._build(settings)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _build(self, s: AppSettings):
        pad = dict(padx=12, pady=6)
        f = ttk.Frame(self, padding=16)
        f.pack(fill=tk.BOTH, expand=True) 

        ttk.Label(f, text="Настройка программы", font=("Segoe UI", 13, "bold")).grid(
            row=0, column=0, columnspan=3, pady=(0, 16), sticky="w"
        )

        self._vars = {"name": tk.StringVar(value=s.operator_name), "month": tk.StringVar(value=s.month_dir)}
        ttk.Label(f, text="Ваше имя:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(f, textvariable=self._vars["name"], width=42).grid(row=1, column=1, columnspan=2, **pad)
        ttk.Label(f, text="Папка месяца:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(f, textvariable=self._vars["month"], width=42).grid(row=2, column=1, columnspan=2, **pad)
        ttk.Button(f, text="Открыть…", command=self._browse).grid(row=3, column=1, sticky="w", padx=12)
        if self._new_month is not None:
            ttk.Button(f, text="Новый месяц…", command=self._create).grid(row=3, column=2, sticky="e", padx=12)

        btns = ttk.Frame(f)
        btns.grid(row=10, column=0, columnspan=3, pady=(16, 0))
        ttk.Button(btns, text="Сохранить", command=self._save).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Отмена",    command=self._cancel).pack(side=tk.LEFT, padx=6)

    def _browse(self):
        p = filedialog.askdirectory(parent=self, title="Папка месяца")
        if p:
            self._vars["month"].set(p)

    def _create(self):
        folder = self._new_month(self, self._vars["name"].get().strip())
        if folder:
            self._vars["month"].set(folder)

    def _save(self):
        name = self._vars["name"].get().strip()
        if not name:
            messagebox.showwarning("Ошибка", "Введите имя оператора", parent=self)
            return
        month = self._vars["month"].get().strip()
        if not month:
            messagebox.showwarning("Папка месяца", "Откройте папку месяца или создайте новый месяц "
                                   "из таблицы компании («Новый месяц…»).", parent=self)
            return
        self.result = AppSettings(operator_name=name, month_dir=month)
        self.destroy()

    def _cancel(self):
        self.destroy()

class PresetDialog(tk.Toplevel):
    """Настройки распознавания месяца (2026-10-11, решение 2а): при создании
    месяца — заполнены как в прошлом, потом — «Месяц» → «Настройки
    распознавания…». result — RecognitionPreset или None («Отмена»)."""
    def __init__(self, parent, preset: RecognitionPreset, title: str = "Настройки распознавания",
                 ok_text: str = "Сохранить", marker_model: bool = False):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.result: Optional[RecognitionPreset] = None
        self._marker_model = marker_model
        self._vars = {
            "missing_digit": tk.StringVar(value=preset.missing_digit),
            "forgive_last": tk.IntVar(value=preset.forgive_last),
            "forgive_with": tk.StringVar(value=preset.forgive_with),
            "serial_pad": tk.DoubleVar(value=preset.serial_pad),
            "account_marker": tk.BooleanVar(value=preset.account_marker and marker_model),
            "drum_rule": tk.BooleanVar(value=preset.drum_rule),
            "first_from_last": tk.BooleanVar(value=preset.first_from_last),
            "serial_by_table": tk.BooleanVar(value=preset.serial_by_table),
            "turn_photo": tk.BooleanVar(value=preset.turn_photo),
            "overlay_notes": tk.BooleanVar(value=preset.overlay_notes),
        }
        f = ttk.Frame(self, padding=16)
        f.pack(fill=tk.BOTH, expand=True)

        def group(text, key, options):
            ttk.Label(f, text=text, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 2))
            for value, label in options:
                ttk.Radiobutton(f, text=label, value=value, variable=self._vars[key]).pack(anchor="w", padx=12)

        group("Детектор не нашёл одну цифру:", "missing_digit", MISSING_MODES.items())
        group("Неуверенные последние цифры:", "forgive_last",
              [(0, "не прощать — к оператору"), (1, "прощать последнюю"), (2, "прощать две последние")])
        group("Прощённую цифру заменять:", "forgive_with", FORGIVE_MODES.items())
        group("Запас вокруг рамки серийника:", "serial_pad", [(p, f"{int(p * 100)}%") for p in SERIAL_PADS])
        ttk.Label(f, text="Лицевой счёт:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 2))
        ttk.Checkbutton(f, text="искать по надписи маркером" + ("" if marker_model else
                                                                " (модель надписи ещё не подключена)"),
                        variable=self._vars["account_marker"],
                        state=tk.NORMAL if marker_model else tk.DISABLED).pack(anchor="w", padx=12)
        ttk.Checkbutton(f, text="искать по надписи поверх фото (счёт — если табличка не прочиталась; "
                                "показание надписи главнее барабана)",
                        variable=self._vars["overlay_notes"]).pack(anchor="w", padx=12)
        ttk.Label(f, text="Дополнительно:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 2))
        for key, label in (("drum_rule", "правило барабана: цифра между двумя — меньшая"),
                           ("first_from_last", "старшая цифра по прошлому показанию"),
                           ("serial_by_table", "серийник по таблице (ближайший номер таблицы)"),
                           ("turn_photo", "на фото ничего не найдено — повернуть (90°, 180°, 270°)")):
            ttk.Checkbutton(f, text=label, variable=self._vars[key]).pack(anchor="w", padx=12)
        ttk.Label(f, text="Как было до настроек: «5», две последние — «0», запас 0%, без надписи.\n"
                          "Действует со следующей обработки.", foreground="#666").pack(anchor="w", pady=(12, 0))
        btns = ttk.Frame(f)
        btns.pack(pady=(12, 0))
        ttk.Button(btns, text=ok_text, command=self._save).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Отмена", command=self.destroy).pack(side=tk.LEFT, padx=6)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.wait_window()

    def _save(self):
        v = self._vars
        self.result = RecognitionPreset(
            missing_digit=v["missing_digit"].get(), forgive_last=int(v["forgive_last"].get()),
            forgive_with=v["forgive_with"].get(), serial_pad=float(v["serial_pad"].get()),
            account_marker=bool(v["account_marker"].get()) and self._marker_model,
            drum_rule=bool(v["drum_rule"].get()), first_from_last=bool(v["first_from_last"].get()),
            serial_by_table=bool(v["serial_by_table"].get()), turn_photo=bool(v["turn_photo"].get()),
            overlay_notes=bool(v["overlay_notes"].get()))
        self.destroy()


class LoginDialog(tk.Toplevel):
    """Подтверждение входа при каждом запуске."""
    def __init__(self, parent, settings: AppSettings):
        super().__init__(parent)
        self.title("Вход")
        self.resizable(False, False)
        self.grab_set()
        self.action = None  # "continue" | "change"
        self._build(settings)
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        self.wait_window()

    def _build(self, s: AppSettings):
        f = ttk.Frame(self, padding=24)
        f.pack()
        ttk.Label(f, text="Вы вошли как:", font=("Segoe UI", 10)).pack()
        ttk.Label(f, text=s.operator_name, font=("Segoe UI", 14, "bold")).pack(pady=(4, 20))
        ttk.Button(f, text="Продолжить", command=self._continue).pack(fill=tk.X, pady=4)
        ttk.Button(f, text="Сменить пользователя", command=self._change).pack(fill=tk.X)

    def _continue(self):
        self.action = "continue"
        self.destroy()

    def _change(self):
        self.action = "change"
        self.destroy()
