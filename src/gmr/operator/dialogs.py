"""Диалоги: первый запуск и смена пользователя, вход, настройки распознавания месяца."""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from src.gmr.domain.preset import (
    CONF_FIELDS,
    CONF_RANGE,
    FORGIVE_MODES,
    MISSING_MODES,
    SERIAL_PADS,
    RecognitionPreset,
)

from src.gmr.ui.run_dialog import center_on

from .settings import AppSettings
from .theme import BG, FONT, action_button


class SettingsDialog(tk.Toplevel):
    """Первый запуск или смена пользователя: имя и папка месяца. «Новый…» —
    создать месяц из таблицы компании (new_month(диалог, имя) → папка | None)."""
    def __init__(self, parent, settings: AppSettings, new_month=None):
        super().__init__(parent)
        self.title("Настройки")
        self.resizable(False, False)
        self.transient(parent)           # поверх главного окна, а не под ним
        self.grab_set()
        self.result: Optional[AppSettings] = None
        self._new_month = new_month
        self._build(settings)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        center_on(self, self.master)
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
    распознавания…». Две вкладки (2026-11-21): «Обычные» — понятные любому
    оператору, «Расширенные» — для владельца программы (цифры, модели и их
    уверенность, номер счётчика). result — RecognitionPreset или None («Отмена»)."""

    MODELS = (   # (название, ключ уверенности, ключ «включена» или None — включена всегда)
        ("Детектор счётчика и номера", "meter_conf", None),
        ("Детектор цифр", "digit_detect_conf", None),
        ("Цифра барабана", "digit_conf", None),
        ("Номер счётчика", "serial_conf", None),
        ("Надпись маркером (лицевой счёт)", "account_conf", "account_marker"),
        ("Надписи поверх фото (не модель — разбор текста)", None, "overlay_notes"),
    )

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
            "missing_digit": tk.StringVar(), "forgive_last": tk.IntVar(), "forgive_with": tk.StringVar(),
            "serial_pad": tk.DoubleVar(), "account_marker": tk.BooleanVar(), "drum_rule": tk.BooleanVar(),
            "first_from_last": tk.BooleanVar(), "serial_by_table": tk.BooleanVar(),
            "turn_photo": tk.BooleanVar(), "overlay_notes": tk.BooleanVar(),
            **{key: tk.StringVar() for key, _, _ in CONF_FIELDS},
        }
        self._fill(preset)

        outer = ttk.Frame(self, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)
        self._tabs = ttk.Notebook(outer)
        self._tabs.pack(fill=tk.BOTH, expand=True)
        simple, advanced = ttk.Frame(self._tabs, padding=14), ttk.Frame(self._tabs, padding=14)
        self._tabs.add(simple, text="Обычные")
        self._tabs.add(advanced, text="Расширенные")
        self._build_simple(simple)
        self._build_advanced(advanced)

        ttk.Label(outer, text="Действует со следующей обработки фото.", foreground="#666").pack(
            anchor="w", pady=(10, 0))
        btns = ttk.Frame(outer)
        btns.pack(pady=(10, 0))
        ttk.Button(btns, text=ok_text, command=self._save).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Отмена", command=self.destroy).pack(side=tk.LEFT, padx=6)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        center_on(self, self.master)
        self.wait_window()

    def _fill(self, p: RecognitionPreset):
        v = self._vars
        for key in ("missing_digit", "forgive_last", "forgive_with", "serial_pad", "drum_rule",
                    "first_from_last", "serial_by_table", "turn_photo", "overlay_notes"):
            v[key].set(getattr(p, key))
        v["account_marker"].set(p.account_marker and self._marker_model)
        for key, _, _ in CONF_FIELDS:
            v[key].set(f"{getattr(p, key):.2f}")

    @staticmethod
    def _title(parent, text, pady=(12, 4)):
        ttk.Label(parent, text=text, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=pady)

    def _build_simple(self, f):
        ttk.Label(f, text="Обычно здесь ничего менять не нужно.", foreground="#666").pack(anchor="w")
        self._title(f, "Что программа читает сама, кроме показания и номера счётчика:")
        ttk.Checkbutton(f, text="лицевой счёт, написанный маркером на счётчике" + (
            "" if self._marker_model else "  (модель надписи не подключена)"),
            variable=self._vars["account_marker"],
            state=tk.NORMAL if self._marker_model else tk.DISABLED).pack(anchor="w", padx=12, pady=2)
        ttk.Checkbutton(f, text="надписи контролёра поверх фото («лс 12345», «пок 6789»)",
                        variable=self._vars["overlay_notes"]).pack(anchor="w", padx=12, pady=2)
        ttk.Checkbutton(f, text="повернуть фото, если на нём ничего не найдено",
                        variable=self._vars["turn_photo"]).pack(anchor="w", padx=12, pady=2)
        ttk.Button(f, text="Вернуть стандартные настройки",
                   command=lambda: self._fill(RecognitionPreset.new_month())).pack(anchor="w", pady=(18, 0))
        ttk.Label(f, text="Остальное — во вкладке «Расширенные» (для владельца программы).",
                  foreground="#666").pack(anchor="w", pady=(10, 0))

    def _build_advanced(self, f):
        tk.Label(f, text="Для владельца программы: эти настройки меняют, как программа читает показания.\n"
                         "Не уверены — не меняйте («Обычные» → «Вернуть стандартные настройки»).",
                 fg="#b45309", justify="left", font=("Segoe UI", 9)).pack(anchor="w")

        def radios(parent, text, key, options):
            self._title(parent, text, pady=(8, 2))
            row = ttk.Frame(parent)
            row.pack(anchor="w", padx=12)
            for value, label in options:
                ttk.Radiobutton(row, text=label, value=value, variable=self._vars[key]).pack(
                    side=tk.LEFT, padx=(0, 12))

        radios(f, "Детектор не нашёл одну цифру — на её место:", "missing_digit", MISSING_MODES.items())
        radios(f, "Неуверенные последние цифры показания:", "forgive_last",
               [(0, "не прощать — оператору"), (1, "прощать последнюю"), (2, "прощать две последние")])
        radios(f, "Прощённую цифру заменять:", "forgive_with", FORGIVE_MODES.items())

        self._title(f, "Модели: включена и уверенность (0,05–0,99)", pady=(14, 2))
        grid = ttk.Frame(f)
        grid.pack(anchor="w", padx=12)
        for col, head in enumerate(("", "включена", "уверенность")):
            ttk.Label(grid, text=head, foreground="#666").grid(row=0, column=col, sticky="w", padx=(0, 14))
        self._conf_boxes = {}
        for r, (name, conf_key, on_key) in enumerate(self.MODELS, start=1):
            ttk.Label(grid, text=name).grid(row=r, column=0, sticky="w", padx=(0, 14), pady=1)
            if on_key is None:                    # без неё программа не читает — включена всегда
                cb = ttk.Checkbutton(grid)
                cb.state(["selected", "disabled"])
            else:
                enabled = on_key != "account_marker" or self._marker_model
                cb = ttk.Checkbutton(grid, variable=self._vars[on_key],
                                     state=tk.NORMAL if enabled else tk.DISABLED)
            cb.grid(row=r, column=1, sticky="w")
            if conf_key is not None:
                box = ttk.Spinbox(grid, from_=CONF_RANGE[0], to=CONF_RANGE[1], increment=0.05, width=6,
                                  textvariable=self._vars[conf_key], format="%.2f")
                box.grid(row=r, column=2, sticky="w")
                self._conf_boxes[conf_key] = box
        ttk.Label(f, text="Первые четыре не выключаются: без них программа не читает показание.\n"
                          "Уверенность ниже — программа сама принимает больше фото, но и ошибается чаще.",
                  foreground="#666", justify="left").pack(anchor="w", pady=(4, 0))

        radios(f, "Запас вокруг рамки номера счётчика:", "serial_pad",
               [(p, f"{int(p * 100)}%") for p in SERIAL_PADS])
        self._title(f, "Опыты (обычно выключены):", pady=(10, 2))
        for key, label in (("serial_by_table", "номер счётчика по таблице (ближайший номер таблицы)"),
                           ("drum_rule", "правило барабана: цифра между двумя — меньшая"),
                           ("first_from_last", "старшая цифра по прошлому показанию")):
            ttk.Checkbutton(f, text=label, variable=self._vars[key]).pack(anchor="w", padx=12)

    def _conf_values(self) -> Optional[dict]:
        out = {}
        for key, _, name in CONF_FIELDS:
            text = self._vars[key].get().strip().replace(",", ".")
            try:
                value = float(text)
            except ValueError:
                value = -1.0
            if not CONF_RANGE[0] <= value <= CONF_RANGE[1]:
                self._tabs.select(1)
                messagebox.showwarning("Уверенность модели", f"«{name}»: нужно число от {CONF_RANGE[0]:.2f} "
                                       f"до {CONF_RANGE[1]:.2f}, сейчас «{text}».", parent=self)
                return None
            out[key] = round(value, 2)
        return out

    def _save(self):
        conf = self._conf_values()
        if conf is None:
            return
        v = self._vars
        self.result = RecognitionPreset(
            missing_digit=v["missing_digit"].get(), forgive_last=int(v["forgive_last"].get()),
            forgive_with=v["forgive_with"].get(), serial_pad=float(v["serial_pad"].get()),
            account_marker=bool(v["account_marker"].get()) and self._marker_model,
            drum_rule=bool(v["drum_rule"].get()), first_from_last=bool(v["first_from_last"].get()),
            serial_by_table=bool(v["serial_by_table"].get()), turn_photo=bool(v["turn_photo"].get()),
            overlay_notes=bool(v["overlay_notes"].get()), **conf)
        self.destroy()


class TextDialog(tk.Toplevel):
    """Ввод строки: кнопки «ОК» / «Отмена» (у simpledialog.askstring — английские
    OK / Cancel). Enter и Esc дальше диалога не уходят: иначе их ловили
    горячие клавиши экрана («Принять», «Назад»)."""
    def __init__(self, parent, title: str, prompt: str, initial: str = ""):
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.resizable(False, False)
        self.configure(bg=BG)
        self.result: Optional[str] = None
        f = ttk.Frame(self, padding=20)
        f.pack(fill=tk.BOTH, expand=True)
        ttk.Label(f, text=prompt, wraplength=440, justify="left").pack(anchor="w")
        self._var = tk.StringVar(value=initial)
        entry = ttk.Entry(f, textvariable=self._var, font=(FONT, 13), width=26)
        entry.pack(anchor="w", pady=(12, 16))
        buttons = ttk.Frame(f)
        buttons.pack(fill=tk.X)
        action_button(buttons, "Отмена", self._cancel).pack(side=tk.RIGHT)
        action_button(buttons, "ОК", self._ok, kind="accent").pack(side=tk.RIGHT, padx=(0, 8))
        self.bind("<Return>", lambda e: self._ok() or "break")
        self.bind("<KP_Enter>", lambda e: self._ok() or "break")
        self.bind("<Escape>", lambda e: self._cancel() or "break")
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 3
        self.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        entry.focus_set()
        entry.select_range(0, tk.END)
        self.wait_visibility()
        self.grab_set()
        center_on(self, self.master)
        self.wait_window()

    def _ok(self):
        self.result = self._var.get()
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


def ask_text(parent, title: str, prompt: str, initial: str = "") -> Optional[str]:
    """Строка от оператора; None — «Отмена»."""
    return TextDialog(parent, title, prompt, initial).result


class ExportDialog(tk.Toplevel):
    """«Выгрузить показания» (2026-11-22): выбор формата, потом — тут же
    «Готово» с кнопками «Открыть файл» / «Открыть папку». export(формат) →
    (путь к файлу, текст отчёта) или исключение (файл открыт в Excel и т. п.).
    chosen — выбранный формат (запоминается в настройках окна)."""

    def __init__(self, parent, formats: dict, current: str, export, open_path):
        super().__init__(parent)
        self.title("Выгрузить показания")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.configure(bg=BG)
        self.chosen: Optional[str] = None
        self._export, self._open_path = export, open_path
        self._fmt = tk.StringVar(value=current if current in formats else next(iter(formats)))
        self._body = ttk.Frame(self, padding=22)
        self._body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(self._body, text="В каком виде выгрузить показания?", font=(FONT, 12, "bold")).pack(anchor="w")
        ttk.Label(self._body, text="Файл появится в папке месяца: «показания» с выбранным расширением.",
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 12))
        for key, label in formats.items():
            ttk.Radiobutton(self._body, text=label, value=key, variable=self._fmt).pack(anchor="w", pady=3)
        self._buttons = ttk.Frame(self._body)
        self._buttons.pack(fill=tk.X, pady=(18, 0))
        action_button(self._buttons, "Отмена", self.destroy).pack(side=tk.RIGHT)
        action_button(self._buttons, "Выгрузить", self._go, kind="accent").pack(side=tk.RIGHT, padx=(0, 8))
        self.bind("<Return>", lambda e: self._go() or "break")
        self.bind("<KP_Enter>", lambda e: self._go() or "break")
        self.bind("<Escape>", lambda e: self.destroy() or "break")
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        center_on(self, parent)
        self.focus_set()
        self.wait_window()

    def _go(self):
        if self.chosen is not None:          # уже выгружено: Enter — закрыть
            self.destroy()
            return
        fmt = self._fmt.get()
        try:
            path, report = self._export(fmt)
        except Exception as e:               # файл открыт в Excel и т. п. — окно остаётся, можно повторить
            messagebox.showwarning("Выгрузка", str(e), parent=self)
            return
        self.chosen = fmt
        for w in self._body.winfo_children():
            w.destroy()
        ttk.Label(self._body, text="✓  Выгружено", font=(FONT, 12, "bold")).pack(anchor="w")
        ttk.Label(self._body, text=report, justify="left", wraplength=560).pack(anchor="w", pady=(8, 16))
        b = ttk.Frame(self._body)
        b.pack(fill=tk.X)
        action_button(b, "Закрыть", self.destroy).pack(side=tk.RIGHT)
        action_button(b, "Открыть папку", lambda: self._open_path(path, folder=True)).pack(side=tk.RIGHT, padx=8)
        action_button(b, "Открыть файл", lambda: self._open_path(path), kind="accent").pack(side=tk.RIGHT)
        center_on(self, self.master)
