"""
program2.py — окно оператора: ручной разбор фото из question/, с которыми не
справился reader.py, и выборочная проверка его автоматических результатов.
Запуск: `python program2.py`. В настройках — «Папка месяца» (её создаёт
`python gmr.py month`): окно показывает фото всех контролёров из
<месяц>/результат, показания и лог пишет в базу месяца (gmr.sqlite) через
src/gmr/application/operator.py. Что program2 читает и пишет —
docs/contract_reader_program2.md.

АРХИТЕКТУРА:
MainWindow (Tk root)
├── SettingsDialog      — первый запуск / смена пользователя
├── LoginDialog         — подтверждение входа
├── ProcessingTab       — список из question/*, кнопки открыть / нечитаемо
├── VerifyTab           — фото из plus/ minus/, авто-строка PLUS/MINUS ещё не проверена
├── EditScreen          — разбор одного фото: Принять / Дубль / Нечитаемо / Нет в базе /
│                         Серийник в базе с ошибкой; подсказка «похожие номера в базе»
└── VerifyScreen        — проверка одного фото: Верно / Исправить (→ база) / Пропустить
ТИХАЯ РАЗМЕТКА (оператор не видит): если оператор исправил модель — в папку
месяца <месяц>/разметка/ (месяц — и в имени файла; фото эталона — нет); на
компьютер разработки её забирает `gmr.py datasets --collect <месяц>`:
CRNN: кроп serial_number + правильный номер — разметка/serials_crnn/images, labels
CNN:  кропы изменённых цифр — разметка/digits_cnn/<цифра>
YOLO: «Нет счётчика», а оператор ввёл показание — исходное фото в разметка/meter_yolo
"""
import json
import logging
import queue
import shutil
import sys
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk

_BASE = Path(__file__).parent

# ── Корень проекта в sys.path — для импорта src.gmr при запуске из другой папки
sys.path.insert(0, str(_BASE))
# С Фазы 6 program2.py не импортирует reader.py: общий код — в src/gmr/.
# Имена с подчёркиванием оставлены локальными псевдонимами, чтобы не трогать
# остальной код окна.
from src.gmr.domain import PipelineConfig, PhotoResult, Outcome, QUESTION_REASONS
from src.gmr.domain.serial_match import normalize_serial as _normalize_serial
from src.gmr.render import draw_annotation as _draw_annotation, read_image, write_image
from src.gmr.console import safe_console
from src.gmr.storage import LOG_COLUMNS as _LOG_COLUMNS
# С этапа 2.3 окно работает с базой папки месяца (src/gmr/storage/month.py):
# каждое действие — одна транзакция (src/gmr/application/operator.py).
from src.gmr.application import month as month_app
from src.gmr.application.month import ExportLocked, NotAMonth
from src.gmr.storage import log_path_for
from src.gmr.application.operator import OperatorSession
# Этап 4: прогон из окна (отдельный процесс gmr.py process) и окно с текстом
from src.gmr.ui.run_dialog import ProcessDialog, process_command, show_text
from src.gmr.domain import find_auto_row_for_output_file
from src.gmr.domain.serial_match import alphabet_for, one_char_matches
# Модели — через контракты и общий с reader.py сервис распознавания (Фаза 3)
from src.gmr.application import digit_crops_by_position, recognize_photo
from src.gmr.ml.loader import default_device, load_models
from src.gmr.domain.datasets import to_etalon

_LOG_FILE = _BASE / "program2.log"


def _log_handlers_for(stderr):
    """С ярлыка (pythonw.exe, этап 4) консоли нет (sys.stderr is None): журнал
    окна — в program2.log в папке программы (последние 4 файла по 1 МБ), иначе
    ошибки пропадали бы. Файл появляется с первой записью."""
    if stderr is not None:
        return None
    from logging.handlers import RotatingFileHandler
    return [RotatingFileHandler(_LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8", delay=True)]


_log_handlers = _log_handlers_for(sys.stderr)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    handlers=_log_handlers)
log = logging.getLogger("program2")

# ─── Константы ────────────────────────────────────────────────────────────────
SETTINGS_FILE = Path(__file__).parent / "settings.json"
PHOTO_EXTS = {".jpg", ".jpeg", ".png"}

# Подпапки question/ в порядке очереди и их читаемые названия — общие с итогом
# месяца (src/gmr/domain/models.py)
QUESTION_SUBFOLDERS = list(QUESTION_REASONS)
REASON_LABELS = dict(QUESTION_REASONS)

# Цвета
CLR_GREEN  = "#2e7d32"
CLR_RED    = "#c62828"
CLR_ORANGE = "#e65100"
CLR_GRAY   = "#757575"
CLR_BROWN  = "#5D4037"   # для кнопки "Нет в базе"
CLR_BG     = "#f5f5f5"
CLR_WHITE  = "#ffffff"

# ─── Настройки ────────────────────────────────────────────────────────────────
@dataclass
class AppSettings:
    operator_name: str = ""
    month_dir:     str = ""   # папка месяца: база, фото, результат
    # «Папки для разметки» больше нет (2026-10-07): исправления оператора идут
    # в папку месяца (<месяц>/разметка); старый ключ в settings.json не читается.

RU_MONTHS = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь", "Июль", "Август",
             "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"]


def default_month_name(day=None) -> str:
    """Имя папки нового месяца по умолчанию: «Октябрь_2026»."""
    day = day or datetime.now()
    return f"{RU_MONTHS[day.month - 1]}_{day.year}"

def load_settings() -> AppSettings:
    if SETTINGS_FILE.exists():
        try:
            d = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            return AppSettings(**{k: d.get(k, "") for k in AppSettings.__dataclass_fields__})
        except Exception:
            pass
    return AppSettings()

def save_settings(s: AppSettings) -> None:
    SETTINGS_FILE.write_text(
        json.dumps(asdict(s), ensure_ascii=False, indent=2), encoding="utf-8"
    )

# ─── Диалоги входа ────────────────────────────────────────────────────────────
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

# ─── Загрузка моделей ─────────────────────────────────────────────────────────
def _abs_model(rel_path: str) -> str:
    """Резолвит путь к модели от директории program2.py."""
    p = Path(rel_path)
    return str(p) if p.is_absolute() else str(_BASE / rel_path)

class ModelBundle:
    """Все модели, загруженные один раз при старте."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        device = default_device()
        log.info(f"Загружаем модели на {device}...")
        self.models = load_models(config, device=device, resolve_path=_abs_model)
        log.info("Модели загружены ✓")

    def run_on_photo(self, photo_path: str) -> dict:
        """
        Запускает все модели на фото. Возвращает dict с результатами.
        Вызывается из фонового потока.
        """
        result = {
            "serial_text": None,
            "serial_conf": None,
            "reading_str": None,   # "56?25" — строка с ? на неуверенных позициях
            "digit_preds": None,   # list[dict] с conf по каждой позиции
            "digit_crops": None,   # list[ndarray|None] — кропы по позициям
            "serial_crop": None,   # ndarray кропа serial_number
            "error": None,
        }
        try:
            rec = recognize_photo(self.models, photo_path, self.config)
            if not rec.detections:
                result["error"] = "YOLO: ничего не найдено"
                return result

            # Серийник
            if rec.serial is not None:
                result["serial_crop"] = rec.serial["crop"]
                result["serial_text"] = rec.serial_prediction.text
                result["serial_conf"] = rec.serial_prediction.confidence

            # Цифры
            if rec.digits is not None:
                result["reading_str"]  = rec.digits.reading_str
                result["digit_preds"]  = rec.digits.digit_results

                # Сохраняем кропы цифр для разметки
                if rec.digits.digit_bboxes and rec.digits.digit_results:
                    result["digit_crops"] = digit_crops_by_position(
                        rec.meter["crop"], rec.digits.digit_bboxes
                    )

        except Exception as e:
            result["error"] = str(e)
            log.exception("Ошибка в run_on_photo")

        return result

# ─── Утилиты ──────────────────────────────────────────────────────────────────
RESULT_SUBFOLDERS = ("question", "plus", "minus", "repeat")


def controller_dirs(results_dir: str) -> list[Path]:
    """
    Папки контролёров в «результате» месяца (<месяц>/результат/<контролёр>), а
    если фото лежали без подпапок — сама папка результата. С этапа 2.3 окно
    показывает всех контролёров сразу («0 из 0» больше не бывает).
    """
    base = Path(results_dir) if results_dir else None
    if base is None or not base.is_dir():
        return []
    dirs = [base] if any((base / f).is_dir() for f in RESULT_SUBFOLDERS) else []
    dirs += sorted(d for d in base.iterdir()
                   if d.is_dir() and any((d / f).is_dir() for f in RESULT_SUBFOLDERS))
    return dirs


def controller_dir(photo_path: str) -> Path:
    """Папка контролёра, в которой лежит фото (…/<контролёр>/question/<причина>/фото
    или …/<контролёр>/plus/фото)."""
    p = Path(photo_path)
    for parent in p.parents:
        if parent.name in RESULT_SUBFOLDERS:
            return parent.parent
    return p.parent


def controller_name(photo_path: str, results_dir: str) -> str:
    """Имя подпапки контролёра (source_folder в логе); "" — фото без подпапок."""
    d = controller_dir(photo_path)
    return "" if Path(results_dir) == d else d.name


def list_question_photos(results_dir: str) -> list[tuple[str, str, str]]:
    """
    Возвращает список (reason, filename, full_path) из question/ всех
    контролёров. Сортировка: по приоритету QUESTION_SUBFOLDERS, затем по
    контролёру и имени.
    """
    result = []
    dirs = controller_dirs(results_dir)
    for subfolder in QUESTION_SUBFOLDERS:
        for d in dirs:
            sub = d / "question" / subfolder
            if not sub.exists():
                continue
            for p in sorted(sub.iterdir()):
                if p.is_file() and p.suffix.lower() in PHOTO_EXTS:
                    result.append((subfolder, p.name, str(p)))
    return result

# Папка в output/ → исход строки лога, которую проверяют в этой папке
VERIFY_FOLDER_OUTCOME = {"plus": "PLUS", "minus": "MINUS"}


def find_verify_row(log_rows: list[dict], file_name: str, folder: str,
                    controller: Optional[str] = None) -> Optional[dict]:
    """
    Авто-строка лога для фото из plus/ или minus/: исход — как у папки
    (PLUS/MINUS), имя файла совпадает с final_filename или original_filename
    (с расширением или без). Строки REPEAT не подходят: у второго фото того же
    счётчика final_filename тот же, но показаний в строке нет (ошибка до
    2026-09-30 — вкладка «Проверка» брала последнюю строку с этим именем).
    Среди подходящих — последняя; если задан controller — предпочитаются
    строки этого контролёра (source_folder).
    """
    wanted = VERIFY_FOLDER_OUTCOME.get(folder)
    stem = Path(file_name).stem
    found = []
    for r in log_rows:
        if r.get("source") != "auto" or r.get("outcome") != wanted:
            continue
        names = {n for n in (r.get("final_filename", ""), r.get("original_filename", "")) if n}
        names |= {Path(n).stem for n in names}
        if file_name in names or stem in names:
            found.append(r)
    if controller is not None:
        same = [r for r in found if (r.get("source_folder") or "") == controller]
        if same:
            return same[-1]
    return found[-1] if found else None


def list_verify_photos(results_dir: str, log_rows: list[dict]) -> list[dict]:
    """Фото из plus/ и minus/ всех контролёров, у которых авто-строка PLUS/MINUS
    ещё не проверена."""
    result = []
    for ctrl in controller_dirs(results_dir):
        name = "" if ctrl == Path(results_dir) else ctrl.name
        for folder in ("plus", "minus"):
            d = ctrl / folder
            if not d.exists():
                continue
            for p in sorted(d.iterdir()):
                if not (p.is_file() and p.suffix.lower() in PHOTO_EXTS):
                    continue
                row = find_verify_row(log_rows, p.name, folder, controller=name)
                if row is not None and not row.get("verified_by"):
                    result.append({"path": str(p), "log_row": row, "folder": folder})
    return result


def _to_float(v) -> Optional[float]:
    s = str(v).strip() if v is not None else ""
    if s in ("", "nan", "None"):
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


@dataclass
class VerifyCorrection:
    """Что изменится, если сохранить исправление на вкладке «Проверка»."""
    error:           Optional[str] = None     # не сохранять, показать оператору
    serial:          str = ""
    reading:         int = 0
    old_account:     str = ""
    new_account:     str = ""
    last_reading:    Optional[float] = None   # последнее показание (нового) абонента
    delta:           Optional[float] = None
    outcome:         str = "PLUS"
    new_account_has_reading: str = ""         # уже записанное показание у нового абонента

    @property
    def account_changed(self) -> bool:
        return self.new_account != self.old_account


def plan_verify_correction(df: Optional[pd.DataFrame], cfg: PipelineConfig, row: dict,
                           new_serial: str, new_reading_str: str) -> VerifyCorrection:
    """
    Исправление при проверке. Серийник тот же — меняется только показание
    этого абонента. Серийник другой — ищем его в таблице (как reader.py:
    номер, '0'+номер, '00'+номер); фото принадлежит найденному абоненту,
    показание переносится к нему.
    """
    serial = new_serial.strip()
    c = VerifyCorrection(serial=serial, reading=int(new_reading_str),
                         old_account=row.get("account_id", ""),
                         new_account=row.get("account_id", ""),
                         last_reading=_to_float(row.get("last_reading")))
    if _normalize_serial(serial) != _normalize_serial(row.get("serial_id", "")):
        if df is None:
            return VerifyCorrection(error="Таблица не загружена — сменить абонента нельзя.")
        match = None
        for cand in (serial, "0" + serial, "00" + serial):
            m = df[df[cfg.col_serial].apply(lambda x: _normalize_serial(str(x)) == _normalize_serial(cand))]
            if not m.empty:
                match = m.iloc[0]
                break
        if match is None:
            return VerifyCorrection(error=f"Серийного номера {serial} нет в таблице.")
        c.new_account = str(match[cfg.col_account_id]).strip()
        c.last_reading = _to_float(match[cfg.col_last_reading])
        if c.account_changed:
            existing = _to_float(match[cfg.col_new_reading])
            c.new_account_has_reading = str(match[cfg.col_new_reading]).strip() if existing is not None else ""
    c.delta = (c.reading - c.last_reading) if c.last_reading is not None else None
    c.outcome = "MINUS" if c.delta is not None and c.delta < 0 else "PLUS"
    return c


# ─── Подсказка «похожие номера в базе» (решение владельца 2026-09-30) ─────────
# Если серийника нет в таблице, но есть номер, отличающийся одним символом, —
# это почти всегда тот же счётчик: модель ошиблась при чтении или в базе
# опечатка. На ответах оператора (Сулиман С) — 7 из 7. Подсказка только
# заполняет лицевой счёт; решает оператор: «Принять» (модель ошиблась),
# «Серийник в базе с ошибкой» или «Нет в базе».
HINT_NOTE = "подсказка"
CHOICE_NOTE = "номер у нескольких абонентов, выбран"   # notes: выбор оператора (2026-10-01)


@dataclass
class SerialIndex:
    """Серийники таблицы для поиска похожих: номер → (лицевой счёт, последнее показание)."""
    by_serial: dict
    alphabet:  str


@dataclass
class SerialHint:
    serial:       str               # номер в таблице
    account:      str
    last_reading: Optional[float]
    delta:        Optional[float]   # показание − последнее показание


def serial_choices(matches: pd.DataFrame, cfg: PipelineConfig, reading_str: str) -> list[SerialHint]:
    """Абоненты, у которых в таблице записан этот номер (по одному на лицевой
    счёт, в порядке таблицы). Больше одного — у одной записи в базе номер с
    ошибкой, выбирает оператор (решение владельца 2026-10-01)."""
    reading = int(reading_str) if reading_str and reading_str.isdigit() else None
    out, seen = [], set()
    for serial, account, last in zip(matches[cfg.col_serial], matches[cfg.col_account_id],
                                     matches[cfg.col_last_reading]):
        account = str(account).strip()
        if account in seen:
            continue
        seen.add(account)
        last = _to_float(last)
        delta = (reading - last) if reading is not None and last is not None else None
        out.append(SerialHint(str(serial).strip(), account, last, delta))
    return out


def build_serial_index(df: Optional[pd.DataFrame], cfg: PipelineConfig) -> Optional[SerialIndex]:
    if df is None:
        return None
    by_serial = {}
    for serial, account, last in zip(df[cfg.col_serial], df[cfg.col_account_id], df[cfg.col_last_reading]):
        key = _normalize_serial(str(serial))
        if key and key != "nan" and key not in by_serial:
            by_serial[key] = (str(account).strip(), _to_float(last))
    return SerialIndex(by_serial, alphabet_for(set(by_serial)))


def serial_hints(serial: str, reading_str: str, index: Optional[SerialIndex], limit: int = 3) -> list[SerialHint]:
    """
    Номера из таблицы «в одном символе» от serial. Первыми — с наименьшим
    расходом (|показание − последнее|), если показание прочитано полностью.
    """
    serial = (serial or "").strip()
    if index is None or len(serial) < 3:
        return []
    reading = int(reading_str) if reading_str and reading_str.isdigit() else None
    hints = []
    for t in sorted({t for _, t in one_char_matches(serial, set(index.by_serial), index.alphabet)}):
        account, last = index.by_serial[t]
        delta = (reading - last) if reading is not None and last is not None else None
        hints.append(SerialHint(t, account, last, delta))
    hints.sort(key=lambda h: (h.delta is None, abs(h.delta) if h.delta is not None else 0))
    return hints[:limit]


# ─── «Серийник в базе с ошибкой» (решения владельца 2026-09-30, 2026-10-01) ───
# На фото серийник верный, по лицевому счёту понятно, что счётчик тот же, но в
# базе номер записан с опечаткой. Номер в базе месяца исправляется на номер с
# фото, показание записывается сразу, фото — в plus/ или minus/. Номер попадает
# в список для компании (db_serial_fix.csv в папке месяца).
DB_SERIAL_FIX_LIST    = "db_serial_fix.csv"   # в папке месяца — список для компании
DB_SERIAL_FIX_COLUMNS = ["Фото", "Контролёр", "Лицевой счёт", "Серийник в базе",
                         "Серийник на фото", "Показание", "Дата", "Оператор"]


def table_serial_for_account(df: Optional[pd.DataFrame], cfg: PipelineConfig, account: str) -> Optional[str]:
    """Серийник абонента в таблице (None — лицевого счёта нет)."""
    if df is None or not account:
        return None
    m = df[df[cfg.col_account_id].apply(lambda x: str(x).strip() == account.strip())]
    return None if m.empty else _normalize_serial(str(m.iloc[0][cfg.col_serial]))


def append_db_serial_fix(list_path: str, entry: dict) -> None:
    """Дописывает строку в список для исправления базы (CSV «;», открывается в Excel)."""
    import csv
    p = Path(list_path)
    new = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DB_SERIAL_FIX_COLUMNS, delimiter=";")
        if new:
            w.writeheader()
        w.writerow({k: entry.get(k, "") for k in DB_SERIAL_FIX_COLUMNS})


def free_photo_path(dst_dir: str, name: str, current: str) -> str:
    """Путь для фото в dst_dir; если имя занято другим файлом — добавляется _2, _3…"""
    p = Path(dst_dir) / name
    k = 2
    while p.exists() and p.resolve() != Path(current).resolve():
        p = Path(dst_dir) / f"{Path(name).stem}_{k}{Path(name).suffix}"
        k += 1
    return str(p)


def open_in_explorer(path) -> None:
    """Открыть папку в проводнике (Windows) или файловом менеджере."""
    import os
    import subprocess
    try:
        if hasattr(os, "startfile"):
            os.startfile(str(path))                       # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError as e:
        log.warning(f"Не удалось открыть {path}: {e}")


def describe_reading(r) -> str:
    """Чьё показание и когда — для вопроса «Заменить?»."""
    who = {"auto": f"записала программа по фото {r.photo}" if r.photo else "записала программа",
           "manual": f"записал оператор {r.updated_by}",
           "table": "было в таблице компании"}.get(r.source, f"записал {r.updated_by}")
    return f"{who}, дата {r.date}" if r.date else who


def place_decision(row: dict, photo_path: str, dst_dir: str) -> str:
    """Имя для фото в dst_dir («Дубль», «Нечитаемо», «Нет в базе»): если имя
    занято другим фото — <имя>_2… (этап 3: файлы не затирают друг друга),
    тогда оно же — final_filename строки лога. Возвращает имя."""
    name = Path(free_photo_path(dst_dir, Path(photo_path).name, photo_path)).name
    if name != Path(photo_path).name:
        row["final_filename"] = name
    return name


def move_photo(src: str, dst_dir: str, new_name: Optional[str] = None) -> str:
    """Перемещает фото в dst_dir, возвращает новый путь."""
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    name = new_name or Path(src).name
    dst = str(Path(dst_dir) / name)
    shutil.move(src, dst)
    return dst

def redraw_annotation(path: str, serial: str, reading_str: str) -> None:
    """
    Перерисовывает аннотацию в левом верхнем углу уже сохранённого фото.
    Использует финальные значения оператора вместо модельных.
    """
    try:
        img = read_image(path)          # пути с кириллицей — см. src/gmr/render/image_io.py
        if img is None:
            log.warning(f"Не удалось открыть фото для перерисовки подписи: {path}")
            return
            
        result = PhotoResult(photo_path=path, outcome=Outcome.MINUS)
        result.serial_text = serial or None
        
        # reading_str → reading если все цифры известны
        if reading_str and reading_str.isdigit() and len(reading_str) == 5:
            result.reading     = int(reading_str)
            result.reading_str = None
        else:
            result.reading     = None
            result.reading_str = reading_str or None
            
        _draw_annotation(img, result)
        if write_image(path, img):
            log.info(f"Аннотация перерисована: {Path(path).name}")
        else:
            log.warning(f"Не удалось сохранить перерисованное фото: {path}")
    except Exception as e:
        log.warning(f"Не удалось перерисовать аннотацию: {e}")

def photo_to_tk(path: str, max_w: int, max_h: int,
                zoom: float = 1.0, offset: tuple = (0, 0)) -> Optional[ImageTk.PhotoImage]:
    try:
        img = Image.open(path)
        base_w, base_h = img.size
        scale = min(max_w / base_w, max_h / base_h)
        new_w = max(1, int(base_w * scale * zoom))
        new_h = max(1, int(base_h * scale * zoom))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None

def now_iso() -> str:
    return datetime.now().strftime("%d.%m.%Y %H:%M")

# ─── Разметка (тихо) ─────────────────────────────────────────────────────────
# root — <месяц>/разметка (с 2026-10-11: на рабочем компьютере папки проекта с
# датасетами нет, папка месяца есть всегда). Внутри — как в датасетах:
# digits_cnn/<цифра>/, serials_crnn/images|labels/, meter_yolo/. Имена — с
# отпечатком фото (обучение делит датасет по фото, models/datasets.py) и с
# месяцем (решение 1а, 2026-10-09). `gmr.py datasets --collect <месяц>` копирует
# это в database/datasets/<датасет>/new/<ГГГГ-ММ>/.
def save_crnn_markup(
    root: Path,
    photo_key: str,
    month: str,
    serial_crop: Optional[np.ndarray],
    correct_text: str,
    model_text: Optional[str],
) -> None:
    """Кроп серийника и правильный номер — только если номер исправили:
    serials_crnn/images/<номер>__<фото>__<месяц>.jpg и labels/… .txt."""
    if serial_crop is None:
        return
    if model_text is not None and correct_text == model_text:
        return
    safe_text = "".join(c for c in correct_text if c.isalnum() or c in "-_") or "serial"
    name = f"{safe_text}__{photo_key}__{month}"
    new = Path(root) / "serials_crnn"
    (new / "images").mkdir(parents=True, exist_ok=True)
    (new / "labels").mkdir(parents=True, exist_ok=True)
    if not write_image(new / "images" / f"{name}.jpg", serial_crop):
        log.warning(f"CRNN разметка: не удалось сохранить {name}.jpg")
        return
    (new / "labels" / f"{name}.txt").write_text(correct_text, encoding="utf-8")
    log.info(f"CRNN разметка: {name}.jpg → '{correct_text}'")

def save_cnn_markup(
    root: Path,
    photo_key: str,
    month: str,
    digit_crops: Optional[list],
    digit_preds: Optional[list],
    model_reading_str: Optional[str],
    final_reading_str: str,
) -> None:
    """Кропы только изменённых цифр: digits_cnn/<цифра>/<фото>__<месяц>__digit_<N>.jpg."""
    if not digit_crops or not digit_preds:
        return
    if len(digit_crops) != len(final_reading_str):
        log.warning(f"CNN разметка: несоответствие длин ({len(digit_crops)} vs {len(final_reading_str)})")
        return
    model_chars = list(model_reading_str or "?" * len(final_reading_str))
    if len(model_chars) != len(final_reading_str):
        model_chars = ["?"] * len(final_reading_str)

    for pos, (crop, final_char, model_char) in enumerate(
        zip(digit_crops, final_reading_str, model_chars)
    ):
        if crop is None:
            continue
        if final_char == model_char:
            continue
        if not final_char.isdigit():
            continue

        out_dir = Path(root) / "digits_cnn" / final_char
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{photo_key}__{month}__digit_{pos + 1}.jpg"
        if not write_image(out_path, crop):
            log.warning(f"CNN разметка: не удалось сохранить {out_path}")
            continue
        log.info(f"CNN разметка: pos={pos} model={model_char!r} → correct={final_char!r}")

def save_meter_markup(root: Path, photo_key: str, month: str, original: Optional[Path]) -> None:
    """«Нет счётчика», а оператор ввёл показание — детектор промахнулся:
    исходное фото в meter_yolo/ на ручную разметку (решение 3а)."""
    if original is None or not Path(original).is_file():
        return
    out = Path(root) / "meter_yolo"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, out / f"{photo_key}__{month}{Path(original).suffix.lower()}")
    log.info(f"YOLO разметка: {photo_key} → meter_yolo/new")

# ─── Виджет показаний (5 полей) ───────────────────────────────────────────────
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

    def focus_first(self):
        self._entries[0].focus_set()

# ─── Главное окно ─────────────────────────────────────────────────────────────
class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Обработка газовых счётчиков")
        self.geometry("960x640")
        self.minsize(800, 500)
        self.configure(bg=CLR_BG)

        self.settings  = load_settings()
        self.session:   Optional[OperatorSession] = None
        self.df: Optional[pd.DataFrame] = None
        self.log_rows:  list[dict] = []
        self.models:    Optional[ModelBundle] = None
        self.config      = PipelineConfig()

        self._do_login()

    def report_callback_exception(self, exc, val, tb):
        """Ошибка в обработчике кнопки: в журнал и оператору (с ярлыка консоли
        нет — без этого ошибка была бы не видна)."""
        log.error("Ошибка в окне", exc_info=(exc, val, tb))
        where = (f"в журнале {_LOG_FILE.name} (папка программы)" if _log_handlers
                 else "в окне PowerShell, из которого запущена программа")
        messagebox.showerror("Ошибка программы", f"{exc.__name__}: {val}\n\nПодробности — {where}.",
                             parent=self)

    def _do_login(self):
        if not self.settings.operator_name or not self.settings.month_dir:
            if not self._open_settings(first_run=True):
                self.destroy()
                return
        else:
            dlg = LoginDialog(self, self.settings)
            if dlg.action == "change":
                self._open_settings()          # «Отмена» — работа с прежними настройками

        while not self._load_data():          # папка месяца не создана — выбрать другую
            if not self._open_settings():
                self.destroy()
                return
        self._load_models_async()
        self._build_ui()

    def _open_settings(self, first_run=False) -> bool:
        dlg = SettingsDialog(self, self.settings, new_month=self.ask_new_month)
        if dlg.result:
            self.settings = dlg.result
            save_settings(self.settings)
            return True
        return False

    def _load_data(self) -> bool:
        """Открывает базу папки месяца (и делает её копию). False — папка не месяц."""
        if self.session is not None:
            self.session.close()
            self.session = None
        try:
            self.session = OperatorSession(self.settings.month_dir, self.config)
        except NotAMonth as e:
            # этап 4: месяц можно создать прямо отсюда, без PowerShell
            if not (self.settings.month_dir and messagebox.askyesno(
                    "Папка месяца",
                    f"{e}\n\nСоздать в этой папке новый месяц из таблицы компании?")):
                return False
            if not self.create_month(self.settings.month_dir):
                return False
            self.session = OperatorSession(self.settings.month_dir, self.config)
        try:
            dest = self.session.backup()
            if dest is not None:
                log.info(f"Копия базы месяца: {dest}")
        except (OSError, sqlite3.Error) as e:
            log.warning(f"Не удалось сделать копию базы месяца: {e}")
        self.reload()
        return True

    def reload(self):
        """Абоненты, показания и лог — заново из базы (их мог изменить reader.py)."""
        self.df = pd.DataFrame(self.session.table_rows(), columns=self.session.table_columns())
        self.log_rows = self.session.log_rows()
        self._serial_index = None
        log.info(f"Месяц: {self.session.folder.root} — абонентов {len(self.df)}, строк лога {len(self.log_rows)}")

    @property
    def results_dir(self) -> str:
        return str(self.session.folder.results)

    def _load_models_async(self):
        self._model_load_error: Optional[str] = None

        def _load():
            try:
                bundle = ModelBundle(self.config)
                self.models = bundle
            except Exception as e:
                log.error(f"Ошибка загрузки моделей: {e}")
                self._model_load_error = str(e)
                self.after(0, self._on_model_load_error)

        threading.Thread(target=_load, daemon=True).start()

    def _on_model_load_error(self):
        err = getattr(self, "_model_load_error", None)
        if err:
            messagebox.showerror(
                "Ошибка загрузки моделей",
                f"Не удалось загрузить модели:\n\n{err}\n\n"
                "Программа продолжит работу, но автораспознавание недоступно.",
            )

    # ─── Месяц: создать, обновить таблицу, итог (этап 4) ───────────────────────
    def _ask_table(self, parent=None) -> Optional[str]:
        return filedialog.askopenfilename(
            parent=parent or self, title="Таблица компании",
            filetypes=[("Таблицы", "*.xls *.xlsx *.csv"), ("Все файлы", "*.*")]) or None

    def ask_new_month(self, parent=None, who: Optional[str] = None) -> Optional[str]:
        """Новый месяц без PowerShell (2026-10-07): таблица компании, потом —
        как «Сохранить как»: где и под каким именем создать папку месяца (по
        умолчанию «Октябрь_2026» рядом с текущим месяцем). Возвращает папку
        созданного месяца или None."""
        parent = parent or self
        table = self._ask_table(parent)
        if not table:
            return None
        here = Path(self.settings.month_dir).parent if self.settings.month_dir else Path.home()
        folder = filedialog.asksaveasfilename(
            parent=parent, title="Где создать папку нового месяца", initialdir=str(here),
            initialfile=default_month_name(), confirmoverwrite=False)
        if not folder:
            return None
        return folder if self.create_month(folder, table, parent=parent, who=who) else None

    def create_month(self, folder: str, table: Optional[str] = None, parent=None,
                     who: Optional[str] = None) -> bool:
        """Новый месяц в папке folder из таблицы компании (то же, что
        `gmr.py month <папка> --table <таблица>`). True — месяц создан."""
        parent = parent or self
        if month_app.is_month(month_app.MonthFolder(Path(folder))):
            messagebox.showerror("Новый месяц", f"В папке {folder} месяц уже есть.\n"
                                 "Новая таблица компании — меню «Месяц» → «Обновить таблицу…».", parent=parent)
            return False
        table = table or self._ask_table(parent)
        if not table:
            return False
        old_log = Path(log_path_for(table))
        import_log = old_log.is_file() and messagebox.askyesno(
            "Старый лог",
            f"Рядом с таблицей лежит лог старой версии программы:\n{old_log.name}\n\n"
            "Перенести его в базу месяца? Тогда фото из этого лога будут считаться уже разобранными.\n"
            "Обычно — «Нет»: месяц начинается с чистого листа.", default="no", parent=parent)
        try:
            n = month_app.table_readings_count(table, self.config)
        except (ValueError, OSError) as e:
            messagebox.showerror("Новый месяц", str(e), parent=parent)
            return False
        clear = n > 0 and messagebox.askyesno(
            "Показания в таблице",
            f"В таблице уже есть показания у {n} абонентов.\n\nОчистить их?\n\n"
            "«Да» — месяц начинается без показаний, программа прочитает все фото.\n"
            "«Нет» — эти показания считаются уже записанными.\n\n"
            "Оригинал таблицы не меняется.", parent=parent)
        try:
            rep = month_app.load_table(folder, table, self.config, who=who or self.settings.operator_name,
                                       import_old_log=import_log, clear_readings=clear)
            exported = month_app.export_month(folder, self.config).text()
        except (ValueError, OSError, ExportLocked) as e:
            messagebox.showerror("Новый месяц", str(e), parent=parent)
            return False
        show_text(parent, "Месяц создан", rep.text() + "\n\n" + exported)
        return True

    def _switch_month(self, folder: str):
        """Окно — на другой папке месяца (настройки сохраняются). Если месяц
        не открылся, окно остаётся в прежнем месяце."""
        self._close_screen()
        old = self.settings
        self.settings = AppSettings(old.operator_name, folder)
        opened = False
        try:
            opened = self._load_data()
        finally:
            if not opened:
                self.settings = old
                self._load_data()
        save_settings(self.settings)
        self._month_var.set(f"Месяц: {self.session.folder.root}")
        self.refresh_tabs()

    def new_month(self):
        """Меню «Месяц» → «Новый месяц…»: таблица компании и папка нового месяца."""
        folder = self.ask_new_month()
        if folder:
            self._switch_month(folder)

    def update_table(self):
        """Меню «Месяц» → «Обновить таблицу…»: компания прислала новую таблицу."""
        table = self._ask_table()
        if not table or not messagebox.askyesno(
                "Обновить таблицу", f"Загрузить таблицу\n{table}\nв месяц {self.session.folder.root}?\n\n"
                "Сверка — по лицевому счёту; показания из базы сохраняются.", parent=self):
            return
        self._close_screen()
        try:
            rep = month_app.load_table(str(self.session.folder.root), table, self.config,
                                       who=self.settings.operator_name)
            exported = month_app.export_month(str(self.session.folder.root), self.config).text()
        except (ValueError, OSError, ExportLocked) as e:
            messagebox.showerror("Обновить таблицу", str(e), parent=self)
            return
        self.refresh_tabs()
        show_text(self, "Таблица обновлена", rep.text() + "\n\n" + exported)

    def choose_month(self):
        """Меню «Месяц» → «Открыть другой месяц…»."""
        folder = filedialog.askdirectory(parent=self, title="Папка месяца")
        if not folder:
            return
        if not month_app.is_month(month_app.MonthFolder(Path(folder))):
            if not messagebox.askyesno("Папка месяца", f"В папке {folder} месяц не создан.\n\n"
                                       "Создать новый месяц из таблицы компании?", parent=self):
                return
            if not self.create_month(folder):
                return
        self._switch_month(folder)

    def show_summary(self):
        """«Итог месяца»: то же, что `gmr.py month <папка>`."""
        show_text(self, "Итог месяца", month_app.month_summary(str(self.session.folder.root)))

    def open_month_folder(self):
        open_in_explorer(self.session.folder.root)

    def reread_errors(self):
        """Меню «Месяц» → «Прочитать заново фото с ошибками…» (после замены модели)."""
        if messagebox.askyesno(
                "Прочитать заново", "Прочитать заново все фото с ошибками, которые ждут оператора?\n\n"
                "Нужно, например, после замены модели. Разобранные фото не трогаются.", parent=self):
            self.process_new(reread=True)

    def _build_ui(self):
        menubar = tk.Menu(self)
        m = tk.Menu(menubar, tearoff=False)
        m.add_command(label="Итог месяца", command=self.show_summary)
        m.add_command(label="Открыть папку месяца", command=self.open_month_folder)
        m.add_separator()
        m.add_command(label="Новый месяц…", command=self.new_month)
        m.add_command(label="Обновить таблицу компании…", command=self.update_table)
        m.add_command(label="Открыть другой месяц…", command=self.choose_month)
        m.add_separator()
        m.add_command(label="Прочитать заново фото с ошибками…", command=self.reread_errors)
        menubar.add_cascade(label="Месяц", menu=m)
        self.configure(menu=menubar)        # self.config — это PipelineConfig, поэтому configure
        self._month_menu = m

        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=(8, 0))
        self._month_var = tk.StringVar(value=f"Месяц: {self.session.folder.root}")
        ttk.Label(top, textvariable=self._month_var, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        ttk.Button(top, text="⇩  Выгрузить показания", command=self.export).pack(side=tk.RIGHT)
        ttk.Button(top, text="Итог месяца", command=self.show_summary).pack(side=tk.RIGHT, padx=(0, 8))
        tk.Button(top, text="▶  Обработать новые", command=self.process_new,
                  bg=CLR_GREEN, fg="white", relief="flat", padx=10).pack(side=tk.RIGHT, padx=8)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._proc_tab = ProcessingTab(self._notebook, self)
        self._verify_tab = VerifyTab(self._notebook, self)

        self._notebook.add(self._proc_tab,   text="  Обработка  ")
        self._notebook.add(self._verify_tab, text="  Проверка  ")

    def refresh_tabs(self):
        self.reload()
        self._proc_tab.refresh()
        self._verify_tab.refresh()

    def _close_screen(self):
        """Вернуться к спискам (экран разбора закрывается)."""
        scr = getattr(self, "_current_screen", None)
        if scr is not None and scr.winfo_exists():
            scr._back()

    def process_new(self, reread: bool = False):
        """«Обработать новые» (этап 4): прогон новых фото месяца с полосой
        прогресса; пока он идёт, разбирать фото нельзя (решение 2а)."""
        if self.session is None:
            return
        self._close_screen()
        dlg = ProcessDialog(self, process_command(_BASE, str(self.session.folder.root), reread),
                            self.session.folder.stop_file, self._after_process, cwd=_BASE,
                            title="Обработка фото с ошибками" if reread else "Обработка новых фото")
        self._process_dialog = dlg
        self.wait_window(dlg)

    def _after_process(self, code: int, lines: list[str]):
        self.refresh_tabs()
        report = self.session.folder.results / "report.txt"
        if code == 0 and report.is_file():
            show_text(self, "Итог обработки", report.read_text(encoding="utf-8"))
        else:
            tail = "\n".join(lines[-25:]) or "(нет вывода)"
            show_text(self, "Обработка не удалась", f"Код завершения: {code}\n\n{tail}")

    def export(self) -> bool:
        """Кнопка «Выгрузить показания»: показания.xlsx и лог.csv."""
        try:
            rep = self.session.export()
        except ExportLocked as e:
            messagebox.showwarning("Выгрузка", str(e))
            return False
        messagebox.showinfo("Выгрузка", rep.text())
        return True

    def _on_close(self):
        """Закрытие окна: выгрузка, потом выход. Файл открыт в Excel —
        «Повторить» после того, как его закроют, или выйти без выгрузки
        (показания в базе, выгрузить можно потом: gmr.py export)."""
        while self.session is not None:
            try:
                self.session.export()
                break
            except ExportLocked as e:
                if not messagebox.askretrycancel(
                    "Выгрузка",
                    f"{e}\n\n«Повторить» — после того как закроете файл.\n"
                    "«Отмена» — выйти без выгрузки (показания сохранены в базе).",
                ):
                    break
        self.destroy()

    def destroy(self):
        if getattr(self, "session", None) is not None:
            self.session.close()
            self.session = None
        super().destroy()

    def _with_photo_identity(self, row: dict, photo_path: str) -> Optional[dict]:
        """
        Отпечаток исходного фото — из автоматической строки, которая создала
        этот файл (файл в question/ пересохранён с аннотацией, по нему отпечаток
        не посчитать). Нужен, чтобы reader.py узнал разобранное фото по
        содержимому, а не только по имени. Возвращает эту строку (или None).
        """
        src = find_auto_row_for_output_file(
            self.log_rows, row.get("original_filename", ""),
            controller_name(photo_path, self.results_dir),
        )
        if src is not None and row.get("source") == "manual" and not row.get("photo_hash"):
            row["photo_hash"] = src.get("photo_hash", "")
            row["source_folder"] = src.get("source_folder", "")
        return src

    def append_log(self, row: dict, photo_path: str):
        """Решение оператора без показания: «Дубль», «Нечитаемо», «Нет в базе»."""
        self._with_photo_identity(row, photo_path)
        self.session.add_row(row)
        self.log_rows.append(row)

    def existing_reading(self, account: str):
        """Показание, уже записанное лицевому счёту из таблицы (None — нет;
        счёта нет в таблице — показание «Принять» и не пишет)."""
        if not account or self.session is None or self.df is None:
            return None
        if not (self.df[self.config.col_account_id].astype(str).str.strip() == account).any():
            return None
        return self.session.reading(account)

    def _sync_df_reading(self, account: str):
        """Текущее показание абонента в таблице окна — как в базе."""
        r = self.session.reading(account)
        acc = self.df[self.config.col_account_id].astype(str).str.strip()
        self.df.loc[acc == account, self.config.col_new_reading] = r.value if r else ""

    def accept_reading(self, row: dict, photo_path: str):
        """«Принять»: показание и строка лога одной записью в базу. Если лицевого
        счёта нет в таблице — только строка лога (как и раньше)."""
        src = self._with_photo_identity(row, photo_path)
        account = row.get("account_id", "")
        in_table = account and (self.df[self.config.col_account_id].astype(str).str.strip() == account).any()
        if in_table:
            date_name = (src or {}).get("original_filename") or row.get("original_filename", "")
            self.session.accept(row, date_name, keep=photo_path)
            self._sync_df_reading(account)
        else:
            self.session.add_row(row)
        self.log_rows.append(row)

    def fix_serial_and_accept(self, row: dict, photo_path: str, photo_serial: str) -> str:
        """«Серийник в базе с ошибкой»: номер в базе — как на фото, показание сразу."""
        src = self._with_photo_identity(row, photo_path)
        date_name = (src or {}).get("original_filename") or row.get("original_filename", "")
        old = self.session.fix_serial_and_accept(row, photo_serial, date_name, keep=photo_path)
        acc = self.df[self.config.col_account_id].astype(str).str.strip()
        self.df.loc[acc == row["account_id"], self.config.col_serial] = photo_serial
        self._sync_df_reading(row["account_id"])
        self._serial_index = None
        self.log_rows.append(row)
        return old

    def serial_index(self) -> Optional[SerialIndex]:
        """Индекс серийников таблицы для подсказки (строится один раз: номера,
        лицевые счета и последние показания program2 не меняет)."""
        if getattr(self, "_serial_index", None) is None and self.df is not None:
            self._serial_index = build_serial_index(self.df, self.config)
        return getattr(self, "_serial_index", None)

    def open_edit_screen(self, photo_path: str, reason: str):
        self._notebook.pack_forget()
        screen = EditScreen(self, photo_path, reason, on_back=self._back_from_screen)
        screen.pack(fill=tk.BOTH, expand=True)
        self._current_screen = screen

    def open_verify_screen(self, item: dict, index: int = 0):
        self._notebook.pack_forget()
        screen = VerifyScreen(self, item, on_back=self._back_from_screen, index=index)
        screen.pack(fill=tk.BOTH, expand=True)
        self._current_screen = screen

    def open_verify_at(self, index: int):
        """
        Следующее фото на вкладке «Проверка» — как во вкладке «Обработка».
        Список строится заново: проверенное фото из него выпадает, поэтому
        после «Верно»/«Сохранить» следующее стоит на том же номере.
        """
        items = list_verify_photos(self.results_dir, self.log_rows)
        if hasattr(self, "_current_screen"):
            self._current_screen.pack_forget()
            self._current_screen.destroy()
            del self._current_screen
        if index < len(items):
            self.open_verify_screen(items[index], index)
        else:
            self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            self.refresh_tabs()
            messagebox.showinfo("Готово", "Все фото на проверке просмотрены.")

    def _back_from_screen(self):
        if hasattr(self, "_current_screen"):
            self._current_screen.pack_forget()
            self._current_screen.destroy()
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.refresh_tabs()

    def open_next_or_back(self, current_path: str):
        photos = list_question_photos(self.results_dir)

        if hasattr(self, "_current_screen"):
            self._current_screen.pack_forget()
            self._current_screen.destroy()

        if photos:
            reason, fname, next_path = photos[0]
            screen = EditScreen(self, next_path, reason, on_back=self._back_from_screen)
            screen.pack(fill=tk.BOTH, expand=True)
            self._current_screen = screen
        else:
            self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            self.refresh_tabs()
            messagebox.showinfo("Готово", "Все фото в очереди обработаны!")

# ─── Таб «Обработка» ──────────────────────────────────────────────────────────
class ProcessingTab(ttk.Frame):
    def __init__(self, parent, app: MainWindow):
        super().__init__(parent)
        self.app = app
        self._build()
        self.refresh()

    def _build(self):
        self._counter_var = tk.StringVar(value="Загрузка...")
        ttk.Label(self, textvariable=self._counter_var, font=("Segoe UI", 10), wraplength=900).pack(
            anchor="w", padx=12, pady=(10, 4)
        )

        cols = ("reason", "controller", "filename")
        self._tree = ttk.Treeview(self, columns=cols, show="headings", selectmode="browse")
        self._tree.heading("reason",     text="Причина")
        self._tree.heading("controller", text="Контролёр")
        self._tree.heading("filename",   text="Имя файла")
        self._tree.column("reason",     width=240, stretch=False)
        self._tree.column("controller", width=160, stretch=False)
        self._tree.column("filename",   width=320)
        self._items: list[tuple[str, str, str]] = []

        vsb = ttk.Scrollbar(self, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)

        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0), pady=(0, 12))
        vsb.pack(side=tk.LEFT, fill=tk.Y, pady=(0, 12))

        btn_frame = ttk.Frame(self)
        btn_frame.pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=12)

        ttk.Button(btn_frame, text="✎  Открыть",    command=self._open,       width=18).pack(pady=6)
        ttk.Button(btn_frame, text="✗  Нечитаемо",  command=self._unreadable, width=18).pack(pady=6)
        ttk.Button(btn_frame, text="⟳  Обновить",   command=self.refresh,     width=18).pack(pady=6)

        self._tree.bind("<Double-Button-1>", lambda e: self._open())

    def refresh(self):
        self._tree.delete(*self._tree.get_children())
        self._items = list_question_photos(self.app.results_dir)
        for reason, fname, path in self._items:
            label = REASON_LABELS.get(reason, reason)
            ctrl = controller_name(path, self.app.results_dir)
            self._tree.insert("", tk.END, values=(label, ctrl, fname))

        remaining = len(self._items)
        processed = sum(
            1 for r in self.app.log_rows
            if r.get("source") == "manual"
            and r.get("outcome") not in ("", None)
        )
        total = remaining + processed
        self._counter_var.set(f"Всего: {total} | Обработано: {processed} | Осталось: {remaining}")

    def _selected_path(self) -> Optional[tuple[str, str]]:
        sel = self._tree.selection()
        if not sel:
            messagebox.showinfo("Выберите фото", "Сначала выберите фото из списка.")
            return None
        idx = self._tree.index(sel[0])
        if idx < len(self._items):
            reason, _, path = self._items[idx]
            return reason, path
        return None

    def _open(self):
        sel = self._selected_path()
        if sel:
            reason, path = sel
            self.app.open_edit_screen(path, reason)

    def _unreadable(self):
        sel = self._selected_path()
        if not sel:
            return
        reason, path = sel
        if not messagebox.askyesno("Нечитаемо", f"Пометить как нечитаемо?\n{Path(path).name}"):
            return
        row = {k: "" for k in _LOG_COLUMNS}
        row.update({
            "original_filename": Path(path).name,
            "outcome":            "UNREADABLE",
            "source":             "manual",
            "processed_by":      self.app.settings.operator_name,
            "processed_at":      now_iso(),
            "notes":              "нечитаемо (из списка)",
        })
        dst_dir = str(controller_dir(path) / "unreadable")
        name = place_decision(row, path, dst_dir)
        self.app.append_log(row, path)
        try:
            move_photo(path, dst_dir, name)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return
        self.refresh()

# ─── Таб «Проверка» ───────────────────────────────────────────────────────────
class VerifyTab(ttk.Frame):
    def __init__(self, parent, app: MainWindow):
        super().__init__(parent)
        self.app = app
        self._build()
        self.refresh()

    def _build(self):
        self._counter_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._counter_var, font=("Segoe UI", 10), wraplength=900).pack(
            anchor="w", padx=12, pady=(10, 4)
        )

        cols = ("outcome", "controller", "account_id", "reading", "processed_at")
        self._tree = ttk.Treeview(self, columns=cols, show="headings", selectmode="browse")
        self._tree.heading("controller",   text="Контролёр")
        self._tree.column("controller",   width=140, stretch=False)
        self._tree.heading("outcome",      text="Результат")
        self._tree.heading("account_id",   text="Account ID")
        self._tree.heading("reading",      text="Показания")
        self._tree.heading("processed_at", text="Обработано")
        self._tree.column("outcome",      width=80,  stretch=False)
        self._tree.column("account_id",   width=140, stretch=False)
        self._tree.column("reading",      width=100, stretch=False)
        self._tree.column("processed_at", width=200)

        vsb = ttk.Scrollbar(self, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0), pady=(0, 12))
        vsb.pack(side=tk.LEFT, fill=tk.Y, pady=(0, 12))

        btn_frame = ttk.Frame(self)
        btn_frame.pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=12)
        ttk.Button(btn_frame, text="✎  Открыть",  command=self._open,   width=18).pack(pady=6)
        ttk.Button(btn_frame, text="⟳  Обновить", command=self.refresh, width=18).pack(pady=6)

        self._tree.bind("<Double-Button-1>", lambda e: self._open())
        self._items: list[dict] = []

    def refresh(self):
        self._tree.delete(*self._tree.get_children())
        self._items = list_verify_photos(self.app.results_dir, self.app.log_rows)
        for item in self._items:
            r = item["log_row"]
            self._tree.insert("", tk.END, values=(
                r.get("outcome", ""),
                controller_name(item["path"], self.app.results_dir),
                r.get("account_id", ""),
                r.get("reading", ""),
                r.get("processed_at", "")[:16],
            ))

        total = len(self._items)
        self._counter_var.set(f"Всего для проверки: {total}")

    def _open(self):
        sel = self._tree.selection()
        if not sel:
            messagebox.showinfo("Выберите фото", "Сначала выберите фото из списка.")
            return
        idx = self._tree.index(sel[0])
        if idx < len(self._items):
            self.app.open_verify_screen(self._items[idx], idx)

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

# ─── Экран редактирования ────────────────────────────────────────────────────
class EditScreen(ttk.Frame):
    """
    Главный рабочий экран оператора.
    Слева — фото, справа — поля ввода и кнопки.
    """
    def __init__(self, parent: MainWindow, photo_path: str, reason: str, on_back):
        super().__init__(parent)
        self.app        = parent
        self.photo_path = photo_path
        self.folder     = controller_dir(photo_path)   # папка контролёра этого фото
        self.reason     = reason
        self._on_back   = on_back
        self._model_result: Optional[dict] = None
        self._tk_img: Optional[ImageTk.PhotoImage] = None
        self._confirmed_suspicious  = False
        self._last_edited: str = "serial"
        self._filling_other: bool = False
        self._model_serial:  Optional[str] = None
        self._model_reading: Optional[str] = None
        self._reading_manually_changed: bool = False

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
                img = self._orig_image.resize((disp_w, disp_h), Image.LANCZOS)
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
        self._serial_match_var.set("")
        for w in self._hint_frame.winfo_children():   # подсказка — только для ненайденного номера
            w.destroy()

        if self.app.df is None or not serial:
            self._update_accept_state()
            return

        cfg = self.app.config
        for candidate in [serial, "0" + serial, "00" + serial]:
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
        self._last_edited = "serial"
        model_serial = getattr(self, "_model_serial", None)
        if model_serial is not None and self._serial_var.get() != model_serial:
            self._serial_source = "Серийник: вручную"
        elif model_serial is not None and self._serial_var.get() == model_serial:
            self._serial_source = "Серийник: модель"
        else:
            self._serial_source = "Серийник: вручную"
        self._serial_match_var.set("")
        self._update_source_label()
        if hasattr(self, "_serial_lookup_job"):
            self.after_cancel(self._serial_lookup_job)
        self._serial_lookup_job = self.after(
            350, lambda: self._lookup_by_serial(self._serial_var.get())
        )

    def _on_account_change(self):
        if self._filling_other:
            return
        self._last_edited = "account"
        self._account_match_var.set("")
        if hasattr(self, "_account_lookup_job"):
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
        ok = (
            self._reading_widget.is_complete()
            and (self._serial_var.get().strip() or self._account_var.get().strip())
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

    def _accept(self):
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
        key = h or Path(self.photo_path).stem
        try:
            month = self.app.session.month_label()
            root = self.app.session.folder.markup
            save_crnn_markup(root, key, month, mr.get("serial_crop"), final_serial, mr.get("serial_text"))
            save_cnn_markup(root, key, month, mr.get("digit_crops"), mr.get("digit_preds"),
                            mr.get("reading_str"), final_reading_str)
            if self.reason == "no_meter":
                save_meter_markup(root, key, month, self._original_photo(row))
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

    def _not_in_db(self):
        serial  = self._serial_var.get().strip()
        account = self._account_var.get().strip()

        if not messagebox.askyesno(
            "Нет в базе",
            f"Счётчик не найден в базе абонентов.\n\n"
            f"Серийник: {serial or '—'}\n"
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

# ─── Экран проверки ──────────────────────────────────────────────────────────
class VerifyScreen(ttk.Frame):
    """Проверка фото из plus/ minus/ которые обработал reader.py автоматически."""
    def __init__(self, parent: MainWindow, item: dict, on_back, index: int = 0):
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
        self.app.bind_all("<Return>", lambda e: self._on_return())
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
            img = self._orig_image.copy()
            orig_w, orig_h = img.size
            base_scale = min(w / orig_w, h / orig_h)
            disp_w = max(1, int(orig_w * base_scale * self._zoom_scale))
            disp_h = max(1, int(orig_h * base_scale * self._zoom_scale))
            img = img.resize((disp_w, disp_h), Image.LANCZOS)
            self._tk_img = ImageTk.PhotoImage(img)
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

# ─── Точка входа ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    safe_console()
    app = MainWindow()
    app.mainloop()