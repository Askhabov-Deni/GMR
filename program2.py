"""
program2.py — ручная обработка фото из question/ которые reader.py не смог обработать.
АРХИТЕКТУРА:
App (Tk root)
├── SettingsDialog      — первый запуск / смена пользователя
├── LoginDialog         — подтверждение входа
├── MainWindow          — главное окно с двумя табами
│     ├── ProcessingTab — список из question/*, кнопки открыть / нечитаемо
│     └── VerifyTab     — список из plus/ minus/ где source=auto, verified_by=""
├── EditScreen          — редактирование одного фото (из processing tab)
└── VerifyScreen        — проверка одного фото (из verify tab)
ТИХАЯ РАЗМЕТКА (клерк не знает):
CRNN: сохраняем кроп serial_number + правильный текст в .txt файл рядом с картинкой
CNN:  сохраняем кропы только изменённых цифр, если показания исправили
"""
import json
import logging
import os
import queue
import shutil
import sys
import threading
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import pandas as pd
import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# ── Добавляем пути моделей ────────────────────────────────────────────────────
_BASE = Path(__file__).parent
sys.path.insert(0, str(_BASE / "models" / "crnn"))
sys.path.insert(0, str(_BASE / "models" / "cnn"))
from models.crnn.infer_crnn import CRNNInferer
from models.yolo_all_detect.infer_yolo import YOLOInferer
from models.cnn.infer_cnn import CNNInferer

# ── Импорт общих утилит из reader.py ──────────────────────────────────────────
sys.path.insert(0, str(_BASE))
from reader import (
    PipelineConfig,
    PhotoResult,
    Outcome,  # <-- ДОБАВЛЕНО для redraw_annotation
    _load_table, _save_table,
    _load_log, _save_log, _append_log_row, _log_path,
    _normalize_serial,
    _find_crop_entry,
    _read_meter_digits,
    _draw_annotation,
    _LOG_COLUMNS,
)
from src.gmr.domain import find_auto_row_for_output_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("program2")

# ─── Константы ────────────────────────────────────────────────────────────────
SETTINGS_FILE = Path(__file__).parent / "settings.json"
PHOTO_EXTS = {".jpg", ".jpeg", ".png"}

# Подпапки question/ которые обрабатываем (кроме repeat/)
QUESTION_SUBFOLDERS = [
    "digits_error",
    "suspicious",
    "serial_low_conf",
    "serial_not_found",
    "no_serial",
    "no_meter",
]

# Читаемые названия причин
REASON_LABELS = {
    "digits_error":      "Ошибка цифр",
    "suspicious":        "Подозрительно",
    "serial_low_conf":   "Серийник (низкая уверенность)",
    "serial_not_found":  "Серийник не найден",
    "no_serial":         "Нет серийника",
    "no_meter":          "Нет счётчика",
}

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
    photos_dir:    str = ""   # корень: рядом лежат question/, plus/, minus/, ...
    table_path:    str = ""
    training_dir:  str = ""

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
    """Первый запуск или смена пользователя."""
    def __init__(self, parent, settings: AppSettings):
        super().__init__(parent)
        self.title("Настройки")
        self.resizable(False, False)
        self.grab_set()
        self.result: Optional[AppSettings] = None
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

        fields = [
            ("Ваше имя:",            "name",     None),
            ("Папка с фото:",        "photos",    "dir"),
            ("Таблица (CSV/Excel):",  "table",    "file"),
            ("Папка для разметки:",  "training",  "dir"),
        ]

        self._vars = {}
        for i, (label, key, mode) in enumerate(fields, start=1):
            ttk.Label(f, text=label).grid(row=i, column=0, sticky="w", **pad)
            var = tk.StringVar(value=getattr(s, {
                "name":  "operator_name",  "photos":  "photos_dir",
                "table":  "table_path",   "training":  "training_dir"
            }[key]))
            self._vars[key] = var
            entry = ttk.Entry(f, textvariable=var, width=42)
            entry.grid(row=i, column=1, **pad)
            if mode:
                ttk.Button(f, text="…", width=3,
                           command=lambda v=var, m=mode: self._browse(v, m)
                           ).grid(row=i, column=2, padx=(0, 12))

        btns = ttk.Frame(f)
        btns.grid(row=10, column=0, columnspan=3, pady=(16, 0))
        ttk.Button(btns, text="Сохранить", command=self._save).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Отмена",    command=self._cancel).pack(side=tk.LEFT, padx=6)

    def _browse(self, var: tk.StringVar, mode: str):
        if mode == "dir":
            p = filedialog.askdirectory(title="Выберите папку")
        else:
            p = filedialog.askopenfilename(
                title="Выберите таблицу",
                filetypes=[("Таблицы", "*.csv *.xlsx *.xls"), ("Все файлы", "*.*")]
            )
        if p:
            var.set(p)

    def _save(self):
        name = self._vars["name"].get().strip()
        if not name:
            messagebox.showwarning("Ошибка", "Введите имя оператора", parent=self)
            return
        self.result = AppSettings(
            operator_name=name,
            photos_dir=self._vars["photos"].get().strip(),
            table_path=self._vars["table"].get().strip(),
            training_dir=self._vars["training"].get().strip(),
        )
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
        ttk.Label(f, text=f"Вы вошли как:", font=("Segoe UI", 10)).pack()
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Загружаем модели на {device}...")
        self.meter_detector = YOLOInferer(
            _abs_model(config.meter_detect_model), conf_thresh=config.meter_conf_thresh
        )
        self.digit_detector = YOLOInferer(
            _abs_model(config.digit_detect_model),
            conf_thresh=config.digit_detect_conf_thresh,
            straighten=False,
        )
        self.digit_ocr  = CNNInferer(_abs_model(config.digit_ocr_model), device=device)
        self.serial_ocr = CRNNInferer(_abs_model(config.serial_ocr_model), device=device)
        log.info("Модели загружены ✓")

    def run_on_photo(self, photo_path: str) -> dict:
        """
        Запускает все модели на фото. Возвращает dict с результатами.
        Вызывается из фонового потока.
        """
        cfg = self.config
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
            crops = self.meter_detector.process_image(photo_path, save_crops=False)
            if not crops:
                result["error"] = "YOLO: ничего не найдено"
                return result

            meter_entry  = _find_crop_entry(crops, "gas_meter")
            serial_entry = _find_crop_entry(crops, "serial_number")

            # Серийник
            if serial_entry is not None:
                result["serial_crop"] = serial_entry["crop"]
                sr = self.serial_ocr.predict_with_details(serial_entry["crop"])
                result["serial_text"] = sr["text"]
                result["serial_conf"] = sr["avg_confidence"]

            # Цифры
            if meter_entry is not None:
                meter_crop = meter_entry["crop"]
                reading, reading_str, err, digit_results, digit_bboxes = _read_meter_digits(
                    self.digit_detector, self.digit_ocr,
                    meter_crop,
                    cfg.digit_conf_thresh,
                    cfg.expected_digits,
                    ignore_last_digits=cfg.ignore_last_digits,
                    missing_placeholder=cfg.missing_digit_placeholder,
                    forgiven_placeholder=cfg.forgiven_digit_placeholder,
                )
                result["reading_str"]  = reading_str
                result["digit_preds"]  = digit_results

                # Сохраняем кропы цифр для разметки
                if digit_bboxes and digit_results:
                    crops_by_pos = []
                    for bbox in digit_bboxes:
                        if bbox is None:
                            crops_by_pos.append(None)
                        else:
                            x1, y1, x2, y2 = bbox
                            crops_by_pos.append(meter_crop[y1:y2, x1:x2].copy())
                    result["digit_crops"] = crops_by_pos

        except Exception as e:
            result["error"] = str(e)
            log.exception("Ошибка в run_on_photo")

        return result

# ─── Утилиты ──────────────────────────────────────────────────────────────────
def list_question_photos(photos_dir: str) -> list[tuple[str, str, str]]:
    """
    Возвращает список (reason, filename, full_path) из question/ подпапок.
    Сортировка по приоритету QUESTION_SUBFOLDERS.
    """
    result = []
    base = Path(photos_dir) / "question"
    for subfolder in QUESTION_SUBFOLDERS:
        sub = base / subfolder
        if not sub.exists():
            continue
        for p in sorted(sub.iterdir()):
            if p.is_file() and p.suffix.lower() in PHOTO_EXTS:
                result.append((subfolder, p.name, str(p)))
    return result

def list_verify_photos(photos_dir: str, log_rows: list[dict]) -> list[dict]:
    """
    Возвращает фото из plus/ и minus/ где source=auto и verified_by="".
    Ищет совпадение по original_filename ИЛИ final_filename.
    """
    # Строим два индекса: по оригинальному и по финальному имени
    log_by_original: dict[str, dict] = {}
    log_by_final: dict[str, dict] = {}
    for r in log_rows:
        orig = r.get("original_filename", "")
        final = r.get("final_filename", "")
        if orig:
            log_by_original[orig] = r
            # Также индексируем без расширения
            log_by_original[Path(orig).stem] = r
        if final:
            log_by_final[final] = r
            log_by_final[Path(final).stem] = r

    result = []
    for folder in ("plus", "minus"):
        d = Path(photos_dir) / folder
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if not (p.is_file() and p.suffix.lower() in PHOTO_EXTS):
                continue
            # Ищем запись в логе: сначала по имени файла, потом по stem
            row = (
                log_by_final.get(p.name)
                or log_by_final.get(p.stem)
                or log_by_original.get(p.name)
                or log_by_original.get(p.stem)
            )
            if row and row.get("source") == "auto" and not row.get("verified_by"):
                result.append({"path": str(p), "log_row": row, "folder": folder})
    return result

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
        img = cv2.imread(path)
        if img is None:
            return
            
        # ИСПРАВЛЕНО: передаём обязательные аргументы photo_path и outcome
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
        cv2.imwrite(path, img)
        log.info(f"Аннотация перерисована: {Path(path).name}")
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
def save_crnn_markup(
    training_dir: str,
    photo_path: str,
    serial_crop: Optional[np.ndarray],
    correct_text: str,
    model_text: Optional[str],
) -> None:
    """Сохраняем разметку CRNN только если серийник исправили."""
    if serial_crop is None:
        return
    if model_text is not None and correct_text == model_text:
        return
    out_dir = Path(training_dir) / "crnn" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_text = "".join(c for c in correct_text if c.isalnum() or c in "-_")
    if not safe_text:
        safe_text = Path(photo_path).stem
    out_fname = f"{safe_text}.jpeg"
    
    out_img = out_dir / out_fname
    if out_img.exists():
        out_img = out_dir / f"{safe_text}_{uuid.uuid4().hex[:6]}.jpeg"
    cv2.imwrite(str(out_img), serial_crop)

    out_txt = out_img.with_suffix(".txt")
    out_txt.write_text(correct_text, encoding="utf-8")
    log.info(f"CRNN разметка: {out_img.name} → '{correct_text}'")

def save_cnn_markup(
    training_dir: str,
    digit_crops: Optional[list],
    digit_preds: Optional[list],
    model_reading_str: Optional[str],
    final_reading_str: str,
) -> None:
    """Сохраняем кропы только изменённых цифр."""
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

        out_dir = Path(training_dir) / "cnn" / final_char
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{uuid.uuid4().hex}.jpg")
        cv2.imwrite(out_path, crop)
        log.info(f"CNN разметка: pos={pos} model={model_char!r} → correct={final_char!r}")

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
            # ИСПРАВЛЕНО: убраны пробелы внутри скобок
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
        self.df: Optional[pd.DataFrame] = None
        self.log_rows:  list[dict] = []
        self.models:    Optional[ModelBundle] = None
        self.config      = PipelineConfig()

        self._do_login()

    def _do_login(self):
        if not self.settings.operator_name or not self.settings.photos_dir:
            self._open_settings(first_run=True)
            if not self.settings.operator_name:
                self.destroy()
                return
        else:
            dlg = LoginDialog(self, self.settings)
            if dlg.action == "change":
                self._open_settings()
                if not self.settings.operator_name:
                    self.destroy()
                    return

        self._load_data()
        self._load_models_async()
        self._build_ui()

    def _open_settings(self, first_run=False):
        dlg = SettingsDialog(self, self.settings)
        if dlg.result:
            self.settings = dlg.result
            save_settings(self.settings)

    def _load_data(self):
        if self.settings.table_path and Path(self.settings.table_path).exists():
            try:
                self.df = _load_table(self.settings.table_path)
                log.info(f"Таблица загружена: {len(self.df)} строк")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить таблицу:\n{e}")
                self.df = None

        log_p = _log_path(self.settings.table_path) if self.settings.table_path else None
        if log_p and Path(log_p).exists():
            self.log_rows = _load_log(log_p)
            log.info(f"Лог загружен: {len(self.log_rows)} записей")

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

    def _build_ui(self):
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._proc_tab = ProcessingTab(self._notebook, self)
        self._verify_tab = VerifyTab(self._notebook, self)

        self._notebook.add(self._proc_tab,   text="  Обработка  ")
        self._notebook.add(self._verify_tab, text="  Проверка  ")

    def refresh_tabs(self):
        self._proc_tab.refresh()
        self._verify_tab.refresh()

    def append_log(self, row: dict):
        if row.get("source") == "manual" and not row.get("photo_hash"):
            # Отпечаток исходного фото — из автоматической строки, которая
            # создала этот файл (файл в question/ пересохранён с аннотацией,
            # по нему отпечаток не посчитать). Нужен, чтобы reader.py узнал
            # разобранное фото по содержимому, а не только по имени.
            src = find_auto_row_for_output_file(
                self.log_rows, row.get("original_filename", ""),
                Path(self.settings.photos_dir).name,
            )
            if src is not None:
                row["photo_hash"] = src.get("photo_hash", "")
                row["source_folder"] = src.get("source_folder", "")
        self.log_rows.append(row)
        log_p = _log_path(self.settings.table_path)
        _append_log_row(log_p, row)

    def save_table(self):
        if self.df is not None and self.settings.table_path:
            try:
                _save_table(self.df, self.settings.table_path)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить таблицу:\n{e}")

    def open_edit_screen(self, photo_path: str, reason: str):
        self._notebook.pack_forget()
        screen = EditScreen(self, photo_path, reason, on_back=self._back_from_screen)
        screen.pack(fill=tk.BOTH, expand=True)
        self._current_screen = screen

    def open_verify_screen(self, item: dict):
        self._notebook.pack_forget()
        screen = VerifyScreen(self, item, on_back=self._back_from_screen)
        screen.pack(fill=tk.BOTH, expand=True)
        self._current_screen = screen

    def _back_from_screen(self):
        if hasattr(self, "_current_screen"):
            self._current_screen.pack_forget()
            self._current_screen.destroy()
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.refresh_tabs()

    def open_next_or_back(self, current_path: str):
        photos = list_question_photos(self.settings.photos_dir)

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
        ttk.Label(self, textvariable=self._counter_var, font=("Segoe UI", 10)).pack(
            anchor="w", padx=12, pady=(10, 4)
        )

        cols = ("reason", "filename")
        self._tree = ttk.Treeview(self, columns=cols, show="headings", selectmode="browse")
        self._tree.heading("reason",   text="Причина")
        self._tree.heading("filename", text="Имя файла")
        self._tree.column("reason",   width=240, stretch=False)
        self._tree.column("filename", width=400)

        vsb = ttk.Scrollbar(self, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)

        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0), pady=(0, 12))
        vsb.pack(side=tk.LEFT, fill=tk.Y, pady=(0, 12))

        btn_frame = ttk.Frame(self)
        btn_frame.pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=12)

        ttk.Button(btn_frame, text="✎  Открыть",    command=self._open,       width=18).pack(pady=6)
        ttk.Button(btn_frame, text="✗  Нечитаемо",  command=self._unreadable, width=18).pack(pady=6)
        ttk.Button(btn_frame, text="⟳  Обновить",   command=self.refresh,     width=18).pack(pady=6)

        # ИСПРАВЛЕНО: убраны пробелы
        self._tree.bind("<Double-Button-1>", lambda e: self._open())

    def refresh(self):
        self._tree.delete(*self._tree.get_children())
        photos = list_question_photos(self.app.settings.photos_dir)
        for reason, fname, path in photos:
            label = REASON_LABELS.get(reason, reason)
            self._tree.insert("", tk.END, values=(label, fname), tags=(path,))

        remaining = len(photos)
        processed = sum(
            1 for r in self.app.log_rows
            if r.get("source") == "manual"
            and r.get("outcome") not in ("", None)
        )
        total = remaining + processed
        self._counter_var.set(
            f"Всего: {total} | Обработано: {processed} | Осталось: {remaining}"
        )

    def _selected_path(self) -> Optional[tuple[str, str]]:
        sel = self._tree.selection()
        if not sel:
            messagebox.showinfo("Выберите фото", "Сначала выберите фото из списка.")
            return None
        item = self._tree.item(sel[0])
        fname = item["values"][1]
        photos = list_question_photos(self.app.settings.photos_dir)
        for reason, fn, path in photos:
            if fn == fname:
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
        self.app.append_log(row)
        dst_dir = str(Path(self.app.settings.photos_dir) / "unreadable")
        try:
            move_photo(path, dst_dir)
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
        ttk.Label(self, textvariable=self._counter_var, font=("Segoe UI", 10)).pack(
            anchor="w", padx=12, pady=(10, 4)
        )

        cols = ("outcome", "account_id", "reading", "processed_at")
        self._tree = ttk.Treeview(self, columns=cols, show="headings", selectmode="browse")
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

        # ИСПРАВЛЕНО: убраны пробелы
        self._tree.bind("<Double-Button-1>", lambda e: self._open())
        self._items: list[dict] = []

    def refresh(self):
        self._tree.delete(*self._tree.get_children())
        self._items = list_verify_photos(self.app.settings.photos_dir, self.app.log_rows)
        for item in self._items:
            r = item["log_row"]
            self._tree.insert("", tk.END, values=(
                r.get("outcome", ""),
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
            self.app.open_verify_screen(self._items[idx])

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
        
        # ИСПРАВЛЕНО: убраны пробелы
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
            bottom, text="→  Пропустить  [Esc]",
            bg=CLR_GRAY, fg="white", font=("Segoe UI", 10),
            relief="flat", padx=12, pady=8,
            command=self._back,
        ).pack(side=tk.RIGHT, padx=4)

    def _bind_keys(self):
        print("HOTKEYS BOUND")
        # ИСПРАВЛЕНО: убраны пробелы
        for seq, cb in [
            ("<Return>",   self._hotkey_accept),
            ("<KP_Enter>", self._hotkey_accept),
            ("<Escape>",   self._hotkey_back),
            ("<Delete>",   self._hotkey_unreadable),
            ("<d>",        self._hotkey_duplicate),
            ("<D>",        self._hotkey_duplicate),
            ("<n>",        self._hotkey_not_in_db),
            ("<N>",        self._hotkey_not_in_db),
        ]:
            self.app.bind_all(seq, cb)

    def _unbind_keys(self):
        # ИСПРАВЛЕНО: убраны пробелы
        for seq in ("<Return>", "<KP_Enter>", "<Escape>", "<Delete>",
                    "<d>", "<D>", "<n>", "<N>"):
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
        print("ENTER PRESSED")
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
        self._update_delta()
        self._update_accept_state()

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

        ext      = Path(self.photo_path).suffix
        new_name = f"{account}{ext}" if account else Path(self.photo_path).name

        if self.app.df is not None and account:
            cfg  = self.app.config
            mask = self.app.df[cfg.col_account_id].apply(
                lambda x: str(x).strip() == account
            )
            if mask.any():
                self.app.df.loc[mask, cfg.col_new_reading] = str(reading)
                self.app.save_table()

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
            "notes":             self.reason,
        })
        self.app.append_log(row)

        self._save_markup_silent(serial, reading_s, mr)

        folder = "minus" if outcome_str == "MINUS" else "plus"
        dst_dir = str(Path(self.app.settings.photos_dir) / folder)
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

    def _save_markup_silent(self, final_serial: str, final_reading_str: str, mr: dict):
        td = self.app.settings.training_dir
        if not td:
            return
        try:
            save_crnn_markup(
                td,
                self.photo_path,
                mr.get("serial_crop"),
                final_serial,
                mr.get("serial_text"),
            )
            save_cnn_markup(
                td,
                mr.get("digit_crops"),
                mr.get("digit_preds"),
                mr.get("reading_str"),
                final_reading_str,
            )
        except Exception as e:
            log.warning(f"Ошибка сохранения разметки: {e}")

    def _duplicate(self):
        if not messagebox.askyesno("Дубль", "Пометить как дубль?", parent=self.app):
            return
        dst_dir = str(Path(self.app.settings.photos_dir) / "repeat")
        row = {k: "" for k in _LOG_COLUMNS}
        row.update({
            "original_filename": Path(self.photo_path).name,
            "outcome":            "REPEAT",
            "source":            "manual",
            "processed_by":      self.app.settings.operator_name,
            "processed_at":      now_iso(),
        })
        self.app.append_log(row)
        try:
            move_photo(self.photo_path, dst_dir)
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
        dst_dir = str(Path(self.app.settings.photos_dir) / "unreadable")
        row = {k: "" for k in _LOG_COLUMNS}
        row.update({
            "original_filename": Path(self.photo_path).name,
            "outcome":            "UNREADABLE",
            "source":            "manual",
            "processed_by":      self.app.settings.operator_name,
            "processed_at":      now_iso(),
        })
        self.app.append_log(row)
        try:
            move_photo(self.photo_path, dst_dir)
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

        dst_dir = str(Path(self.app.settings.photos_dir) / "not_in_db")
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
        self.app.append_log(row)
        try:
            move_photo(self.photo_path, dst_dir)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return
        try:
            self.app.open_next_or_back(self.photo_path)
        finally:
            self._unbind_keys()

# ─── Экран проверки ──────────────────────────────────────────────────────────
class VerifyScreen(ttk.Frame):
    """Проверка фото из plus/ minus/ которые обработал reader.py автоматически."""
    def __init__(self, parent: MainWindow, item: dict, on_back):
        super().__init__(parent)
        self.app      = parent
        self.item     = item
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
        
        # ИСПРАВЛЕНО: убраны пробелы
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

        tk.Button(
            bottom, text="✓  Верно  [Enter]",
            bg=CLR_GREEN, fg="white", font=("Segoe UI", 11, "bold"),
            relief="flat", padx=16, pady=8,
            command=self._verify_ok,
        ).pack(side=tk.LEFT, padx=(0, 8))

        self._edit_btn = tk.Button(
            bottom, text="✎  Исправить  [E]",
            bg="#1565C0", fg="white", font=("Segoe UI", 10),
            relief="flat", padx=12, pady=8,
            command=self._enter_edit,
        )
        self._edit_btn.pack(side=tk.LEFT, padx=4)

        self._save_edit_btn = tk.Button(
            bottom, text="💾  Сохранить правку",
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
        # ИСПРАВЛЕНО: убраны пробелы
        self.app.bind_all("<Return>", lambda e: self._verify_ok())
        self.app.bind_all("<Escape>", lambda e: self._back())
        self.app.bind_all("<e>",      lambda e: self._enter_edit())
        self.app.bind_all("<E>",      lambda e: self._enter_edit())
        self.app.bind_all("<Right>",  lambda e: self._skip())

    def _unbind_keys(self):
        # ИСПРАВЛЕНО: убраны пробелы
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

    def _skip(self):
        try:
            self._on_back()
        finally:
            self._unbind_keys()

    def _verify_ok(self):
        r = self.item["log_row"]
        for i, row in enumerate(self.app.log_rows):
            if row.get("original_filename") == r.get("original_filename"):
                self.app.log_rows[i]["verified_by"] = self.app.settings.operator_name
                self.app.log_rows[i]["verified_at"] = now_iso()
                break

        log_p = _log_path(self.app.settings.table_path)
        _save_log(log_p, self.app.log_rows)

        try:
            self._on_back()
        finally:
            self._unbind_keys()

    def _enter_edit(self):
        if self._edit_mode:
            return
        self._edit_mode = True
        self._edit_frame.pack(fill=tk.X, pady=(12, 0))
        self._edit_btn.pack_forget()
        self._save_edit_btn.pack(side=tk.LEFT, padx=4)

    def _save_edit(self):
        r        = self.item["log_row"]
        new_ser  = self._serial_var.get().strip()
        new_read = self._reading_widget.get_string()
        if not self._reading_widget.is_complete():
            messagebox.showwarning("Ошибка", "Введите все 5 цифр показаний.", parent=self.app)
            return

        for i, row in enumerate(self.app.log_rows):
            if row.get("original_filename") == r.get("original_filename"):
                self.app.log_rows[i]["serial_id"]   = new_ser
                self.app.log_rows[i]["reading"]      = new_read
                self.app.log_rows[i]["verified_by"]  = self.app.settings.operator_name
                self.app.log_rows[i]["verified_at"]  = now_iso()
                notes = self.app.log_rows[i].get("notes", "")
                self.app.log_rows[i]["notes"] = (notes + " | исправлено при проверке").strip(" |")
                break

        _save_log(_log_path(self.app.settings.table_path), self.app.log_rows)

        try:
            self._on_back()
        finally:
            self._unbind_keys()

# ─── Точка входа ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()