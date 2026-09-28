"""
reader.py — главный пайплайн обработки фотографий газовых счётчиков.

АРХИТЕКТУРА:
  process_photo(photo_path, df, config) -> PhotoResult
    │
    ├── YOLOInferer (meter_detector)    → кроп gas_meter + кроп serial_numbers
    ├── CRNNInferer (serial_ocr)        → текст серийного номера
    ├── lookup_in_table(serial, df)     → строка таблицы (лицевой ID, последние показания)
    ├── YOLOInferer (digit_detector)    → bbox-ы цифр на кропе счётчика
    ├── CNNInferer  (digit_ocr)         → класс каждой цифры → собираем число
    └── decide_outcome(...)             → категория + действие (запись / папка)

ПАПКИ-РЕЗУЛЬТАТЫ:
  plus/     — показания выросли, данные записаны
  minus/    — показания упали (< 0), данные записаны
  repeat/   — счётчик уже обработан (дубль)
  question/
    no_meter/        — YOLO не нашёл gas_meter
    no_serial/       — YOLO не нашёл serial_numbers
    serial_low_conf/ — CRNN уверенность ниже порога
    serial_not_found/— серийник не найден в таблице
    digits_error/    — не удалось прочитать 5 цифр
    suspicious/      — отклонение > DELTA_THRESHOLD

DEBUG-РЕЖИМ (config.debug_digits = True):
  question/digits_error/<account_id или photo_stem>/
    digit_0.jpg … digit_N.jpg   — кроп каждой найденной цифры
    meta.json                   — предсказание и конфиданс по каждой позиции
"""

import os
import csv
import json
import shutil
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np
import torch
import pandas as pd

from models.crnn.infer_crnn import CRNNInferer
from models.yolo_all_detect.infer_yolo import YOLOInferer
from models.cnn.infer_cnn import CNNInferer


# ─── Конфиг ──────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # Пути к моделям
    meter_detect_model:       str   = "meter_detect/runs/detect/gas_meter_all_classes_s_v1/weights/best.pt"
    digit_detect_model:       str   = "meter_ocr/runs/yolo/digits_detect_v4/weights/best.pt"
    digit_ocr_model:          str   = "meter_ocr/runs/cnn/runs/v2_gold/best.pth"
    serial_ocr_model:         str   = "serial_id_ocr/runs/crnn/2026-06-05_01-09/best.pt"

    # Пути к данным.
    # Раньше здесь были захардкожены личные Windows-пути оператора (D:\...) —
    # это делало "чистую установку" на другой машине невозможной (см.
    # docs/MIGRATION_TZ.md §0). Теперь дефолты — безопасные относительные
    # пути внутри репозитория; реальные пути задаются через переменные
    # окружения (см. .env.example) либо передаются явно при создании
    # PipelineConfig(...).
    input_dir:                str   = field(default_factory=lambda: os.environ.get("GMR_INPUT_DIR", "data/input"))
    output_base_dir:          str   = field(default_factory=lambda: os.environ.get("GMR_OUTPUT_DIR", "data/output"))
    table_path:               str   = field(default_factory=lambda: os.environ.get("GMR_TABLE_PATH", "data/meters_table.csv"))

    # Имена столбцов в таблице
    col_serial:               str   = "Номер счетчика"
    col_account_id:           str   = "Лицевой счет"
    col_last_reading:         str   = "Последние показания"
    col_new_reading:          str   = "Текущие показания"

    # Пороги
    meter_conf_thresh:        float = 0.7
    digit_detect_conf_thresh: float = 0.7
    serial_conf_thresh:       float = 0.6
    digit_conf_thresh:        float = 0.6
    expected_digits:          int   = 5
    delta_threshold:          float = 10_000.0

    # Прочее
    move_photos:              bool  = False  # True=перемещать, False=копировать

    # Заглушки для цифр.
    # missing_digit_placeholder — для восстановленной (пропущенной детектором) позиции.
    #   Детектор иногда пропускает одну цифру посередине — её позиция вычисляется
    #   геометрически, а значение подставляется как placeholder.
    #   '5' — нейтральное среднее для цифры около середины шкалы.
    missing_digit_placeholder:  str = "5"

    # forgiven_digit_placeholder — для прощённых позиций (ignore_last_digits).
    #   Последние цифры часто не имеют веса на счётчике — при низком конфидансе
    #   подставляем '0' (минимальное влияние на итоговое число).
    forgiven_digit_placeholder: str = "0"

    # Режим мягкого чтения: игнорировать ошибки на последних N цифрах счётчика.
    # Последние цифры (десятые/сотые доли) часто не имеют веса на счётчике —
    # при низком конфидансе туда подставляется forgiven_digit_placeholder.
    #   0 — выключено (строгий режим)
    #   1 — игнорируем последнюю цифру
    #   2 — игнорируем последние две  ← рекомендуется
    #   3 — последние три
    #   5 — все цифры (фактически всегда успех, осторожно)
    # Несовместимо с debug_digits по смыслу: при ignore_last_digits > 0
    # ошибки на хвостовых позициях не попадают в digits_error и не будут продебажены.
    ignore_last_digits:         int = 2

    # Дебаг CNN: сохранять кропы цифр и meta.json при ошибке распознавания
    debug_digits:              bool = False

    # Отрисовка боксов на итоговом фото.
    # gas_meter    — зелёный
    # serial_number — синий
    # digit         — оранжевый
    # Подписи классов не рисуются.
    draw_boxes:                bool = True


# ─── Цвета боксов ────────────────────────────────────────────────────────────

_BOX_COLORS = {
    "gas_meter":     (0,   200,  0),    # зелёный
    "serial_number": (200,  0,   0),    # синий (BGR)
    "digit":         (0,   140, 255),   # оранжевый (BGR)
}
_BOX_THICKNESS = 2


# ─── Типы результата ──────────────────────────────────────────────────────────

class Outcome(Enum):
    PLUS             = auto()
    MINUS            = auto()
    REPEAT           = auto()
    NO_METER         = auto()
    NO_SERIAL        = auto()
    SERIAL_LOW_CONF  = auto()
    SERIAL_NOT_FOUND = auto()
    DIGITS_ERROR     = auto()
    SUSPICIOUS       = auto()


OUTCOME_FOLDER = {
    Outcome.PLUS:             "plus",
    Outcome.MINUS:            "minus",
    Outcome.REPEAT:           "repeat",
    Outcome.NO_METER:         "question/no_meter",
    Outcome.NO_SERIAL:        "question/no_serial",
    Outcome.SERIAL_LOW_CONF:  "question/serial_low_conf",
    Outcome.SERIAL_NOT_FOUND: "question/serial_not_found",
    Outcome.DIGITS_ERROR:     "question/digits_error",
    Outcome.SUSPICIOUS:       "question/suspicious",
}


@dataclass
class PhotoResult:
    photo_path:     str
    outcome:        Outcome
    serial_text:    Optional[str]   = None
    serial_conf:    Optional[float] = None
    account_id:     Optional[str]   = None
    reading:        Optional[int]   = None
    # ↓ Частично распознанные показания, например "5?3?1"
    #   заполняется даже при DIGITS_ERROR — для аннотации на фото
    reading_str:    Optional[str]   = None
    last_reading:   Optional[float] = None
    delta:          Optional[float] = None
    new_photo_name: Optional[str]   = None
    error_detail:   Optional[str]   = None
    # ↓ Данные для отрисовки боксов (заполняются в process_photo)
    # meter_crops_raw — список dict из meter_detector (gas_meter, serial_number)
    meter_crops_raw: Optional[list] = None
    # digit_bboxes_in_orig — боксы цифр в координатах оригинала [(x1,y1,x2,y2), ...]
    digit_bboxes_in_orig: Optional[list] = None


# ─── Нормализация серийного номера ───────────────────────────────────────────

def _normalize_serial(s: str) -> str:
    """
    Приводит серийный номер к каноническому виду для сравнения.

    Убирает только пробелы по краям — внутренние символы (включая ведущие
    нули) остаются нетронутыми. Намеренно НЕ вызываем int() / lstrip('0'),
    чтобы не потерять ведущие нули у номеров вроде "007123".
    """
    return s.strip()


# ─── Вспомогательные функции ─────────────────────────────────────────────────

def _load_table(path: str) -> pd.DataFrame:
    """
    Загружает таблицу счётчиков.

    ВАЖНО — ведущие нули:

    CSV:  dtype=str достаточно — pandas читает ячейки как текст.

    Excel: dtype=str НЕ спасает числовые ячейки. Если ячейка хранится
    как число (openpyxl отдаёт int), dtype=str превратит 42306 в "42306",
    а не в "0042306". Используем converters={col: str} — это заставляет
    pandas вызывать str() на сыром значении из openpyxl. Текстовые ячейки
    вернут строку как есть, числовые — str(int), т.е. без ведущих нулей.
    Полная защита от потери нулей обеспечивается fallback-ом с добавлением
    нулей в process_photo (шаг 3).
    """
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        _tmp = pd.read_excel(path, nrows=0)
        converters = {col: str for col in _tmp.columns}
        return pd.read_excel(path, converters=converters)
    return pd.read_csv(path, dtype=str)


def _save_table(df: pd.DataFrame, path: str) -> None:
    """
    Сохраняет таблицу обратно на диск.

    Для CSV принудительно используем quoting=QUOTE_ALL, чтобы ведущие нули
    в строковых столбцах не потерялись при следующей загрузке сторонними
    инструментами (Excel, pandas без dtype=str и т.д.).
    """
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)


# ─── Processing log ──────────────────────────────────────────────────────────

_LOG_COLUMNS = [
    "original_filename", "final_filename", "serial_id", "account_id",
    "reading", "last_reading", "delta", "outcome", "source", "processed_by",
    "processed_at", "verified_by", "verified_at", "model_serial_conf",
    "model_reading_str", "notes",
]


def _log_path(table_path: str) -> str:
    """<table_name>_log.csv рядом с таблицей."""
    p = Path(table_path)
    return str(p.parent / (p.stem + "_log.csv"))


def _load_log(log_path: str) -> list[dict]:
    """Загружает лог; возвращает [] если файл не существует."""
    if not Path(log_path).exists():
        return []
    with open(log_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save_log(log_path: str, rows: list[dict]) -> None:
    """Перезаписывает весь лог."""
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_LOG_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_log_row(log_path: str, row: dict) -> None:
    """Дописывает одну строку в лог (создаёт файл с заголовком если нет)."""
    exists = Path(log_path).exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_LOG_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _log_filenames(rows: list[dict]) -> set[str]:
    """Множество original_filename из лога (непустые)."""
    return {r["original_filename"] for r in rows if r.get("original_filename")}


def _maybe_init_log(
    table_path: str,
    log_path: str,
    df: "pd.DataFrame",
    config: "PipelineConfig",
) -> list[dict]:
    """
    При первом запуске: если лог не существует, но в таблице уже есть
    заполненные col_new_reading — предлагает инициализировать лог.
    Возвращает итоговые строки лога (может быть []).
    """
    if Path(log_path).exists():
        return _load_log(log_path)

    filled = df[df[config.col_new_reading].apply(
        lambda v: pd.notna(v) and str(v).strip() not in ("", "nan")
    )]

    if filled.empty:
        return []

    print(
        f"\nНайдены данные в таблице без лога ({len(filled)} строк).\n"
        "Инициализировать лог из таблицы? [Да/Нет]: ",
        end="", flush=True,
    )
    try:
        answer = input().strip().lower()
    except EOFError:
        answer = "нет"

    if answer not in ("да", "д", "y", "yes"):
        return []

    rows = []
    for _, row in filled.iterrows():
        rows.append({
            # Намеренно НЕ пустая строка: _log_filenames() фильтрует пустые,
            # поэтому при инициализации из таблицы ставим специальный маркер.
            # reader.py проверяет дубль по _log_filenames_cache (set строк),
            # а при повторном прогоне имя фото НЕ совпадёт с "__pre_existing__"
            # — это нормально: нас интересует только проверка _table_filled,
            # которую мы подавляем через флаг source="pre_existing" в notes.
            # Реальный эффект: _table_filled=True → SUSPICIOUS, но только для
            # строк без маркера. Маркер здесь нужен чтобы set был непустым
            # и не давал False-срабатываний.
            # → Правильное поведение достигается через отдельную проверку ниже.
            "original_filename": "__pre_existing__",
            "final_filename":    "",
            "serial_id":         str(row.get(config.col_serial, "")),
            "account_id":        str(row.get(config.col_account_id, "")),
            "reading":           str(row.get(config.col_new_reading, "")),
            "last_reading":      str(row.get(config.col_last_reading, "")),
            "delta":             "",
            "outcome":           "UNKNOWN",
            "source":            "pre_existing",
            "processed_by":      "unknown",
            "processed_at":      "",
            "verified_by":       "",
            "verified_at":       "",
            "model_serial_conf": "",
            "model_reading_str": "",
            "notes":             "инициализировано из таблицы",
        })

    _save_log(log_path, rows)
    logging.getLogger("reader").info(f"Лог инициализирован: {len(rows)} записей → {log_path}")
    return rows


def _make_log_row(
    original_filename: str,
    result: "PhotoResult",
    config: "PipelineConfig",
    source: str = "auto",
    processed_by: str = "auto",
) -> dict:
    """Строит dict для записи в лог по результату process_photo."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "original_filename": original_filename,
        "final_filename":    result.new_photo_name or "",
        "serial_id":         result.serial_text or "",
        "account_id":        result.account_id or "",
        "reading":           str(result.reading) if result.reading is not None else "",
        "last_reading":      str(result.last_reading) if result.last_reading is not None else "",
        "delta":             str(result.delta) if result.delta is not None else "",
        "outcome":           result.outcome.name,
        "source":            source,
        "processed_by":      processed_by,
        "processed_at":      ts,
        "verified_by":       "",
        "verified_at":       "",
        "model_serial_conf": f"{result.serial_conf:.4f}" if result.serial_conf is not None else "",
        "model_reading_str": result.reading_str or "",
        "notes":             result.error_detail or "",
    }


def _find_crop(crops: list[dict], class_name: str) -> Optional[np.ndarray]:
    for c in crops:
        if c["class"] == class_name:
            return c["crop"]
    return None


def _find_crop_entry(crops: list[dict], class_name: str) -> Optional[dict]:
    """Возвращает весь dict кропа (включая bbox), а не только crop."""
    for c in crops:
        if c["class"] == class_name:
            return c
    return None


def _draw_annotation(img: np.ndarray, result: PhotoResult) -> None:
    """Рисует серийник и показания в левом верхнем углу изображения (in-place).

    Показания отображаются как reading_str (например "5?3?1") если число не
    удалось распознать полностью, или как целое число при успехе.
    """
    if result.reading is not None:
        reading_display = str(result.reading)
    elif result.reading_str is not None:
        reading_display = result.reading_str   # например "5?3?1"
    else:
        reading_display = "?"

    lines = [
        f"Serial:  {result.serial_text or '?'}",
        f"Reading: {reading_display}",
    ]

    _, w = img.shape[:2]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, w / 1000)
    thickness  = 2
    pad        = 8
    line_h     = int(30 * font_scale)

    box_h = line_h * len(lines) + pad * 2
    box_w = int(340 * font_scale)
    cv2.rectangle(img, (0, 0), (box_w, box_h), (0, 0, 0), -1)

    for i, line in enumerate(lines):
        y = pad + line_h * i + line_h - 4
        cv2.putText(img, line, (pad, y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)


def _draw_boxes(img: np.ndarray, result: PhotoResult) -> None:
    """
    Рисует боксы детекций на изображении (in-place).

    gas_meter     — зелёный   — из meter_crops_raw
    serial_number — синий     — из meter_crops_raw
    digit         — оранжевый — из digit_bboxes_in_orig (уже в координатах оригинала)

    Подписи классов не рисуются.
    """
    # Боксы от meter_detector (gas_meter и serial_number)
    if result.meter_crops_raw:
        for entry in result.meter_crops_raw:
            cls   = entry["class"]
            color = _BOX_COLORS.get(cls, (128, 128, 128))
            x1, y1, x2, y2 = entry["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, _BOX_THICKNESS)

    # Боксы цифр (уже пересчитаны в координаты оригинала)
    if result.digit_bboxes_in_orig:
        color = _BOX_COLORS["digit"]
        for (x1, y1, x2, y2) in result.digit_bboxes_in_orig:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, _BOX_THICKNESS)


def _save_annotated(
    src: str,
    dst_dir: str,
    new_name: str,
    result: PhotoResult,
    move: bool,
    draw_boxes: bool = False,
) -> None:
    """
    Читает фото, рисует аннотацию (и боксы если draw_boxes=True), сохраняет в dst.
    Оригинал удаляет если move=True — только после успешной записи.
    """
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    dst_path = str(Path(dst_dir) / new_name)

    img = cv2.imread(src)
    if img is not None:
        if draw_boxes:
            _draw_boxes(img, result)
        _draw_annotation(img, result)
        written = cv2.imwrite(dst_path, img)
        if written:
            if move:
                os.remove(src)
        else:
            # imwrite провалился — fallback без аннотации
            if move:
                shutil.move(src, dst_path)
            else:
                shutil.copy2(src, dst_path)
    else:
        # Не смогли прочитать — просто переносим как есть
        if move:
            shutil.move(src, dst_path)
        else:
            shutil.copy2(src, dst_path)


def _save_digit_debug(
    debug_dir: str,
    digit_crops_sorted: list[Optional[dict]],
    digit_results: list[dict],
) -> None:
    """
    Сохраняет кропы цифр и meta.json в папку debug_dir.

    digit_results — список dict по каждой позиции:
      {
        "position":   int,           # индекс 0..N-1
        "digit":      str,           # предсказанная цифра или "?" если None-заглушка
        "confidence": float | None,  # None для заглушки
        "ok":         bool,          # прошёл ли порог конфиданса
      }

    Кроп сохраняется только если у позиции есть реальный crop (не None-заглушка).
    """
    Path(debug_dir).mkdir(parents=True, exist_ok=True)

    for pos, (dc, res) in enumerate(zip(digit_crops_sorted, digit_results)):
        if dc is not None:
            crop_path = str(Path(debug_dir) / f"digit_{pos}.jpg")
            cv2.imwrite(crop_path, dc["crop"])

    meta_path = str(Path(debug_dir) / "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(digit_results, f, ensure_ascii=False, indent=2)


# ─── Чтение цифр счётчика ────────────────────────────────────────────────────

def _read_meter_digits(
    digit_detector: YOLOInferer,
    digit_ocr: CNNInferer,
    meter_crop: np.ndarray,
    conf_thresh: float,
    expected_digits: int,
    ignore_last_digits: int = 0,
    missing_placeholder: str = "5",
    forgiven_placeholder: str = "0",
) -> tuple[Optional[int], Optional[str], Optional[str], Optional[list], Optional[list]]:
    """
    Прогоняет кроп счётчика через детектор цифр + CNN классификатор.

    Возвращает кортеж:
      (number, reading_str, error_msg, digit_results, digit_bboxes)

    - number:         int   если все значимые цифры распознаны успешно, иначе None
    - reading_str:    строка вида "56?76" — всегда заполняется при частичном/полном
                      успехе (? на месте нераспознанных позиций), None только если
                      детектор вернул ноль кропов
    - error_msg:      описание ошибки или None при успехе
    - digit_results:  список dict для каждой позиции (для debug), None при
                      фатальной ошибке до классификации
    - digit_bboxes:   список (x1,y1,x2,y2) в координатах кропа для каждой позиции
                      (None для заглушек), None при фатальной ошибке

    missing_placeholder  — символ/цифра для восстановленной пропущенной позиции
    forgiven_placeholder — символ/цифра для прощённых хвостовых позиций

    ignore_last_digits — сколько последних позиций прощаем при низком конфидансе.

    При ровно 4 найденных цифрах пытается восстановить пропущенную позицию
    (только если пропала 3-я, 4-я или 5-я цифра).
    """
    digit_crops = digit_detector.process_array(meter_crop, save_crops=False, max_per_class=5)

    if not digit_crops:
        return None, None, "digit detector found nothing", None, None

    digit_crops_sorted = sorted(digit_crops, key=lambda c: c["bbox"][0])
    n = len(digit_crops_sorted)

    if n == expected_digits:
        pass  # всё хорошо

    elif n == expected_digits - 1:
        centers = [(c["bbox"][0] + c["bbox"][2]) // 2 for c in digit_crops_sorted]
        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        avg_step = sum(gaps) / len(gaps)

        # Ищем gap значительно больше среднего — пропуск в середине
        missing_idx = None
        for i, gap in enumerate(gaps):
            if gap > avg_step * 1.6:
                missing_idx = i + 1
                break

        # Если gap не найден — пропала крайняя цифра
        if missing_idx is None:
            crop_w = meter_crop.shape[1]
            left_margin  = centers[0]
            right_margin = crop_w - centers[-1]

            if right_margin > left_margin * 1.6:
                missing_idx = expected_digits - 1   # пропала последняя
            elif left_margin > right_margin * 1.6:
                missing_idx = 0                      # пропала первая
            else:
                return None, None, f"expected {expected_digits} digits, got {n} (gap not found)", None, None

        if missing_idx < 2:
            return None, None, (
                f"expected {expected_digits} digits, got {n} "
                f"(missing digit at critical position {missing_idx})"
            ), None, None

        digit_crops_sorted.insert(missing_idx, None)  # None = заглушка

    else:
        return None, None, f"expected {expected_digits} digits, got {n}", None, None

    # ── Классифицируем цифры ──────────────────────────────────────────────────
    forgiven_positions = set(range(expected_digits - ignore_last_digits, expected_digits))

    digits        = []
    digit_results = []
    digit_bboxes  = []   # (x1,y1,x2,y2) в координатах кропа, None для заглушек
    has_error     = False

    for pos, dc in enumerate(digit_crops_sorted):
        if dc is None:
            # Заглушка восстановленной позиции
            digits.append(missing_placeholder)
            digit_results.append({
                "position":   pos,
                "digit":      missing_placeholder,
                "confidence": None,
                "ok":         True,
                "note":       "inserted placeholder",
            })
            digit_bboxes.append(None)
            continue

        pred = digit_ocr.predict(dc["crop"])
        conf = pred["confidence"]
        digit_char = str(pred["digit"])
        ok = conf >= conf_thresh

        if not ok and pos in forgiven_positions:
            digit_results.append({
                "position":   pos,
                "digit":      digit_char,
                "confidence": round(float(conf), 4),
                "ok":         False,
                "note":       f"forgiven→{forgiven_placeholder}",
            })
            digits.append(forgiven_placeholder)
        else:
            digit_results.append({
                "position":   pos,
                "digit":      digit_char,
                "confidence": round(float(conf), 4),
                "ok":         ok,
            })
            if ok:
                digits.append(digit_char)
            else:
                digits.append("?")
                has_error = True

        digit_bboxes.append(dc["bbox"])

    reading_str = "".join(digits)

    if has_error:
        bad = [
            f"pos{r['position']}(pred={r['digit']},conf={r['confidence']:.3f})"
            for r in digit_results
            if not r["ok"]
            and r.get("note", "").startswith("forgiven") is False
            and r.get("confidence") is not None
        ]
        error_msg = f"low conf digits: {', '.join(bad)}  →  '{reading_str}'"
        return None, reading_str, error_msg, digit_results, digit_bboxes

    number = int(reading_str)
    return number, reading_str, None, digit_results, digit_bboxes


def _digit_bboxes_to_orig(
    digit_bboxes: list[Optional[tuple]],
    meter_bbox: tuple[int, int, int, int],
    meter_crop_shape: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """
    Пересчитывает bbox-ы цифр из координат кропа в координаты оригинального фото.

    digit_bboxes      — [(x1,y1,x2,y2) в пикселях кропа | None для заглушек]
    meter_bbox        — (x1,y1,x2,y2) кропа счётчика в оригинале (от meter_detector)
    meter_crop_shape  — (h, w) кропа (после возможного выпрямления)

    Возвращает список только реальных bbox-ов (заглушки пропускаются).
    """
    orig_x1, orig_y1, orig_x2, orig_y2 = meter_bbox
    crop_h, crop_w = meter_crop_shape[:2]

    # Размер bbox счётчика в оригинале
    bbox_w = orig_x2 - orig_x1
    bbox_h = orig_y2 - orig_y1

    # Масштаб: кроп мог быть выпрямлен и изменить размер,
    # поэтому пересчитываем через соотношение сторон
    scale_x = bbox_w / crop_w
    scale_y = bbox_h / crop_h

    result = []
    for bbox in digit_bboxes:
        if bbox is None:
            continue
        dx1, dy1, dx2, dy2 = bbox
        x1 = orig_x1 + int(dx1 * scale_x)
        y1 = orig_y1 + int(dy1 * scale_y)
        x2 = orig_x1 + int(dx2 * scale_x)
        y2 = orig_y1 + int(dy2 * scale_y)
        result.append((x1, y1, x2, y2))
    return result


# ─── Основная функция обработки одного фото ──────────────────────────────────

def process_photo(
    photo_path: str,
    df: pd.DataFrame,
    config: PipelineConfig,
    meter_detector: YOLOInferer,
    digit_detector: YOLOInferer,
    digit_ocr: CNNInferer,
    serial_ocr: CRNNInferer,
) -> PhotoResult:
    """
    Обрабатывает одну фотографию. Возвращает PhotoResult.
    НЕ изменяет df и не трогает файлы — это делает caller (run_pipeline).
    """
    result = PhotoResult(photo_path=photo_path, outcome=Outcome.NO_METER)
    ext = Path(photo_path).suffix

    # ── Шаг 1: детекция трёх классов ─────────────────────────────────────────
    crops = meter_detector.process_image(photo_path, save_crops=False)
    if not crops:
        result.outcome = Outcome.NO_METER
        result.error_detail = "YOLO returned no crops"
        return result

    meter_entry  = _find_crop_entry(crops, "gas_meter")
    serial_entry = _find_crop_entry(crops, "serial_number")

    # Сохраняем все найденные детекции для отрисовки боксов
    result.meter_crops_raw = crops

    if meter_entry is None:
        result.outcome = Outcome.NO_METER
        result.error_detail = "class gas_meter not detected"
        return result

    if serial_entry is None:
        result.outcome = Outcome.NO_SERIAL
        result.error_detail = "class serial_number not detected"
        return result

    meter_crop = meter_entry["crop"]
    serial_crop = serial_entry["crop"]
    meter_bbox  = meter_entry["bbox"]   # (x1,y1,x2,y2) в оригинале

    # ── Шаг 2: читаем серийный номер ─────────────────────────────────────────
    serial_res  = serial_ocr.predict_with_details(serial_crop)
    serial_text = serial_res["text"]
    serial_conf = serial_res["avg_confidence"]

    result.serial_text = serial_text
    result.serial_conf = serial_conf

    if serial_conf < config.serial_conf_thresh:
        result.outcome = Outcome.SERIAL_LOW_CONF
        result.error_detail = f"serial conf={serial_conf:.3f} < {config.serial_conf_thresh}"
        # Читаем цифры для информативной аннотации
        _r, _rs, _, _, _bboxes = _read_meter_digits(
            digit_detector, digit_ocr,
            meter_crop,
            config.digit_conf_thresh,
            config.expected_digits,
            ignore_last_digits=config.ignore_last_digits,
            missing_placeholder=config.missing_digit_placeholder,
            forgiven_placeholder=config.forgiven_digit_placeholder,
        )
        result.reading_str = _rs
        result.reading     = _r
        if _bboxes is not None:
            result.digit_bboxes_in_orig = _digit_bboxes_to_orig(
                _bboxes, meter_bbox, meter_crop.shape
            )
        return result

    # ── Шаг 3: ищем в таблице ────────────────────────────────────────────────
    def _find_serial_in_df(serial: str) -> pd.DataFrame:
        norm = _normalize_serial(serial)
        return df[df[config.col_serial].apply(
            lambda x: _normalize_serial(str(x)) == norm
        )]

    serial_candidates = [
        serial_text,
        "0"  + serial_text,
        "00" + serial_text,
    ]

    matches     = pd.DataFrame()
    serial_used = serial_text
    for candidate in serial_candidates:
        matches = _find_serial_in_df(candidate)
        if not matches.empty:
            serial_used = candidate
            break

    if not matches.empty and serial_used != serial_text:
        logging.getLogger("reader").info(
            f"  ℹ️  serial fallback: '{serial_text}' → '{serial_used}' (добавлены ведущие нули)"
        )

    if matches.empty:
        result.outcome = Outcome.SERIAL_NOT_FOUND
        result.error_detail = (
            f"serial '{serial_text}' not in table "
            f"(tried: {', '.join(repr(c) for c in serial_candidates)})"
        )
        _r, _rs, _, _, _bboxes = _read_meter_digits(
            digit_detector, digit_ocr,
            meter_crop,
            config.digit_conf_thresh,
            config.expected_digits,
            ignore_last_digits=config.ignore_last_digits,
            missing_placeholder=config.missing_digit_placeholder,
            forgiven_placeholder=config.forgiven_digit_placeholder,
        )
        result.reading_str = _rs
        result.reading     = _r
        if _bboxes is not None:
            result.digit_bboxes_in_orig = _digit_bboxes_to_orig(
                _bboxes, meter_bbox, meter_crop.shape
            )
        return result

    result.serial_text = serial_used

    row_idx    = matches.index[0]
    account_id = str(df.at[row_idx, config.col_account_id])
    last_val   = df.at[row_idx, config.col_last_reading]

    result.account_id = account_id

    try:
        last_reading = (
            float(str(last_val).replace(",", "."))
            if pd.notna(last_val) and str(last_val).strip() != ""
            else None
        )
    except ValueError:
        last_reading = None
    result.last_reading = last_reading

    # ── Шаг 4: проверяем дубль ───────────────────────────────────────────────
    # Дубль определяем по логу, а не по col_new_reading.
    #
    # Кейс «инициализация из таблицы»:
    #   _maybe_init_log записывает строки с original_filename="__pre_existing__"
    #   и source="pre_existing". Такие строки означают: счётчик уже обработан
    #   до нашего прогона (данные в таблице были заранее). Это тоже дубль.
    #   Ищем их по совпадению account_id в логе с source=pre_existing.
    _orig_fname   = Path(photo_path).name
    _log_fnames   = getattr(config, "_log_filenames_cache", set())
    _log_rows     = getattr(config, "_log_rows_cache", [])
    _new_val      = df.at[row_idx, config.col_new_reading]
    _table_filled = pd.notna(_new_val) and str(_new_val).strip() not in ("", "nan")

    if _orig_fname in _log_fnames:
        # Файл уже есть в логе — настоящий дубль
        result.outcome = Outcome.REPEAT
        result.error_detail = f"already in log: {_orig_fname}"
        result.new_photo_name = f"{account_id}{ext}"
        return result

    _processed_accounts = getattr(config, "_processed_accounts_cache", set())
    if account_id in _processed_accounts:
        # Счётчик уже обработан в этом прогоне — два фото одного счётчика
        result.outcome = Outcome.REPEAT
        result.error_detail = f"duplicate in current run: account={account_id}"
        result.new_photo_name = f"{account_id}{ext}"
        return result

    if _table_filled:
        # Проверяем: может это pre_existing (инициализировано из таблицы)?
        # Нормализуем account_id: pandas может прочитать числовую колонку как float
        # → "1300000013.0" вместо "1300000013". Срезаем ".0" через int(float(...)).
        def _norm_account(v: str) -> str:
            v = v.strip()
            try:
                return str(int(float(v)))
            except (ValueError, OverflowError):
                return v

        _account_id_norm = _norm_account(str(account_id))
        _is_pre_existing = any(
            r.get("source") == "pre_existing"
            and _norm_account(str(r.get("account_id", ""))) == _account_id_norm
            for r in _log_rows
        )
        if _is_pre_existing:
            # Данные в таблице были до нашего прогона — это дубль, не аномалия
            result.outcome = Outcome.REPEAT
            result.error_detail = f"pre-existing entry in table (account={account_id})"
            result.new_photo_name = f"{account_id}{ext}"
            return result

        # В таблице есть данные, в логе нет и не pre_existing → рассинхронизация
        result.outcome = Outcome.SUSPICIOUS
        result.error_detail = (
            "аномалия: данные в таблице есть, фото не найдено в логе — "
            "возможна рассинхронизация"
        )
        result.new_photo_name = f"{account_id}{ext}"
        # Показания НЕ перезаписываем
        return result

    # ── Шаг 5: читаем показания счётчика ─────────────────────────────────────
    reading, reading_str, err, digit_results, digit_bboxes = _read_meter_digits(
        digit_detector, digit_ocr,
        meter_crop,
        config.digit_conf_thresh,
        config.expected_digits,
        ignore_last_digits=config.ignore_last_digits,
        missing_placeholder=config.missing_digit_placeholder,
        forgiven_placeholder=config.forgiven_digit_placeholder,
    )

    result.reading_str = reading_str

    # Пересчитываем digit bbox-ы в координаты оригинала (для отрисовки)
    if digit_bboxes is not None:
        result.digit_bboxes_in_orig = _digit_bboxes_to_orig(
            digit_bboxes, meter_bbox, meter_crop.shape
        )

    if reading is None:
        result.outcome = Outcome.DIGITS_ERROR
        result.error_detail = err
        result.new_photo_name = f"{account_id}{ext}"

        # ── Debug: сохраняем кропы цифр и meta.json ──────────────────────────
        if config.debug_digits and digit_results is not None:
            debug_dir = str(
                Path(config.output_base_dir)
                / "question/digits_error"
                / account_id
            )
            _raw_crops = digit_detector.process_array(meter_crop, save_crops=False, max_per_class=5)

            if _raw_crops:
                _sorted = sorted(_raw_crops, key=lambda c: c["bbox"][0])
                _with_placeholders: list[Optional[dict]] = []
                src_idx = 0
                for res in digit_results:
                    if res.get("note") == "inserted placeholder":
                        _with_placeholders.append(None)
                    else:
                        _with_placeholders.append(
                            _sorted[src_idx] if src_idx < len(_sorted) else None
                        )
                        src_idx += 1
                _save_digit_debug(debug_dir, _with_placeholders, digit_results)

        return result

    result.reading     = reading
    result.reading_str = reading_str

    # ── Шаг 6: проверяем подозрительное отклонение ───────────────────────────
    if last_reading is not None:
        delta = reading - last_reading
        result.delta = delta

        if abs(delta) > config.delta_threshold:
            result.outcome = Outcome.SUSPICIOUS
            result.error_detail = f"delta={delta:+.0f} > ±{config.delta_threshold:.0f}"
            result.new_photo_name = f"{account_id}{ext}"
            return result

        result.outcome = Outcome.PLUS if delta >= 0 else Outcome.MINUS
    else:
        result.delta = None
        result.outcome = Outcome.PLUS

    result.new_photo_name = f"{account_id}{ext}"
    return result


# ─── Тест одного фото ────────────────────────────────────────────────────────

def test_one(photo_path: str, config: PipelineConfig = None):
    """Быстрый тест одного фото — без таблицы, папок и перемещений."""
    config = config or PipelineConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meter_detector = YOLOInferer(config.meter_detect_model, conf_thresh=config.meter_conf_thresh)
    digit_detector = YOLOInferer(config.digit_detect_model, conf_thresh=config.digit_detect_conf_thresh, straighten=False)
    digit_ocr      = CNNInferer(config.digit_ocr_model, device=device)
    serial_ocr     = CRNNInferer(config.serial_ocr_model, device=device)

    crops = meter_detector.process_image(photo_path, save_crops=False)
    print("Найденные классы:", [c['class'] for c in crops] if crops else "ничего")

    meter_entry  = _find_crop_entry(crops, "gas_meter")
    serial_entry = _find_crop_entry(crops, "serial_number")

    serial_crop = serial_entry["crop"] if serial_entry else None
    serial_res  = serial_ocr.predict_with_details(serial_crop)
    print(f"Серийник: '{serial_res['text']}'  conf={serial_res['avg_confidence']:.3f}")

    meter_crop = meter_entry["crop"] if meter_entry else None
    reading, reading_str, err, digit_results, digit_bboxes = _read_meter_digits(
        digit_detector, digit_ocr,
        meter_crop,
        config.digit_conf_thresh,
        config.expected_digits,
        ignore_last_digits=config.ignore_last_digits,
        missing_placeholder=config.missing_digit_placeholder,
        forgiven_placeholder=config.forgiven_digit_placeholder,
    )
    print(f"Показания: {reading}  строка: {reading_str!r}  ошибка: {err or '—'}")
    if digit_results:
        print("Детали по цифрам:")
        for r in digit_results:
            flag = "✓" if r["ok"] else "✗"
            conf_str = f"{r['confidence']:.3f}" if r["confidence"] is not None else "N/A"
            print(f"  [{flag}] pos={r['position']}  digit={r['digit']}  conf={conf_str}")


# ─── Вспомогательные функции для определения структуры входной папки ────────

_PHOTO_EXTS = {".jpg", ".jpeg", ".png"}


def _list_photos(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _PHOTO_EXTS
    )


def _resolve_input_folders(input_dir: str) -> list[tuple[str, Path]]:
    """
    Определяет режим работы с input_dir.

    Возвращает список пар (subfolder_name, folder_path):
      - если input_dir сам содержит фото (хотя бы одно) → [("", input_dir)]
        subfolder_name = "" означает "без подпапки", обычный режим
      - если input_dir содержит подпапки с фото (глубина 1) →
        [(f1.name, f1), (f2.name, f2), ...]

    Подпапки без фото внутри игнорируются.
    """
    base = Path(input_dir)

    if _list_photos(base):
        return [("", base)]

    result = []
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        if _list_photos(sub):
            result.append((sub.name, sub))

    return result


# ─── Формирование отчёта ──────────────────────────────────────────────────────

# Группы исходов для сводных метрик отчёта.
_SUCCESS_OUTCOMES = (Outcome.PLUS, Outcome.MINUS)
_QUESTION_OUTCOMES = (
    Outcome.NO_METER,
    Outcome.NO_SERIAL,
    Outcome.SERIAL_LOW_CONF,
    Outcome.SERIAL_NOT_FOUND,
    Outcome.DIGITS_ERROR,
    Outcome.SUSPICIOUS,
)


def _format_report(title: str, stats: dict["Outcome", int]) -> str:
    """
    Формирует текстовый отчёт по статистике: количество и процент по каждому
    исходу + сводные метрики (успешно / на проверку / дубли).
    """
    total = sum(stats.values())
    lines = []
    lines.append("=" * 55)
    lines.append(title)
    lines.append("=" * 55)
    lines.append(f"Всего фото: {total}")
    lines.append("")

    if total == 0:
        lines.append("(нет данных)")
        lines.append("=" * 55)
        return "\n".join(lines)

    for outcome, count in stats.items():
        if count:
            pct = count / total * 100
            lines.append(f"  {outcome.name:<22} {count:>6}   {pct:5.1f}%")

    success = sum(stats.get(o, 0) for o in _SUCCESS_OUTCOMES)
    question = sum(stats.get(o, 0) for o in _QUESTION_OUTCOMES)
    repeat = stats.get(Outcome.REPEAT, 0)

    lines.append("")
    lines.append("-" * 55)
    lines.append(
        f"  Успешно (PLUS+MINUS):    {success:>6}   {success / total * 100:5.1f}%"
    )
    lines.append(
        f"  На проверку (question/): {question:>6}   {question / total * 100:5.1f}%"
    )
    lines.append(
        f"  Дубли (REPEAT):          {repeat:>6}   {repeat / total * 100:5.1f}%"
    )
    lines.append("=" * 55)

    return "\n".join(lines)


def _save_report(output_base: str, report_text: str) -> str:
    """Сохраняет отчёт в output_base/report.txt, возвращает путь к файлу."""
    Path(output_base).mkdir(parents=True, exist_ok=True)
    report_path = str(Path(output_base) / "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    return report_path


def _format_folder_ranking(per_folder_stats: dict[str, dict["Outcome", int]]) -> str:
    """
    Формирует рейтинг подпапок (контролёров) по доле успешных обработок
    (PLUS+MINUS от общего числа фото). Помогает увидеть, кто фотографирует
    качественнее, а кто хуже.
    """
    rows = []
    for name, stats in per_folder_stats.items():
        total = sum(stats.values())
        if total == 0:
            continue
        success = sum(stats.get(o, 0) for o in _SUCCESS_OUTCOMES)
        question = sum(stats.get(o, 0) for o in _QUESTION_OUTCOMES)
        rows.append((name, total, success, question, success / total * 100))

    if not rows:
        return ""

    rows.sort(key=lambda r: r[4], reverse=True)

    lines = []
    lines.append("=" * 55)
    lines.append("РЕЙТИНГ ПАПОК ПО % УСПЕШНЫХ ОБРАБОТОК")
    lines.append("=" * 55)
    lines.append(f"{'Папка':<30} {'Всего':>6} {'Успех':>6} {'На пров.':>9} {'%':>7}")
    lines.append("-" * 55)
    for name, total, success, question, pct in rows:
        lines.append(f"{name:<30} {total:>6} {success:>6} {question:>9} {pct:>6.1f}%")
    lines.append("=" * 55)

    return "\n".join(lines)


# ─── Пайплайн для целой папки (или набора папок) ─────────────────────────────

def run_pipeline(config: PipelineConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("reader")

    log.info(f"Загружаем таблицу: {config.table_path}")
    df = _load_table(config.table_path)
    log.info(f"Строк в таблице: {len(df)}")

    # ── Лог обработки ────────────────────────────────────────────────────────
    log_path = _log_path(config.table_path)
    log_rows = _maybe_init_log(config.table_path, log_path, df, config)
    config._log_filenames_cache      = _log_filenames(log_rows)
    config._log_rows_cache           = log_rows   # нужен для проверки pre_existing
    config._processed_accounts_cache = set()       # account_id обработанных в этом прогоне
    log.info(f"Processing log: {log_path} ({len(log_rows)} записей)")

    log.info("Загружаем модели...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Устройство: {device}")

    meter_detector = YOLOInferer(config.meter_detect_model,  conf_thresh=config.meter_conf_thresh)
    digit_detector = YOLOInferer(config.digit_detect_model,  conf_thresh=config.digit_detect_conf_thresh, straighten=False)
    digit_ocr      = CNNInferer(config.digit_ocr_model,      device=device)
    serial_ocr     = CRNNInferer(config.serial_ocr_model,    device=device)
    log.info("Модели загружены ✓")

    if config.ignore_last_digits > 0:
        log.info(
            f"⚙️  ignore_last_digits={config.ignore_last_digits}: "
            f"последние {config.ignore_last_digits} цифр(ы) — ошибки конфиданса прощаются, "
            f"подставляется '{config.forgiven_digit_placeholder}'"
        )
    if config.debug_digits:
        log.info("⚙️  debug_digits=True: кропы цифр и meta.json будут сохраняться в question/digits_error/<account_id>/")
    if config.ignore_last_digits > 0 and config.debug_digits:
        log.warning(
            "⚠️  ignore_last_digits и debug_digits включены одновременно: "
            "ошибки на прощённых позициях НЕ попадут в digits_error и не будут продебажены."
        )
    if config.draw_boxes:
        log.info("⚙️  draw_boxes=True: на фото будут нарисованы боксы детекций")

    # ── Определяем структуру входной папки ──────────────────────────────────
    input_groups = _resolve_input_folders(config.input_dir)

    if not input_groups:
        log.warning(f"В {config.input_dir} не найдено ни фото, ни подпапок с фото.")
        return

    if len(input_groups) == 1 and input_groups[0][0] == "":
        log.info(f"Режим: одна папка с фото ({config.input_dir})")
    else:
        names = ", ".join(name for name, _ in input_groups)
        log.info(f"Режим: папка с подпапками. Найдено подпапок с фото: {len(input_groups)} ({names})")

    total_stats: dict[Outcome, int] = {o: 0 for o in Outcome}
    per_folder_stats: dict[str, dict[Outcome, int]] = {}

    for subfolder_name, folder_path in input_groups:
        if subfolder_name:
            log.info(f"\n{'=' * 55}")
            log.info(f"Обрабатываем подпапку: {subfolder_name}")
            log.info("=" * 55)
            output_base = str(Path(config.output_base_dir) / subfolder_name)
        else:
            output_base = config.output_base_dir

        photos = _list_photos(folder_path)
        log.info(f"Найдено фото: {len(photos)}\n")

        stats: dict[Outcome, int] = {o: 0 for o in Outcome}
        db_updated = False

        for i, photo_path_obj in enumerate(photos, 1):
            photo_path = photo_path_obj  # совместимость с process_photo (ожидает str)
            log.info(f"[{i}/{len(photos)}] {photo_path.name}")

            result = process_photo(
                str(photo_path), df, config,
                meter_detector, digit_detector, digit_ocr, serial_ocr,
            )
            stats[result.outcome] += 1
            total_stats[result.outcome] += 1

            detail = result.error_detail or ""
            log.info(
                f"  serial={result.serial_text!r}({result.serial_conf or 0:.2f})  "
                f"reading={result.reading_str!r}  delta={result.delta}  "
                f"→ {result.outcome.name}  {detail}"
            )

            # Записываем в таблицу (только для PLUS / MINUS)
            if result.outcome in (Outcome.PLUS, Outcome.MINUS) and result.account_id:
                serial_norm = _normalize_serial(result.serial_text or "")
                mask = df[config.col_serial].apply(
                    lambda x: _normalize_serial(str(x)) == serial_norm
                )
                df.loc[mask, config.col_new_reading] = str(result.reading)
                db_updated = True
                log.info(f"  ✅ Записано в таблицу: account={result.account_id}, reading={result.reading}")

            # Сохраняем аннотированное фото в нужную папку
            dst_dir  = str(Path(output_base) / OUTCOME_FOLDER[result.outcome])
            new_name = result.new_photo_name or photo_path.name
            _save_annotated(
                str(photo_path), dst_dir, new_name, result,
                move=config.move_photos,
                draw_boxes=config.draw_boxes,
            )
            log.info(f"  📁 → {output_base}/{OUTCOME_FOLDER[result.outcome]}/{new_name}")

            # Запись в лог после каждого фото
            log_row = _make_log_row(photo_path_obj.name, result, config)
            _append_log_row(log_path, log_row)
            config._log_filenames_cache.add(photo_path_obj.name)
            config._log_rows_cache.append(log_row)
            if result.account_id:
                config._processed_accounts_cache.add(result.account_id)

            # Промежуточное сохранение таблицы и лога каждые 50 фото
            if db_updated and i % 50 == 0:
                _save_table(df, config.table_path)
                log.info(f"  💾 Промежуточное сохранение таблицы ({i} фото обработано)")

        if db_updated:
            _save_table(df, config.table_path)
            log.info(f"\n💾 Таблица сохранена: {config.table_path}")

        per_folder_stats[subfolder_name or Path(config.input_dir).name] = stats

        report_title = f"ОТЧЁТ: {subfolder_name or Path(config.input_dir).name}"
        report_text = _format_report(report_title, stats)
        report_path = _save_report(output_base, report_text)

        log.info("\n" + report_text)
        log.info(f"💾 Отчёт сохранён: {report_path}")

    if len(input_groups) > 1 or input_groups[0][0]:
        overall_report_text = _format_report("ОБЩИЙ ОТЧЁТ ПО ВСЕМ ПАПКАМ", total_stats)

        if len(per_folder_stats) > 1:
            ranking_text = _format_folder_ranking(per_folder_stats)
            overall_report_text = overall_report_text + "\n\n" + ranking_text

        overall_report_path = _save_report(config.output_base_dir, overall_report_text)

        log.info("\n" + overall_report_text)
        log.info(f"💾 Общий отчёт сохранён: {overall_report_path}")


# ─── Точка входа ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = PipelineConfig(
        move_photos         = False,  # True=перемещать, False=копировать (для теста)
        ignore_last_digits  = 2,      # прощаем последние 2 цифры (без веса на счётчике)
        debug_digits        = False,  # включить для анализа ошибок CNN
        draw_boxes          = False,   # включить для визуальной проверки детекций
    )

    # test_one('database/raw_photos/1300000013.jpeg')
    run_pipeline(config)