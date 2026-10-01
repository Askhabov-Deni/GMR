"""
reader.py — главный пайплайн обработки фотографий газовых счётчиков.

Запуск: `python gmr.py process [--input --output --table]` или `python reader.py`
(одинаковые настройки — значения по умолчанию PipelineConfig). Как результаты
используются окном оператора — docs/contract_reader_program2.md.

АРХИТЕКТУРА:
  process_photo(photo_path, df, config) -> PhotoResult
    │
    ├── шаг 0: фото уже разобрано / прочитано?  → REPEAT без запуска моделей
    ├── MeterDetector    (YOLO)         → кроп gas_meter + кроп serial_numbers
    ├── SerialRecognizer (CRNN)         → текст серийного номера
    ├── lookup_in_table(serial, df)     → строка таблицы (лицевой ID, последние показания)
    ├── DigitDetector    (YOLO)         → bbox-ы цифр на кропе счётчика
    ├── DigitRecognizer  (CNN)          → класс каждой цифры → собираем число
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
import json
import shutil
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from src.gmr.domain import (
    PipelineConfig,
    Outcome,
    OUTCOME_FOLDER,
    PhotoResult,
    DeltaThresholdPolicy,
    DuplicatePolicy,
    ProcessedPhotoPolicy,
    DigitDetector,
    DigitRecognizer,
    MeterDetector,
    SerialRecognizer,
)
from src.gmr.application import (
    RecognitionModels,
    find_detection,
    read_meter_digits_for_config,
    describe_substitutions,
)
from src.gmr.ml.loader import default_device, load_models
from src.gmr.console import safe_console
from src.gmr.storage import (
    CsvLogStore, ShadowLogStore, SqliteLogStore, backup_dir_for, backup_files,
    photo_fingerprint, append_log_row, load_table, log_path_for, save_table,
)
from src.gmr.storage.backup import KEEP, remove_old
from src.gmr.domain.serial_match import normalize_serial
from src.gmr.render import draw_annotation, read_image, write_image

# Модели вызываются только через контракты src/gmr/domain/ml.py:
# загрузка — src/gmr/ml/loader.py, чтение цифр — src/gmr/application/.
# domain-слой не знает ни про torch, ни про cv2, ни про YOLO/CNN/CRNN.
# PipelineConfig, Outcome, PhotoResult, OUTCOME_FOLDER доступны и как
# reader.<имя> (их так используют тесты).
_delta_policy        = DeltaThresholdPolicy()
_duplicate_policy    = DuplicatePolicy()
_processed_photo_policy = ProcessedPhotoPolicy()


# ─── Цвета боксов ────────────────────────────────────────────────────────────

_BOX_COLORS = {
    "gas_meter":     (0,   200,  0),    # зелёный
    "serial_number": (200,  0,   0),    # синий (BGR)
    "digit":         (0,   140, 255),   # оранжевый (BGR)
}
_BOX_THICKNESS = 2


# ─── Таблица, лог, серийники ─────────────────────────────────────────────────
# Таблица — src/gmr/storage/table.py, лог — src/gmr/storage/log_store.py,
# нормализация серийника — src/gmr/domain/serial_match.py (Фаза 6).
# Старые имена с подчёркиванием (_load_table, _load_log, …) удалены в Фазе 7.


def _maybe_init_log(
    table_path: str,
    log_path: str,
    df: "pd.DataFrame",
    config: "PipelineConfig",
    store: Optional["CsvLogStore"] = None,
) -> list[dict]:
    """
    При первом запуске: если лог не существует, но в таблице уже есть
    заполненные col_new_reading — предлагает инициализировать лог.
    Возвращает итоговые строки лога (может быть []).

    store — куда писать/откуда читать (по умолчанию CsvLogStore(log_path),
    как было всегда). run_pipeline передаёт сюда ShadowLogStore в
    shadow-run режиме (Фаза 2b), чтобы инициализация лога из таблицы тоже
    дублировалась в SQLite. Существование лога по-прежнему проверяется по
    CSV-файлу (log_path) — источник истины CSV (переход на SQLite — docs/BACKLOG.md).
    """
    store = store or CsvLogStore(log_path)

    if Path(log_path).exists():
        return store.load()

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
            # Маркер вместо имени фото: строка описывает показание, которое
            # уже было в таблице до первого запуска, а не обработанное фото.
            # DuplicatePolicy (п. 2a) по source="pre_existing" считает такой
            # счёт закрытым → REPEAT, а не SUSPICIOUS.
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

    store.save(rows)
    logging.getLogger("reader").info(f"Лог инициализирован: {len(rows)} записей → {log_path}")
    return rows


def _make_log_row(
    original_filename: str,
    result: "PhotoResult",
    config: "PipelineConfig",
    source: str = "auto",
    processed_by: str = "auto",
    photo_hash: str = "",
    source_folder: str = "",
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
        # notes: причина исхода + какие цифры подставлены (вариант А, 2026-09-30)
        "notes":             " | ".join(n for n in (result.error_detail, result.digit_notes) if n),
        "photo_hash":        photo_hash,
        "source_folder":     source_folder,
    }


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

    img = read_image(src)          # пути с кириллицей — см. src/gmr/render/image_io.py
    if img is not None:
        if draw_boxes:
            _draw_boxes(img, result)
        draw_annotation(img, result)
        written = write_image(dst_path, img)
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
            write_image(crop_path, dc["crop"])

    meta_path = str(Path(debug_dir) / "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(digit_results, f, ensure_ascii=False, indent=2)


# ─── Чтение цифр счётчика ────────────────────────────────────────────────────

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
    meter_detector: MeterDetector,
    digit_detector: DigitDetector,
    digit_recognizer: DigitRecognizer,
    serial_recognizer: SerialRecognizer,
    photo_hash: Optional[str] = None,
) -> PhotoResult:
    """
    Обрабатывает одну фотографию. Возвращает PhotoResult.
    НЕ изменяет df и не трогает файлы — это делает caller (run_pipeline).

    photo_hash — отпечаток содержимого фото (photo_fingerprint). Если задан,
    шаг 0 узнаёт фото в логе по нему, иначе по имени файла.

    Модели — в виде контрактов src/gmr/domain/ml.py (Фаза 3); реальные
    модели создаёт src.gmr.ml.loader.load_models.
    """
    models = RecognitionModels(
        meter_detector=meter_detector,
        digit_detector=digit_detector,
        digit_recognizer=digit_recognizer,
        serial_recognizer=serial_recognizer,
    )
    result = PhotoResult(photo_path=photo_path, outcome=Outcome.NO_METER)
    ext = Path(photo_path).suffix

    # ── Шаг 0: фото уже обрабатывалось? ──────────────────────────────────────
    # До запуска моделей. Разобранное оператором или успешно прочитанное фото
    # сразу REPEAT; фото только с авто-ошибками в логе обрабатывается заново
    # (ProcessedPhotoPolicy, решение владельца 2026-09-29, вариант Г).
    _seen = _processed_photo_policy.decide(
        Path(photo_path).name, getattr(config, "_log_rows_cache", []), photo_hash,
        hashes_in_run=frozenset(getattr(config, "_processed_hashes_cache", ())),
    )
    if _seen.skip:
        row = _seen.row or {}
        result.outcome = Outcome.REPEAT
        result.error_detail = _seen.reason
        result.account_id = row.get("account_id") or None
        result.serial_text = row.get("serial_id") or None
        result.new_photo_name = row.get("final_filename") or Path(photo_path).name
        return result

    # ── Шаг 1: детекция трёх классов ─────────────────────────────────────────
    crops = meter_detector.detect(photo_path)
    if not crops:
        result.outcome = Outcome.NO_METER
        result.error_detail = "YOLO returned no crops"
        return result

    meter_entry  = find_detection(crops, "gas_meter")
    serial_entry = find_detection(crops, "serial_number")

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
    serial_res  = serial_recognizer.recognize(serial_crop)
    serial_text = serial_res.text
    serial_conf = serial_res.confidence

    result.serial_text = serial_text
    result.serial_conf = serial_conf

    if serial_conf < config.serial_conf_thresh:
        result.outcome = Outcome.SERIAL_LOW_CONF
        result.error_detail = f"serial conf={serial_conf:.3f} < {config.serial_conf_thresh}"
        # Читаем цифры для информативной аннотации
        _r, _rs, _, _dres, _bboxes = read_meter_digits_for_config(models, meter_crop, config).as_tuple()
        result.reading_str = _rs
        result.reading     = _r
        result.digit_notes = describe_substitutions(_dres)
        if _bboxes is not None:
            result.digit_bboxes_in_orig = _digit_bboxes_to_orig(
                _bboxes, meter_bbox, meter_crop.shape
            )
        return result

    # ── Шаг 3: ищем в таблице ────────────────────────────────────────────────
    def _find_serial_in_df(serial: str) -> pd.DataFrame:
        norm = normalize_serial(serial)
        return df[df[config.col_serial].apply(
            lambda x: normalize_serial(str(x)) == norm
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
        _r, _rs, _, _dres, _bboxes = read_meter_digits_for_config(models, meter_crop, config).as_tuple()
        result.reading_str = _rs
        result.reading     = _r
        result.digit_notes = describe_substitutions(_dres)
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
    # Проверка "это фото уже в логе" — на шаге 0 (ProcessedPhotoPolicy).
    _log_rows     = getattr(config, "_log_rows_cache", [])
    _processed_accounts = getattr(config, "_processed_accounts_cache", set())
    _new_val      = df.at[row_idx, config.col_new_reading]
    _table_filled = pd.notna(_new_val) and str(_new_val).strip() not in ("", "nan")

    _dup = _duplicate_policy.decide(
        account_id=account_id,
        table_new_reading_filled=_table_filled,
        processed_accounts=_processed_accounts,
        log_rows=_log_rows,
    )
    if _dup.outcome is not None:
        # SUSPICIOUS здесь означает рассинхронизацию таблицы и лога —
        # показания НЕ перезаписываем (см. DuplicatePolicy).
        result.outcome = _dup.outcome
        result.error_detail = _dup.reason
        result.new_photo_name = f"{account_id}{ext}"
        return result

    # ── Шаг 5: читаем показания счётчика ─────────────────────────────────────
    reading, reading_str, err, digit_results, digit_bboxes = (
        read_meter_digits_for_config(models, meter_crop, config).as_tuple()
    )

    result.reading_str = reading_str
    result.digit_notes = describe_substitutions(digit_results)

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
            _raw_crops = digit_detector.detect(meter_crop)

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
    result.outcome, result.delta = _delta_policy.decide(reading, last_reading, config.delta_threshold)

    if result.outcome == Outcome.SUSPICIOUS:
        result.error_detail = f"delta={result.delta:+.0f} > ±{config.delta_threshold:.0f}"
        result.new_photo_name = f"{account_id}{ext}"
        return result

    result.new_photo_name = f"{account_id}{ext}"
    return result


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

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"


def run_pipeline(config: PipelineConfig) -> None:
    """
    Прогон папки с фото. Перед ним — копия таблицы и лога
    (src/gmr/storage/backup.py). Всё, что прогон пишет в консоль, пишется и в
    журнал `<папка результатов>/run_logs/run_<дата_время>.txt` (последние 30),
    вместе с причиной, если прогон прервался (2026-10-01).
    """
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt="%H:%M:%S")
    log = logging.getLogger("reader")
    log.setLevel(logging.INFO)

    run_logs = Path(config.output_base_dir) / "run_logs"
    run_logs.mkdir(parents=True, exist_ok=True)
    journal = run_logs / f"run_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.txt"
    handler = logging.FileHandler(journal, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(handler)
    try:
        log.info(f"Журнал прогона: {journal}")
        _backup_before_run(config, log)
        _run_pipeline(config, log)
    except BaseException:
        log.exception("Прогон прерван")
        raise
    finally:
        log.removeHandler(handler)
        handler.close()
        remove_old(run_logs, KEEP)


def _backup_before_run(config: PipelineConfig, log: logging.Logger) -> None:
    try:
        dest = backup_files(
            [config.table_path, log_path_for(config.table_path)], backup_dir_for(config.table_path)
        )
    except OSError as e:
        log.warning(f"Не удалось сделать копию таблицы и лога: {e}")
        return
    if dest is not None:
        log.info(f"Копия таблицы и лога: {dest}")


def _run_pipeline(config: PipelineConfig, log: logging.Logger) -> None:
    log.info(f"Загружаем таблицу: {config.table_path}")
    df = load_table(config.table_path)
    log.info(f"Строк в таблице: {len(df)}")

    # ── Лог обработки ────────────────────────────────────────────────────────
    log_path = log_path_for(config.table_path)
    if CsvLogStore(log_path).upgrade_if_needed():
        log.info(f"Лог переведён на новый формат (столбцы photo_hash, source_folder): {log_path}")

    # Фаза 2b: в shadow-run режиме каждая запись лога дублируется в SQLite.
    # CSV (log_path) остаётся источником истины — чтение идёт только из него.
    shadow_store = None
    if config.shadow_sqlite_log:
        sqlite_path = str(Path(log_path).with_suffix(".sqlite"))
        if Path(sqlite_path).exists():
            # SQLite от прошлого shadow-прогона пересоздаём из текущего CSV,
            # иначе после ручных правок CSV (program2.py) сверка расходилась
            # бы не из-за бага, а из-за устаревшей копии.
            Path(sqlite_path).unlink()
        shadow_store = ShadowLogStore(CsvLogStore(log_path), SqliteLogStore(sqlite_path))
        if Path(log_path).exists():
            shadow_store.shadow.save(shadow_store.primary.load())
        log.info(f"Shadow-run: SQLite-лог → {sqlite_path}")

    log_rows = _maybe_init_log(config.table_path, log_path, df, config, store=shadow_store)
    config._log_rows_cache           = log_rows   # нужен для проверки pre_existing
    config._processed_accounts_cache = set()       # account_id обработанных в этом прогоне
    config._processed_hashes_cache   = set()       # отпечатки фото, обработанных в этом прогоне
    log.info(f"Processing log: {log_path} ({len(log_rows)} записей)")

    log.info("Загружаем модели...")
    device = default_device()
    log.info(f"Устройство: {device}")

    models = load_models(config, device=device)
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

            photo_hash = photo_fingerprint(str(photo_path))
            result = process_photo(
                str(photo_path), df, config,
                models.meter_detector, models.digit_detector,
                models.digit_recognizer, models.serial_recognizer,
                photo_hash=photo_hash,
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
                serial_norm = normalize_serial(result.serial_text or "")
                mask = df[config.col_serial].apply(
                    lambda x: normalize_serial(str(x)) == serial_norm
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
            log_row = _make_log_row(
                photo_path_obj.name, result, config,
                photo_hash=photo_hash, source_folder=subfolder_name,
            )
            if shadow_store is not None:
                shadow_store.append(log_row)
            else:
                append_log_row(log_path, log_row)
            config._log_rows_cache.append(log_row)
            if result.account_id:
                config._processed_accounts_cache.add(result.account_id)
            config._processed_hashes_cache.add(photo_hash)

            # Промежуточное сохранение таблицы и лога каждые 50 фото
            if db_updated and i % 50 == 0:
                save_table(df, config.table_path)
                log.info(f"  💾 Промежуточное сохранение таблицы ({i} фото обработано)")

        if db_updated:
            save_table(df, config.table_path)
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

    if shadow_store is not None:
        _report_shadow_run(shadow_store, log_path, log)


def _report_shadow_run(shadow_store: ShadowLogStore, log_path: str, log: logging.Logger) -> None:
    """Сверяет CSV и SQLite после прогона и пишет результат рядом с логом."""
    divergences = shadow_store.compare()
    report_path = str(Path(log_path).with_name(Path(log_path).stem + "_shadow_report.txt"))
    n_rows = len(shadow_store.primary.load())

    lines = [f"Shadow-run: CSV vs SQLite, строк в CSV-логе: {n_rows}"]
    if not divergences:
        lines.append("РЕЗУЛЬТАТ: совпадение построчно, расхождений нет")
    else:
        lines.append(f"РЕЗУЛЬТАТ: РАСХОЖДЕНИЯ — {len(divergences)}")
        for idx, p_row, s_row in divergences[:50]:
            diff_cols = sorted(k for k in set(p_row) | set(s_row) if p_row.get(k) != s_row.get(k))
            lines.append(f"  строка {idx}: {', '.join(diff_cols)}")
            for c in diff_cols:
                lines.append(f"    {c}: csv={p_row.get(c)!r}  sqlite={s_row.get(c)!r}")
        if len(divergences) > 50:
            lines.append(f"  ... и ещё {len(divergences) - 50}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if divergences:
        log.warning(f"⚠️  Shadow-run: {len(divergences)} расхождений CSV/SQLite → {report_path}")
    else:
        log.info(f"✅ Shadow-run: CSV и SQLite совпадают ({n_rows} строк) → {report_path}")


# ─── Точка входа ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    safe_console()
    run_pipeline(PipelineConfig())
