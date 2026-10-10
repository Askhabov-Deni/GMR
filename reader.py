"""
reader.py — главный пайплайн обработки фотографий газовых счётчиков.

Запуск: `python gmr.py process <папка месяца>` или `python reader.py <папка
месяца>` (одинаковые настройки — значения по умолчанию PipelineConfig). Фото —
из <месяц>/фото, результат — в <месяц>/результат, абоненты, показания и лог —
в базе месяца. Как результаты используются окном оператора —
docs/contract_reader_program2.md.

АРХИТЕКТУРА:
  process_photo(photo_path, df, config) -> PhotoResult
    │
    ├── шаг 0: фото уже разобрано / прочитано?  → REPEAT без запуска моделей
    ├── MeterDetector    (YOLO)         → кроп gas_meter + кроп serial_numbers (+ надпись маркером)
    ├── AccountRecognizer (CRNN, если подключена) → лицевой счёт по надписи маркером
    ├── SerialRecognizer (CRNN)         → текст серийного номера
    ├── lookup_in_table(serial, df)     → строка таблицы (лицевой ID, последние показания);
    │                                     серийник + надпись — src/gmr/domain/account_match.py
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
    error/           — программа упала на этом фото (битый файл, ошибка модели)
    serial_ambiguous/— номер счётчика в таблице у нескольких абонентов

DEBUG-РЕЖИМ (config.debug_digits = True):
  question/digits_error/<account_id или photo_stem>/
    digit_0.jpg … digit_N.jpg   — кроп каждой найденной цифры
    meta.json                   — предсказание и конфиданс по каждой позиции
"""

import os
import sys
import json
import shutil
import sqlite3
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
    RunSelection,
    RunSelectionPolicy,
    DigitDetector,
    DigitRecognizer,
    MeterDetector,
    SerialRecognizer,
    AccountPrediction,
    AccountRecognizer,
)
from src.gmr.domain.account_match import AccountMarkerPolicy, account_groups as build_account_groups
from src.gmr.application import (
    RecognitionModels,
    detect_meter,
    find_detection,
    read_meter_digits_for_config,
    describe_substitutions,
)
from src.gmr.ml.loader import default_device, load_models
from src.gmr.application.cycle import close_waiting, free_name, move_files, result_file
from src.gmr.application.month import ExportLocked, NotAMonth, export_month, is_month, month_preset
from src.gmr.console import safe_console
from src.gmr.domain.photo_date import reading_date
from src.gmr.storage.month import MonthDB, MonthFolder, Reading, now_text
from src.gmr.storage import photo_fingerprint
from src.gmr.storage.backup import KEEP, backup_sqlite, remove_old
from src.gmr.domain.serial_match import build_serial_groups, normalize_serial
from src.gmr.render import draw_annotation, read_image, turn_image, write_image

# Модели вызываются только через контракты src/gmr/domain/ml.py:
# загрузка — src/gmr/ml/loader.py, чтение цифр — src/gmr/application/.
# domain-слой не знает ни про torch, ни про cv2, ни про YOLO/CNN/CRNN.
# PipelineConfig, Outcome, PhotoResult, OUTCOME_FOLDER доступны и как
# reader.<имя> (их так используют тесты).
_delta_policy        = DeltaThresholdPolicy()
_duplicate_policy    = DuplicatePolicy()
_processed_photo_policy = ProcessedPhotoPolicy()
_run_selection       = RunSelectionPolicy()
_account_policy      = AccountMarkerPolicy()


# ─── Цвета боксов ────────────────────────────────────────────────────────────

_BOX_COLORS = {
    "gas_meter":     (0,   200,  0),    # зелёный
    "serial_number": (200,  0,   0),    # синий (BGR)
    "digit":         (0,   140, 255),   # оранжевый (BGR)
}
_BOX_THICKNESS = 2


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
        "notes":             " | ".join(n for n in (result.error_detail, _turn_note(result), result.serial_notes,
                                                    result.account_notes, result.digit_notes) if n),
        "photo_hash":        photo_hash,
        "source_folder":     source_folder,
    }


def _turn_note(result: PhotoResult) -> Optional[str]:
    """Пометка в notes: фото пришлось повернуть (этап 7b)."""
    if not result.photo_turn:
        return None
    return f"фото повёрнуто на {result.photo_turn}°: на исходном детектор ничего не нашёл"


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
        # фото пришлось повернуть (этап 7b) — в результат кладётся повёрнутое:
        # счётчик стоит прямо, рамки и подпись — на своих местах
        img = turn_image(img, result.photo_turn)
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


# ─── Серийник по таблице (этап 6b) ───────────────────────────────────────────

def serial_groups_of(df: pd.DataFrame, config: PipelineConfig) -> dict:
    """Словарь для серийника по таблице (этап 6b): {номер: варианты записи}."""
    return build_serial_groups(df[config.col_serial])


def _rows_with_serial(df: pd.DataFrame, config: PipelineConfig, serial: str) -> pd.DataFrame:
    norm = normalize_serial(serial)
    return df[df[config.col_serial].apply(lambda x: normalize_serial(str(x)) == norm)]


# ─── Надпись маркером (лицевой счёт) ─────────────────────────────────────────

def account_groups_of(df: pd.DataFrame, config: PipelineConfig) -> dict:
    """Словарь модели надписи: {счёт: варианты записи} по всей таблице."""
    return build_account_groups(df[config.col_account_id])


def _account_rows(df: pd.DataFrame, config: PipelineConfig, account: str) -> pd.DataFrame:
    return df[df[config.col_account_id].astype(str).str.strip() == str(account)]


def _read_marker(recognizer: Optional[AccountRecognizer], crops: list, groups: Optional[dict],
                 config: PipelineConfig) -> Optional[AccountPrediction]:
    """Что написано маркером и какой это счёт; None — нет модели или надписи."""
    if recognizer is None:
        return None
    det = find_detection(crops, config.account_class)
    return None if det is None else recognizer.recognize(det["crop"], groups)


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
    reread: bool = False,
    account_recognizer: Optional[AccountRecognizer] = None,
    account_groups: Optional[dict] = None,
    serial_groups: Optional[dict] = None,
) -> PhotoResult:
    """
    Обрабатывает одну фотографию. Возвращает PhotoResult.
    НЕ изменяет df и не трогает файлы — это делает caller (run_pipeline).

    photo_hash — отпечаток содержимого фото (photo_fingerprint). Если задан,
    шаг 0 узнаёт фото в логе по нему, иначе по имени файла.
    reread — прогон уже решил прочитать фото заново (RunSelectionPolicy:
    --reread, номер появился в таблице), шаг 0 пропускается.

    Модели — в виде контрактов src/gmr/domain/ml.py (Фаза 3); реальные
    модели создаёт src.gmr.ml.loader.load_models.

    account_recognizer — модель надписи маркером (None — её нет, всё как до
    неё); account_groups — словарь счетов таблицы для неё
    (account_groups_of(df, config): прогон строит его один раз).
    """
    models = RecognitionModels(
        meter_detector=meter_detector,
        digit_detector=digit_detector,
        digit_recognizer=digit_recognizer,
        serial_recognizer=serial_recognizer,
        account_recognizer=account_recognizer,
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
    if _seen.skip and not reread:
        row = _seen.row or {}
        result.outcome = Outcome.REPEAT
        result.error_detail = _seen.reason
        result.account_id = row.get("account_id") or None
        result.serial_text = row.get("serial_id") or None
        result.new_photo_name = row.get("final_filename") or Path(photo_path).name
        return result

    # ── Шаг 1: детекция трёх классов ─────────────────────────────────────────
    # (ничего не найдено и включено turn_if_nothing — фото поворачивается, этап 7b)
    crops, result.photo_turn = detect_meter(meter_detector, photo_path, config)
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

    meter_crop = meter_entry["crop"]
    meter_bbox  = meter_entry["bbox"]   # (x1,y1,x2,y2) в оригинале

    def _read_digits_for_annotation() -> None:
        """Показание для аннотации и оператору, когда исход — не запись."""
        _r, _rs, _, _dres, _bboxes = read_meter_digits_for_config(models, meter_crop, config).as_tuple()
        result.reading_str = _rs
        result.reading     = _r
        result.digit_notes = describe_substitutions(_dres)
        if _bboxes is not None:
            result.digit_bboxes_in_orig = _digit_bboxes_to_orig(_bboxes, meter_bbox, meter_crop.shape)

    # ── Шаг 1b: надпись маркером (лицевой счёт) — только если есть модель ───
    if account_recognizer is not None and account_groups is None:
        account_groups = account_groups_of(df, config)
    marker = _read_marker(account_recognizer, crops, account_groups, config)

    # ── Шаги 2–3: серийник → строка таблицы ─────────────────────────────────
    # failure — почему серийник не дал ровно один счёт (исход и пояснение)
    failure: Optional[tuple[Outcome, str]] = None
    ambiguous: list[str] = []
    matches = pd.DataFrame()

    if serial_entry is None:
        failure = (Outcome.NO_SERIAL, "class serial_number not detected")
    else:
        serial_res  = serial_recognizer.recognize(serial_entry["crop"])
        serial_text = serial_res.text
        serial_conf = serial_res.confidence

        result.serial_text = serial_text
        result.serial_conf = serial_conf

        if serial_conf < config.serial_conf_thresh:
            failure = (Outcome.SERIAL_LOW_CONF,
                       f"serial conf={serial_conf:.3f} < {config.serial_conf_thresh}")
        else:
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
                failure = (Outcome.SERIAL_NOT_FOUND, (
                    f"serial '{serial_text}' not in table "
                    f"(tried: {', '.join(repr(c) for c in serial_candidates)})"
                ))
            else:
                result.serial_text = serial_used
                # Номер в таблице у нескольких лицевых счетов — у одной записи в базе номер
                # с ошибкой; чей счётчик, решает оператор (решение владельца 2026-10-01).
                accounts = list(dict.fromkeys(str(a).strip() for a in matches[config.col_account_id]))
                if len(accounts) > 1:
                    failure = (Outcome.SERIAL_AMBIGUOUS,
                               f"номер {serial_used} у нескольких абонентов: {', '.join(accounts)}")
                    ambiguous = accounts

    # ── Шаг 3a: серийник по таблице (этап 6b, пресет месяца; выключено) ─────
    if (failure is not None and failure[0] in (Outcome.SERIAL_NOT_FOUND, Outcome.SERIAL_LOW_CONF)
            and config.serial_by_table and hasattr(serial_recognizer, "match_table")):
        if serial_groups is None:
            serial_groups = serial_groups_of(df, config)
        m = serial_recognizer.match_table(serial_entry["crop"], serial_groups)
        if m.serial is not None and m.confidence >= config.serial_table_conf_thresh:
            result.serial_notes = (f"серийник по таблице: прочитано {result.serial_text}, "
                                   f"взят {m.serial} (доля {m.confidence:.2f})")
            result.serial_text = m.serial
            matches = _rows_with_serial(df, config, m.serial)
            accounts = list(dict.fromkeys(str(a).strip() for a in matches[config.col_account_id]))
            failure = None
            if len(accounts) > 1:
                failure = (Outcome.SERIAL_AMBIGUOUS,
                           f"номер {m.serial} у нескольких абонентов: {', '.join(accounts)}")
                ambiguous = accounts
        elif m.serial is not None:
            result.serial_notes = (f"серийник по таблице: не уверен — ближе всех {m.serial} "
                                   f"(доля {m.confidence:.2f})")

    if failure is not None:
        # Серийник счёт не дал — может быть, его даёт надпись маркером
        # (src/gmr/domain/account_match.py, правило 2). Без модели надписи — как раньше.
        rows_m = (_account_rows(df, config, marker.account)
                  if _account_policy.confident(marker, config.account_conf_thresh) else df.iloc[0:0])
        decision = _account_policy.rescue(
            failure[0], marker, config.account_conf_thresh, result.serial_text,
            [str(v) for v in rows_m[config.col_serial]], ambiguous,
        )
        result.account_notes = decision.note
        if decision.account is None or rows_m.empty:
            result.outcome, result.error_detail = failure
            if failure[0] != Outcome.NO_SERIAL:
                # Читаем цифры для информативной аннотации
                _read_digits_for_annotation()
            return result
        matches = rows_m
        result.serial_text = normalize_serial(str(rows_m.iloc[0][config.col_serial]))

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

    # ── Шаг 3b: серийник нашёл счёт сам — сверить с надписью маркером ───────
    # Надпись уверенно указывает на другой счёт → показание не пишем, решает
    # оператор (account_match.py, правило 1). Счёт фото не присваивается: чей
    # это счётчик, неизвестно, и показание счёта по серийнику с другого фото
    # не должно молча закрыть это фото (close_waiting ищет по account_id).
    if failure is None:
        conflict = _account_policy.conflict(account_id, marker, config.account_conf_thresh)
        if conflict is not None:
            result.outcome = Outcome.SUSPICIOUS
            result.error_detail = conflict
            result.account_id = None
            result.last_reading = None
            _read_digits_for_annotation()
            return result

    # ── Шаг 4: проверяем дубль ───────────────────────────────────────────────
    # Дубль определяем по логу, а не по col_new_reading.
    #
    # Кейс «показание было в таблице до программы»:
    #   _MonthRun добавляет к логу строки с original_filename="__pre_existing__"
    #   и source="pre_existing" для показаний, внесённых до программы (в базе
    #   source=table). Такой счёт уже закрыт — это тоже дубль. Ищем их по
    #   совпадению account_id в логе с source=pre_existing.
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
    # (прошлое показание — для старшей цифры по прошлому показанию, этап 6b)
    reading, reading_str, err, digit_results, digit_bboxes = (
        read_meter_digits_for_config(models, meter_crop, config, last_reading=last_reading).as_tuple()
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


class PhotosInRoot(ValueError):
    """В корне <месяц>/фото лежат фото — фото кладут только в папки
    контролёров (решение владельца 2026-10-03, этап 3, пункт 4б)."""


def _check_no_photos_in_root(photos_dir: Path) -> None:
    loose = _list_photos(photos_dir) if photos_dir.is_dir() else []
    if loose:
        names = ", ".join(p.name for p in loose[:5]) + (" …" if len(loose) > 5 else "")
        raise PhotosInRoot(
            f"В папке {photos_dir} лежат фото без папки контролёра ({len(loose)} шт.: {names}). "
            f"Фото кладут только в папки контролёров: фото\\<контролёр>\\. "
            f"Разложите их и запустите снова — прогон не начинался.")


# служебные файлы Windows/macOS и временные файлы Office — не «файлы контролёров»
_SYSTEM_FILES = {"thumbs.db", "desktop.ini", ".ds_store"}


def _not_taken(folder: Path, with_dirs: bool = True) -> list[str]:
    """Что в папке программа не возьмёт (этап 3, решение владельца
    2026-10-03 — перечислять в отчёте): файлы не .jpg/.jpeg/.png и папки
    внутри папки контролёра (фото в них не читаются; имя — с «\\» на конце).
    Служебные файлы не считаются."""
    if not folder.is_dir():
        return []
    out = []
    for p in sorted(folder.iterdir(), key=lambda x: x.name):
        if p.name.lower() in _SYSTEM_FILES or p.name.startswith("~$"):
            continue
        if p.is_dir():
            if with_dirs:
                out.append(p.name + "\\")
        elif p.suffix.lower() not in _PHOTO_EXTS:
            out.append(p.name)
    return out


def _resolve_input_folders(input_dir: str) -> list[tuple[str, Path]]:
    """Папки контролёров с фото: [(имя, путь), …]. Фото в корне — ошибка
    (_check_no_photos_in_root, до начала прогона); папки без фото не берутся."""
    base = Path(input_dir)
    if not base.is_dir():
        return []
    return [(sub.name, sub) for sub in sorted(p for p in base.iterdir() if p.is_dir())
            if _list_photos(sub)]


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
    Outcome.ERROR,
    Outcome.SERIAL_AMBIGUOUS,
)


_SKIP_LABELS = {
    RunSelection.SKIP_DONE:    "Уже разобраны раньше (пропущены)",
    RunSelection.SKIP_WAITING: "Ждут оператора с прошлых прогонов",
    RunSelection.SKIP_COPY:    "Тот же файл ещё раз (копия)",
}


def _skip_lines(skipped: Optional[dict]) -> list[str]:
    return [f"{label}: {skipped[k]}" for k, label in _SKIP_LABELS.items() if skipped and skipped.get(k)]


_NOT_TAKEN_SHOWN = 10


def _where(folder: str, folder_report: bool = False) -> str:
    if folder_report:
        return ""
    return f" в «{folder}»" if folder else " в корне фото\\"


def _not_taken_line(names: list[str], where: str = "") -> str:
    shown = ", ".join(names[:_NOT_TAKEN_SHOWN]) + (" …" if len(names) > _NOT_TAKEN_SHOWN else "")
    return f"⚠ Не взяты{where} (не фото .jpg/.jpeg/.png или папка внутри): {len(names)} — {shown}"


def _format_report(title: str, stats: dict["Outcome", int], skipped: Optional[dict] = None,
                   not_taken: Optional[dict] = None, folder_report: bool = False) -> str:
    """
    Формирует текстовый отчёт по статистике этого прогона: количество и
    процент по каждому исходу + сводные метрики (успешно / на проверку /
    дубли); skipped — сколько фото пропущено (RunSelection → число);
    not_taken — что программа не взяла: {папка: [имена]} ("" — корень фото\\);
    в отчёте папки контролёра (folder_report) — без имени папки.
    """
    total = sum(stats.values())
    lines = []
    lines.append("=" * 55)
    lines.append(title)
    lines.append("=" * 55)
    lines.append(f"Новых фото в этом прогоне: {total}")
    lines += _skip_lines(skipped)
    for folder, names in (not_taken or {}).items():
        if names:
            lines.append(_not_taken_line(names, _where(folder, folder_report)))
    lines.append("")

    if total == 0:
        lines.append("(новых фото нет)")
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
    Прогон по папке месяца config.month_dir (этап 2.2b; старый режим «таблица
    и CSV-лог» убран на этапе 2.3b): фото из <месяц>/фото, результат в
    <месяц>/результат, абоненты, показания и лог — в базе gmr.sqlite;
    показание и строка лога пишутся одной транзакцией; в конце — выгрузка
    показания.xlsx (_MonthRun). Перед прогоном — копия базы месяца. Всё, что
    прогон пишет в консоль, пишется и в журнал
    `<месяц>/результат/run_logs/run_<дата_время>.txt` (последние 30), вместе
    с причиной, если прогон прервался (2026-10-01).
    """
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt="%H:%M:%S")
    log = logging.getLogger("reader")
    log.setLevel(logging.INFO)

    if not config.month_dir:
        raise NotAMonth("Не указана папка месяца: python gmr.py process <папка месяца>")
    month = MonthFolder(Path(config.month_dir))
    if not is_month(month):
        raise NotAMonth(f"Месяц не создан: {month.root}. Сначала: "
                        f"python gmr.py month \"{month.root}\" --table <таблица компании>")
    _check_no_photos_in_root(month.photos)
    config.input_dir = str(month.photos)
    config.output_base_dir = str(month.results)
    month.stop_file.unlink(missing_ok=True)          # старая просьба остановиться — не в счёт

    run_logs = Path(config.output_base_dir) / "run_logs"
    run_logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    journal, n = run_logs / f"run_{stamp}.txt", 2
    while journal.exists():                          # два прогона в одну секунду — свой журнал у каждого
        journal, n = run_logs / f"run_{stamp}_{n}.txt", n + 1
    handler = logging.FileHandler(journal, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(handler)
    try:
        log.info(f"Журнал прогона: {journal}")
        _apply_month_preset(config, month, log)
        _backup_month(month, log)
        store = _MonthRun(config, log, month)
        try:
            _run_pipeline(config, log, store)
        finally:
            store.close()
    except BaseException:
        log.exception("Прогон прерван")
        raise
    finally:
        month.stop_file.unlink(missing_ok=True)
        log.removeHandler(handler)
        handler.close()
        remove_old(run_logs, KEEP)


_PRESET_FIELDS = ("missing_digit_mode", "ignore_last_digits", "forgiven_digit_mode",
                  "serial_crop_pad", "account_ocr_model", "drum_rule", "first_digit_from_last",
                  "serial_by_table", "turn_if_nothing")


def _apply_month_preset(config: PipelineConfig, month: MonthFolder, log: logging.Logger) -> None:
    """Настройки распознавания месяца (src/gmr/domain/preset.py) — в config.
    Месяц без них — всё как в PipelineConfig."""
    preset = month_preset(month.root)
    if preset is None:
        log.info("Настройки распознавания: по умолчанию (как до настроек месяца)")
        return
    applied = preset.apply(config)
    for name in _PRESET_FIELDS:
        setattr(config, name, getattr(applied, name))
    log.info(f"Настройки распознавания месяца: {preset.describe()}")


def _backup_month(month: MonthFolder, log: logging.Logger) -> None:
    try:
        dest = backup_sqlite(month.db, month.backups)
    except (OSError, sqlite3.Error) as e:
        log.warning(f"Не удалось сделать копию базы месяца: {e}")
        return
    if dest is not None:
        log.info(f"Копия базы месяца: {dest}")


class _MonthRun:
    """
    Прогон по папке месяца (этап 2.2b). Абоненты и показания — из базы;
    показание (PLUS/MINUS), запись в журнал изменений и строка лога — одной
    транзакцией: сбой посреди прогона не теряет показаний, при перезапуске
    фото узнаётся по логу. Таблицу во время прогона никто не пишет, поэтому
    открытый Excel ему не мешает; в конце — выгрузка показания.xlsx (если
    файл открыт — показания остаются в базе, выгрузить позже: gmr.py export).
    """

    def __init__(self, config: PipelineConfig, log: logging.Logger, month: MonthFolder):
        self.config, self.log, self.month = config, log, month
        self.db = MonthDB(month.db)
        abonents = self.db.abonents(only_in_table=True)
        readings = self.db.readings()
        cfg = config
        self.df = pd.DataFrame(
            [{cfg.col_serial: a.serial, cfg.col_account_id: a.account,
              cfg.col_last_reading: a.last_reading,
              cfg.col_new_reading: readings[a.account].value if a.account in readings else ""}
             for a in abonents.values()],
            columns=[cfg.col_serial, cfg.col_account_id, cfg.col_last_reading, cfg.col_new_reading],
        )
        # Показания, которые были в таблице при загрузке, — как строки
        # pre_existing старого лога: по такому счёту новое фото → REPEAT
        # (DuplicatePolicy). В базу эти строки не пишутся.
        self.log_rows = self.db.log_rows() + [
            {"original_filename": "__pre_existing__", "source": "pre_existing",
             "outcome": "UNKNOWN", "account_id": account}
            for account, r in readings.items() if r.source == "table"
        ]
        log.info(f"Месяц: {month.root}; абонентов: {len(abonents)}, с показанием: "
                 f"{sum(1 for a in abonents if a in readings)}, строк лога: {len(self.log_rows)}")

    def record(self, result: PhotoResult, log_row: dict, photo_name: str) -> list[dict]:
        """Показание и строка лога — одной транзакцией. Если у счёта появилось
        показание, его фото, ждущие оператора, закрываются (этап 3, пункт 3) —
        возвращаются их строки REPEAT."""
        cfg = self.config
        closed, moves = [], []
        with self.db.transaction():
            if result.outcome in (Outcome.PLUS, Outcome.MINUS) and result.account_id:
                account, value = result.account_id, str(result.reading)
                old = self.db.reading(account)
                self.db.put_reading(Reading(account, value, reading_date(photo_name), "auto",
                                            photo_name, now_text(), "auto"))
                self.db.add_change("auto", "показание записано", account, cfg.col_new_reading,
                                   old.value if old else "", value, note=photo_name)
                self.df.loc[self.df[cfg.col_account_id] == account, cfg.col_new_reading] = value
                self.log.info(f"  ✅ Записано в базу: account={account}, reading={value}")
            self.db.append_log_rows([log_row])
            if result.outcome in (Outcome.PLUS, Outcome.MINUS) and result.account_id:
                closed, moves = close_waiting(self.db, self.month.results, result.account_id,
                                              photo_name, "auto")
        for row in closed:
            self.log.info(f"  ↪ {row['original_filename']}: ждало оператора, у л/с "
                          f"{row['account_id']} теперь есть показание → repeat/{row['final_filename']}")
        for err in move_files(moves):
            self.log.warning(f"  ⚠️  Не удалось перенести фото в repeat/: {err}")
        return closed

    def finish(self) -> None:
        try:
            self.log.info(export_month(str(self.month.root), self.config).text())
        except ExportLocked as e:
            self.log.warning(f"⚠️  {e} Показания сохранены в базе.")

    def close(self) -> None:
        self.db.close()


class _PhotoIndex:
    """Строки лога по фото — как ProcessedPhotoPolicy.row_matches: строка с
    отпечатком совпадает по отпечатку, без отпечатка — по имени файла."""

    def __init__(self, rows: list[dict]):
        self._n = 0
        self._by_hash: dict[str, list] = {}
        self._no_hash_by_name: dict[str, list] = {}
        self._by_name: dict[str, list] = {}
        for r in rows:
            self.add(r)

    def add(self, row: dict) -> None:
        self._n += 1
        item = (self._n, row)
        name = row.get("original_filename") or ""
        self._by_name.setdefault(name, []).append(item)
        if row.get("photo_hash"):
            self._by_hash.setdefault(row["photo_hash"], []).append(item)
        else:
            self._no_hash_by_name.setdefault(name, []).append(item)

    def rows(self, name: str, photo_hash: str) -> list[dict]:
        if photo_hash:
            found = self._by_hash.get(photo_hash, []) + self._no_hash_by_name.get(name, [])
        else:
            found = self._by_name.get(name, [])
        return [r for _, r in sorted(found, key=lambda x: x[0])]


def _serial_lookup(df: pd.DataFrame, config: PipelineConfig):
    """Есть ли номер в таблице — так же, как ищет process_photo (шаг 3):
    номер, '0'+номер, '00'+номер."""
    serials = {normalize_serial(str(s)) for s in df[config.col_serial]}

    def in_table(serial: str) -> bool:
        s = normalize_serial(serial)
        return any(c in serials for c in (s, "0" + s, "00" + s))
    return in_table


def _run_pipeline(config: PipelineConfig, log: logging.Logger, store) -> None:
    df = store.df
    config._log_rows_cache           = store.log_rows   # нужен для проверки pre_existing
    config._processed_accounts_cache = set()            # счета, получившие показание в этом прогоне
    config._processed_hashes_cache   = set()            # отпечатки фото, прочитанных в этом прогоне
    index = _PhotoIndex(store.log_rows)
    serial_in_table = _serial_lookup(df, config)
    models = None                                       # загружаются, только если есть что читать
    account_groups_cache = None                         # словарь счетов для модели надписи (один на прогон)
    serial_groups_cache = None                          # словарь номеров для серийника по таблице (этап 6b)

    def load():
        log.info("Загружаем модели...")
        device = default_device()
        log.info(f"Устройство: {device}")
        loaded = load_models(config, device=device)
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
        if loaded.account_recognizer is not None:
            log.info(f"⚙️  надпись маркером: {config.account_ocr_model} (класс «{config.account_class}», "
                     f"порог {config.account_conf_thresh})")
        return loaded

    # ── Папки контролёров ────────────────────────────────────────────────────
    base = Path(config.input_dir)
    input_groups = _resolve_input_folders(config.input_dir)
    # что программа не возьмёт — в журнал и в отчёты (этап 3)
    not_taken = {"": _not_taken(base, with_dirs=False)}
    if base.is_dir():
        not_taken |= {d.name: _not_taken(d) for d in sorted(base.iterdir(), key=lambda x: x.name)
                      if d.is_dir()}
    for folder, names_ in not_taken.items():
        if names_:
            log.warning(_not_taken_line(names_, _where(folder)))
    if not input_groups:
        log.warning(f"В {config.input_dir} нет папок контролёров с фото.")
    else:
        names = ", ".join(name for name, _ in input_groups)
        log.info(f"Папок контролёров с фото: {len(input_groups)} ({names})")
    if config.reread_errors:
        log.info("⚙️  --reread: фото с ошибками, которые ждут оператора, читаются заново")

    # ── Сколько новых фото прочитать — для полосы прогресса окна (этап 4).
    # Отпечатки считаются один раз и запоминаются.
    hashed: dict[Path, tuple[str, Optional[Exception]]] = {}
    expected, seen = 0, set()
    for _, folder_path in input_groups:
        for p in _list_photos(folder_path):
            try:
                h, err = photo_fingerprint(str(p)), None
            except Exception as e:      # файл не читается — при чтении станет ERROR
                h, err = "", e
            hashed[p] = (h, err)
            if _run_selection.decide(index.rows(p.name, h), bool(h) and h in seen, serial_in_table,
                                     reread=config.reread_errors) == RunSelection.READ:
                expected += 1
                seen.add(h)
    log.info(f"Новых фото к чтению: {expected}")
    stop_file = MonthFolder(Path(config.month_dir)).stop_file if config.month_dir else None
    done, stopped = 0, False

    total_stats: dict[Outcome, int] = {o: 0 for o in Outcome}
    total_skipped: dict[str, int] = {}
    per_folder_stats: dict[str, dict[Outcome, int]] = {}

    for subfolder_name, folder_path in input_groups:
        if stopped:
            break
        log.info(f"\n{'=' * 55}")
        log.info(f"Папка контролёра: {subfolder_name}")
        log.info("=" * 55)
        output_base = str(Path(config.output_base_dir) / subfolder_name)

        photos = _list_photos(folder_path)
        log.info(f"Фото в папке: {len(photos)}\n")

        stats: dict[Outcome, int] = {o: 0 for o in Outcome}
        skipped: dict[str, int] = {}

        for photo_path_obj in photos:
            photo_path = photo_path_obj  # совместимость с process_photo (ожидает str)

            # ── Читать или пропустить (этап 3: каждое фото разбирается один раз)
            photo_hash, hash_error = hashed[photo_path]
            rows = index.rows(photo_path.name, photo_hash)
            choice = _run_selection.decide(
                rows, bool(photo_hash) and photo_hash in config._processed_hashes_cache,
                serial_in_table, reread=config.reread_errors)
            if choice != RunSelection.READ:
                skipped[choice] = skipped.get(choice, 0) + 1
                total_skipped[choice] = total_skipped.get(choice, 0) + 1
                continue

            if stop_file is not None and stop_file.exists():   # «Остановить» в окне (этап 4)
                stopped = True
                log.warning(f"⏹ Остановлено оператором: прочитано {done} из {expected}. "
                            "Остальные фото прочитает следующий запуск.")
                break
            if models is None:                       # до строки «Фото 1 из N»: окно покажет «Загружаем модели…»
                models = load()
                if models.account_recognizer is not None:
                    account_groups_cache = account_groups_of(df, config)
                if config.serial_by_table:
                    serial_groups_cache = serial_groups_of(df, config)
            done += 1
            log.info(f"Фото {done} из {expected}: {photo_path.name}" + ("  (читается заново)" if rows else ""))
            if rows:
                # прежний файл результата этого фото (question/…, not_in_db/) убрать:
                # сейчас фото разложится заново
                old = result_file(Path(config.output_base_dir), rows[-1])
                if old is not None and old.is_file():
                    old.unlink()

            try:
                if hash_error is not None:
                    raise hash_error
                result = process_photo(
                    str(photo_path), df, config,
                    models.meter_detector, models.digit_detector,
                    models.digit_recognizer, models.serial_recognizer,
                    photo_hash=photo_hash, reread=bool(rows),
                    account_recognizer=models.account_recognizer,
                    account_groups=account_groups_cache,
                    serial_groups=serial_groups_cache,
                )
            except Exception as e:
                # Ошибка на одном фото не останавливает прогон: фото → question/error,
                # при следующем прогоне читается заново. Ctrl+C (KeyboardInterrupt)
                # сюда не попадает и останавливает прогон, как раньше.
                log.exception(f"  Ошибка программы на фото {photo_path.name}")
                result = PhotoResult(
                    photo_path=str(photo_path), outcome=Outcome.ERROR,
                    error_detail=f"ошибка программы: {type(e).__name__}: {e}",
                )
            stats[result.outcome] += 1
            total_stats[result.outcome] += 1

            detail = result.error_detail or ""
            log.info(
                f"  serial={result.serial_text!r}({result.serial_conf or 0:.2f})  "
                f"reading={result.reading_str!r}  delta={result.delta}  "
                f"→ {result.outcome.name}  {detail}"
            )

            # Сохраняем аннотированное фото в нужную папку. Имя занято другим
            # фото — <имя>_2, <имя>_3… (этап 3: файлы не затирают друг друга)
            dst_dir  = str(Path(output_base) / OUTCOME_FOLDER[result.outcome])
            new_name = free_name(Path(dst_dir), result.new_photo_name or photo_path.name)
            if new_name != (result.new_photo_name or photo_path.name):
                result.new_photo_name = new_name
            try:
                _save_annotated(
                    str(photo_path), dst_dir, new_name, result,
                    move=config.move_photos,
                    draw_boxes=config.draw_boxes,
                )
            except OSError:
                if result.outcome != Outcome.ERROR:
                    raise
                # файл не читается совсем — остаётся во входной папке, в логе ERROR
                log.exception(f"  Не удалось скопировать {photo_path.name} в {dst_dir}")
            log.info(f"  📁 → {output_base}/{OUTCOME_FOLDER[result.outcome]}/{new_name}")

            # Показание (PLUS/MINUS) и строка лога — после каждого фото
            log_row = _make_log_row(
                photo_path_obj.name, result, config,
                photo_hash=photo_hash, source_folder=subfolder_name,
            )
            closed = store.record(result, log_row, photo_path_obj.name)
            for row in [log_row, *closed]:
                config._log_rows_cache.append(row)
                index.add(row)
            # Счёт «обработан в прогоне», только если получил показание: неудачное
            # фото не блокирует следующее фото того же счётчика (этап 3, пункт 3)
            if result.account_id and result.outcome in (Outcome.PLUS, Outcome.MINUS):
                config._processed_accounts_cache.add(result.account_id)
            config._processed_hashes_cache.add(photo_hash)

        if skipped:
            log.info("Пропущено: " + "; ".join(_skip_lines(skipped)))
        per_folder_stats[subfolder_name] = stats

        report_text = _format_report(f"ОТЧЁТ: {subfolder_name}", stats, skipped,
                                     {subfolder_name: not_taken.get(subfolder_name, [])}, folder_report=True)
        report_path = _save_report(output_base, report_text)

        log.info("\n" + report_text)
        log.info(f"💾 Отчёт сохранён: {report_path}")

    overall_report_text = _format_report("ОБЩИЙ ОТЧЁТ ПО ВСЕМ ПАПКАМ", total_stats, total_skipped,
                                         {k: v for k, v in not_taken.items() if v} or None)
    if len(per_folder_stats) > 1:
        ranking_text = _format_folder_ranking(per_folder_stats)
        if ranking_text:
            overall_report_text = overall_report_text + "\n\n" + ranking_text

    overall_report_path = _save_report(config.output_base_dir, overall_report_text)
    log.info("\n" + overall_report_text)
    log.info(f"💾 Общий отчёт сохранён: {overall_report_path}")

    store.finish()


# ─── Точка входа ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    safe_console()
    if len(sys.argv) != 2:
        print("Запуск: python reader.py <папка месяца>  (то же, что python gmr.py process <папка месяца>)")
        sys.exit(2)
    try:
        run_pipeline(PipelineConfig(month_dir=sys.argv[1]))
    except (NotAMonth, PhotosInRoot) as e:
        print(f"ОШИБКА: {e}")
        sys.exit(1)
