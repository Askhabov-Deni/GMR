"""
Что прогон пишет по каждому фото: строка лога, фото с подписью в папке
результата, отладочные кропы цифр (config.debug_digits).
"""
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.gmr.domain import PhotoResult, PipelineConfig
from src.gmr.domain.datasets import standard_class
from src.gmr.render import draw_annotation, read_image, turn_image, write_image

# ─── Цвета боксов ────────────────────────────────────────────────────────────

_BOX_COLORS = {                         # классы — одним словом (standard_class)
    "meter":  (0,   200,  0),           # зелёный
    "serial": (200,  0,   0),           # синий (BGR)
    "digit":  (0,   140, 255),          # оранжевый (BGR)
}
_BOX_THICKNESS = 2


def make_log_row(
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

    meter         — зелёный   — из meter_crops_raw
    serial        — синий     — из meter_crops_raw
    digit         — оранжевый — из digit_bboxes_in_orig (уже в координатах оригинала)

    Подписи классов не рисуются.
    """
    # Боксы от meter_detector (meter и serial)
    if result.meter_crops_raw:
        for entry in result.meter_crops_raw:
            cls   = entry["class"]
            color = _BOX_COLORS.get(standard_class(cls), (128, 128, 128))
            x1, y1, x2, y2 = entry["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, _BOX_THICKNESS)

    # Боксы цифр (уже пересчитаны в координаты оригинала)
    if result.digit_bboxes_in_orig:
        color = _BOX_COLORS["digit"]
        for (x1, y1, x2, y2) in result.digit_bboxes_in_orig:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, _BOX_THICKNESS)


def save_annotated(
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


def save_digit_debug(
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
