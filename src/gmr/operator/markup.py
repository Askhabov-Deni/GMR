"""Тихая разметка: исправления оператора — в <месяц>/разметка для дообучения моделей."""
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from src.gmr.domain.datasets import DIGIT, SERIAL, free_crop_name, photo_id
from src.gmr.render import write_image

from .common import log


def _free_name(folder: Path, photo: str, kind: str, n: int, answer: str = "") -> Path:
    taken = {p.name.lower() for p in folder.iterdir()} if folder.is_dir() else set()
    return folder / free_crop_name(taken, photo, kind, n, answer)

def save_crnn_markup(
    root: Path,
    photo: str,
    serial_crop: Optional[np.ndarray],
    correct_text: str,
    model_text: Optional[str],
) -> None:
    """Кроп серийника — только если номер исправили:
    serials_crnn/images/<счёт>__serial1__<номер>.jpg (номер — в имени)."""
    if serial_crop is None:
        return
    if model_text is not None and correct_text == model_text:
        return
    if not correct_text.isdigit():
        log.warning(f"CRNN разметка: номер «{correct_text}» не из цифр — не сохраняю")
        return
    folder = Path(root) / "serials_crnn" / "images"
    folder.mkdir(parents=True, exist_ok=True)
    path = _free_name(folder, photo, SERIAL, 1, correct_text)
    if not write_image(path, serial_crop):
        log.warning(f"CRNN разметка: не удалось сохранить {path.name}")
        return
    log.info(f"CRNN разметка: {path.name}")

def save_cnn_markup(
    root: Path,
    photo: str,
    digit_crops: Optional[list],
    digit_preds: Optional[list],
    model_reading_str: Optional[str],
    final_reading_str: str,
) -> None:
    """Кропы только изменённых цифр: digits_cnn/<цифра>/<счёт>__digit<N>.jpg."""
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
        out_path = _free_name(out_dir, photo, DIGIT, pos + 1)
        if not write_image(out_path, crop):
            log.warning(f"CNN разметка: не удалось сохранить {out_path}")
            continue
        log.info(f"CNN разметка: pos={pos} model={model_char!r} → correct={final_char!r}")

def save_meter_markup(root: Path, photo: str, original: Optional[Path]) -> None:
    """«Нет счётчика», а оператор ввёл показание — детектор промахнулся:
    исходное фото в meter_yolo/<счёт>.jpg на ручную разметку (решение 3а)."""
    if original is None or not Path(original).is_file():
        return
    out = Path(root) / "meter_yolo"
    out.mkdir(parents=True, exist_ok=True)
    ext = Path(original).suffix.lower()
    base, k = photo_id(photo), 1
    while (out / f"{base if k == 1 else f'{base}-{k}'}{ext}").exists():
        k += 1
    name = f"{base if k == 1 else f'{base}-{k}'}{ext}"
    shutil.copy2(original, out / name)
    log.info(f"YOLO разметка: {name} → meter_yolo")
