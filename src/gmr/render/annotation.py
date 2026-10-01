"""
src/gmr/render/annotation.py — подпись «Serial / Reading» в левом верхнем углу
фото. Перенесено из reader.py (_draw_annotation) без изменений (Фаза 6):
её использует и reader.py, и program2.py (перерисовка после ручной правки).
"""
import cv2
import numpy as np

from src.gmr.domain.models import PhotoResult


def draw_annotation(img: np.ndarray, result: PhotoResult) -> None:
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
