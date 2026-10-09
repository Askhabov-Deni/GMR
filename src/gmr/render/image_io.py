"""
src/gmr/render/image_io.py — чтение и запись картинок при любых путях
(2026-10-01, решение владельца «сделать надёжнее»).

cv2.imread/cv2.imwrite на Windows в некоторых сборках OpenCV не открывают
пути с кириллицей (папки контролёров: «Сулиман С», «Аюб»): imread молча
возвращает None, imwrite — False. Эти функции читают и пишут байты через
Python (np.fromfile / ndarray.tofile), а кодирование делает OpenCV
(imdecode / imencode) — путь до OpenCV не доходит.

Поведение при ошибке — как у cv2: read_image → None, write_image → False.
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def read_image(path) -> Optional[np.ndarray]:
    """Как cv2.imread(path): BGR-массив или None, если файла нет / не картинка."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# поворот по часовой стрелке, градусы → код OpenCV (этап 7b: фото, где детектор ничего не нашёл)
TURNS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def turn_image(img: np.ndarray, degrees: int) -> np.ndarray:
    """Картинка, повёрнутая на 90 / 180 / 270° по часовой стрелке (0 — как есть)."""
    return cv2.rotate(img, TURNS[degrees]) if degrees else img


def write_image(path, img: np.ndarray) -> bool:
    """Как cv2.imwrite(path, img): True при успехе. Формат — по расширению (.jpg по умолчанию)."""
    try:
        ok, buf = cv2.imencode(Path(str(path)).suffix or ".jpg", img)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except (OSError, cv2.error):
        return False
