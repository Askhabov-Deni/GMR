"""src/gmr/render — отрисовка на фото (cv2). Не domain: зависит от OpenCV."""
from .annotation import draw_annotation
from .image_io import read_image, turn_image, write_image

__all__ = ["draw_annotation", "read_image", "turn_image", "write_image"]
