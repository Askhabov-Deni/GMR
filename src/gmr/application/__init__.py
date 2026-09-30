"""
src/gmr/application — сценарии поверх domain и контрактов моделей.
Не импортирует torch/cv2/pandas/tkinter.
"""
from .recognition import (
    DigitReading,
    PhotoRecognition,
    RecognitionModels,
    SUBSTITUTED_PREFIX,
    describe_substitutions,
    digit_crops_by_position,
    find_detection,
    read_meter_digits,
    read_meter_digits_for_config,
    recognize_photo,
)

__all__ = [
    "DigitReading",
    "PhotoRecognition",
    "RecognitionModels",
    "SUBSTITUTED_PREFIX",
    "describe_substitutions",
    "digit_crops_by_position",
    "find_detection",
    "read_meter_digits",
    "read_meter_digits_for_config",
    "recognize_photo",
]
