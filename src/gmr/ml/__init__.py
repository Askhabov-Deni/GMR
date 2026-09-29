"""
src/gmr/ml — адаптеры реальных моделей к контрактам src/gmr/domain/ml.py.

Загрузчик моделей (torch, ultralytics) — в src/gmr/ml/loader.py и сюда
намеренно не реэкспортируется: `import src.gmr.ml` не тянет torch.
"""
from .adapters import (
    CnnDigitRecognizer,
    CrnnSerialRecognizer,
    YoloDigitDetector,
    YoloMeterDetector,
)

__all__ = [
    "CnnDigitRecognizer",
    "CrnnSerialRecognizer",
    "YoloDigitDetector",
    "YoloMeterDetector",
]
