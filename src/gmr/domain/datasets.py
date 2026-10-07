"""
src/gmr/domain/datasets.py — где лежат датасеты моделей и какое трудное фото
идёт в эталон (этап 5, решения владельца 2026-10-04). Без тяжёлых библиотек:
это нужно и окну оператора (тихая разметка), и инструментам обучения
(models/datasets.py, tools/etalon.py).
"""
import hashlib
from pathlib import Path

DATASETS_DIR = Path("database") / "datasets"
ETALON_SHARE = 1 / 3


def to_etalon(photo_hash: str) -> bool:
    """Трудное фото — в эталон или в обучение: каждое третье по отпечатку
    фото. Одно и то же фото — всегда в одно место: эталон (tools/etalon.py)
    и папки new/ (окно оператора) не пересекаются."""
    x = int(hashlib.sha256(f"etalon:{photo_hash}".encode("utf-8")).hexdigest()[:12], 16) / 16 ** 12
    return x < ETALON_SHARE
