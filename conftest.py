"""Корневой conftest.py: гарантирует, что репозиторий (не site-packages) первым
в sys.path, чтобы локальный пакет tests/ не перекрывался одноимённым пакетом,
который тянут некоторые ML-зависимости (albumentations/ultralytics-thop)."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))
