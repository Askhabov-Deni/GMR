"""Общее для окна: журнал окна (program2.log), цвета, папки очереди."""
import logging
import sys
from pathlib import Path

from src.gmr.domain import QUESTION_REASONS

_BASE = Path(__file__).resolve().parents[3]   # папка проекта (рядом — program2.py)
_LOG_FILE = _BASE / "program2.log"


def _log_handlers_for(stderr):
    """С ярлыка (pythonw.exe, этап 4) консоли нет (sys.stderr is None): журнал
    окна — в program2.log в папке программы (последние 4 файла по 1 МБ), иначе
    ошибки пропадали бы. Файл появляется с первой записью."""
    if stderr is not None:
        return None
    from logging.handlers import RotatingFileHandler
    return [RotatingFileHandler(_LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8", delay=True)]


_log_handlers = _log_handlers_for(sys.stderr)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    handlers=_log_handlers)
log = logging.getLogger("program2")

# ─── Константы ────────────────────────────────────────────────────────────────
PHOTO_EXTS = {".jpg", ".jpeg", ".png"}

# Подпапки question/ в порядке очереди и их читаемые названия — общие с итогом
# месяца (src/gmr/domain/models.py)
QUESTION_SUBFOLDERS = list(QUESTION_REASONS)
REASON_LABELS = dict(QUESTION_REASONS)

# Цвета
CLR_GREEN  = "#2e7d32"
CLR_RED    = "#c62828"
CLR_ORANGE = "#e65100"
CLR_GRAY   = "#757575"
CLR_BROWN  = "#5D4037"   # для кнопки "Нет в базе"
CLR_BG     = "#f5f5f5"
CLR_WHITE  = "#ffffff"
