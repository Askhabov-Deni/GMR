"""Настройки окна (settings.json рядом с program2.py): оператор и папка месяца."""
import json
from dataclasses import asdict, dataclass
from datetime import datetime

from .common import _BASE

SETTINGS_FILE = _BASE / "settings.json"

# ─── Настройки ────────────────────────────────────────────────────────────────
@dataclass
class AppSettings:
    operator_name: str = ""
    month_dir:     str = ""   # папка месяца: база, фото, результат
    export_format: str = "xlsx"   # «Выгрузить»: в каком виде в прошлый раз (month.EXPORT_FORMATS)
    # «Папки для разметки» больше нет (2026-10-07): исправления оператора идут
    # в папку месяца (<месяц>/разметка); старый ключ в settings.json не читается.

RU_MONTHS = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь", "Июль", "Август",
             "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"]


def default_month_name(day=None) -> str:
    """Имя папки нового месяца по умолчанию: «Октябрь_2026»."""
    day = day or datetime.now()
    return f"{RU_MONTHS[day.month - 1]}_{day.year}"

# до 1.3.0 программа называлась «GMR Оператор» и лежала в папке с этим именем:
# новая версия, распакованная рядом (папка «GMR»), берёт имя и месяц оттуда
OLD_SETTINGS = SETTINGS_FILE.parent.parent / "GMR Оператор" / "settings.json"


def load_settings() -> AppSettings:
    for path in (SETTINGS_FILE, OLD_SETTINGS):
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                return AppSettings(**{k: d[k] for k in AppSettings.__dataclass_fields__ if k in d})
            except Exception:
                pass
    return AppSettings()

def save_settings(s: AppSettings) -> None:
    SETTINGS_FILE.write_text(
        json.dumps(asdict(s), ensure_ascii=False, indent=2), encoding="utf-8"
    )
