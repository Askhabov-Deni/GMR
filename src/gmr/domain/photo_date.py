"""
src/gmr/domain/photo_date.py — дата показания по имени файла (2026-10-01).

WhatsApp стирает из фото дату съёмки, но дата получения есть в имени файла:
  IMG-20261001-WA0012.jpg                 (Android, папка WhatsApp Images)
  WhatsApp Image 2026-10-01 at 14.32.05.jpeg   (WhatsApp на компьютере)
Решение владельца 2026-10-01: в «Дату» реестра пишется эта дата, а если в
имени её нет — дата обработки.
"""
import datetime as dt
import re

DATE_FORMAT = "%d %m %Y"          # как в столбце «Дата» реестра
_DATE_IN_NAME = re.compile(r"(?<!\d)(20\d\d)[-_.]?([01]\d)[-_.]?([0-3]\d)(?!\d)")


def date_from_filename(name: str):
    """Дата из имени файла (datetime.date) или None."""
    for y, m, d in _DATE_IN_NAME.findall(name):
        try:
            return dt.date(int(y), int(m), int(d))
        except ValueError:
            continue
    return None


def reading_date(name: str, today=None) -> str:
    """Текст для «Даты»: из имени файла, иначе today (по умолчанию — сегодня)."""
    found = date_from_filename(name)
    return (found or today or dt.date.today()).strftime(DATE_FORMAT)
