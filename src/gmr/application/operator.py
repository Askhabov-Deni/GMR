"""
src/gmr/application/operator.py — действия окна оператора с базой месяца
(этап 2.3, 2026-10-01). Окна здесь нет: program2.py только вызывает эти
методы, а проверяются они обычными тестами.

Каждое действие — одна транзакция: показание, запись в журнал изменений
«кто, когда, было → стало» и строка лога пишутся вместе или не пишутся вовсе.
Пишется только то, что меняется (показание лицевого счёта, строка лога по её
номеру), а не таблица и лог целиком, — поэтому окно не затирает того, что
в это же время записал reader.py (известные ошибки этапа 2).
"""
from pathlib import Path
from typing import Optional

from src.gmr.application.month import (
    SERIAL_FIX_ACTION, ExportReport, NotAMonth, export_month, is_month,
)
from src.gmr.domain.config import PipelineConfig
from src.gmr.domain.photo_date import reading_date
from src.gmr.storage.backup import backup_sqlite
from src.gmr.storage.month import MonthDB, MonthFolder, Reading, now_text

CORRECTED_NOTE = "исправлено при проверке"   # по этой пометке tools/analyze_log.py считает исправления


def _num(v) -> Optional[float]:
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        return None


class OperatorSession:
    """База месяца для окна оператора."""

    def __init__(self, month_dir: str, config: Optional[PipelineConfig] = None):
        self.folder = MonthFolder(Path(month_dir))
        if not is_month(self.folder):
            raise NotAMonth(f"Месяц не создан: {self.folder.root}. Сначала: "
                            f"python gmr.py month \"{self.folder.root}\" --table <таблица компании>")
        self.cfg = config or PipelineConfig()
        self.db = MonthDB(self.folder.db)

    def close(self) -> None:
        self.db.close()

    def backup(self) -> Optional[Path]:
        return backup_sqlite(self.folder.db, self.folder.backups)

    # ─── чтение ─────────────────────────────────────────────────────────────
    def table_columns(self) -> list[str]:
        cfg = self.cfg
        return [cfg.col_serial, cfg.col_account_id, cfg.col_last_reading, cfg.col_new_reading]

    def table_rows(self) -> list[dict]:
        """Абоненты в таблице: номер, лицевой счёт, прошлое и текущее показание
        (для поиска в окне — столбцы table_columns(), как у таблицы компании)."""
        cfg = self.cfg
        readings = self.db.readings()
        return [{cfg.col_serial: a.serial, cfg.col_account_id: a.account,
                 cfg.col_last_reading: a.last_reading,
                 cfg.col_new_reading: readings[a.account].value if a.account in readings else ""}
                for a in self.db.abonents(only_in_table=True).values()]

    def log_rows(self) -> list[dict]:
        return self.db.log_rows(with_id=True)

    def reading(self, account: str) -> Optional[Reading]:
        return self.db.reading(account)

    # ─── запись ─────────────────────────────────────────────────────────────
    def add_row(self, row: dict) -> None:
        """Строка лога без показания: «Дубль», «Нечитаемо», «Нет в базе»."""
        with self.db.transaction():
            self.db.append_log_rows([row])

    def _put_reading(self, account: str, value: str, date: str, photo: str, who: str,
                     action: str) -> None:
        old = self.db.reading(account)
        self.db.put_reading(Reading(account, value, date, "manual", photo, now_text(), who))
        self.db.add_change(who, action, account, self.cfg.col_new_reading,
                           old.value if old else "", value, note=photo)

    def accept(self, row: dict, date_name: str) -> None:
        """«Принять»: показание row["reading"] лицевому счёту row["account_id"]
        и строка лога. date_name — имя исходного фото (из него — «Дата»)."""
        with self.db.transaction():
            self._put_reading(row["account_id"], row["reading"], reading_date(date_name),
                              row.get("original_filename", ""), row["processed_by"], "показание записано")
            self.db.append_log_rows([row])

    def fix_serial_and_accept(self, row: dict, photo_serial: str, date_name: str) -> str:
        """«Серийник в базе с ошибкой» (решение владельца 2026-10-01): номер
        абонента в базе исправляется на номер с фото, показание записывается
        сразу. Возвращает номер, который был в базе."""
        account, who = row["account_id"], row["processed_by"]
        with self.db.transaction():
            ab = self.db.abonent(account)
            if ab is None:
                raise ValueError(f"Лицевого счёта {account} нет в базе месяца")
            old_serial = ab.serial
            self.db.set_serial(account, photo_serial)
            self.db.add_change(who, SERIAL_FIX_ACTION, account, self.cfg.col_serial,
                               old_serial, photo_serial, note=row.get("original_filename", ""))
            self._put_reading(account, row["reading"], reading_date(date_name),
                              row.get("original_filename", ""), who, "показание записано")
            self.db.append_log_rows([row])
        return old_serial

    def mark_verified(self, row: dict, who: str, at: str) -> None:
        """«Верно» на вкладке «Проверка» — отметка ровно в эту строку лога."""
        with self.db.transaction():
            self.db.update_log_row(row["_id"], {"verified_by": who, "verified_at": at})
        row.update(verified_by=who, verified_at=at)

    def apply_correction(self, row: dict, c, who: str, at: str) -> None:
        """
        Исправление на вкладке «Проверка» (c — program2.VerifyCorrection).
        Показание переносится к новому абоненту; у прежнего стирается, только
        если там всё ещё то, что записал reader.py. Строка лога меняется на
        месте (row — с "_id"), исходное чтение модели остаётся в
        model_reading_str.
        """
        photo = row.get("original_filename", "")
        with self.db.transaction():
            if c.account_changed:
                old = self.db.reading(c.old_account)
                if old is not None and _num(old.value) == _num(row.get("reading")):
                    self.db.delete_reading(c.old_account)
                    self.db.add_change(who, "показание стёрто при проверке", c.old_account,
                                       self.cfg.col_new_reading, old.value, "",
                                       note=f"{photo}: абонент {c.old_account} → {c.new_account}")
            prev = self.db.reading(c.new_account)
            date = prev.date if prev is not None and prev.date else reading_date(photo)
            self._put_reading(c.new_account, str(c.reading), date, photo, who,
                              "показание исправлено при проверке")
            note = CORRECTED_NOTE
            if c.account_changed:
                note += f" (абонент {c.old_account} → {c.new_account})"
            ext = Path(row.get("final_filename") or photo).suffix
            fields = {
                "serial_id":    c.serial,
                "account_id":   c.new_account,
                "reading":      str(c.reading),
                "last_reading": str(c.last_reading) if c.last_reading is not None else "",
                "delta":        str(c.delta) if c.delta is not None else "",
                "outcome":      c.outcome,
                "final_filename": f"{c.new_account}{ext}" if c.account_changed else row.get("final_filename", ""),
                "verified_by":  who,
                "verified_at":  at,
                "notes":        (row.get("notes", "") + " | " + note).strip(" |"),
            }
            self.db.update_log_row(row["_id"], fields)
        row.update(fields)

    def rename_photo_in_log(self, row: dict, final_filename: str) -> None:
        """Имя фото в plus/ или minus/ после переноса (если имя было занято)."""
        with self.db.transaction():
            self.db.update_log_row(row["_id"], {"final_filename": final_filename})
        row["final_filename"] = final_filename

    def export(self) -> ExportReport:
        return export_month(str(self.folder.root), self.cfg)
