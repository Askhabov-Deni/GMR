"""
src/gmr/storage/log_store.py — лог обработки фото (Фаза 2b, этап 2.3b).

  - LOG_COLUMNS    — столбцы лога; порядок не менять, новые — только в конец
                     (docs/contract_reader_program2.md, раздел 4).
  - SqliteLogStore — таблица processing_log; её создаёт база месяца
                     (src/gmr/storage/month.py). Все столбцы TEXT, порядок
                     строк — порядок записи.
  - CsvLogStore, load_log, save_log, log_path_for — CSV-лог старого режима
                     (`<таблица>_log.csv`): при создании месяца рядом лежащий
                     лог переносится в базу (src/gmr/application/month.py).

Дозапись в CSV-лог и shadow-run (CSV + SQLite с построчной сверкой) убраны
вместе со старым режимом на этапе 2.3b: лог живёт в базе месяца, CSV — только
выгрузка `лог.csv`.
"""
import csv
import sqlite3
from pathlib import Path


LOG_COLUMNS = [
    "original_filename", "final_filename", "serial_id", "account_id",
    "reading", "last_reading", "delta", "outcome", "source", "processed_by",
    "processed_at", "verified_by", "verified_at", "model_serial_conf",
    "model_reading_str", "notes",
    # добавлены 2026-09-29 (идентичность фото, см. MIGRATION_STATUS.md):
    # отпечаток содержимого исходного фото и подпапка, из которой оно пришло.
    # Всегда в конце — порядок прежних столбцов не меняется.
    "photo_hash", "source_folder",
]


# ─── CSV-лог старого режима (чтение для переноса в базу месяца) ──────────────

class CsvLogStore:
    """
    CSV-лог в формате старого reader.py (тот же формат у выгрузки лог.csv).
    Лог старого формата (без photo_hash, source_folder) читается как есть —
    недостающих столбцов в строках просто нет.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path

    def load(self) -> list[dict]:
        if not Path(self.log_path).exists():
            return []
        with open(self.log_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def save(self, rows: list[dict]) -> None:
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


# ─── SQLite (таблица processing_log базы месяца) ──────────────────────────────

class SqliteLogStore:
    """
    processing_log в SQLite. Все столбцы TEXT (как в CSV — там тоже всё
    строки после csv.DictReader), плюс id autoincrement для сохранения
    порядка вставки при .load() — CSV гарантирует порядок вставки файлом
    построчно, SQLite без явного ORDER BY этого не гарантирует.

    Отсутствующие в row ключи (extrasaction="ignore" в CSV-аналоге)
    сохраняются как '' — та же семантика, что и csv.DictWriter с
    extrasaction="ignore" при отсутствующем значении (KeyError там на самом
    деле бы возник при insert, но reader.py всегда передаёт полный dict через
    _make_log_row — расхождение с CSV в этом крайнем случае не ожидается,
    но .append() ниже защищается через .get(col, "")).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        cols_sql = ", ".join(f'"{c}" TEXT NOT NULL DEFAULT \'\'' for c in LOG_COLUMNS)
        with self._connect() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS processing_log "
                f"(id INTEGER PRIMARY KEY AUTOINCREMENT, {cols_sql})"
            )
            # база, созданная до добавления новых столбцов, — дополняем
            existing = {r[1] for r in conn.execute("PRAGMA table_info(processing_log)")}
            for c in LOG_COLUMNS:
                if c not in existing:
                    conn.execute(
                        f'ALTER TABLE processing_log ADD COLUMN "{c}" TEXT NOT NULL DEFAULT \'\''
                    )

    def load(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {', '.join(LOG_COLUMNS)} FROM processing_log ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def save(self, rows: list[dict]) -> None:
        """Полная перезапись."""
        placeholders = ", ".join(f":{c}" for c in LOG_COLUMNS)
        col_list = ", ".join(LOG_COLUMNS)
        with self._connect() as conn:
            conn.execute("DELETE FROM processing_log")
            conn.executemany(
                f"INSERT INTO processing_log ({col_list}) VALUES ({placeholders})",
                [{c: str(r.get(c, "")) for c in LOG_COLUMNS} for r in rows],
            )

    def append(self, row: dict) -> None:
        col_list = ", ".join(LOG_COLUMNS)
        placeholders = ", ".join(f":{c}" for c in LOG_COLUMNS)
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO processing_log ({col_list}) VALUES ({placeholders})",
                {c: str(row.get(c, "")) for c in LOG_COLUMNS},
            )


# ─── Функции-обёртки над CSV-логом ───────────────────────────────────────────

def log_path_for(table_path: str) -> str:
    """<table_name>_log.csv рядом с таблицей."""
    p = Path(table_path)
    return str(p.parent / (p.stem + "_log.csv"))


def load_log(log_path: str) -> list[dict]:
    """Загружает лог; возвращает [] если файл не существует."""
    return CsvLogStore(log_path).load()


def save_log(log_path: str, rows: list[dict]) -> None:
    """Перезаписывает весь лог."""
    CsvLogStore(log_path).save(rows)
