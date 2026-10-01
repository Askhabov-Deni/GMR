"""
src/gmr/storage/log_store.py — хранилища для processing log (Фаза 2b,
docs/MIGRATION_TZ.md).

Область действия этой фазы — ТОЛЬКО processing log (append-only журнал
обработанных фото, файл `<table>_log.csv`). Таблица счётчиков
(config.table_path, CSV/XLSX) не трогается: это import/export слой данных
оператора, а не журнал приложения (см. docs/MIGRATION_STATUS.md, раздел
"Фаза 2b").

Три класса:
  - CsvLogStore    — канонический перенос текущего CSV-поведения reader.py
                      (_load_log/_save_log/_append_log_row) без изменения
                      формата файла. Остаётся source of truth в этой фазе.
  - SqliteLogStore  — новое хранилище на sqlite3 (стандартная библиотека,
                      без новых зависимостей). Схема — те же _LOG_COLUMNS,
                      все поля TEXT, порядок строк = порядок вставки (id
                      autoincrement), чтобы .load() возвращал список в том
                      же порядке, что и CSV.
  - ShadowLogStore  — пишет одновременно в primary (CSV, источник истины)
                      и shadow (SQLite), сравнивает построчно на лету и
                      копит расхождения в .divergences. .load()/.save()
                      делегируются в primary — ShadowLogStore не подменяет
                      источник истины, только проверяет параллельную запись.

Все три реализуют один и тот же неявный протокол: load() -> list[dict],
save(rows) -> None, append(row) -> None — совместимый с тем, как reader.py
и program2.py уже используют _load_log/_save_log/_append_log_row.
"""
import csv
import sqlite3
from pathlib import Path
from typing import Optional, Protocol


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


class LogStore(Protocol):
    def load(self) -> list[dict]: ...
    def save(self, rows: list[dict]) -> None: ...
    def append(self, row: dict) -> None: ...


# ─── CSV (текущее поведение reader.py, без изменений) ────────────────────────

class CsvLogStore:
    """
    Прямой перенос reader.py:_load_log/_save_log/_append_log_row.
    Формат файла на диске не менялся ни на байт — это та же CSV-схема,
    что была раньше, просто оформленная как класс.
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

    def append(self, row: dict) -> None:
        self.upgrade_if_needed()
        exists = Path(self.log_path).exists()
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def header(self) -> Optional[list[str]]:
        """Шапка файла лога, или None если файла нет / он пустой."""
        if not Path(self.log_path).exists():
            return None
        with open(self.log_path, newline="", encoding="utf-8") as f:
            return next(csv.reader(f), None)

    def upgrade_if_needed(self) -> bool:
        """
        Лог старого формата (без столбцов, добавленных позже) переписывается
        с текущей шапкой: старые значения сохраняются, новые столбцы пустые.
        Без этого дозапись строки из LOG_COLUMNS под старую шапку сдвинула
        бы столбцы. Возвращает True, если файл был переписан.
        """
        head = self.header()
        if head is None or head == LOG_COLUMNS:
            return False
        rows = self.load()
        self.save(rows)
        return True


# ─── SQLite (новое) ───────────────────────────────────────────────────────────

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
        """Полная перезапись — аналог _save_log (используется при инициализации
        лога из таблицы, см. reader.py:_maybe_init_log)."""
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


# ─── Shadow-run harness ────────────────────────────────────────────────────────

class ShadowLogStore:
    """
    Пишет каждую строку одновременно в primary (source of truth — CSV,
    CsvLogStore) и shadow (SQLite, SqliteLogStore); построчная сверка —
    отдельным вызовом compare() в конце прогона. primary остаётся единственным источником истины —
    ShadowLogStore ничего не переключает, только проверяет, что SQLite
    воспроизводит то же самое.

    .divergences — список (row_index, primary_row, shadow_row) для строк,
    где primary и shadow разошлись. Пустой список после прогона — и есть
    условие готовности Фазы 2b ("shadow-run совпал построчно").

    load()/save() делегируются в primary — чтение всегда идёт из источника
    истины, shadow не участвует в принятии решений внутри process_photo/
    DuplicatePolicy (важно для кейса 8, раздел 2 ТЗ — рассинхронизация
    таблицы и лога не должна тихо замаскироваться вторым источником данных).
    """

    def __init__(self, primary: LogStore, shadow: LogStore):
        self.primary = primary
        self.shadow = shadow
        self.divergences: list[tuple[int, dict, dict]] = []

    def load(self) -> list[dict]:
        return self.primary.load()

    def save(self, rows: list[dict]) -> None:
        self.primary.save(rows)
        self.shadow.save(rows)

    def append(self, row: dict) -> None:
        self.primary.append(row)
        self.shadow.append(row)

    def compare(self) -> list[tuple[int, dict, dict]]:
        """Полная построчная сверка primary vs shadow. Вызывается один раз
        в конце прогона (run_pipeline), а не после каждой записи: сверка
        читает оба хранилища целиком, и вызов на каждом append давал бы
        O(n²) чтений CSV за прогон на тысячу фото."""
        self.divergences.clear()
        p_rows = self.primary.load()
        s_rows = self.shadow.load()
        for i, (p, s) in enumerate(zip(p_rows, s_rows)):
            p_norm = {c: str(p.get(c, "")) for c in LOG_COLUMNS}
            s_norm = {c: str(s.get(c, "")) for c in LOG_COLUMNS}
            if p_norm != s_norm:
                self.divergences.append((i, p_norm, s_norm))
        if len(p_rows) != len(s_rows):
            self.divergences.append((
                -1,
                {"_row_count": str(len(p_rows))},
                {"_row_count": str(len(s_rows))},
            ))
        return self.divergences

    def is_clean(self) -> bool:
        return not self.divergences


# ─── Функции-обёртки над CSV-логом (Фаза 6) ─────────────────────────────────
# Раньше жили в reader.py как _log_path/_load_log/_save_log/_append_log_row
# (program2.py импортировал их оттуда). Поведение то же.

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


def append_log_row(log_path: str, row: dict) -> None:
    """Дописывает одну строку в лог (создаёт файл с заголовком если нет)."""
    CsvLogStore(log_path).append(row)
