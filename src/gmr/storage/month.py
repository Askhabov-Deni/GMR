"""
src/gmr/storage/month.py — папка месяца и её база (этап 2.2a, 2026-10-01).

Решения владельца 2026-10-01 (docs/BACKLOG.md, этап 2): один месяц — одна
папка; таблица компании копируется в базу; лог — тоже в базе; таблицу и лог
руками не правят; результат — выгрузка показания.xlsx.

    <папка месяца>/
      фото/<контролёр>/…     сюда докладываются фото
      результат/…            раскладка reader.py (plus/, question/…)
      gmr.sqlite             база месяца
      показания.xlsx         выгрузка для компании
      лог.csv                выгрузка лога для просмотра и `gmr.py analyze`
      таблицы/               копии загруженных таблиц компании и отчёты загрузки
      gmr_backups/           копии базы перед изменениями

База — SQLite (стандартная библиотека). Таблицы:
  meta       — служебное: столбцы таблицы компании, когда создан месяц;
  abonents   — абоненты из таблицы компании (все её столбцы — в data, JSON);
  readings   — текущее показание по лицевому счёту (то, что уйдёт в выгрузку);
  changes    — журнал изменений: кто, когда, что было и что стало;
  processing_log — лог обработки фото, те же столбцы, что CSV-лог
               (src/gmr/storage/log_store.py, SqliteLogStore).
"""
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from .log_store import LOG_COLUMNS, SqliteLogStore

SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class MonthFolder:
    root: Path

    @property
    def db(self) -> Path: return self.root / "gmr.sqlite"
    @property
    def photos(self) -> Path: return self.root / "фото"
    @property
    def results(self) -> Path: return self.root / "результат"
    @property
    def export_xlsx(self) -> Path: return self.root / "показания.xlsx"
    @property
    def export_log(self) -> Path: return self.root / "лог.csv"
    @property
    def tables(self) -> Path: return self.root / "таблицы"
    @property
    def backups(self) -> Path: return self.root / "gmr_backups"
    # окно оператора просит прогон остановиться (кнопка «Остановить», этап 4):
    # reader.py проверяет файл между фото и заканчивает аккуратно
    @property
    def stop_file(self) -> Path: return self.root / ".gmr_stop"

    def exists(self) -> bool:
        return self.db.is_file()


@dataclass
class Abonent:
    account: str
    serial: str
    last_reading: str
    in_table: bool
    row_order: int
    data: dict


@dataclass
class Reading:
    account: str
    value: str
    date: str
    source: str        # table — было в таблице при загрузке; auto / manual — с этапа 2.2b
    photo: str
    updated_at: str
    updated_by: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS abonents (
    account      TEXT PRIMARY KEY,
    serial       TEXT NOT NULL,
    last_reading TEXT NOT NULL,
    in_table     INTEGER NOT NULL,
    row_order    INTEGER NOT NULL,
    data         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS abonents_serial ON abonents (serial);
CREATE TABLE IF NOT EXISTS readings (
    account    TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    date       TEXT NOT NULL DEFAULT '',
    source     TEXT NOT NULL,
    photo      TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL,
    updated_by TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS changes (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    at      TEXT NOT NULL,
    who     TEXT NOT NULL,
    action  TEXT NOT NULL,
    account TEXT NOT NULL DEFAULT '',
    field   TEXT NOT NULL DEFAULT '',
    old     TEXT NOT NULL DEFAULT '',
    new     TEXT NOT NULL DEFAULT '',
    note    TEXT NOT NULL DEFAULT ''
);
"""


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class MonthDB:
    """База месяца. Все изменения внутри `with db.transaction():` записываются
    целиком или не записываются вовсе."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path), timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        SqliteLogStore(str(self.path))          # таблица processing_log
        if self.meta("schema") is None:
            with self.transaction():
                self.set_meta("schema", SCHEMA_VERSION)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.conn:                          # commit или rollback целиком
            yield self.conn

    # ─── meta ───────────────────────────────────────────────────────────────
    def meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def columns(self) -> list[str]:
        return json.loads(self.meta("columns") or "[]")

    # ─── абоненты ───────────────────────────────────────────────────────────
    def abonents(self, only_in_table: bool = False) -> dict[str, Abonent]:
        sql = "SELECT * FROM abonents" + (" WHERE in_table=1" if only_in_table else "") + " ORDER BY row_order"
        return {r["account"]: Abonent(r["account"], r["serial"], r["last_reading"], bool(r["in_table"]),
                                      r["row_order"], json.loads(r["data"]))
                for r in self.conn.execute(sql)}

    def abonent(self, account: str) -> Optional[Abonent]:
        r = self.conn.execute("SELECT * FROM abonents WHERE account=?", (account,)).fetchone()
        return Abonent(r["account"], r["serial"], r["last_reading"], bool(r["in_table"]),
                       r["row_order"], json.loads(r["data"])) if r else None

    def put_abonent(self, a: Abonent) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO abonents (account, serial, last_reading, in_table, row_order, data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (a.account, a.serial, a.last_reading, int(a.in_table), a.row_order,
             json.dumps(a.data, ensure_ascii=False)),
        )

    def set_serial(self, account: str, serial: str) -> None:
        self.conn.execute("UPDATE abonents SET serial=? WHERE account=?", (serial, account))

    def accounts_by_serial(self, serial: str) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT account FROM abonents WHERE serial=? AND in_table=1 ORDER BY row_order", (serial,))]

    # ─── показания ──────────────────────────────────────────────────────────
    def readings(self) -> dict[str, Reading]:
        return {r["account"]: Reading(**dict(r)) for r in self.conn.execute("SELECT * FROM readings")}

    def reading(self, account: str) -> Optional[Reading]:
        row = self.conn.execute("SELECT * FROM readings WHERE account=?", (account,)).fetchone()
        return Reading(**dict(row)) if row else None

    def put_reading(self, r: Reading) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO readings (account, value, date, source, photo, updated_at, updated_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (r.account, r.value, r.date, r.source, r.photo, r.updated_at, r.updated_by),
        )

    def delete_reading(self, account: str) -> None:
        self.conn.execute("DELETE FROM readings WHERE account=?", (account,))

    # ─── журнал изменений ───────────────────────────────────────────────────
    def add_change(self, who: str, action: str, account: str = "", field: str = "",
                   old: str = "", new: str = "", note: str = "") -> None:
        self.conn.execute(
            "INSERT INTO changes (at, who, action, account, field, old, new, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (now_text(), who, action, account, field, old, new, note),
        )

    def changes(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM changes ORDER BY id")]

    # ─── лог обработки ──────────────────────────────────────────────────────
    def log_rows(self, with_id: bool = False) -> list[dict]:
        """Строки лога по порядку. with_id — с ключом "_id" (номер строки в базе:
        по нему окно оператора меняет отметку проверки и исправления)."""
        return self._log_where("1", (), with_id)

    def log_rows_of_account(self, account: str, outcomes: tuple) -> list[dict]:
        """Авто-строки лога лицевого счёта с этими исходами (по порядку)."""
        marks = ", ".join("?" for _ in outcomes)
        return self._log_where(f'source=\'auto\' AND account_id=? AND outcome IN ({marks})',
                               (account, *outcomes))

    def log_rows_of_photo(self, photo_hash: str, name: str) -> list[dict]:
        """Строки лога одного фото — как ProcessedPhotoPolicy.row_matches: по
        отпечатку, а строки без отпечатка — по имени исходного файла."""
        if photo_hash:
            return self._log_where("photo_hash=? OR (photo_hash='' AND original_filename=?)",
                                   (photo_hash, name))
        return self._log_where("original_filename=?", (name,))

    def _log_where(self, where: str, params: tuple, with_id: bool = False) -> list[dict]:
        cols = ", ".join(f'"{c}"' for c in LOG_COLUMNS)
        rows = self.conn.execute(f"SELECT id, {cols} FROM processing_log WHERE {where} ORDER BY id", params)
        out = []
        for r in rows:
            d = dict(r)
            row_id = d.pop("id")
            if with_id:
                d["_id"] = row_id
            out.append(d)
        return out

    def update_log_row(self, row_id: int, fields: dict) -> None:
        fields = {k: str(v or "") for k, v in fields.items() if k in LOG_COLUMNS}
        if fields:
            sets = ", ".join(f'"{k}"=?' for k in fields)
            self.conn.execute(f"UPDATE processing_log SET {sets} WHERE id=?", (*fields.values(), row_id))

    def append_log_rows(self, rows: list[dict]) -> None:
        cols = ", ".join(f'"{c}"' for c in LOG_COLUMNS)
        marks = ", ".join("?" for _ in LOG_COLUMNS)
        self.conn.executemany(f"INSERT INTO processing_log ({cols}) VALUES ({marks})",
                              [[str(r.get(c) or "") for c in LOG_COLUMNS] for r in rows])
