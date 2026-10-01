"""
Тесты storage-слоя processing log (Фаза 2b, docs/MIGRATION_TZ.md).

1. CsvLogStore пишет файл байт-в-байт как старый _save_log из reader.py
   (эталон скопирован ниже из коммита 92199d2 как есть) — так же устроена
   выгрузка лог.csv.
2. SqliteLogStore (лог в базе месяца) возвращает те же строки, что и CSV, в
   том же порядке, включая неудобные значения: кириллица, запятые, кавычки,
   переводы строк, пустые поля, ведущие нули.
Дозапись в CSV-лог и shadow-run (сверка CSV/SQLite) убраны со старым режимом
на этапе 2.3b.
"""
import csv

from src.gmr.storage import LOG_COLUMNS, CsvLogStore, SqliteLogStore
from src.gmr.storage import load_log, save_log  # noqa: E402


# ─── Эталон: старая реализация из reader.py (коммит 92199d2), без изменений ──

def _old_save_log(log_path, rows):
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _row(**overrides):
    row = {c: "" for c in LOG_COLUMNS}
    row.update(overrides)
    return row


TRICKY_ROWS = [
    _row(original_filename="IMG_0001.jpg", final_filename="1300000013.jpg",
         serial_id="007123", account_id="1300000013", reading="1200",
         last_reading="1000.0", delta="200.0", outcome="PLUS", source="auto",
         processed_by="auto", processed_at="2026-09-28T10:00:00+00:00",
         model_serial_conf="0.9512", model_reading_str="01200"),
    _row(original_filename="фото с пробелом, и запятой.jpg", outcome="DIGITS_ERROR",
         notes='expected 5 digits, got 3; "кавычки"', model_reading_str="5?3?1"),
    _row(original_filename="multi.jpg", outcome="SUSPICIOUS",
         notes="аномалия: данные в таблице есть,\nфото не найдено в логе"),
    _row(original_filename="__pre_existing__", source="pre_existing",
         outcome="UNKNOWN", account_id="1300000014",
         notes="инициализировано из таблицы"),
    _row(original_filename="empty.jpg", outcome="NO_METER"),
    # пробелы по краям: серийник из таблицы как в golden test case15,
    # хранилище обязано сохранить значение как есть, без strip()
    _row(original_filename=" spaced.jpg ", serial_id=" 007123 ", notes="  хвост  "),
]


# ─── 1. CSV: байт-в-байт как старый код ──────────────────────────────────────

def test_csv_save_identical_bytes_to_old_implementation(tmp_path):
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    _old_save_log(old, TRICKY_ROWS)
    CsvLogStore(str(new)).save(TRICKY_ROWS)
    assert old.read_bytes() == new.read_bytes()


def test_csv_load_missing_file_returns_empty(tmp_path):
    assert CsvLogStore(str(tmp_path / "nope.csv")).load() == []


def test_wrappers_use_same_format(tmp_path):
    # load_log/save_log (раньше reader._load_log и т.д.) — формат файла прежний
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    _old_save_log(old, TRICKY_ROWS)
    save_log(str(new), TRICKY_ROWS)
    assert old.read_bytes() == new.read_bytes()
    assert load_log(str(new)) == CsvLogStore(str(old)).load()
    # порядок столбцов — контракт с program2.py и старыми логами
    # (docs/contract_reader_program2.md, раздел 4): новые — только в конец
    assert LOG_COLUMNS == [
        "original_filename", "final_filename", "serial_id", "account_id", "reading",
        "last_reading", "delta", "outcome", "source", "processed_by", "processed_at",
        "verified_by", "verified_at", "model_serial_conf", "model_reading_str", "notes",
        "photo_hash", "source_folder",
    ]


# ─── 2. SQLite == CSV ────────────────────────────────────────────────────────

def test_sqlite_append_roundtrip_matches_csv(tmp_path):
    csv_store = CsvLogStore(str(tmp_path / "log.csv"))
    sql_store = SqliteLogStore(str(tmp_path / "log.sqlite"))
    csv_store.save(TRICKY_ROWS)
    for r in TRICKY_ROWS:
        sql_store.append(r)
    assert sql_store.load() == csv_store.load()


def test_sqlite_save_replaces_everything(tmp_path):
    sql_store = SqliteLogStore(str(tmp_path / "log.sqlite"))
    sql_store.append(_row(original_filename="old.jpg"))
    sql_store.save(TRICKY_ROWS[:2])
    assert [r["original_filename"] for r in sql_store.load()] == [
        r["original_filename"] for r in TRICKY_ROWS[:2]
    ]


def test_sqlite_keeps_leading_zeros_as_text(tmp_path):
    sql_store = SqliteLogStore(str(tmp_path / "log.sqlite"))
    sql_store.append(_row(serial_id="007123", reading="01200"))
    loaded = sql_store.load()[0]
    assert loaded["serial_id"] == "007123"
    assert loaded["reading"] == "01200"


def test_sqlite_missing_keys_become_empty_string(tmp_path):
    csv_store = CsvLogStore(str(tmp_path / "log.csv"))
    sql_store = SqliteLogStore(str(tmp_path / "log.sqlite"))
    partial = {"original_filename": "x.jpg", "outcome": "PLUS"}
    csv_store.save([partial])
    sql_store.append(partial)
    assert sql_store.load() == csv_store.load()


def test_sqlite_reopen_preserves_data(tmp_path):
    path = str(tmp_path / "log.sqlite")
    SqliteLogStore(path).append(TRICKY_ROWS[0])
    assert SqliteLogStore(path).load()[0]["reading"] == "1200"
