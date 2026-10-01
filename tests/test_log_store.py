"""
Тесты storage-слоя processing log (Фаза 2b, docs/MIGRATION_TZ.md).

1. CsvLogStore пишет файл байт-в-байт как старые _save_log/_append_log_row
   из reader.py (эталон скопирован ниже из коммита 92199d2 как есть).
2. SqliteLogStore возвращает те же строки, что и CSV, в том же порядке,
   включая неудобные значения: кириллица, запятые, кавычки, переводы строк,
   пустые поля, ведущие нули.
3. ShadowLogStore.compare() действительно ловит расхождение, а не молчит.
"""
import csv
import sqlite3
from pathlib import Path

from src.gmr.storage import LOG_COLUMNS, CsvLogStore, ShadowLogStore, SqliteLogStore
from src.gmr.storage import append_log_row, load_log  # noqa: E402


# ─── Эталон: старая реализация из reader.py (коммит 92199d2), без изменений ──

def _old_save_log(log_path, rows):
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _old_append_log_row(log_path, row):
    exists = Path(log_path).exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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


def test_csv_append_identical_bytes_to_old_implementation(tmp_path):
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    store = CsvLogStore(str(new))
    for r in TRICKY_ROWS:
        _old_append_log_row(old, r)
        store.append(r)
    assert old.read_bytes() == new.read_bytes()


def test_csv_load_missing_file_returns_empty(tmp_path):
    assert CsvLogStore(str(tmp_path / "nope.csv")).load() == []


def test_reader_wrappers_still_use_same_format(tmp_path):
    # load_log/save_log/append_log_row (раньше reader._load_log и т.д.) —
    # формат файла должен остаться прежним.
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    for r in TRICKY_ROWS:
        _old_append_log_row(old, r)
        append_log_row(str(new), r)
    assert old.read_bytes() == new.read_bytes()
    assert load_log(str(new)) == CsvLogStore(str(old)).load()
    assert LOG_COLUMNS == LOG_COLUMNS


# ─── 2. SQLite == CSV ────────────────────────────────────────────────────────

def test_sqlite_append_roundtrip_matches_csv(tmp_path):
    csv_store = CsvLogStore(str(tmp_path / "log.csv"))
    sql_store = SqliteLogStore(str(tmp_path / "log.sqlite"))
    for r in TRICKY_ROWS:
        csv_store.append(r)
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
    csv_store.append(partial)
    sql_store.append(partial)
    assert sql_store.load() == csv_store.load()


def test_sqlite_reopen_preserves_data(tmp_path):
    path = str(tmp_path / "log.sqlite")
    SqliteLogStore(path).append(TRICKY_ROWS[0])
    assert SqliteLogStore(path).load()[0]["reading"] == "1200"


# ─── 3. Shadow-run ───────────────────────────────────────────────────────────

def _shadow(tmp_path):
    return ShadowLogStore(
        CsvLogStore(str(tmp_path / "log.csv")),
        SqliteLogStore(str(tmp_path / "log.sqlite")),
    )


def test_shadow_clean_when_both_written_identically(tmp_path):
    shadow = _shadow(tmp_path)
    shadow.save(TRICKY_ROWS[:2])
    for r in TRICKY_ROWS[2:]:
        shadow.append(r)
    assert shadow.compare() == []
    assert shadow.is_clean()


def test_shadow_detects_changed_value(tmp_path):
    shadow = _shadow(tmp_path)
    for r in TRICKY_ROWS:
        shadow.append(r)
    with sqlite3.connect(str(tmp_path / "log.sqlite")) as conn:
        conn.execute("UPDATE processing_log SET reading='9999' WHERE id=1")
    divergences = shadow.compare()
    assert len(divergences) == 1
    idx, p, s = divergences[0]
    assert idx == 0 and p["reading"] == "1200" and s["reading"] == "9999"


def test_shadow_detects_row_count_mismatch(tmp_path):
    shadow = _shadow(tmp_path)
    for r in TRICKY_ROWS:
        shadow.append(r)
    shadow.primary.append(_row(original_filename="only_in_csv.jpg"))
    assert any(idx == -1 for idx, _, _ in shadow.compare())


def test_shadow_reads_only_from_primary(tmp_path):
    # Кейс 8 (рассинхронизация таблицы и лога) решается по данным лога.
    # Лишняя строка в SQLite не должна попасть в то, что читает пайплайн.
    shadow = _shadow(tmp_path)
    shadow.append(TRICKY_ROWS[0])
    shadow.shadow.append(_row(original_filename="only_in_sqlite.jpg", source="pre_existing"))
    names = [r["original_filename"] for r in shadow.load()]
    assert names == ["IMG_0001.jpg"]
