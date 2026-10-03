"""
src/gmr/application/cycle.py — месячный цикл (этап 3, решения владельца
2026-10-03): где лежит файл результата фото и что делать с фото, которые
ждут оператора, когда у счёта появилось показание.

Пункт 3: если у лицевого счёта появилось показание (прочитал reader.py или
принял оператор), его фото в question/ с причиной «цифры» (DIGITS_ERROR) или
«подозрительно» (SUSPICIOUS) больше не нужны оператору — они уходят в
repeat/ своего контролёра, в лог пишется строка REPEAT с пометкой.
"""
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.gmr.domain import OUTCOME_FOLDER, Outcome, meaningful_rows
from src.gmr.storage import LOG_COLUMNS
from src.gmr.storage.month import MonthDB

WAITING_FOR_READING = ("DIGITS_ERROR", "SUSPICIOUS")
CLOSED_NOTE = "закрыто: у счёта появилось показание"
_MANUAL_FOLDER = {"NOT_IN_DB": "not_in_db", "UNREADABLE": "unreadable", "REPEAT": "repeat"}


def result_file(results: Path, row: dict) -> Optional[Path]:
    """Файл результата, который оставила строка лога: авто-строка —
    <контролёр>/<папка исхода>/<final_filename или original_filename>,
    решение оператора без показания — <контролёр>/<not_in_db|unreadable|repeat>/<имя>.
    None — у строки файла нет (например, строка-копия)."""
    outcome = row.get("outcome") or ""
    if row.get("source") == "manual":
        # имя занято — program2.py пишет новое в final_filename (place_decision)
        folder, name = _MANUAL_FOLDER.get(outcome), row.get("final_filename") or row.get("original_filename")
    else:
        folder = OUTCOME_FOLDER.get(Outcome[outcome]) if outcome in Outcome.__members__ else None
        name = row.get("final_filename") or row.get("original_filename")
    if not folder or not name:
        return None
    return Path(results) / (row.get("source_folder") or "") / folder / name


def free_name(folder: Path, name: str, taken: frozenset = frozenset()) -> str:
    """Имя для файла в folder, не занятое ни файлом, ни именами из taken:
    name, потом <имя>_2, <имя>_3… (два разных фото не затирают друг друга)."""
    stem, ext = Path(name).stem, Path(name).suffix
    candidate, k = name, 2
    while candidate in taken or (Path(folder) / candidate).exists():
        candidate, k = f"{stem}_{k}{ext}", k + 1
    return candidate


def close_waiting(db: MonthDB, results: Path, account: str, by_photo: str, who: str,
                  keep: Optional[str] = None) -> tuple[list[dict], list[tuple[Path, Path]]]:
    """
    Пункт 3: фото лицевого счёта account, которые ждут оператора (последняя
    строка фото — DIGITS_ERROR или SUSPICIOUS по этому счёту), закрываются:
    в лог (в транзакции вызывающего) пишется строка REPEAT. Возвращает эти
    строки и переносы файлов question/… → repeat/ — их делает move_files
    после записи в базу. keep — файл, который оператор разбирает сейчас (не
    трогать).
    """
    rows_out, moves, seen, taken = [], [], set(), set()
    for cand in db.log_rows_of_account(account, WAITING_FOR_READING):
        key = cand.get("photo_hash") or ("name", cand.get("original_filename"))
        if key in seen:
            continue
        seen.add(key)
        rows = meaningful_rows(db.log_rows_of_photo(cand.get("photo_hash") or "",
                                                    cand.get("original_filename") or ""))
        last = rows[-1] if rows else {}
        # последнее о фото — всё ещё эта ошибка по этому счёту (такие исходы
        # бывают только у авто-строк; решение оператора, показание или повтор
        # — значит, фото уже не ждёт)
        if last.get("outcome") not in WAITING_FOR_READING or last.get("account_id") != account:
            continue
        src = result_file(results, last)
        if keep and src is not None and Path(src).resolve() == Path(keep).resolve():
            continue
        dst_dir = Path(results) / (last.get("source_folder") or "") / "repeat"
        name = free_name(dst_dir, src.name if src else last.get("original_filename") or "",
                         frozenset(taken))
        taken.add(name)
        rows_out.append({c: "" for c in LOG_COLUMNS} | {
            "original_filename": last.get("original_filename", ""),
            "final_filename":    name,
            "serial_id":         last.get("serial_id", ""),
            "account_id":        account,
            "outcome":           "REPEAT",
            "source":            "auto",
            "processed_by":      who,
            "processed_at":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "notes":             f"{CLOSED_NOTE} (л/с {account}, фото {by_photo})",
            "photo_hash":        last.get("photo_hash", ""),
            "source_folder":     last.get("source_folder", ""),
        })
        if src is not None and src.exists():
            moves.append((src, dst_dir / name))
    if rows_out:
        db.append_log_rows(rows_out)
    return rows_out, moves


def move_files(moves: list[tuple[Path, Path]]) -> list[str]:
    """Переносит файлы; возвращает тексты ошибок (пусто — всё перенесено)."""
    errors = []
    for src, dst in moves:
        try:
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        except OSError as e:
            errors.append(f"{src} → {dst}: {e}")
    return errors
