"""
tools/datasets_standard.py — РАЗОВЫЙ переход кропов в датасетах на стандарт
имён (docs/DATASETS.md, решение владельца 2026-10-15). После перехода модуль
удаляется: новые кропы сразу называются по стандарту.

  python gmr.py datasets --standard [--month <папка месяца> …]          показать
  python gmr.py datasets --standard [--month <папка месяца> …] --apply  переименовать
  python gmr.py datasets --standard-undo                                вернуть как было

Что переименовывается (images/ и new/ каждого датасета):
  serials_crnn  1300000143__serial_number_1_set1.jpeg + labels/….txt «0070351»
                → 1300000143__serial1__0070351.jpg (номер — из подписи);
                разметка окна «0070351__<отпечаток>__2026-10.jpg»
                → <счёт>__serial1__0070351.jpg;
  accounts_crnn 1300000093__marker_id_1.jpeg → 1300000093__account1.jpg;
  digits_cnn    1300000013__gas_meter_1__digit_3.jpg → 1300000013__digit3.jpg;
                разметка окна «<отпечаток>__2026-10__digit_3.jpg» → <счёт>__digit3.jpg.
Лицевой счёт кропа разметки — по отпечатку фото из базы месяца (--month);
не нашёлся — вместо счёта остаётся отпечаток. Имя, которое не удалось
разобрать, не трогается и показывается. Подписи серийников (labels/)
переезжают в serials_crnn/labels_before_standard/; что во что переименовано —
<датасеты>/standard_renames.csv: по нему --standard-undo возвращает всё.
"""
import csv
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Optional

from models.datasets import DIGITS, IMAGE_EXTS
from src.gmr.domain.datasets import ACCOUNT, DIGIT, SERIAL, free_crop_name, parse_crop_name

RENAMES = "standard_renames.csv"
LABELS_BACKUP = "labels_before_standard"
_MARKUP_HASH = re.compile(r"^[0-9a-f]{16}$")
_MONTH = re.compile(r"^\d{4}-\d{2}$")
_EXAMPLES = 5


def accounts_by_photo(months) -> dict[str, str]:
    """{отпечаток фото: лицевой счёт} из логов месяцев (последняя запись со счётом)."""
    from src.gmr.storage.month import MonthDB, MonthFolder
    out = {}
    for m in months:
        folder = MonthFolder(Path(m))
        if not folder.db.is_file():
            raise ValueError(f"{m} — не папка месяца (нет {folder.db.name})")
        with MonthDB(folder.db) as db:
            for row in db.log_rows():
                if row.get("photo_hash") and row.get("account_id"):
                    out[row["photo_hash"]] = row["account_id"]
    return out


def _ext(p: Path) -> str:
    e = p.suffix.lower()
    return ".jpg" if e in (".jpg", ".jpeg") else e


def _photo(part: str, accounts: dict, stat: Counter) -> str:
    """Фото кропа разметки — счёт по отпечатку; остальное как есть."""
    if _MARKUP_HASH.match(part):
        stat["разметка: счёт по отпечатку найден" if part in accounts else
             "разметка: счёт не найден (оставлен отпечаток)"] += 1
        return accounts.get(part, part)
    return part


def _serial_parts(stem: str, accounts: dict, stat: Counter) -> Optional[tuple[str, int]]:
    parts = stem.split("__")
    if len(parts) == 3 and _MARKUP_HASH.match(parts[1]) and _MONTH.match(parts[2]):   # <номер>__<отпечаток>__<месяц>
        return _photo(parts[1], accounts, stat), 1
    if len(parts) < 2:
        return None
    m = re.search(r"serial\D*?(\d+)", parts[1])
    return _photo(parts[0], accounts, stat), int(m.group(1)) if m else 1


def _account_parts(stem: str, accounts: dict, stat: Counter) -> Optional[tuple[str, int]]:
    parts = stem.split("__")
    if len(parts) < 2:
        return None
    m = re.search(r"(\d+)$", parts[1])
    return _photo(parts[0], accounts, stat), int(m.group(1)) if m else 1


def _digit_parts(stem: str, accounts: dict, stat: Counter) -> Optional[tuple[str, int]]:
    parts = stem.split("__")
    m = re.search(r"digit_?(\d+)$", stem)
    if len(parts) < 2 or m is None:
        return None
    return _photo(parts[0], accounts, stat), int(m.group(1))


def _folders(ds_root: Path, ds: str) -> list[Path]:
    """Папки с кропами датасета: images/ (у цифр — 0…9) и то же внутри new/."""
    def leaf(base: Path) -> list[Path]:
        return [base / d for d in DIGITS] if ds == "digits_cnn" else [base / "images"]
    out = leaf(ds_root)
    new = ds_root / "new"
    if new.is_dir():
        out += leaf(new)
        out += [f for month in sorted(p for p in new.iterdir() if p.is_dir() and p.name not in DIGITS + ["images"])
                for f in leaf(month)]
    return [f for f in out if f.is_dir()]


def plan(root: Path, months=()) -> tuple[list[tuple[Path, Path, Optional[Path]]], dict, dict]:
    """[(кроп, новое имя, подпись серийника или None)], счётчики и примеры
    нераспознанных — по датасетам."""
    from models.crnn.dataset_crnn import _parse_label_txt
    root = Path(root)
    accounts = accounts_by_photo(months)
    moves, stats, odd = [], {}, {}
    for ds, kind, parts_of in (("serials_crnn", SERIAL, _serial_parts), ("accounts_crnn", ACCOUNT, _account_parts),
                               ("digits_cnn", DIGIT, _digit_parts)):
        stat, examples = Counter(), []
        folders = {f: sorted(p for p in f.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
                   for f in _folders(root / ds, ds)}
        taken = {p.name.lower() for files in folders.values() for p in files}   # имена — без повторов во всём датасете
        for folder, files in folders.items():
            for p in files:
                crop = parse_crop_name(p.name)
                if crop is not None and crop.kind == kind:
                    stat["уже по стандарту"] += 1
                    continue
                parts = parts_of(p.stem, accounts, stat)
                label, answer = None, ""
                if kind == SERIAL and parts is not None:
                    label = folder.parent / "labels" / (p.stem + ".txt")
                    answer = _parse_label_txt(label) if label.is_file() else None
                    if answer is None:
                        stat["нет годной подписи — не трогаем"] += 1
                        examples.append(p.name)
                        continue
                if parts is None:
                    stat["имя не разобрать — не трогаем"] += 1
                    examples.append(p.name)
                    continue
                taken.discard(p.name.lower())
                new = free_crop_name(taken, parts[0], kind, parts[1], answer, _ext(p))
                taken.add(new.lower())
                if new == p.name:
                    stat["уже по стандарту"] += 1
                    continue
                moves.append((p, p.with_name(new), label))
                stat["переименовать"] += 1
        stats[ds], odd[ds] = stat, examples[:_EXAMPLES]
    return moves, stats, odd


def run(root: Path, months=(), apply: bool = False) -> str:
    root = Path(root)
    moves, stats, odd = plan(root, months)
    lines = ["Переход на стандарт имён (docs/DATASETS.md)" + ("" if apply else " — проверка, ничего не изменено")]
    for ds, stat in stats.items():
        lines.append(f"{ds}: " + (", ".join(f"{k} — {n}" for k, n in sorted(stat.items())) or "кропов нет"))
        lines += [f"    {n}" for n in odd[ds]]
    if not apply:
        lines.append("Переименовать — тот же запуск с --apply (вернуть потом — --standard-undo).")
        return "\n".join(lines)
    log = root / RENAMES
    new_log = not log.exists()
    with open(log, "a", encoding="utf-8-sig" if new_log else "utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        if new_log:
            w.writerow(["было", "стало", "подпись было", "подпись стала"])
        for src, dst, label in moves:
            backup = ""
            if label is not None:
                rel = label.relative_to(root / "serials_crnn")
                b = root / "serials_crnn" / LABELS_BACKUP / rel
                if label.is_file():                  # у «x.jpg» и «x.jpeg» подпись одна — переносится раз
                    b.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(label), str(b))
                    if not any(label.parent.iterdir()):
                        label.parent.rmdir()
                backup = b.relative_to(root).as_posix()
            src.rename(dst)
            w.writerow([src.relative_to(root).as_posix(), dst.relative_to(root).as_posix(),
                        label.relative_to(root).as_posix() if label else "", backup])
    lines.append(f"Переименовано: {len(moves)}. Что во что — {log}")
    return "\n".join(lines)


def undo(root: Path) -> str:
    """Вернуть имена и подписи серийников, как были до --standard --apply."""
    root = Path(root)
    log = root / RENAMES
    if not log.is_file():
        raise ValueError(f"нет {log}: переход не делался или уже отменён")
    with open(log, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    back, skipped = 0, []
    for r in reversed(rows):
        src, dst = root / r["было"], root / r["стало"]
        if not dst.is_file() or src.exists():
            skipped.append(r["стало"])
            continue
        dst.rename(src)
        if r["подпись было"] and (root / r["подпись стала"]).is_file():
            (root / r["подпись было"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(root / r["подпись стала"]), str(root / r["подпись было"]))
        back += 1
    log.rename(log.with_suffix(".csv.undone"))
    return (f"Возвращено: {back}" + (f"; не вернуть (файла нет или имя занято): {len(skipped)}" if skipped else "")
            + f". Список переименований — {log.with_suffix('.csv.undone')}")
