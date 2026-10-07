"""
tools/etalon.py — эталон: трудные фото с правильным ответом оператора, чтобы
сравнивать модели «было → стало» (этап 5b, решения владельца 2026-10-04).

  python gmr.py etalon add <папка месяца | лог.csv> [--photos <папка с фото>]
  python gmr.py etalon check [--serial-pad 0.05] [--meter-weights …] [--digit-detect-weights …]
                             [--digit-weights …] [--serial-weights …]
  python gmr.py etalon                 сколько фото в эталоне, по причинам

Трудное фото — модель не справилась (авто-ошибка: нет счётчика, нет или не
найден серийник, ошибка цифр, подозрительно, номер у нескольких абонентов),
а оператор ввёл показание («Принять», «Серийник в базе с ошибкой»); или
модель прочитала, а оператор исправил показание на «Проверке». Правильный
ответ — серийник и показание из строки оператора.

Эталон и обучение не пересекаются: каждое третье трудное фото (по отпечатку
фото, одно и то же фото — всегда одинаково) идёт в эталон, остальные — в
обучение (этап 5c, папки new/ датасетов).

Папка эталона (database/datasets/etalon, в git не попадает):
  photos/<отпечаток>.<ext>   исходные фото; имя — отпечаток, без имён и номеров
  answers.csv                отпечаток, файл, серийник, показание, причина, откуда, когда
  checks/<дата-время>.csv    результат check по каждому фото
В выводе команд только числа: его можно присылать.
"""
import argparse
import csv
import dataclasses
import hashlib
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from models.datasets import DATASETS_DIR, IMAGE_EXTS
from src.gmr.application.operator import CORRECTED_NOTE
from src.gmr.domain import PipelineConfig
from src.gmr.domain.models import QUESTION_REASONS
from src.gmr.domain.serial_match import normalize_serial

ETALON_DIR = DATASETS_DIR / "etalon"
ETALON_SHARE = 1 / 3
# авто-исходы «модель не справилась»; ERROR — сбой программы, а не модели
FAILED = ("NO_METER", "NO_SERIAL", "SERIAL_LOW_CONF", "SERIAL_NOT_FOUND", "SERIAL_AMBIGUOUS",
          "DIGITS_ERROR", "SUSPICIOUS")
CORRECTED = "CORRECTED"
REASON_NAMES = {o: QUESTION_REASONS[o.lower()] for o in FAILED} | {CORRECTED: "Исправлено на «Проверке»"}
ANSWER_COLUMNS = ["photo_hash", "file", "serial", "reading", "reason", "source", "added_at"]


@dataclass
class Answer:
    photo_hash: str
    original_filename: str
    source_folder: str
    serial: str
    reading: str
    reason: str


def to_etalon(photo_hash: str) -> bool:
    """Эталон или обучение: каждое третье фото по отпечатку (решение 1а)."""
    x = int(hashlib.sha256(f"etalon:{photo_hash}".encode("utf-8")).hexdigest()[:12], 16) / 16 ** 12
    return x < ETALON_SHARE


def _reading(value) -> str:
    """Показание из лога → целое число строкой ("01234", "1234.0" → "1234")."""
    try:
        return str(int(float(str(value).strip().replace(",", "."))))
    except ValueError:
        return ""


def hard_answers(rows: list[dict]) -> list[Answer]:
    """Трудные фото с ответом оператора — по строкам лога (порядок записи).
    Фото узнаётся по отпечатку; в старых логах (до 2026-09-29) отпечатка нет —
    тогда по папке контролёра и имени файла, а photo_hash пустой (его
    посчитает add по найденному файлу)."""
    by_photo: dict[str, list[dict]] = {}
    for r in rows:
        name = r.get("original_filename") or ""
        key = (r.get("photo_hash") or "").strip() or f"name:{r.get('source_folder', '')}/{name}"
        by_photo.setdefault(key, []).append(r)
    out = []
    for key, rs in by_photo.items():
        h = "" if key.startswith("name:") else key
        failed = [r for r in rs if r.get("source") == "auto" and r.get("outcome") in FAILED]
        manual = [r for r in rs if r.get("source") == "manual" and r.get("outcome") in ("PLUS", "MINUS")]
        corrected = [r for r in rs if r.get("source") == "auto" and r.get("outcome") in ("PLUS", "MINUS")
                     and CORRECTED_NOTE in (r.get("notes") or "")]
        if failed and manual:
            ans, reason, origin = manual[-1], failed[-1]["outcome"], failed[-1]
        elif corrected:
            ans, reason, origin = corrected[-1], CORRECTED, corrected[-1]
        else:
            continue
        serial, reading = (ans.get("serial_id") or "").strip(), _reading(ans.get("reading", ""))
        if serial and reading:
            out.append(Answer(h, origin.get("original_filename", ""), origin.get("source_folder", ""),
                              serial, reading, reason))
    return out


def read_answers(etalon: Path) -> list[dict]:
    path = Path(etalon) / "answers.csv"
    if not path.is_file():
        return []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _log_rows(source: Path) -> list[dict]:
    if source.suffix.lower() == ".csv":
        from src.gmr.storage import load_log
        return load_log(str(source))
    from src.gmr.storage.month import MonthDB, MonthFolder
    folder = MonthFolder(source)
    if not folder.exists():
        raise ValueError(f"{source} — не папка месяца и не лог .csv")
    with MonthDB(folder.db) as db:
        return db.log_rows()


def _find_photo(a: Answer, photos: Path, index: dict) -> Optional[Path]:
    """Исходное фото: по имени в папке контролёра, иначе — по отпечатку среди
    всех фото. Без отпечатка (старый лог) — по имени, потом отпечаток
    записывается в a.photo_hash."""
    from src.gmr.storage import photo_fingerprint
    guess = photos / a.source_folder / a.original_filename
    if not a.photo_hash:
        found = None
        if a.original_filename:
            found = guess if guess.is_file() else next(
                (p for p in sorted(photos.rglob(a.original_filename)) if p.is_file()), None)
        if found is not None:
            a.photo_hash = photo_fingerprint(str(found))
        return found
    if a.original_filename and guess.is_file() and photo_fingerprint(str(guess)) == a.photo_hash:
        return guess
    if not index:
        for p in sorted(photos.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                index.setdefault(photo_fingerprint(str(p)), p)
        index.setdefault("", None)          # индекс построен (даже если фото нет)
    return index.get(a.photo_hash)


@dataclass
class AddReport:
    found: int = 0
    etalon: int = 0
    added: int = 0
    already: int = 0
    missing: int = 0
    training: int = 0

    def text(self) -> str:
        return "\n".join([
            f"Трудных фото с ответом оператора: {self.found}",
            f"  в эталон (каждое третье): {self.etalon} — добавлено {self.added}, уже были {self.already}",
            f"  в обучение (этап 5c, папки new/): {self.training}",
        ] + ([f"  ⚠ фото не найдено на диске: {self.missing}"] if self.missing else []))


def add(source: Path, photos: Optional[Path] = None, etalon: Path = ETALON_DIR) -> AddReport:
    source, etalon = Path(source), Path(etalon)
    if photos is None:
        if source.suffix.lower() == ".csv":
            raise ValueError("для лога .csv укажите --photos <папка с исходными фото>")
        from src.gmr.storage.month import MonthFolder
        photos = MonthFolder(source).photos
    answers = hard_answers(_log_rows(source))
    have = {r["photo_hash"] for r in read_answers(etalon)}
    rep, index, new_rows = AddReport(found=len(answers)), {}, []
    for a in answers:
        src = None
        if not a.photo_hash:                  # старый лог: отпечаток — по найденному файлу
            src = _find_photo(a, Path(photos), index)
            if src is None:
                rep.missing += 1
                continue
        if not to_etalon(a.photo_hash):
            rep.training += 1
            continue
        rep.etalon += 1
        if a.photo_hash in have:
            rep.already += 1
            continue
        src = src or _find_photo(a, Path(photos), index)
        if src is None:
            rep.missing += 1
            continue
        name = f"{a.photo_hash}{src.suffix.lower()}"
        (etalon / "photos").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, etalon / "photos" / name)
        new_rows.append({"photo_hash": a.photo_hash, "file": name, "serial": a.serial, "reading": a.reading,
                         "reason": a.reason, "source": source.name,
                         "added_at": datetime.now().isoformat(timespec="seconds")})
        rep.added += 1
    if new_rows:
        path = etalon / "answers.csv"
        new_file = not path.is_file()
        etalon.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=ANSWER_COLUMNS)
            if new_file:
                w.writeheader()
            w.writerows(new_rows)
    return rep


# ─── Проверка моделей на эталоне ─────────────────────────────────────────────

def serial_matches(text: str, truth: str) -> bool:
    """Как reader.py ищет номер в таблице: как есть, с «0» или «00» спереди."""
    return any(normalize_serial(p + (text or "")) == normalize_serial(truth) for p in ("", "0", "00")) \
        and bool(text)


def reading_matches(number: Optional[int], truth: str, config: PipelineConfig) -> bool:
    """Показание верно: совпадают все цифры, кроме последних ignore_last_digits
    (их программа прощает)."""
    if number is None:
        return False
    n = config.expected_digits - config.ignore_last_digits
    return str(number).zfill(config.expected_digits)[:n] == str(truth).zfill(config.expected_digits)[:n]


@dataclass
class PhotoCheck:
    photo_hash: str
    reason: str
    meter_found: bool
    serial_found: bool
    serial_ok: bool          # номер верный и уверенный — reader.py нашёл бы его в таблице
    serial_wrong_sure: bool  # номер неверный, но уверенный — опасно: чужой абонент
    reading_ok: bool         # показание прочитано и верно
    reading_wrong_sure: bool  # показание прочитано (все цифры уверенно), но неверно — записалось бы
    model_serial: str = ""
    model_reading: str = ""

    @property
    def all_ok(self) -> bool:
        return self.serial_ok and self.reading_ok


def check_photo(models, path: Path, answer: dict, config: PipelineConfig) -> PhotoCheck:
    from src.gmr.application import recognize_photo
    rec = recognize_photo(models, str(path), config)
    sp = rec.serial_prediction
    text, sure = (sp.text, sp.confidence >= config.serial_conf_thresh) if sp else ("", False)
    number = rec.digits.number if rec.digits is not None else None
    s_ok = serial_matches(text, answer["serial"])
    r_ok = reading_matches(number, answer["reading"], config)
    return PhotoCheck(answer["photo_hash"], answer["reason"], rec.meter is not None, rec.serial is not None,
                      s_ok and sure, sure and not s_ok, r_ok, number is not None and not r_ok,
                      text, "" if number is None else str(number))


def summary(results: list[PhotoCheck], title: str) -> str:
    cols = ("фото", "счётчик найден", "серийник верно", "показание верно", "прочитано бы верно",
            "уверенно, но неверно")

    def row(name: str, rs: list[PhotoCheck]) -> list[str]:
        n = len(rs)

        def pct(k: int) -> str:
            return f"{k} ({100 * k / n:.0f}%)" if n else "0"
        return [name, str(n), pct(sum(r.meter_found for r in rs)), pct(sum(r.serial_ok for r in rs)),
                pct(sum(r.reading_ok for r in rs)), pct(sum(r.all_ok for r in rs)),
                pct(sum(r.serial_wrong_sure or r.reading_wrong_sure for r in rs))]
    by_reason = Counter(r.reason for r in results)
    table = [["причина", *cols]]
    table += [row(REASON_NAMES.get(reason, reason), [r for r in results if r.reason == reason])
              for reason in [o for o in (*FAILED, CORRECTED) if o in by_reason]]
    table.append(row("ВСЕГО", results))
    widths = [max(len(t[i]) for t in table) for i in range(len(table[0]))]
    lines = [title, ""] + ["  ".join(c.ljust(w) for c, w in zip(t, widths)).rstrip() for t in table]
    lines += ["",
              "серийник верно — номер прочитан верно и уверенно (программа нашла бы его в таблице);",
              "показание верно — все цифры, кроме двух последних (их программа прощает);",
              "прочитано бы верно — и то и другое: фото не ушло бы оператору;",
              "уверенно, но неверно — опасные ошибки: неверный номер или показание с уверенностью."]
    return "\n".join(lines)


def check(config: PipelineConfig, etalon: Path = ETALON_DIR, models=None, title: str = "") -> tuple[str, Path]:
    etalon = Path(etalon)
    answers = read_answers(etalon)
    if not answers:
        raise ValueError(f"эталон пуст: {etalon} — сначала `python gmr.py etalon add <папка месяца>`")
    if models is None:
        from src.gmr.ml.loader import load_models
        models = load_models(config)
    results, missing = [], 0
    for a in answers:
        path = etalon / "photos" / a["file"]
        if not path.is_file():
            missing += 1
            continue
        results.append(check_photo(models, path, a, config))
    out = etalon / "checks" / f"{datetime.now():%Y-%m-%d_%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        fields = [f.name for f in dataclasses.fields(PhotoCheck)]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(dataclasses.asdict(r) for r in results)
    text = summary(results, title)
    if missing:
        text += f"\n\n⚠ нет файла фото в эталоне: {missing}"
    return text + f"\n\nПо каждому фото: {out}", out


def status(etalon: Path = ETALON_DIR) -> str:
    answers = read_answers(etalon)
    by = Counter(a["reason"] for a in answers)
    lines = [f"Эталон: {Path(etalon).resolve()} — фото: {len(answers)}"]
    lines += [f"  {REASON_NAMES.get(r, r)}: {n}" for r, n in by.most_common()]
    return "\n".join(lines)


def _config(args) -> tuple[PipelineConfig, list[str]]:
    cfg, notes = PipelineConfig(), []
    for arg, field in (("meter_weights", "meter_detect_model"), ("digit_detect_weights", "digit_detect_model"),
                       ("digit_weights", "digit_ocr_model"), ("serial_weights", "serial_ocr_model")):
        if getattr(args, arg):
            cfg = dataclasses.replace(cfg, **{field: getattr(args, arg)})
            notes.append(f"{field} = {getattr(args, arg)}")
    if args.serial_pad is not None:
        cfg = dataclasses.replace(cfg, serial_crop_pad=args.serial_pad)
        notes.append(f"запас вокруг рамки серийника = {args.serial_pad}")
    return cfg, notes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python gmr.py etalon",
                                description="Эталон трудных фото: пополнить и проверить на нём модели.")
    p.add_argument("--etalon", default=str(ETALON_DIR), help=f"папка эталона (по умолчанию {ETALON_DIR})")
    sub = p.add_subparsers(dest="cmd")
    # --etalon можно писать и до, и после add/check
    either = argparse.ArgumentParser(add_help=False)
    either.add_argument("--etalon", default=argparse.SUPPRESS, help="папка эталона")
    a = sub.add_parser("add", parents=[either], help="добавить трудные фото с ответом оператора из месяца или лога")
    a.add_argument("source", help="папка месяца или лог .csv")
    a.add_argument("--photos", help="папка с исходными фото (по умолчанию <месяц>/фото)")
    c = sub.add_parser("check", parents=[either], help="как текущие (или другие) модели читают эталон")
    c.add_argument("--serial-pad", type=float, default=None, help="запас вокруг рамки серийника, например 0.05")
    for name in ("meter", "digit-detect", "digit", "serial"):
        c.add_argument(f"--{name}-weights", default=None, help="другой файл весов")
    args = p.parse_args(argv)
    try:
        if args.cmd == "add":
            print(add(Path(args.source), Path(args.photos) if args.photos else None, Path(args.etalon)).text())
        elif args.cmd == "check":
            cfg, notes = _config(args)
            title = "Модели: " + ("; ".join(notes) if notes else "как в работе (PipelineConfig)")
            print(check(cfg, Path(args.etalon), title=title)[0])
        else:
            print(status(Path(args.etalon)))
    except ValueError as e:
        print(f"ОШИБКА: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
