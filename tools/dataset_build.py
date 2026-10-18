"""
tools/dataset_build.py — сборка кропов для датасетов из папки фото и таблицы
(шаг 2 стандарта датасетов, docs/DATASETS.md; решение владельца 2026-10-15).

  python gmr.py datasets --build <папка с фото> --table <таблица> [--only serials accounts digits]
                         [--out <папка>]

Фото называется по лицевому счёту (1300000375.jpg; «1300000375 (2).jpg» —
тот же счёт). Кропы режут те же детекторы и так же, как в работе. Ответы:
  serials_crnn/images/<счёт>__serial1__<номер>.jpg  — номер из таблицы (текстом, с нулями);
  accounts_crnn/images/<счёт>__account1.jpg          — счёт из имени фото;
  digits_cnn/<цифра>/<счёт>__digit<N>.jpg            — цифры «Текущих показаний»
                                                       (5 знаков, с нулями слева).
Кроп не делается, если ответа нет или он неоднозначен (счёта нет в таблице,
номеров или показаний у счёта несколько, показаний нет), детектор не нашёл
нужное или нашёл не 5 цифр. Эталон архива (etalon from-table) — раньше
сборки: он забирает свои фото из папки. Тот же счёт ещё раз — «-2»: имена не
совпадают ни между собой, ни с кропами в датасетах. По каждому фото — <папка>/сборка.csv (пишется сразу:
прерванную сборку тот же запуск с той же --out продолжает с места обрыва).
Фото, на котором программа упала, пропускается (кропов от него нет) — при
продолжении обрабатывается снова. Результат — в отдельной папке (раскладка как
в database/datasets): просмотрите и перенесите в датасеты сами. Работу
программы не меняет.
"""
import csv
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from models.datasets import IMAGE_EXTS, time_left
from src.gmr.application.recognition import find_detection
from src.gmr.domain import PipelineConfig
from src.gmr.domain.datasets import ACCOUNT, DIGIT, METER, SERIAL, free_crop_name
from src.gmr.render.image_io import write_image

KINDS = {"serials": SERIAL, "accounts": ACCOUNT, "digits": DIGIT}
_FOLDER = {SERIAL: "serials_crnn", ACCOUNT: "accounts_crnn", DIGIT: "digits_cnn"}
_ACCOUNT_IN_NAME = re.compile(r"^\d{3,}")
_MIN_SERIAL, _MAX_SERIAL = 4, 10                        # как models/crnn/config_crnn.py
REPORT = "сборка.csv"
COLUMNS = ["фото", "счёт", "номер", "надпись", "цифры"]


def read_answers(table: Path, config: PipelineConfig) -> tuple[dict, dict, dict]:
    """({счёт: номера счётчика}, {счёт: показания}, {счёт: прошлые показания})
    из таблицы — текстом, как в Excel. Прошлые показания — столбец
    «Последние показания» (в выгрузке реестра — «… значение»); его может не быть."""
    from src.gmr.storage.register import clean_serial, read_register
    reg = read_register(str(table))
    missing = [c for c in (config.col_account_id, config.col_serial) if c not in reg.columns]
    if missing:
        raise ValueError(f"в таблице {table} нет столбцов: {', '.join(missing)}")
    last_col = next((c for c in (config.col_last_reading, f"{config.col_last_reading} значение")
                     if c in reg.columns), "")
    serials, readings, lasts = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in reg.rows:
        account = row[config.col_account_id]
        if not account:
            continue
        if serial := clean_serial(row[config.col_serial]):
            serials[account].add(serial)
        if reading := row.get(config.col_new_reading, "").strip():
            readings[account].add(reading)
        if last := row.get(last_col, "").strip():
            lasts[account].add(last)
    return serials, readings, lasts


def reading_digits(text: str, expected: int) -> Optional[str]:
    """«2353» / «2353,0» → «02353»; None — не число или длиннее expected цифр."""
    try:
        value = float(text.replace(",", "."))
    except ValueError:
        return None
    if value < 0 or value != int(value) or len(str(int(value))) > expected:
        return None
    return str(int(value)).zfill(expected)


def photos_in(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def account_of_photo(path: Path) -> Optional[str]:
    """Лицевой счёт из имени фото: цифры в начале («1300000375 (2).jpg»)."""
    m = _ACCOUNT_IN_NAME.match(Path(path).stem)
    return m.group() if m else None


def only_one(values: set) -> Optional[str]:
    return next(iter(values)) if len(values) == 1 else None


class Builder:
    def __init__(self, out: Path, taken: set, only: set):
        self.out, self.taken, self.only = Path(out), taken, only
        self.stat = {k: Counter() for k in (SERIAL, ACCOUNT, DIGIT)}
        self.pending: list[tuple[Path, object]] = []

    def save(self, kind: str, account: str, crop, n: int = 1, answer: str = "", digit: str = "") -> str:
        """Имя занято сразу, файл пишется, когда фото обработано целиком."""
        name = free_crop_name(self.taken, account, kind, n, answer)
        self.taken.add(name.lower())
        self.pending.append((self.out / _FOLDER[kind] / (digit if kind == DIGIT else "images") / name, crop))
        return name

    def note(self, kind: str, what: str) -> str:
        self.stat[kind][what] += 1
        return what

    def photo(self, path: Path, models, serials: dict, readings: dict, config: PipelineConfig) -> dict:
        """Кропы одного фото и строка сборка.csv. Упало — ни кропов, ни
        счётчиков от этого фото (ошибка уходит дальше)."""
        stat = {k: Counter(c) for k, c in self.stat.items()}
        self.pending = []
        try:
            row = self._photo(path, models, serials, readings, config)
        except Exception:
            for dst, _ in self.pending:
                self.taken.discard(dst.name.lower())
            self.stat, self.pending = stat, []
            raise
        for dst, crop in self.pending:
            dst.parent.mkdir(parents=True, exist_ok=True)
            write_image(dst, crop)
        self.pending = []
        return row

    def _photo(self, path: Path, models, serials: dict, readings: dict, config: PipelineConfig) -> dict:
        row = dict.fromkeys(COLUMNS, "")
        account = account_of_photo(path)
        row["счёт"] = account or ""
        if account is None:
            for kind, col in ((SERIAL, "номер"), (ACCOUNT, "надпись"), (DIGIT, "цифры")):
                if kind in self.only:
                    row[col] = self.note(kind, "в имени фото нет лицевого счёта")
            return row
        found = models.meter_detector.detect(str(path)) or []
        if SERIAL in self.only:
            row["номер"] = self._serial(account, find_detection(found, SERIAL), serials)
        if ACCOUNT in self.only:
            row["надпись"] = self._account(account, find_detection(found, ACCOUNT))
        if DIGIT in self.only:
            row["цифры"] = self._digits(account, find_detection(found, METER), readings, models, config)
        return row

    def _account(self, account: str, det) -> str:
        if det is None:
            return self.note(ACCOUNT, "детектор не нашёл надпись")
        self.note(ACCOUNT, "сделан")
        return self.save(ACCOUNT, account, det["crop"])

    def _serial(self, account: str, det, serials: dict) -> str:
        if account not in serials:
            return self.note(SERIAL, "счёта нет в таблице (или номер пустой)")
        serial = only_one(serials[account])
        if serial is None:
            return self.note(SERIAL, "у счёта несколько номеров")
        if not serial.isdigit() or not _MIN_SERIAL <= len(serial) <= _MAX_SERIAL:
            return self.note(SERIAL, "номер в таблице не из 4–10 цифр")
        if det is None:
            return self.note(SERIAL, "детектор не нашёл табличку")
        self.note(SERIAL, "сделан")
        return self.save(SERIAL, account, det["crop"], answer=serial)

    def _digits(self, account: str, det, readings: dict, models, config: PipelineConfig) -> str:
        values = readings.get(account, set())
        if not values:
            return self.note(DIGIT, "показаний в таблице нет")
        text = only_one(values)
        if text is None:
            return self.note(DIGIT, "у счёта несколько показаний")
        reading = reading_digits(text, config.expected_digits)
        if reading is None:
            return self.note(DIGIT, f"показание не число из ≤{config.expected_digits} цифр")
        if det is None:
            return self.note(DIGIT, "детектор не нашёл показания")
        found = sorted(models.digit_detector.detect(det["crop"]) or [], key=lambda d: d["bbox"][0])
        if len(found) != config.expected_digits:
            return self.note(DIGIT, f"найдено цифр не {config.expected_digits}")
        self.note(DIGIT, "сделаны")
        for pos, (d, digit) in enumerate(zip(found, reading), 1):
            self.save(DIGIT, account, d["crop"], n=pos, digit=digit)
        return reading


def _load_detectors(config: PipelineConfig):
    from types import SimpleNamespace

    from src.gmr.ml.loader import load_digit_detector, load_meter_detector
    return SimpleNamespace(meter_detector=load_meter_detector(config),
                           digit_detector=load_digit_detector(config))


_COLUMN = {"serials": "номер", "accounts": "надпись", "digits": "цифры"}


def _done_before(out: Path, only) -> set:
    """Фото, уже записанные в сборка.csv прошлого запуска (путь от папки фото).
    Продолжать — тем же набором кропов: иначе у части фото их не будет."""
    with open(out / REPORT, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    before = {k for k, col in _COLUMN.items() if any(r.get(col) for r in rows)}
    if rows and before != set(only):
        raise ValueError(f"в {out} собирались кропы: {' '.join(sorted(before))} — продолжайте с тем же "
                         f"--only или укажите другую --out")
    return {r["фото"] for r in rows}


def run(photos: Path, table: Path, out: Optional[Path] = None, only=("serials", "accounts", "digits"),
        dataset_root: Optional[Path] = None, models=None) -> str:
    photos = Path(photos)
    if not photos.is_dir():
        raise ValueError(f"нет папки {photos}")
    out = Path(out) if out else photos.parent / f"{photos.name}_датасет"
    resume = (out / REPORT).is_file()
    if out.exists() and any(out.iterdir()) and not resume:
        raise ValueError(f"папка {out} уже есть и не пустая (и в ней нет {REPORT}) — "
                         f"укажите другую (--out) или удалите её")
    paths = photos_in(photos)
    if not paths:
        raise ValueError(f"в {photos} нет фото")
    config = PipelineConfig()
    serials, readings, _ = read_answers(Path(table), config)
    taken = {p.name.lower() for p in Path(dataset_root).rglob("*") if p.is_file()} \
        if dataset_root and Path(dataset_root).is_dir() else set()
    done = set()
    if resume:
        done = _done_before(out, only)
        taken |= {p.name.lower() for p in out.rglob("*") if p.is_file()}
    todo = [p for p in paths if p.relative_to(photos).as_posix() not in done]
    builder = Builder(out, taken, {KINDS[k] for k in only})
    errors: list[str] = []
    if todo:
        models = models or _load_detectors(config)
        out.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with open(out / REPORT, "a", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter=";")
            if not resume:
                w.writeheader()
            for i, p in enumerate(todo, 1):
                try:
                    row = builder.photo(p, models, serials, readings, config)
                except Exception as e:                     # битое фото не останавливает сборку
                    errors.append(f"{p.relative_to(photos).as_posix()}: {type(e).__name__}: {e}")
                else:
                    row["фото"] = p.relative_to(photos).as_posix()
                    w.writerow(row)
                    fh.flush()
                if i % 25 == 0 or i == len(todo):
                    print(f"Фото {i} из {len(todo)}{time_left(started, i, len(todo))}", flush=True)
    lines = [f"Сборка датасета: {len(paths)} фото ({photos}) → {out}"]
    if resume:
        lines.append(f"Продолжение: уже были в {REPORT} — {len(paths) - len(todo)}, сейчас — {len(todo)}")
    for kind in (SERIAL, ACCOUNT, DIGIT):
        if builder.stat[kind]:
            lines.append(f"{_FOLDER[kind]}: " + ", ".join(f"{k} — {n}" for k, n in builder.stat[kind].most_common()))
    if errors:
        lines.append(f"⚠ программа упала на фото: {len(errors)} (кропов от них нет; при продолжении — "
                     f"снова), первое: {errors[0]}")
    lines.append(f"По каждому фото: {out / REPORT}")
    lines.append("Просмотрите кропы и перенесите в database\\datasets\\ (раскладка та же).")
    return "\n".join(lines)
