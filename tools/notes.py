"""
tools/notes.py — замер: что программа поняла бы из надписей контролёров
поверх фото (src/gmr/overlay.py — найти и прочитать, src/gmr/domain/
note_match.py — понять по таблице). Работу программы не меняет.

  python gmr.py notes <папка фото> --table <таблица> [--out <папка>]
  python gmr.py notes database\\datasets\\etalon_archive --table <таблица>

Папка фото — фото названы по лицевому счёту (как архив месяца): правильный
ответ — этот счёт и его «Текущие показания» из таблицы. Папка эталона
архива (answers.csv + photos\\) — фото по отпечатку: ответ — номер счётчика и
показание из answers.csv, счёт — владелец этого номера по таблице.

По каждому фото — две проверки, как в работе:
  - счёт по надписи — как если бы таблички не было (показание с барабана —
    правильное, им счёт подтверждается);
  - показание по надписи — как если бы барабан не прочитался (счёт
    известен).
Итог фото: «опасно» — надпись дала не тот счёт или не то показание; «спор»
— пошло бы оператору; «верно»; «не понять» — надпись есть, но понята не
целиком (обрывки, мусор, лишнее число): программа её не использует, фото
идёт как без надписи. Картинки для разбора — <out>\\<итог>\\: фото, зелёным —
принятые строки, оранжевым — похоже на текст, но не прочитано, красным — не
текст (тень, труба); под фото — что прочитано и почему такой итог. По
каждому фото — <out>\\надписи.csv.
"""
import argparse
import csv
import shutil
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from models.datasets import time_left
from src.gmr.domain import PipelineConfig
from src.gmr.render.image_io import read_image, write_image

RIGHT, DANGER, CONFLICT, UNCLEAR, NO_TRUTH, NOT_TEXT = \
    "верно", "опасно", "спор", "не_понять", "нет_ответа", "не_текст"
STATUSES = (RIGHT, DANGER, CONFLICT, UNCLEAR, NO_TRUTH, NOT_TEXT)
_OLD = ("не_совпало", "неуверенно")       # папки прошлых версий замера
REPORT = "надписи.csv"
_FIELDS = ["фото", "итог", "счёт", "счёт по надписи", "показание", "показание по надписи",
           "строки", "отброшено", "почему", "правильно"]
GREEN, ORANGE, RED = (40, 170, 40), (0, 140, 255), (40, 40, 220)


def _int(text) -> Optional[int]:
    try:
        return int(float(str(text).replace(",", ".")))
    except ValueError:
        return None


def truths(folder: Path, table: Path) -> tuple[Path, list[Path], dict, dict]:
    """(папка фото, фото, {фото: (счета, показание)}, {счёт: прошлое
    показание}) — правильные ответы из таблицы."""
    from tools.dataset_build import account_of_photo, only_one, photos_in, read_answers
    serials, readings, lasts = read_answers(Path(table), PipelineConfig())
    past = {a: _int(only_one(v)) for a, v in lasts.items() if only_one(v) is not None}
    past.update({a: None for a in set(serials) | set(readings) if a not in past})
    answers = Path(folder) / "answers.csv"
    if answers.is_file() and (Path(folder) / "photos").is_dir():         # эталон архива
        owners: dict = {}
        for account, ss in serials.items():
            for s in ss:
                owners.setdefault(s.lstrip("0"), set()).add(account)
        with open(answers, encoding="utf-8-sig", newline="") as fh:
            rows = {r["file"]: r for r in csv.DictReader(fh)}
        photos = photos_in(Path(folder) / "photos")
        truth = {p: (owners.get(rows[p.name]["serial"].lstrip("0"), set()), rows[p.name]["reading"])
                 for p in photos if p.name in rows}
        return Path(folder) / "photos", photos, {p: t for p, t in truth.items() if t[0]}, past
    photos = photos_in(folder)
    truth = {}
    for p in photos:
        account = account_of_photo(p)
        if account and (account in serials or account in readings):
            truth[p] = ({account}, only_one(readings.get(account, set())))
    return Path(folder), photos, truth, past


def _verdict(ans, ok: Optional[bool]) -> str:
    """Итог одной проверки: спор; верно или опасно (ok); "" — надпись ответа не дала (ok None)."""
    if ans.conflict:
        return CONFLICT
    return "" if ok is None else RIGHT if ok else DANGER


def judge(lines, dropped, truth, index, lasts) -> dict:
    """Что программа поняла бы из надписи на одном фото и верно ли это.
    truth — (счета, показание) или None (в таблице ответа нет)."""
    from src.gmr.domain.note_match import interpret
    texts = [ln.text for ln in lines]
    unread = sum(d.text_like for d in dropped)
    row = {"строки": " / ".join(ln.text for ln in lines),
           "отброшено": " / ".join(f"{d.text} ({d.why})" for d in dropped)}
    if not lines and not unread:
        return {**row, "итог": NOT_TEXT if dropped else ""}
    accounts, reading = truth or (set(), None)
    rd = _int(reading) if reading else None
    by_note = interpret(texts, index, lasts, None, rd, unread)
    row["счёт по надписи"] = by_note.account or by_note.conflict or "—"
    why = [f"без таблички: {by_note.why or by_note.conflict}"]
    if truth:
        row["правильно"] = f"л/с {', '.join(sorted(accounts))}, показание {reading or '—'}"
        row["счёт"] = _verdict(by_note, None if by_note.account is None else by_note.account in accounts)
        if len(accounts) == 1 and rd is not None:
            on_plate = interpret(texts, index, lasts, next(iter(accounts)), None, unread)
            row["показание по надписи"] = on_plate.reading if on_plate.reading is not None else \
                on_plate.conflict or "—"
            row["показание"] = _verdict(on_plate, None if on_plate.reading is None else on_plate.reading == rd)
            why.append(f"с табличкой: {on_plate.why or on_plate.conflict}")
    got = {row.get("счёт", ""), row.get("показание", "")}
    row["итог"] = NO_TRUTH if not truth else next((s for s in (DANGER, CONFLICT, RIGHT) if s in got), UNCLEAR)
    row["почему"] = "; ".join(why)
    return row


def _font(size: int):
    """Шрифт с кириллицей (Windows — Arial, Linux — DejaVu) или None."""
    from PIL import ImageFont
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return None


def picture(img: np.ndarray, lines, dropped, row: dict) -> np.ndarray:
    """Фото (как его видит поиск надписи) с рамками строк и текстом под ним."""
    from PIL import Image, ImageDraw
    from src.gmr.overlay import MAX_SIDE
    s = MAX_SIDE / max(img.shape[:2])
    out = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    found = [(ln.box, ln.text, GREEN, "принято") for ln in lines] + \
            [(d.box, d.text, ORANGE if d.text_like else RED, d.why) for d in dropped]
    for n, (box, _, color, _) in enumerate(found, 1):
        x, y, w, h = box
        cv2.rectangle(out, (x - 3, y - 3), (x + w + 3, y + h + 3), color, 3)
        cv2.putText(out, str(n), (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    font = _font(22)
    if font is None:
        return out
    text = [(f"{n}  {t}  — {why}", color) for n, (_, t, color, why) in enumerate(found, 1)]
    text += [(part, (0, 0, 0)) for k in ("итог", "счёт по надписи", "показание по надписи", "правильно", "почему")
             if row.get(k) for part in textwrap.wrap(f"{k}: {row[k]}", max(20, out.shape[1] // 12))]
    panel = Image.new("RGB", (out.shape[1], 12 + 30 * len(text)), "white")
    draw = ImageDraw.Draw(panel)
    for i, (t, color) in enumerate(text):
        draw.text((10, 6 + 30 * i), t, fill=color[::-1], font=font)
    return np.vstack([out, cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)])


def run(folder: Path, table: Path, out: Optional[Path] = None, bank=None) -> str:
    from src.gmr.domain.note_match import AccountIndex
    from src.gmr.overlay import GlyphBank, read
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"нет папки {folder}")
    base, photos, truth, lasts = truths(folder, table)
    if not photos:
        raise ValueError(f"в {folder} нет фото")
    out = Path(out) if out else folder.parent / f"{folder.name}_надписи"
    for s in STATUSES + _OLD:                           # прошлый замер — заменяется (только свои папки)
        shutil.rmtree(out / s, ignore_errors=True)
    bank = bank or GlyphBank.load()
    index = AccountIndex(lasts)
    stat, tests, unread_files = Counter(), Counter(), 0
    rows = []
    started = time.monotonic()
    for i, p in enumerate(photos, 1):
        img = read_image(p)
        if img is None:
            unread_files += 1
            continue
        dropped: list = []
        lines = read(img, bank, dropped)
        row = judge(lines, dropped, truth.get(p), index, lasts)
        if row["итог"]:
            name = "__".join(p.relative_to(base).with_suffix("").parts) + ".jpg"
            row["фото"] = p.relative_to(base).as_posix()
            stat[row["итог"]] += 1
            tests.update((k, row[k]) for k in ("счёт", "показание") if row.get(k))
            (out / row["итог"]).mkdir(parents=True, exist_ok=True)
            write_image(out / row["итог"] / name, picture(img, lines, dropped, row))
            rows.append(row)
        if i % 25 == 0 or i == len(photos):
            print(f"Фото {i} из {len(photos)}{time_left(started, i, len(photos))}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / REPORT, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS, delimiter=";", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    noted = sum(stat[s] for s in (RIGHT, DANGER, CONFLICT, UNCLEAR, NO_TRUTH))
    checked = noted - stat[NO_TRUTH]

    def counts(kind: str) -> str:
        return ", ".join(f"{s} {tests[(kind, s)]}" for s in (RIGHT, DANGER, CONFLICT))

    text = [f"Надписи: {folder} — фото {len(photos)}",
            f"  надпись есть: {noted} ({100 * noted / len(photos):.0f}%)",
            f"  по фото с ответом в таблице ({checked}): верно {stat[RIGHT]}, опасно {stat[DANGER]}, "
            f"спор (к оператору) {stat[CONFLICT]}, не понять (надпись не используется) {stat[UNCLEAR]}",
            f"    счёт по надписи (как без таблички): {counts('счёт')}",
            f"    показание по надписи (как без барабана): {counts('показание')}"]
    if stat[NO_TRUTH]:
        text.append(f"  надпись есть, но ответа в таблице нет: {stat[NO_TRUTH]}")
    if stat[NOT_TEXT]:
        text.append(f"  найдено только не-текст (тени, трубы — отброшено): {stat[NOT_TEXT]}")
    if unread_files:
        text.append(f"  не открылись как картинка: {unread_files}")
    text.append(f"Картинки для разбора — {out} ({', '.join(STATUSES)}), по каждому фото — {out / REPORT}")
    return "\n".join(text)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python gmr.py notes",
                                description="Замер: что программа поняла бы из надписей контролёров поверх фото")
    p.add_argument("folder", help="папка фото (имя — лицевой счёт) или папка эталона архива")
    p.add_argument("--table", required=True, help="таблица: лицевой счёт, номер счётчика, показания")
    p.add_argument("--out", help="куда картинки и отчёт (по умолчанию <папка>_надписи рядом)")
    args = p.parse_args(argv)
    try:
        print(run(Path(args.folder), Path(args.table), Path(args.out) if args.out else None))
    except (ValueError, OSError) as e:
        print(f"ОШИБКА: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
