"""
tools/notes.py — замер: надписи контролёров поверх фото (src/gmr/overlay.py)
находятся и читаются верно? Работу программы не меняет.

  python gmr.py notes <папка фото> --table <таблица> [--out <папка>]
  python gmr.py notes database\\datasets\\etalon_archive --table <таблица>

Папка фото — фото названы по лицевому счёту (как архив месяца): правильный
ответ — этот счёт и его «Текущие показания» из таблицы. Папка эталона
архива (answers.csv + photos\\) — фото по отпечатку: ответ — номер счётчика и
показание из answers.csv, счёт — владелец этого номера по таблице.

Строка надписи «верно», если каждое число в ней — запись правильного счёта
(is_account_form: без нулей, начало счёта + остаток, хвост из ≥4 значащих
цифр) или правильное показание; приписка
«лс» требует счёт, «пок»/«пк» — показание. Число, перенесённое на
следующую строку («лс04290» / «05357»), сверяется и вместе. «Не уверен» — какой-то символ
почти одинаково похож на два (в работе такое фото пошло бы оператору).

Вывод — только числа. Вырезки строк — <out>\\верно\\, не_совпало\\,
неуверенно\\, нет_ответа\\ (имя — фото и номер строки), по каждой строке —
<out>\\надписи.csv. В не_совпало\\ видно: ошибка чтения, обрезанная строка,
не надпись (найдено лишнее) или ответ таблицы не тот.
"""
import argparse
import csv
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from models.datasets import time_left
from src.gmr.domain import PipelineConfig
from src.gmr.domain.account_match import account_variants
from src.gmr.render.image_io import read_image, write_image

STATUSES = ("верно", "не_совпало", "неуверенно", "нет_ответа")
REPORT = "надписи.csv"
_MIN_TAIL = 4


def is_account_form(number: str, account: str) -> bool:
    """Так можно записать счёт (контролёры пишут как придётся):
    - без нулей, как на корпусе маркером (account_variants: «17777»);
    - начало счёта как есть, остаток — с другим числом нулей
      («0429» + «00013» — счёт 0429000013);
    - конец счёта с ≥4 значащими цифрами («9005357» — счёт 0429005357)."""
    if number in account_variants(account):
        return True
    if not (number.isdigit() and account.isdigit()):
        return False
    for k in range(1, len(account)):
        head, rest = account[:k], account[k:]
        if number.startswith(head) and len(number) > k and int(rest) and int(number[k:]) == int(rest):
            return True
    return len(number.lstrip("0")) >= _MIN_TAIL and any(
        int(number) == int(account[k:]) for k in range(len(account)))


def _int(text) -> Optional[int]:
    try:
        return int(float(str(text).replace(",", ".")))
    except ValueError:
        return None


def agrees(tokens: list, accounts: set, reading: Optional[str]) -> bool:
    """Каждое число строки — запись правильного счёта или правильное показание."""
    from src.gmr.overlay import ACCOUNT, READING
    want = _int(reading) if reading else None
    for kind, number in tokens:
        is_acc = any(is_account_form(number, a) for a in accounts)
        is_rd = want is not None and int(number) == want
        if not {ACCOUNT: is_acc, READING: is_rd}.get(kind, is_acc or is_rd):
            return False
    return bool(tokens)


def readings_of(texts: list, n: int) -> list[str]:
    """Как можно понять строку n: сама по себе или вместе с соседней, если
    число перенесено на следующую строку (overlay.wrapped); и каждое — с «З»
    вместо «3» после «п»/«к» («ПКЗ», overlay.spellings)."""
    from src.gmr.overlay import spellings, wrapped
    out = [texts[n]]
    if n + 1 < len(texts) and wrapped(texts[n], texts[n + 1]):
        out.append(texts[n] + texts[n + 1])
    if n > 0 and wrapped(texts[n - 1], texts[n]):
        out.append(texts[n - 1] + texts[n])
    return list(dict.fromkeys(v for t in out for v in spellings(t)))


def truths(folder: Path, table: Path) -> tuple[list[Path], dict]:
    """(фото, {фото: (счета, показание)}) — правильные ответы из таблицы."""
    from tools.dataset_build import account_of_photo, only_one, photos_in, read_answers
    serials, readings, _ = read_answers(Path(table), PipelineConfig())
    answers = Path(folder) / "answers.csv"
    if answers.is_file() and (Path(folder) / "photos").is_dir():         # эталон архива
        owners: dict = {}
        for account, ss in serials.items():
            for s in ss:
                owners.setdefault(s.lstrip("0"), set()).add(account)
        with open(answers, encoding="utf-8-sig", newline="") as fh:
            rows = {r["file"]: r for r in csv.DictReader(fh)}
        photos = photos_in(Path(folder) / "photos")
        return photos, {p: (owners.get(rows[p.name]["serial"].lstrip("0"), set()), rows[p.name]["reading"])
                        for p in photos if p.name in rows}
    photos = photos_in(folder)
    truth = {}
    for p in photos:
        account = account_of_photo(p)
        if account and (account in serials or account in readings):
            truth[p] = ({account}, only_one(readings.get(account, set())))
    return photos, truth


def run(folder: Path, table: Path, out: Optional[Path] = None, bank=None) -> str:
    from src.gmr.overlay import GlyphBank, parse, read
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"нет папки {folder}")
    photos, truth = truths(folder, table)
    if not photos:
        raise ValueError(f"в {folder} нет фото")
    out = Path(out) if out else folder.parent / f"{folder.name}_надписи"
    for s in STATUSES:                                  # прошлый замер — заменяется (только свои папки)
        shutil.rmtree(out / s, ignore_errors=True)
    bank = bank or GlyphBank.load()
    stat, with_note, unread = Counter(), 0, 0
    rows = []
    started = time.monotonic()
    for i, p in enumerate(photos, 1):
        img = read_image(p)
        if img is None:
            unread += 1
            continue
        lines = read(img, bank)
        with_note += bool(lines)
        texts = [ln.text for ln in lines]
        for n, line in enumerate(lines, 1):
            if not line.sure:
                status = "неуверенно"
            elif p not in truth:
                status = "нет_ответа"
            else:
                status = "верно" if any(agrees(parse(t), *truth[p]) for t in readings_of(texts, n - 1)) \
                    else "не_совпало"
            stat[status] += 1
            name = f"{p.stem}__{n}.png"
            (out / status).mkdir(parents=True, exist_ok=True)
            write_image(out / status / name, line.crop)
            rows.append({"фото": p.relative_to(folder).as_posix(), "строка": n, "текст": line.text,
                         "уверенность": f"{line.margin:.3f}".replace(".", ","), "итог": status})
        if i % 25 == 0 or i == len(photos):
            print(f"Фото {i} из {len(photos)}{time_left(started, i, len(photos))}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / REPORT, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["фото", "строка", "текст", "уверенность", "итог"], delimiter=";")
        w.writeheader()
        w.writerows(rows)
    total = sum(stat.values())
    sure = stat["верно"] + stat["не_совпало"]
    lines = [f"Надписи: {folder} — фото {len(photos)}",
             f"  с надписью: {with_note} ({100 * with_note / len(photos):.0f}%), строк: {total}",
             f"  уверенно прочитано и сверено с таблицей: {sure} — верно {stat['верно']}, "
             f"не совпало {stat['не_совпало']}",
             f"  не уверен (пошло бы оператору): {stat['неуверенно']}"]
    if stat["нет_ответа"]:
        lines.append(f"  уверенно, но ответа в таблице нет: {stat['нет_ответа']}")
    if unread:
        lines.append(f"  не открылись как картинка: {unread}")
    lines.append(f"Вырезки строк — {out} ({', '.join(STATUSES)}), по каждой строке — {out / REPORT}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python gmr.py notes",
                                description="Замер: надписи контролёров поверх фото — найдены и прочитаны верно?")
    p.add_argument("folder", help="папка фото (имя — лицевой счёт) или папка эталона архива")
    p.add_argument("--table", required=True, help="таблица: номер счётчика, текущие показания")
    p.add_argument("--out", help="куда вырезки и отчёт (по умолчанию <папка>_надписи рядом)")
    args = p.parse_args(argv)
    try:
        print(run(Path(args.folder), Path(args.table), Path(args.out) if args.out else None))
    except (ValueError, OSError) as e:
        print(f"ОШИБКА: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
