"""
tools/quality_report.py — качество распознавания по месяцам (по работе
оператора: «Проверка» и «Очередь», src/gmr/application/quality.py).

  python gmr.py quality <месяц | папка с месяцами> [...] [--csv качество.csv]

Для каждого месяца: сколько фото программа записала сама, сколько ушло
оператору, сколько её показаний оператор проверил и исправил — процент ошибок
с пределами. Так видно, как программа работает «в поле» от месяца к месяцу.
"""
import argparse
import csv
import sys
from pathlib import Path


def months_in(paths) -> list[Path]:
    from src.gmr.storage.month import MonthFolder
    out = []
    for p in map(Path, paths):
        if (p / "gmr.sqlite").is_file():
            out.append(p)
        elif p.is_dir():
            out += sorted(d for d in p.iterdir() if (MonthFolder(d).db).is_file())
    return out


def main(argv=None) -> int:
    from src.gmr.application.quality import month_quality, quality_row
    from src.gmr.storage.month import MonthDB, MonthFolder
    p = argparse.ArgumentParser(prog="python gmr.py quality", description=__doc__.split("\n\n")[0])
    p.add_argument("paths", nargs="+", help="папка месяца или папка, где лежат месяцы")
    p.add_argument("--csv", help="записать таблицу в CSV (для Excel)")
    a = p.parse_args(argv)
    months = months_in(a.paths)
    if not months:
        print("Месяцев не найдено (папка месяца — там, где gmr.sqlite).")
        return 1
    rows = []
    for m in months:
        with MonthDB(MonthFolder(m).db) as db:
            rows.append(quality_row(m.name, month_quality(db.log_rows())))
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    if a.csv:
        with open(a.csv, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, delimiter=";")
            w.writeheader()
            w.writerows(rows)
        print(f"Таблица: {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
