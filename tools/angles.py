"""
tools/angles.py — замер: насколько повёрнуты счётчики на фото (этап 7a,
решение владельца 2026-10-12, «а»: сначала замер, потом — повёрнутые рамки
YOLO-OBB). Работу программы не меняет: только читает фото и считает.

  python gmr.py angles <папка месяца | папка с фото> [--csv <файл>] [--limit N]

Для каждого фото:
  - наклон показаний и таблички с номером — тем же способом, каким программа
    выпрямляет кропы (угол ищется от −25° до +25°; «25° и больше» — на
    пределе поиска, на деле, скорее всего, больше);
  - рамка показаний выше, чем шире — счётчик на фото стоит боком;
  - счётчик не найден → нашёлся бы, если повернуть фото на 90°, 180°, 270°;
  - показания или номер не прочитаны уверенно → прочитались бы уверенно,
    если перевернуть кроп на 180° (верно ли — замер не проверяет).
В выводе только числа — его можно присылать. По каждому фото — CSV для
Excel (`;`, углы с запятой): путь к фото, углы, что нашлось. По нему можно
отобрать ровные фото (наклон до 2°) для обучения повёрнутых рамок.
"""
import argparse
import contextlib
import csv
import dataclasses
import io
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from models.datasets import IMAGE_EXTS
from src.gmr.application.recognition import find_detection, read_meter_digits_for_config
from src.gmr.domain import PipelineConfig
from src.gmr.render.image_io import read_image

TURNS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
EDGE = 25.0                                   # предел поиска угла (YOLOInferer.find_best_angle)
BINS = ((2, "до 2°"), (5, "2–5°"), (10, "5–10°"), (15, "10–15°"), (EDGE, "15–25°"), (None, "25° и больше"))


@dataclass
class PhotoAngles:
    file: str
    meter: bool = False
    meter_angle: Optional[float] = None
    sideways: bool = False                    # рамка показаний выше, чем шире
    digits_ok: Optional[bool] = None          # показания прочитаны (как в работе)
    digits_ok_turned: Optional[bool] = None   # не прочитаны, а перевёрнутые на 180° — прочитаны бы
    serial: bool = False
    serial_angle: Optional[float] = None
    serial_sure: Optional[bool] = None
    serial_sure_turned: Optional[bool] = None
    found_turned: Optional[int] = None        # счётчик не найден, а на фото, повёрнутом на N°, — найден
    opened: bool = True                       # файл открылся как картинка


HEADERS = {"file": "фото", "meter": "показания найдены", "meter_angle": "наклон показаний",
           "sideways": "боком", "digits_ok": "прочитаны", "digits_ok_turned": "перевёрнутые прочитаны",
           "serial": "табличка найдена", "serial_angle": "наклон таблички", "serial_sure": "номер уверенный",
           "serial_sure_turned": "перевёрнутый уверенный", "found_turned": "найден после поворота фото",
           "opened": "открылось"}


def _detect(models, img) -> list:
    with contextlib.redirect_stdout(io.StringIO()):     # «Детектор ничего не нашёл» — на каждый поворот
        return models.meter_detector.detect_array(img) or []


def _digits_ok(models, crop, config) -> bool:
    return read_meter_digits_for_config(models, crop, config).number is not None


def _serial_sure(models, crop, config) -> bool:
    return models.serial_recognizer.recognize(crop).confidence >= config.serial_conf_thresh


def measure(models, img, config: PipelineConfig, file: str) -> PhotoAngles:
    r = PhotoAngles(file)
    found = _detect(models, img)
    meter, serial = find_detection(found, "gas_meter"), find_detection(found, "serial_number")
    if meter is None:
        r.found_turned = next((turn for turn, code in TURNS.items()
                               if find_detection(_detect(models, cv2.rotate(img, code)), "gas_meter")), None)
    else:
        x1, y1, x2, y2 = meter["bbox"]
        r.meter, r.meter_angle, r.sideways = True, float(meter["angle"]), (y2 - y1) > (x2 - x1)
        r.digits_ok = _digits_ok(models, meter["crop"], config)
        if not r.digits_ok:
            r.digits_ok_turned = _digits_ok(models, cv2.rotate(meter["crop"], cv2.ROTATE_180), config)
    if serial is not None:
        r.serial, r.serial_angle = True, float(serial["angle"])
        r.serial_sure = _serial_sure(models, serial["crop"], config)
        if not r.serial_sure:
            r.serial_sure_turned = _serial_sure(models, cv2.rotate(serial["crop"], cv2.ROTATE_180), config)
    return r


def angle_bins(angles: list) -> str:
    counts = Counter()
    for a in angles:
        a = abs(a)
        counts[next(name for top, name in BINS if top is None or a < top - 1e-9)] += 1
    return "  ".join(f"{name}: {counts[name]}" for _, name in BINS)


def _pct(k: int, n: int) -> str:
    return f"{k} ({100 * k / n:.0f}%)" if n else "0"


def summary(rows: list[PhotoAngles], where: str) -> str:
    n = len(rows)
    meters = [r for r in rows if r.meter]
    serials = [r for r in rows if r.serial]
    lost = [r for r in rows if r.opened and not r.meter]
    unread = [r for r in meters if not r.digits_ok]
    unsure = [r for r in serials if not r.serial_sure]
    turned = Counter(r.found_turned for r in lost)
    return "\n".join([
        f"Замер наклона: {n} фото ({where})",
        "",
        f"Показания найдены: {_pct(len(meters), n)}",
        f"  наклон: {angle_bins([r.meter_angle for r in meters])}",
        f"  счётчик боком (рамка выше, чем шире): {sum(r.sideways for r in meters)}",
        f"  не прочитаны: {len(unread)} — прочитались бы, если перевернуть на 180°: "
        f"{sum(bool(r.digits_ok_turned) for r in unread)}",
        f"Табличка с номером найдена: {_pct(len(serials), n)}",
        f"  наклон: {angle_bins([r.serial_angle for r in serials])}",
        f"  номер неуверенный: {len(unsure)} — уверенный, если перевернуть на 180°: "
        f"{sum(bool(r.serial_sure_turned) for r in unsure)}",
        f"Счётчик не найден: {_pct(len(lost), n)}",
        "  нашёлся бы, если повернуть фото: " + ", ".join(f"на {t}° — {turned[t]}" for t in TURNS)
        + f"; никак — {turned[None]}",
        *([f"Не открылись как картинка: {sum(not r.opened for r in rows)}"] if not all(r.opened for r in rows)
          else []),
        "",
        "Наклон — тем же способом, каким программа выпрямляет кропы (поиск от −25° до +25°);",
        "«25° и больше» — на пределе поиска. «Прочитались бы» — уверенно, верно ли — не проверено.",
    ])


def photos_of(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "да" if v else "нет"
    if isinstance(v, float):
        return f"{v:.1f}".replace(".", ",")          # Excel с русскими настройками
    return str(v)


def write_csv(path: Path, rows: list[PhotoAngles]) -> None:
    fields = [f.name for f in dataclasses.fields(PhotoAngles)]
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow([HEADERS[f] for f in fields])
        w.writerows([_fmt(getattr(r, f)) for f in fields] for r in rows)


def _source(path: Path) -> tuple[Path, Path, PipelineConfig]:
    """Папка с фото, CSV по умолчанию и настройки: у папки месяца — её фото\\
    и её настройки распознавания (пресет)."""
    from src.gmr.application.month import month_preset
    from src.gmr.storage.month import MonthFolder
    month = MonthFolder(path)
    if month.exists():
        preset = month_preset(path)
        cfg = preset.apply(PipelineConfig()) if preset else PipelineConfig()
        return month.photos, month.root / "наклон.csv", cfg
    return path, path.parent / f"наклон_{path.name}.csv", PipelineConfig()


def run(path: Path, csv_path: Optional[Path] = None, limit: Optional[int] = None, models=None) -> str:
    folder, default_csv, config = _source(Path(path))
    if not folder.is_dir():
        raise ValueError(f"нет папки {folder}")
    paths = photos_of(folder)[:limit] if limit else photos_of(folder)
    if not paths:
        raise ValueError(f"в {folder} нет фото")
    if models is None:
        from src.gmr.ml.loader import load_models
        models = load_models(config)
    rows = []
    for i, p in enumerate(paths, 1):
        img = read_image(p)
        rel = p.relative_to(folder).as_posix()
        rows.append(measure(models, img, config, rel) if img is not None else PhotoAngles(rel, opened=False))
        if i % 25 == 0 or i == len(paths):
            print(f"Фото {i} из {len(paths)}", flush=True)
    out = Path(csv_path) if csv_path else default_csv
    write_csv(out, rows)
    return summary(rows, str(folder)) + f"\n\nПо каждому фото: {out}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python gmr.py angles",
                                description="Замер: насколько повёрнуты счётчики на фото (ничего не меняет).")
    p.add_argument("folder", help="папка месяца или любая папка с фото (с подпапками)")
    p.add_argument("--csv", default=None, help="куда записать результат по каждому фото "
                                               "(по умолчанию <месяц>\\наклон.csv или рядом с папкой)")
    p.add_argument("--limit", type=int, default=None, help="только первые N фото — попробовать быстро")
    args = p.parse_args(argv)
    try:
        print(run(Path(args.folder), Path(args.csv) if args.csv else None, args.limit))
    except ValueError as e:
        print(f"ОШИБКА: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
