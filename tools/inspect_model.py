"""
tools/inspect_model.py — посмотреть, что делает с изображением одна модель
(Фаза 4, docs/MIGRATION_TZ.md). Запускать из папки проекта:

  python -m tools.inspect_model meter  <фото|папка>   рамки gas_meter/serial_number + кропы
  python -m tools.inspect_model digits <фото|папка>   рамки цифр на кропе счётчика + кропы цифр
  python -m tools.inspect_model serial <фото|папка>   текст серийника + уверенность по символам
  python -m tools.inspect_model digit  <кроп|папка>   цифра + уверенность
  python -m tools.inspect_model photo  <фото|папка>   всё вместе — как видит фото program2.py

Общие флаги:
  --out ПАПКА       куда сохранить результат (по умолчанию inspect_out/<модель>_<время>)
  --from photo|crop что подаётся на вход: целое фото или уже готовый кроп для этой
                    модели. По умолчанию photo, для digit — crop. С photo недостающие
                    шаги (найти счётчик, найти цифры) делают прод-модели.
  --conf X          поменять порог (детекторы — порог детекции, распознаватели —
                    порог «уверенно»). По умолчанию — тот же, что в reader.py.
  --weights ФАЙЛ    другой файл весов для этой модели — сравнить с текущей.

По умолчанию всё как в reader.py: те же веса, пороги и выпрямление кропов
(модели грузятся через src/gmr/ml/loader.py). Папка обходится с подпапками.
В выходной папке: картинки с рамками/подписями, вырезанные кропы и
summary.csv (разделитель «;», открывается в Excel).
"""
import argparse
import csv
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from src.gmr.application import find_detection, read_meter_digits
from src.gmr.domain import PipelineConfig

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MODES = ("meter", "digits", "serial", "digit", "photo")

_GREEN, _BLUE, _ORANGE, _RED = (0, 200, 0), (200, 0, 0), (0, 140, 255), (0, 0, 255)
_CLASS_COLOR = {"gas_meter": _GREEN, "serial_number": _BLUE}


# ─── Файлы ───────────────────────────────────────────────────────────────────
# Через imdecode/imencode: cv2.imread/imwrite на Windows не открывают пути
# с кириллицей (папки операторов).

def read_image(path: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_image(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if ok:
        buf.tofile(str(path))


def collect_images(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def output_stem(path: Path, base: Path) -> str:
    """Имя для результатов: путь относительно входной папки, '/' → '__'."""
    rel = path.relative_to(base) if base.is_dir() else Path(path.name)
    return "__".join(rel.with_suffix("").parts)


# ─── Рисование ───────────────────────────────────────────────────────────────

def _scale_up(img: np.ndarray, min_h: int = 160) -> tuple[np.ndarray, float]:
    h = img.shape[0]
    if h >= min_h:
        return img.copy(), 1.0
    k = min_h / h
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST), k


def draw_box(img: np.ndarray, bbox, label: str, color, scale: float = 1.0) -> None:
    x1, y1, x2, y2 = (int(round(v * scale)) for v in bbox)
    thick = max(1, img.shape[1] // 400)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    font = max(0.4, img.shape[1] / 1600)
    cv2.putText(img, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                font, color, max(1, thick), cv2.LINE_AA)


def with_caption(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    """Кроп (увеличенный, если маленький) с подписью на тёмной полосе снизу."""
    big, _ = _scale_up(img)
    font = 0.6
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, 1)
    bar = np.zeros((th + 14, max(big.shape[1], tw + 10), 3), dtype=np.uint8)
    cv2.putText(bar, text, (5, th + 6), cv2.FONT_HERSHEY_SIMPLEX, font, color, 1, cv2.LINE_AA)
    if bar.shape[1] > big.shape[1]:
        pad = np.zeros((big.shape[0], bar.shape[1] - big.shape[1], 3), dtype=np.uint8)
        big = np.hstack([big, pad])
    return np.vstack([big, bar])


def _chars(details: list) -> str:
    return " ".join(f"{d['char']}({d['conf']:.2f})" for d in details or [])


# ─── Модели для одного запуска ───────────────────────────────────────────────

@dataclasses.dataclass
class Models:
    """Лениво грузит только те модели, которые нужны режиму."""
    config: PipelineConfig
    loaders: dict[str, Callable[[], object]]
    _cache: dict = dataclasses.field(default_factory=dict)

    def get(self, name: str):
        if name not in self._cache:
            self._cache[name] = self.loaders[name]()
        return self._cache[name]


def _sorted_digits(detections) -> list[dict]:
    return sorted(detections or [], key=lambda c: c["bbox"][0])


# ─── Режимы: по одному на модель ─────────────────────────────────────────────
# Каждый получает путь к изображению, пишет картинки в out и возвращает
# строки для summary.csv.

def inspect_meter(models: Models, path: Path, stem: str, out: Path, from_crop: bool) -> list[dict]:
    img = read_image(path)
    dets = models.get("meter").detect(str(path)) or []
    rows = []
    if img is not None:
        vis = img.copy()
        for d in dets:
            draw_box(vis, d["bbox"], f"{d['class']} {d['conf']:.2f}",
                     _CLASS_COLOR.get(d["class"], _ORANGE))
        write_image(out / f"{stem}.jpg", vis)
    for i, d in enumerate(dets):
        crop_name = f"{stem}__{d['class']}_{i}.jpg"
        write_image(out / crop_name, d["crop"])
        rows.append({"file": str(path), "class": d["class"], "conf": round(float(d["conf"]), 4),
                     "angle": round(float(d.get("angle") or 0.0), 2),
                     "bbox": " ".join(str(int(v)) for v in d["bbox"]), "result_file": crop_name})
    if not dets:
        rows.append({"file": str(path), "class": "", "note": "ничего не найдено"})
    return rows


def _meter_crop(models: Models, path: Path, from_crop: bool):
    """Кроп счётчика: сам файл (--from crop) или gas_meter от детектора счётчика."""
    if from_crop:
        return read_image(path), None
    det = find_detection(models.get("meter").detect(str(path)), "gas_meter")
    if det is None:
        return None, "детектор счётчика не нашёл gas_meter"
    return det["crop"], None


def inspect_digits(models: Models, path: Path, stem: str, out: Path, from_crop: bool) -> list[dict]:
    crop, err = _meter_crop(models, path, from_crop)
    if crop is None:
        return [{"file": str(path), "note": err or "не удалось открыть"}]
    dets = _sorted_digits(models.get("digits").detect(crop))
    expected = models.config.expected_digits
    vis, k = _scale_up(crop, 240)
    rows = []
    for pos, d in enumerate(dets):
        draw_box(vis, d["bbox"], f"{pos}:{d['conf']:.2f}", _ORANGE, scale=k)
        crop_name = f"{stem}__d{pos}.jpg"
        write_image(out / crop_name, d["crop"])
        rows.append({"file": str(path), "n_digits": len(dets), "expected": expected,
                     "position": pos, "conf": round(float(d["conf"]), 4),
                     "bbox": " ".join(str(int(v)) for v in d["bbox"]), "result_file": crop_name})
    write_image(out / f"{stem}.jpg", with_caption(vis, f"found {len(dets)} of {expected}"))
    if not dets:
        rows.append({"file": str(path), "n_digits": 0, "expected": expected, "note": "цифр не найдено"})
    return rows


def inspect_serial(models: Models, path: Path, stem: str, out: Path, from_crop: bool) -> list[dict]:
    if from_crop:
        crop = read_image(path)
        if crop is None:
            return [{"file": str(path), "note": "не удалось открыть"}]
    else:
        det = find_detection(models.get("meter").detect(str(path)), "serial_number")
        if det is None:
            return [{"file": str(path), "note": "детектор не нашёл serial_number"}]
        crop = det["crop"]
    pred = models.get("serial").recognize(crop)
    thr = models.config.serial_conf_thresh
    ok = pred.confidence >= thr
    write_image(out / f"{stem}.jpg", with_caption(
        crop, f"'{pred.text}' {pred.confidence:.3f} {'OK' if ok else 'LOW'}", _GREEN if ok else _RED))
    return [{"file": str(path), "text": pred.text, "conf": round(float(pred.confidence), 4),
             "threshold": thr, "ok": ok, "length": len(pred.text),
             "chars": _chars(pred.details), "result_file": f"{stem}.jpg"}]


def inspect_digit(models: Models, path: Path, stem: str, out: Path, from_crop: bool) -> list[dict]:
    if from_crop:
        crop = read_image(path)
        if crop is None:
            return [{"file": str(path), "note": "не удалось открыть"}]
        crops = [(stem, crop)]
    else:
        meter, err = _meter_crop(models, path, False)
        if meter is None:
            return [{"file": str(path), "note": err}]
        dets = _sorted_digits(models.get("digits").detect(meter))
        if not dets:
            return [{"file": str(path), "note": "цифр не найдено"}]
        crops = [(f"{stem}__d{i}", d["crop"]) for i, d in enumerate(dets)]
    thr = models.config.digit_conf_thresh
    rows = []
    for name, crop in crops:
        pred = models.get("digit").recognize(crop)
        ok = pred.confidence >= thr
        write_image(out / f"{name}.jpg", with_caption(
            crop, f"{pred.digit} {pred.confidence:.2f}", _GREEN if ok else _RED))
        rows.append({"file": str(path), "crop": name, "digit": pred.digit,
                     "conf": round(float(pred.confidence), 4), "threshold": thr, "ok": ok,
                     "result_file": f"{name}.jpg"})
    return rows


def inspect_photo(models: Models, path: Path, stem: str, out: Path, from_crop: bool) -> list[dict]:
    """Все модели по очереди, с порогами и заглушками reader.py (без таблицы и лога)."""
    cfg = models.config
    img = read_image(path)
    dets = models.get("meter").detect(str(path)) or []
    meter = find_detection(dets, "gas_meter")
    serial = find_detection(dets, "serial_number")
    row = {"file": str(path), "classes": " ".join(d["class"] for d in dets)}

    if serial is not None:
        sp = models.get("serial").recognize(serial["crop"])
        row.update(serial_text=sp.text, serial_conf=round(float(sp.confidence), 4),
                   serial_ok=sp.confidence >= cfg.serial_conf_thresh)
    if meter is not None:
        reading = read_meter_digits(
            models.get("digits"), models.get("digit"), meter["crop"],
            cfg.digit_conf_thresh, cfg.expected_digits,
            ignore_last_digits=cfg.ignore_last_digits,
            missing_placeholder=cfg.missing_digit_placeholder,
            forgiven_placeholder=cfg.forgiven_digit_placeholder,
        )
        row.update(reading_str=reading.reading_str, reading=reading.number, error=reading.error,
                   digits=" ".join(
                       f"{r['digit']}({r['confidence']:.2f})" if r["confidence"] is not None
                       else f"{r['digit']}(заглушка)" for r in reading.digit_results or []))
        vis, k = _scale_up(meter["crop"], 240)
        for pos, (bbox, res) in enumerate(zip(reading.digit_bboxes or [], reading.digit_results or [])):
            if bbox is not None:
                draw_box(vis, bbox, f"{res['digit']}", _GREEN if res["ok"] else _RED, scale=k)
        write_image(out / f"{stem}__meter.jpg",
                    with_caption(vis, f"{reading.reading_str} {'' if reading.number is not None else 'ERR'}"))
    if img is not None:
        vis = img.copy()
        for d in dets:
            draw_box(vis, d["bbox"], f"{d['class']} {d['conf']:.2f}",
                     _CLASS_COLOR.get(d["class"], _ORANGE))
        write_image(out / f"{stem}.jpg", vis)
    if not dets:
        row["note"] = "ничего не найдено"
    return [row]


INSPECTORS = {
    "meter": inspect_meter, "digits": inspect_digits, "serial": inspect_serial,
    "digit": inspect_digit, "photo": inspect_photo,
}

# режим → (какую модель смотрим, поле порога, поле весов)
_MODE_MODEL = {
    "meter":  ("meter",  "meter_conf_thresh",        "meter_detect_model"),
    "digits": ("digits", "digit_detect_conf_thresh", "digit_detect_model"),
    "serial": ("serial", "serial_conf_thresh",       "serial_ocr_model"),
    "digit":  ("digit",  "digit_conf_thresh",        "digit_ocr_model"),
}


def run(mode: str, target: Path, out: Path, models: Models, from_crop: bool) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    paths = collect_images(target)
    for i, path in enumerate(paths, 1):
        stem = output_stem(path, target)
        try:
            new = INSPECTORS[mode](models, path, stem, out, from_crop)
        except Exception as e:   # одно битое фото не должно останавливать папку
            new = [{"file": str(path), "note": f"ошибка: {e}"}]
        rows += new
        print(f"[{i}/{len(paths)}] {path.name}: " + "; ".join(_short(r) for r in new))
    write_summary(out / "summary.csv", rows)
    print(f"\nГотово: {len(paths)} изображений → {out}")
    return rows


def _short(row: dict) -> str:
    keys = ("class", "text", "digit", "reading_str", "position", "conf", "ok", "note", "error")
    return " ".join(f"{k}={row[k]}" for k in keys if row.get(k) not in (None, ""))


def write_summary(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["file"], delimiter=";")
        w.writeheader()
        w.writerows(rows)


def build_config(mode: str, conf: Optional[float], weights: Optional[str]) -> PipelineConfig:
    cfg = PipelineConfig()
    if mode in _MODE_MODEL:
        _, conf_field, weights_field = _MODE_MODEL[mode]
        if conf is not None:
            cfg = dataclasses.replace(cfg, **{conf_field: conf})
        if weights is not None:
            cfg = dataclasses.replace(cfg, **{weights_field: weights})
    return cfg


def default_loaders(cfg: PipelineConfig) -> dict[str, Callable[[], object]]:
    from src.gmr.ml import loader   # torch/ultralytics — только при реальном запуске

    def resolve(p: str) -> str:
        return p if Path(p).is_absolute() else str(ROOT / p)
    return {
        "meter":  lambda: loader.load_meter_detector(cfg, resolve),
        "digits": lambda: loader.load_digit_detector(cfg, resolve),
        "serial": lambda: loader.load_serial_recognizer(cfg, resolve_path=resolve),
        "digit":  lambda: loader.load_digit_recognizer(cfg, resolve_path=resolve),
    }


def main(argv: Optional[list[str]] = None, loaders_factory=default_loaders) -> list[dict]:
    p = argparse.ArgumentParser(
        prog="python -m tools.inspect_model",
        description="Посмотреть, что делает с изображением одна модель (настройки — как в reader.py).",
    )
    p.add_argument("mode", choices=MODES)
    p.add_argument("target", help="изображение или папка (с подпапками)")
    p.add_argument("--out", default=None, help="папка для результатов")
    p.add_argument("--from", dest="source", choices=("photo", "crop"), default=None,
                   help="на входе целые фото или готовые кропы (по умолчанию photo, для digit — crop)")
    p.add_argument("--conf", type=float, default=None, help="свой порог вместо прод-порога")
    p.add_argument("--weights", default=None, help="другой файл весов для этой модели")
    args = p.parse_args(argv)

    if args.mode == "photo" and (args.conf is not None or args.weights is not None):
        p.error("режим photo всегда с прод-настройками; --conf/--weights — для одной модели")
    if args.mode in ("meter", "photo") and args.source == "crop":
        p.error(f"режиму {args.mode} нужно целое фото")
    source = args.source or ("crop" if args.mode == "digit" else "photo")

    target = Path(args.target)
    if not target.exists():
        p.error(f"не найдено: {target}")
    out = Path(args.out) if args.out else (
        ROOT / "inspect_out" / f"{args.mode}_{datetime.now():%Y-%m-%d_%H-%M-%S}")

    cfg = build_config(args.mode, args.conf, args.weights)
    if args.mode in _MODE_MODEL:
        _, conf_field, weights_field = _MODE_MODEL[args.mode]
        print(f"Модель: {getattr(cfg, weights_field)}")
        print(f"Порог {conf_field}: {getattr(cfg, conf_field)}"
              + ("  (свой)" if args.conf is not None else "  (как в reader.py)"))
    models = Models(cfg, loaders_factory(cfg))
    return run(args.mode, target, out, models, from_crop=(source == "crop"))


if __name__ == "__main__":
    main()
