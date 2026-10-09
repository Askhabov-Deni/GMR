"""
tools/datasets.py — папки датасетов и их проверка (этап 5).

  python gmr.py datasets [--root database/datasets] [--list]

Создаёт недостающие папки раскладки (models/datasets.py) — ничего не
переносит и не удаляет — и показывает по каждому датасету: сколько файлов,
что не так (фото без разметки, разметка без фото, ошибки в разметке,
одинаковые файлы) и как его поделит обучение. В выводе только числа: его
можно прислать, персональных данных в нём нет. --list — имена проблемных
файлов в <root>/datasets_problems.txt (остаётся у вас на диске).
"""
import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

from models.datasets import DATASETS_DIR, DIGITS, IMAGE_EXTS, ensure_layout, images_in, photo_key, split_by_photo

YOLO_VAL, YOLO_SEED = 0.2, 67            # как models/yolo_all_detect/train_yolo.py
CRNN_TRAIN, CRNN_VAL, CRNN_SEED = 0.7, 0.15, 42   # как models/crnn/train_crnn.py


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _duplicates(files) -> list[list[Path]]:
    """Группы одинаковых по содержимому файлов."""
    by_size = defaultdict(list)
    for f in files:
        by_size[f.stat().st_size].append(f)
    groups = defaultdict(list)
    for same in by_size.values():
        if len(same) > 1:
            for f in same:
                groups[_hash(f)].append(f)
    return [g for g in groups.values() if len(g) > 1]


def _files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []


class Report:
    def __init__(self):
        self.lines: list[str] = []
        self.problems: list[str] = []

    def add(self, text: str = ""):
        self.lines.append(text)

    def problem(self, title: str, names) -> None:
        names = [str(n) for n in names]
        if names:
            self.add(f"  ⚠ {title}: {len(names)}")
            self.problems += [f"[{title}]"] + [f"  {n}" for n in names]


def _split_line(parts: dict) -> str:
    def cnt(v):
        return f"{len(v)} ({len({photo_key(Path(str(f)).name) for f in v})} фото)"
    names = {"train": "обучение", "val": "проверка", "test": "тест"}
    return "  деление при обучении: " + ", ".join(f"{names[k]} {cnt(v)}" for k, v in parts.items() if k in names)


def _new_images(root: Path) -> list[Path]:
    """Картинки в new/ — по папкам месяцев new/<ГГГГ-ММ>/… (2026-10-09, 1а)."""
    new = root / "new"
    return sorted(p for p in new.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS) \
        if new.is_dir() else []


def _new_line(root: Path, what: str) -> str:
    files = _new_images(root)
    months = Counter(p.relative_to(root / "new").parts[0] if len(p.relative_to(root / "new").parts) > 1
                     else "без месяца" for p in files)
    return f"  new/ ({what}): {len(files)}" + (
        " — " + ", ".join(f"{m}: {n}" for m, n in sorted(months.items())) if files else "")


def check_yolo(root: Path, rep: Report) -> None:
    images_dir, labels_dir = root / "images", root / "labels"
    images = images_in(images_dir)
    labels = sorted(p for p in _files(labels_dir) if p.suffix == ".txt")
    stems = {p.stem for p in images}
    names = []
    if (root / "classes.txt").is_file():
        names = [x.strip() for x in (root / "classes.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    labelled = [p for p in images if (labels_dir / (p.stem + ".txt")).is_file()]
    rep.add(f"  фото: {len(images)}, с разметкой: {len(labelled)}")
    rep.add("  классы (classes.txt): " + (", ".join(f"{i} — {n}" for i, n in enumerate(names)) if names
                                          else "нет файла classes.txt — обучение не начнётся"))
    objects, empty, bad = Counter(), [], []
    for lab in labels:
        if lab.stem not in stems:
            continue
        rows = [r.split() for r in lab.read_text(encoding="utf-8", errors="replace").splitlines() if r.strip()]
        if not rows:
            empty.append(lab.name)
        for r in rows:
            try:
                cls, box = int(r[0]), [float(v) for v in r[1:]]
                ok = len(r) == 5 and all(0.0 <= v <= 1.0 for v in box) and (not names or 0 <= cls < len(names))
            except ValueError:
                ok, cls = False, None
            if ok:
                objects[cls] += 1
            else:
                bad.append(lab.name)
    if objects:
        rep.add("  объектов по классам: " + ", ".join(
            f"{c}{' — ' + names[c] if c < len(names) else ''}: {n}" for c, n in sorted(objects.items())))
    if empty:
        rep.add(f"  пустая разметка (на фото ничего нет — так тоже учат): {len(empty)}")
    rep.problem("фото без разметки (в обучение не пойдут)", [p.name for p in images if p not in labelled])
    rep.problem("разметка без фото", [p.name for p in labels if p.stem not in stems])
    rep.problem("ошибки в разметке (не «класс cx cy w h», числа вне 0…1, класс вне classes.txt)",
                sorted(set(bad)))
    rep.problem("одинаковые фото", [" = ".join(p.name for p in g) for g in _duplicates(images)])
    if labelled:
        parts = split_by_photo(labelled, lambda p: p.name, YOLO_SEED, 1 - YOLO_VAL, YOLO_VAL)
        rep.add(_split_line({"train": parts["train"], "val": parts["val"]}))
    if (root / "new").is_dir():
        rep.add(_new_line(root, "на разметку"))


def check_cnn(root: Path, rep: Report) -> None:
    samples, other = [], []
    per_digit = {}
    for d in DIGITS:
        files = _files(root / d)
        imgs = [f for f in files if f.suffix.lower() in IMAGE_EXTS]
        other += [f"{d}/{f.name}" for f in files if f not in imgs]
        per_digit[d] = len(imgs)
        samples += imgs
    extra = [p.name for p in root.iterdir() if p.is_dir() and p.name not in DIGITS + ["new"]] if root.is_dir() else []
    rep.add(f"  кропов: {len(samples)} — " + ", ".join(f"{d}: {n}" for d, n in per_digit.items()))
    no_key = sum(1 for p in samples if "__" not in p.stem)
    rep.add(f"  фото (по имени до «__»): {len({photo_key(p.name) for p in samples})}"
            + (f"; кропов без «__» в имени: {no_key} — каждый считается отдельным фото" if no_key else ""))
    dups = _duplicates(samples)
    rep.problem("одинаковые кропы в разных цифрах (одна из меток неверна)",
                [" = ".join(f"{p.parent.name}/{p.name}" for p in g) for g in dups
                 if len({p.parent.name for p in g}) > 1])
    rep.problem("одинаковые кропы в одной цифре", [" = ".join(f"{p.parent.name}/{p.name}" for p in g) for g in dups
                                                   if len({p.parent.name for p in g}) == 1])
    rep.problem("не картинки в папках цифр", other)
    rep.problem("лишние папки (обучение их не читает)", extra)
    if samples:
        from models.cnn.config_cnn import SEED, TRAIN_RATIO, VAL_RATIO
        rep.add(_split_line(split_by_photo(samples, lambda p: p.name, SEED, TRAIN_RATIO, VAL_RATIO)))
    rep.add(_new_line(root, "исправил оператор"))
    by_digit = Counter(p.parent.name for p in _new_images(root))
    if by_digit:
        rep.add("    по цифрам: " + ", ".join(f"{d}: {by_digit[d]}" for d in DIGITS))


def check_crnn(root: Path, rep: Report) -> None:
    from models.crnn.dataset_crnn import _parse_label_txt
    images_dir, labels_dir = root / "images", root / "labels"
    images = images_in(images_dir)
    labels = sorted(p for p in _files(labels_dir) if p.suffix == ".txt")
    stems = {p.stem for p in images}
    good, lengths, bad = [], Counter(), []
    for p in images:
        lab = labels_dir / (p.stem + ".txt")
        if not lab.is_file():
            continue
        text = _parse_label_txt(lab)
        if text is None:
            bad.append(lab.name)
        else:
            good.append(p)
            lengths[len(text)] += 1
    rep.add(f"  фото: {len(images)}, с годным номером: {len(good)}")
    if lengths:
        rep.add("  длина номера: " + ", ".join(f"{n} цифр — {c}" for n, c in sorted(lengths.items())))
    rep.problem("фото без разметки", [p.name for p in images if not (labels_dir / (p.stem + ".txt")).is_file()])
    rep.problem("разметка без фото", [p.name for p in labels if p.stem not in stems])
    rep.problem("номер не годится (не цифры или длина не 4–10) — в обучение не пойдёт", bad)
    rep.problem("одинаковые фото", [" = ".join(p.name for p in g) for g in _duplicates(images)])
    if good:
        rep.add(_split_line(split_by_photo(good, lambda p: p.name, CRNN_SEED, CRNN_TRAIN, CRNN_VAL)))
    rep.add(_new_line(root, "исправил оператор"))


CHECKS = {"meter_yolo": check_yolo, "digits_yolo": check_yolo, "digits_cnn": check_cnn,
          "serials_crnn": check_crnn}
TITLES = {"meter_yolo": "детектор счётчика (YOLO)", "digits_yolo": "детектор цифр (YOLO)",
          "digits_cnn": "цифры (CNN)", "serials_crnn": "серийники (CRNN)", "etalon": "эталон"}


def run(root: Path, list_problems: bool = False) -> str:
    root = Path(root)
    created = ensure_layout(root)
    rep = Report()
    rep.add(f"Датасеты: {root.resolve()}")
    if created:
        rep.add(f"Создано: {len(created)} (папки раскладки)")
    for name, title in TITLES.items():
        rep.add("")
        rep.add(f"{name} — {title}")
        if name in CHECKS:
            CHECKS[name](root / name, rep)
        else:
            rep.add(f"  файлов: {sum(1 for p in (root / name).rglob('*') if p.is_file())}")
    rep.add("")
    if rep.problems and list_problems:
        out = root / "datasets_problems.txt"
        out.write_text("\n".join(rep.problems) + "\n", encoding="utf-8")
        rep.add(f"Имена проблемных файлов: {out}")
    elif rep.problems:
        rep.add("Имена проблемных файлов — с --list (файл datasets_problems.txt в папке датасетов).")
    else:
        rep.add("Проблем не найдено.")
    return "\n".join(rep.lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python gmr.py datasets",
                                description="Папки датасетов (создать недостающие) и их проверка.")
    p.add_argument("--root", default=str(DATASETS_DIR), help=f"папка датасетов (по умолчанию {DATASETS_DIR})")
    p.add_argument("--list", action="store_true", help="записать имена проблемных файлов в datasets_problems.txt")
    args = p.parse_args(argv)
    print(run(Path(args.root), args.list))
    return 0


if __name__ == "__main__":
    sys.exit(main())
