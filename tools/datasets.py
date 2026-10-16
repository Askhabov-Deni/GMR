"""
tools/datasets.py — папки датасетов и их проверка (этап 5).

  python gmr.py datasets [--root database/datasets] [--list]
  python gmr.py datasets --collect <папка месяца> [<папка месяца> …]
  python gmr.py datasets --review serials|accounts|digits [--conf 0.9] [--weights <best>]
  python gmr.py datasets --review-done serials|accounts|digits
  python gmr.py datasets --build <папка с фото> --table <таблица> [--only serials accounts digits] [--out <папка>]

Создаёт недостающие папки раскладки (models/datasets.py) — ничего не
переносит и не удаляет — и показывает по каждому датасету: сколько файлов,
что не так (фото без разметки, разметка без фото, ошибки в разметке,
одинаковые файлы) и как его поделит обучение. В выводе только числа: его
можно прислать, персональных данных в нём нет. --list — имена проблемных
файлов в <root>/datasets_problems.txt (остаётся у вас на диске).

--collect — забрать разметку месяца (исправления оператора, которые окно
кладёт в <месяц>/разметка) в <датасет>/new/<ГГГГ-ММ>/. Копирует, папку месяца
не меняет; повторный запуск ничего не дублирует.

Имена кропов — по стандарту датасетов (docs/DATASETS.md):
<фото>__serial1__<номер>.jpg, <счёт>__account1.jpg, digits_cnn/<цифра>/<фото>__digit3.jpg.
--review — кропы, где модель не согласна с ответом или не уверена, — в
<датасет>/проверить/ на просмотр; --review-done — обратно (tools/dataset_review.py).
--build — кропы для датасетов из папки фото и таблицы (tools/dataset_build.py).
"""
import argparse
import hashlib
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from models.datasets import (
    DATASETS_DIR, DIGITS, IMAGE_EXTS, LAYOUT, ensure_layout, images_in, parse_crop_name, photo_key,
    split_by_photo,
)
from src.gmr.domain.datasets import DIGIT, SERIAL, free_crop_name

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


def _review_line(root: Path, rep: Report) -> None:
    """Кропы на проверке (gmr.py datasets --review): обучение их не читает."""
    files = [p for p in (root / "проверить").rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS] \
        if (root / "проверить").is_dir() else []
    if files:
        rep.add(f"  ⚠ на проверке (проверить\\, обучение не читает): {len(files)} — "
                f"вернуть: gmr.py datasets --review-done")


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
    extra = [p.name for p in root.iterdir() if p.is_dir() and p.name not in DIGITS + ["new", "проверить"]] if root.is_dir() else []
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
    rep.problem("имя не по стандарту (<фото>__digit3.jpg)",
                [f"{p.parent.name}/{p.name}" for p in samples
                 if (c := parse_crop_name(p.name)) is None or c.kind != DIGIT])
    rep.problem("не картинки в папках цифр", other)
    rep.problem("лишние папки (обучение их не читает)", extra)
    if samples:
        from models.cnn.config_cnn import SEED, TRAIN_RATIO, VAL_RATIO
        rep.add(_split_line(split_by_photo(samples, lambda p: p.name, SEED, TRAIN_RATIO, VAL_RATIO)))
    rep.add(_new_line(root, "исправил оператор"))
    _review_line(root, rep)
    by_digit = Counter(p.parent.name for p in _new_images(root))
    if by_digit:
        rep.add("    по цифрам: " + ", ".join(f"{d}: {by_digit[d]}" for d in DIGITS))


def check_crnn(root: Path, rep: Report) -> None:
    from models.crnn.dataset_crnn import label_of
    images = images_in(root / "images")
    good, lengths, odd, bad = [], Counter(), [], []
    for p in images:
        crop, label = parse_crop_name(p.name), label_of(p.name)
        if label is not None:
            good.append(p)
            lengths[len(label)] += 1
        elif crop is None or crop.kind != SERIAL:
            odd.append(p.name)
        else:
            bad.append(p.name)
    rep.add(f"  кропов: {len(images)}, с годным номером в имени: {len(good)}")
    if lengths:
        rep.add("  длина номера: " + ", ".join(f"{n} цифр — {c}" for n, c in sorted(lengths.items())))
    rep.problem("имя не по стандарту (<фото>__serial1__<номер>.jpg) — в обучение не пойдёт", odd)
    rep.problem("номер в имени не годится (не 4–10 цифр) — в обучение не пойдёт", bad)
    rep.problem("одинаковые кропы", [" = ".join(p.name for p in g) for g in _duplicates(images)])
    if good:
        rep.add(_split_line(split_by_photo(good, lambda p: p.name, CRNN_SEED, CRNN_TRAIN, CRNN_VAL)))
    if (root / "labels").is_dir():
        rep.add("  labels\\ не нужна: номер — в имени кропа (docs\\DATASETS.md)")
    rep.add(_new_line(root, "исправил оператор"))
    _review_line(root, rep)


def check_account(root: Path, rep: Report) -> None:
    from models.account import config_account as C
    from models.account.dataset_account import load_dataset, split
    images = images_in(root / "images")
    items = load_dataset(root / "images", verbose=False)
    named = {Path(it["file"]).name for it in items}
    rep.add(f"  кропов: {len(images)}, с лицевым счётом в имени: {len(items)}, разных счетов: "
            f"{len({it['account'] for it in items})}")
    rep.problem("имя не по стандарту (<счёт>__account1.jpg) — в обучение не пойдёт",
                [p.name for p in images if p.name not in named])
    rep.problem("одинаковые кропы", [" = ".join(p.name for p in g) for g in _duplicates(images)])
    if items:
        tr, va, te = split(items, C.TRAIN_RATIO, C.VAL_RATIO, C.SEED)
        rep.add(f"  деление по счёту: обучение {len(tr)}, проверка {len(va)}, тест {len(te)}")
    if (root / "labels").is_dir():
        rep.add("  labels\\ не нужна: что написано, модель узнаёт сама по счёту из имени (папку можно удалить)")
    rep.add(_new_line(root, "исправил оператор"))
    _review_line(root, rep)


CHECKS = {"meter_yolo": check_yolo, "digits_yolo": check_yolo, "digits_cnn": check_cnn,
          "serials_crnn": check_crnn, "accounts_crnn": check_account}
TITLES = {"meter_yolo": "детектор счётчика (YOLO)", "digits_yolo": "детектор цифр (YOLO)",
          "digits_cnn": "цифры (CNN)", "serials_crnn": "серийники (CRNN)",
          "accounts_crnn": "лицевой счёт маркером (CRNN)", "etalon": "эталон"}


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


def collect(month_dir: Path, root: Path) -> str:
    """Разметка месяца (<месяц>/разметка/<датасет>/…) → <root>/<датасет>/new/<ГГГГ-ММ>/…"""
    from src.gmr.application.month import is_month, month_label
    from src.gmr.storage.month import MonthDB, MonthFolder
    folder = MonthFolder(Path(month_dir))
    if not is_month(folder):
        raise ValueError(f"{month_dir} — не папка месяца")
    with MonthDB(folder.db) as db:
        label = month_label(db.meta("created_at"))
    copied, already, taken, hashes = Counter(), 0, {}, {}
    src = folder.markup
    for p in sorted(src.rglob("*")) if src.is_dir() else []:
        rel = p.relative_to(src)
        if not p.is_file() or rel.parts[0] not in LAYOUT or len(rel.parts) < 2:
            continue
        dst = Path(root) / rel.parts[0] / "new" / label / Path(*rel.parts[1:])
        crop = parse_crop_name(p.name)
        if crop is None:                                 # не кроп (фото для YOLO) — как раньше
            if dst.exists():
                already += p.suffix.lower() in IMAGE_EXTS
                continue
        else:
            # кроп: имя не должно совпасть ни с одним в датасете (тот же счёт
            # в другом месяце → «-2»); уже забранный — узнаётся по содержимому
            if dst.parent not in hashes:
                hashes[dst.parent] = {_hash(f) for f in _files(dst.parent)}
            if _hash(p) in hashes[dst.parent]:
                already += 1
                continue
            ds = rel.parts[0]
            if ds not in taken:
                taken[ds] = {f.name.lower() for f in (Path(root) / ds).rglob("*") if f.is_file()}
            name = p.name if p.name.lower() not in taken[ds] else free_crop_name(
                taken[ds], crop.photo, crop.kind, crop.n, crop.answer, p.suffix.lower())
            dst = dst.with_name(name)
            taken[ds].add(name.lower())
            hashes[dst.parent].add(_hash(p))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        copied[rel.parts[0]] += p.suffix.lower() in IMAGE_EXTS
    return (f"Разметка месяца {folder.root.name} ({label}): "
            + (", ".join(f"{k} — {n}" for k, n in sorted(copied.items())) or "новых нет")
            + (f"; уже были: {already}" if already else ""))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python gmr.py datasets",
                                description="Папки датасетов (создать недостающие) и их проверка.")
    p.add_argument("--root", default=str(DATASETS_DIR), help=f"папка датасетов (по умолчанию {DATASETS_DIR})")
    p.add_argument("--list", action="store_true", help="записать имена проблемных файлов в datasets_problems.txt")
    p.add_argument("--collect", nargs="+", metavar="МЕСЯЦ",
                   help="забрать разметку из папок месяцев (<месяц>/разметка) в new/<ГГГГ-ММ>/")
    p.add_argument("--review", choices=["serials", "accounts", "digits"],
                   help="кропы, где модель не согласна с ответом или не уверена, — в <датасет>/проверить/")
    p.add_argument("--review-done", choices=["serials", "accounts", "digits"],
                   help="вернуть проверенные кропы из <датасет>/проверить/ в датасет")
    p.add_argument("--conf", type=float, default=0.9, help="с --review: ниже этой уверенности — на проверку")
    p.add_argument("--weights", help="с --review: веса модели (по умолчанию — из настроек программы)")
    p.add_argument("--build", metavar="ФОТО", help="собрать кропы из папки фото (имя фото — лицевой счёт)")
    p.add_argument("--table", metavar="ТАБЛИЦА", help="с --build: таблица (номер счётчика, текущие показания)")
    p.add_argument("--only", nargs="+", choices=["serials", "accounts", "digits"],
                   default=["serials", "accounts", "digits"], help="с --build: какие кропы (по умолчанию все)")
    p.add_argument("--out", metavar="ПАПКА", help="с --build: куда (по умолчанию <папка фото>_датасет)")
    args = p.parse_args(argv)
    root = Path(args.root)
    if args.build:
        from tools import dataset_build
        if not args.table:
            p.error("--build: нужна таблица (--table)")
        try:
            print(dataset_build.run(Path(args.build), Path(args.table), Path(args.out) if args.out else None,
                                    args.only, dataset_root=root))
        except (ValueError, OSError) as e:
            print(f"ОШИБКА: {e}")
            return 1
        return 0
    if args.review or args.review_done:
        from tools import dataset_review
        try:
            print(dataset_review.done(root, args.review_done) if args.review_done else
                  dataset_review.review(root, args.review, args.conf, args.weights))
        except (ValueError, OSError) as e:
            print(f"ОШИБКА: {e}")
            return 1
        return 0
    for month_dir in args.collect or []:
        ensure_layout(root)
        try:
            print(collect(Path(month_dir), root))
        except ValueError as e:
            print(f"ОШИБКА: {e}")
            return 1
    if args.collect:
        print()
    print(run(root, args.list))
    return 0


if __name__ == "__main__":
    sys.exit(main())
