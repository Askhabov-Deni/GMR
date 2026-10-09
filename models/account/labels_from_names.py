"""
Разметка надписей маркером по именам файлов — без ручного ввода.

Кропы режет YOLO из фото, которые программа уже назвала лицевым счётом
(plus/, minus/: «1300000065.jpg» → кроп «1300000065__account_1.jpg»). На
счётчике маркером написаны последние цифры этого счёта — значит, метка
известна заранее: 00065.

  python models/account/labels_from_names.py database/datasets/accounts_crnn/images
  python models/account/labels_from_names.py <папка> --digits 5 --dry-run

- Пишет labels/<имя>.txt только для кропов, у которых разметки ещё нет;
  готовую разметку не трогает.
- Счёт берётся из начала имени (цифры до «__», «_» или «.»); если там нет
  номера длиной хотя бы --digits — кроп пропускается (разметьте сам).
- Это догадка: контролёр мог написать другое. После обучения смотрите
  eval/suspect_labels.txt (evaluate_account.py) — там кропы, где модель
  уверенно читает не то, что в метке.
"""
import argparse
import re
import sys
from pathlib import Path

try:
    from ..datasets import images_in, photo_key
except ImportError:  # запуск как отдельный скрипт: нужна папка проекта в sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from models.datasets import images_in, photo_key

_LEADING_NUMBER = re.compile(r"\d+")


def label_from_name(name: str, digits: int):
    """«1300000065__account_1.jpg», 5 → «00065»; None — в имени нет счёта."""
    m = _LEADING_NUMBER.match(photo_key(name))
    if m is None or len(m.group()) < digits:
        return None
    return m.group()[-digits:]


def labels_dir_for(images: Path) -> Path:
    return images.parent / "labels" if images.name == "images" else images


def write_labels(images: Path, labels: Path, digits: int, dry_run: bool = False) -> dict:
    stats = {"written": 0, "had_label": 0, "no_number": 0}
    if not dry_run:
        labels.mkdir(parents=True, exist_ok=True)
    for img in images_in(images):
        out = labels / (img.stem + ".txt")
        if out.exists():
            stats["had_label"] += 1
            continue
        label = label_from_name(img.name, digits)
        if label is None:
            stats["no_number"] += 1
            continue
        if not dry_run:
            out.write_text(label + "\n", encoding="utf-8")
        stats["written"] += 1
    return stats


def main(argv=None) -> int:
    from src.gmr.domain import PipelineConfig
    p = argparse.ArgumentParser(description="Метки надписей маркером из имён файлов (последние цифры счёта)")
    p.add_argument("images", help="папка кропов (например database/datasets/accounts_crnn/images)")
    p.add_argument("--labels", help="куда писать (по умолчанию соседняя labels/)")
    p.add_argument("--digits", type=int, default=PipelineConfig().account_marker_digits)
    p.add_argument("--dry-run", action="store_true", help="только посчитать, ничего не писать")
    args = p.parse_args(argv)
    images = Path(args.images)
    if not images.is_dir():
        print(f"ОШИБКА: нет папки {images}")
        return 1
    labels = Path(args.labels) if args.labels else labels_dir_for(images)
    s = write_labels(images, labels, args.digits, args.dry_run)
    print(f"{'Было бы записано' if args.dry_run else 'Записано'} меток: {s['written']}  (в {labels})")
    print(f"Уже была разметка (не тронута): {s['had_label']}")
    print(f"В имени нет счёта (разметьте вручную): {s['no_number']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
