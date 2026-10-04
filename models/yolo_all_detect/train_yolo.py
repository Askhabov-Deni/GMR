"""
Обучение YOLO-детектора на датасете без папок train/val (этап 5, решение
владельца 2026-10-04: «виртуальное деление», как у цифр и серийников).

Датасет — одна папка (models/datasets.py):
  <датасет>/images/*.jpg      фото
  <датасет>/labels/*.txt      разметка YOLO: «класс cx cy w h» (доли 0…1); пустой файл — на фото
                              ничего нет (так тоже учат)
  <датасет>/classes.txt       названия классов по строкам: строка 1 — класс 0, …
Фото без файла разметки в обучение не идут (ещё не размечены).

При запуске датасет делится по фото (val — примерно --val от фото, по
отпечатку имени: новые фото не перетасовывают старые), списки и data.yaml
пишутся в папку результата <project>/<name>/split/ — в папке датасета
ничего не появляется. Там же run_info.json: датасет (отпечаток), коммит
кода, пакеты, деление.

ИСПОЛЬЗОВАНИЕ (из папки проекта):
  Детектор счётчика (показания, серийник, лицевой счёт):
    python models/yolo_all_detect/train_yolo.py --data database/datasets/meter_yolo \\
        --model yolov8s.pt --project meter_detect/runs/detect --name meter_v2 --imgsz 640 --epochs 200
  Детектор цифр на кропе показаний:
    python models/yolo_all_detect/train_yolo.py --data database/datasets/digits_yolo \\
        --model yolov8n.pt --project meter_ocr/runs/yolo --name digits_v5 --imgsz 480 --epochs 50
  Дообучить текущую модель — --model <путь к best.pt>.
"""
import argparse
import sys
from pathlib import Path

try:
    from ..datasets import images_in, require_parts, split_by_photo, write_run_info, write_split_lists
except ImportError:  # запуск как отдельный скрипт: нужна папка проекта в sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from models.datasets import images_in, require_parts, split_by_photo, write_run_info, write_split_lists


class DatasetError(Exception):
    """Датасет нельзя обучать: сообщение — что исправить."""


def read_classes(data: Path) -> list[str]:
    path = Path(data) / "classes.txt"
    if not path.is_file():
        raise DatasetError(f"нет файла {path}: названия классов по строкам (строка 1 — класс 0). "
                           "Для детектора счётчика его пишет предразметка "
                           "(models/yolo_all_detect/prelabel.py) из текущей модели.")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise DatasetError(f"{path} пустой")
    return names


def labelled_images(data: Path) -> tuple[list[Path], int]:
    """(фото с файлом разметки, сколько фото без разметки)."""
    images = images_in(Path(data) / "images")
    labels = Path(data) / "labels"
    have = [p for p in images if (labels / (p.stem + ".txt")).is_file()]
    return have, len(images) - len(have)


def prepare(data: Path, run_dir: Path, val: float, seed: int, args: dict) -> Path:
    """Делит датасет, пишет split/train.txt, split/val.txt, split/data.yaml и
    run_info.json в run_dir. Возвращает путь к data.yaml."""
    data, run_dir = Path(data).resolve(), Path(run_dir).resolve()
    names = read_classes(data)
    images, unlabelled = labelled_images(data)
    if not images:
        raise DatasetError(f"в {data / 'images'} нет фото с разметкой в {data / 'labels'}")
    parts = split_by_photo(images, lambda p: p.name, seed, 1.0 - val, val)
    del parts["test"]                      # у YOLO только обучение и проверка (доли в сумме 1)
    try:
        require_parts(parts, ("train", "val"))
    except ValueError as e:
        raise DatasetError(str(e)) from None
    labels = [data / "labels" / (p.stem + ".txt") for p in images]
    write_split_lists(run_dir, parts, data)
    split_dir = run_dir / "split"
    # ultralytics берёт списки фото из файлов; разметку ищет, заменяя /images/ на /labels/
    for part in parts:
        (split_dir / f"{part}_abs.txt").write_text("".join(f"{p}\n" for p in parts[part]), encoding="utf-8")
    yaml = split_dir / "data.yaml"
    yaml.write_text(
        f"path: {data.as_posix()}\n"
        f"train: {(split_dir / 'train_abs.txt').as_posix()}\n"
        f"val: {(split_dir / 'val_abs.txt').as_posix()}\n"
        f"nc: {len(names)}\n"
        "names:\n" + "".join(f"  {i}: {n!r}\n" for i, n in enumerate(names)),
        encoding="utf-8")
    write_run_info(run_dir, data, images + labels, parts, seed,
                   dict(args, classes=names, unlabelled_images=unlabelled))
    return yaml


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Обучение YOLO на датасете без train/val (деление при запуске)")
    p.add_argument("--data", required=True, help="папка датасета: images/, labels/, classes.txt")
    p.add_argument("--model", required=True, help="с чего начать: yolov8s.pt / yolov8n.pt или best.pt для дообучения")
    p.add_argument("--project", required=True, help="папка прогонов, например meter_detect/runs/detect")
    p.add_argument("--name", required=True, help="имя прогона (новая папка в --project)")
    p.add_argument("--val", type=float, default=0.2, help="доля фото на проверку (по умолчанию 0.2)")
    p.add_argument("--seed", type=int, default=67)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--device", default=None, help="cpu или 0 (видеокарта); по умолчанию — что есть")
    args = p.parse_args(argv)

    run_dir = Path(args.project).resolve() / args.name
    if run_dir.exists():
        print(f"ОШИБКА: папка {run_dir} уже есть — выберите другое --name")
        return 1
    try:
        yaml = prepare(Path(args.data), run_dir, args.val, args.seed, vars(args))
    except DatasetError as e:
        print(f"ОШИБКА: {e}")
        return 1
    print(f"Деление: {run_dir / 'split'}")

    from ultralytics import YOLO
    # ultralytics кладёт labels.cache рядом с labels/ — в папке датасета его быть не должно
    cache = Path(args.data).resolve() / "labels.cache"
    had_cache = cache.exists()
    try:
        YOLO(args.model).train(
            data=str(yaml), project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
            epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience,
            seed=args.seed, device=args.device,
        )
    finally:
        if not had_cache:
            cache.unlink(missing_ok=True)
    print(f"Готово: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
