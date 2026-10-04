"""
Предразметка: текущая модель-детектор расставляет рамки на фото без
разметки, человек потом только поправляет их в программе разметки (этап 5,
2026-10-04: разметка трёх классов детектора счётчика потерялась — модель
поможет разметить заново).

  python models/yolo_all_detect/prelabel.py <папка с фото> [--weights <best.pt>] [--conf 0.25]

- Пишет файл разметки YOLO «класс cx cy w h» только для фото, у которых его
  ещё нет: готовую разметку никогда не трогает.
- Куда: если папка фото называется images — в соседнюю labels/, иначе — в
  ту же папку (рядом с фото). Можно указать --labels.
- Если модель ничего не нашла — файл не пишется: пустой файл значил бы
  «на фото ничего нет», и такое фото пошло бы в обучение без разметки.
- classes.txt (названия классов модели по строкам) — в папку разметки и в
  папку датасета, если его там нет.
- По умолчанию веса — детектор счётчика из PipelineConfig, порог — 0.25:
  ниже, чем в reader.py (0.7), чтобы человек видел и неуверенные рамки.
"""
import argparse
import sys
from pathlib import Path

try:
    from ..datasets import images_in
except ImportError:  # запуск как отдельный скрипт: нужна папка проекта в sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from models.datasets import images_in


def labels_dir_for(images: Path) -> Path:
    images = Path(images)
    return images.parent / "labels" if images.name == "images" else images


def yolo_lines(classes, boxes_xywhn) -> list[str]:
    """Строки разметки YOLO: класс и рамка в долях (6 знаков)."""
    return [f"{int(c)} " + " ".join(f"{float(v):.6f}" for v in box)
            for c, box in zip(classes, boxes_xywhn)]


def write_classes(folder: Path, names: list[str]) -> bool:
    path = Path(folder) / "classes.txt"
    if path.exists():
        return False
    path.write_text("".join(f"{n}\n" for n in names), encoding="utf-8")
    return True


def prelabel(images: Path, labels: Path, predict, names: list[str]) -> dict:
    """predict(path) → (классы, рамки xywhn). Возвращает счётчики."""
    labels.mkdir(parents=True, exist_ok=True)
    stats = {"written": 0, "had_labels": 0, "nothing_found": 0}
    for img in images_in(images):
        out = labels / (img.stem + ".txt")
        if out.exists():
            stats["had_labels"] += 1
            continue
        classes, boxes = predict(img)
        lines = yolo_lines(classes, boxes)
        if not lines:
            stats["nothing_found"] += 1
            continue
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        stats["written"] += 1
    write_classes(labels, names)
    if images.name == "images":
        write_classes(images.parent, names)
    return stats


def main(argv=None) -> int:
    from src.gmr.domain import PipelineConfig
    p = argparse.ArgumentParser(description="Предразметка YOLO текущей моделью: рамки на фото без разметки")
    p.add_argument("images", help="папка с фото (например database/datasets/meter_yolo/images)")
    p.add_argument("--labels", help="куда писать разметку (по умолчанию соседняя labels/ или та же папка)")
    p.add_argument("--weights", default=PipelineConfig().meter_detect_model,
                   help="веса детектора (по умолчанию детектор счётчика из PipelineConfig)")
    p.add_argument("--conf", type=float, default=0.25, help="порог уверенности (по умолчанию 0.25)")
    args = p.parse_args(argv)

    images = Path(args.images)
    if not images.is_dir():
        print(f"ОШИБКА: нет папки {images}")
        return 1
    labels = Path(args.labels) if args.labels else labels_dir_for(images)

    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = [model.names[i] for i in sorted(model.names)]

    def predict(path: Path):
        r = model.predict(str(path), conf=args.conf, verbose=False)[0]
        return r.boxes.cls.tolist(), r.boxes.xywhn.tolist()

    s = prelabel(images, labels, predict, names)
    print(f"Классы модели: {', '.join(f'{i} — {n}' for i, n in enumerate(names))}")
    print(f"Разметка записана: {s['written']}  (в {labels})")
    print(f"Уже была разметка (не тронута): {s['had_labels']}")
    print(f"Модель ничего не нашла (разметьте вручную): {s['nothing_found']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
