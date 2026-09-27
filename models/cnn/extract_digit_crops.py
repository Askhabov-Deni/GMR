"""
Вырезает отдельные цифры из размеченных изображений.
Из каждого бокса YOLO-разметки вырезает кроп и сохраняет в папку по классу.

Структура на выходе:
  digit_crops/
    0/  *.jpg
    1/  *.jpg
    ...
    9/  *.jpg
"""

from pathlib import Path
import cv2

# ── Настройки ─────────────────────────────────────────────────────────────────
IMAGES_DIR = "database/model_ocr/gas_meter/images"
LABELS_DIR = "database/model_ocr/gas_meter/labels"
OUTPUT_DIR  = "digit_crops"
CROP_SIZE   = (32, 64)   # (ширина, высота)
VAL_SPLIT   = 0.2
SEED        = 67
# ─────────────────────────────────────────────────────────────────────────────

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    import random
    random.seed(SEED)

    images_dir = Path(IMAGES_DIR)
    labels_dir = Path(LABELS_DIR)
    output_dir = Path(OUTPUT_DIR)

    # Создаём папки train/val для каждого класса
    for split in ("train", "val"):
        for i in range(10):
            (output_dir / split / str(i)).mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in EXTENSIONS
        and (labels_dir / (f.stem + ".txt")).exists()
    ])

    print(f"Изображений с разметкой: {len(image_files)}")

    # Сначала собираем все кропы по классам в память
    class_crops = {i: [] for i in range(10)}

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        label_path = labels_dir / (img_path.stem + ".txt")
        with open(label_path) as f:
            lines = f.read().strip().splitlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx  = float(parts[1])
            cy  = float(parts[2])
            bw  = float(parts[3])
            bh  = float(parts[4])

            x1 = max(0, int((cx - bw / 2) * w))
            x2 = min(w, int((cx + bw / 2) * w))
            y1 = max(0, int((cy - bh / 2) * h))
            y2 = min(h, int((cy + bh / 2) * h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, CROP_SIZE, interpolation=cv2.INTER_LANCZOS4)
            name = f"{img_path.stem}__x{x1}.jpg"
            class_crops[cls].append((name, crop))

    # Стратифицированный сплит — каждый класс 80/20 независимо
    total_train = total_val = 0
    print("\nПо классам (train / val):")
    for cls, items in class_crops.items():
        random.shuffle(items)
        n_val   = max(1, int(len(items) * VAL_SPLIT))
        n_train = len(items) - n_val
        train_items = items[n_val:]
        val_items   = items[:n_val]

        for name, crop in train_items:
            cv2.imwrite(str(output_dir / "train" / str(cls) / name), crop)
        for name, crop in val_items:
            cv2.imwrite(str(output_dir / "val" / str(cls) / name), crop)

        print(f"  {cls}: {n_train} train / {n_val} val")
        total_train += n_train
        total_val   += n_val

    print(f"\nИтого: {total_train} train / {total_val} val")
    print(f"Сохранено в: {output_dir.resolve()}")


if __name__ == "__main__":
    main()