import argparse
import os
import random
import shutil

# ===== НАСТРОЙКИ =====
# Пути — относительно папки проекта (запускать из неё). Раньше были зашиты
# абсолютные C:\AD\gas-meter-reader\... (MIGRATION_TZ.md, 3bis, находка 6).
_p = argparse.ArgumentParser(description="Train/val split датасета детектора цифр (YOLO)")
_p.add_argument("--images_dir", default="database/meter_ocr_data/digits_detect/images")
_p.add_argument("--labels_dir", default="database/meter_ocr_data/digits_detect/labels")
_p.add_argument("--output_dir", default="database/meter_ocr_data/digits_detect/dataset")
_p.add_argument("--train_ratio", type=float, default=0.8)
_args = _p.parse_args()
images_dir = _args.images_dir
labels_dir = _args.labels_dir
output_dir = _args.output_dir
train_ratio = _args.train_ratio
# Фиксированный seed: split воспроизводим при повторном запуске (как SEED=67
# в dataset_cnn.py). Текущий split модели digits_detect_v4 был сделан без
# seed и этим НЕ восстанавливается (MIGRATION_TZ.md, 3bis, находка 5).
SEED = 67

# Проверяем что папки существуют
if not os.path.exists(images_dir):
    print(f"Ошибка: папка {images_dir} не найдена")
    exit(1)
if not os.path.exists(labels_dir):
    print(f"Ошибка: папка {labels_dir} не найдена")
    exit(1)

# Получаем фото у которых есть разметка
photos = []
for f in os.listdir(images_dir):
    if f.endswith(('.jpg', '.png', '.jpeg')):
        name = os.path.splitext(f)[0]
        label_file = os.path.join(labels_dir, f"{name}.txt")
        if os.path.exists(label_file):
            photos.append(f)

print(f"Найдено пар фото+разметка: {len(photos)}")

# Перемешиваем и делим. sorted — порядок os.listdir зависит от файловой
# системы, без сортировки один и тот же seed давал бы разный split.
photos.sort()
random.Random(SEED).shuffle(photos)
split_idx = int(len(photos) * train_ratio)
train_photos = photos[:split_idx]
val_photos = photos[split_idx:]

# Создаем папки
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

# Копируем файлы
for photo in train_photos:
    name = os.path.splitext(photo)[0]
    shutil.copy(os.path.join(images_dir, photo), os.path.join(output_dir, 'train', 'images', photo))
    shutil.copy(os.path.join(labels_dir, f"{name}.txt"), os.path.join(output_dir, 'train', 'labels', f"{name}.txt"))

for photo in val_photos:
    name = os.path.splitext(photo)[0]
    shutil.copy(os.path.join(images_dir, photo), os.path.join(output_dir, 'val', 'images', photo))
    shutil.copy(os.path.join(labels_dir, f"{name}.txt"), os.path.join(output_dir, 'val', 'labels', f"{name}.txt"))

# Создаем data.yaml
yaml_content = f"""path: {os.path.abspath(output_dir)}
train: train/images
val: val/images
nc: 1
names: ['digit']
"""

with open(os.path.join(output_dir, "data.yaml"), "w") as f:
    f.write(yaml_content)

print(f"Готово! Train: {len(train_photos)}, Val: {len(val_photos)}")
print(f"Путь: {os.path.join(output_dir, 'data.yaml')}")