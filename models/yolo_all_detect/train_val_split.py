import os
import random
import shutil

# ===== НАСТРОЙКИ =====
# Укажи полные пути
images_dir = "C:\\AD\\gas-meter-reader\\database\\meter_ocr_data\\digits_detect\\images"
labels_dir = "C:\\AD\\gas-meter-reader\\database\\meter_ocr_data\\digits_detect\\labels"
output_dir = "C:\\AD\\gas-meter-reader\\database\\meter_ocr_data\\digits_detect\\dataset"
train_ratio = 0.8

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

# Перемешиваем и делим
random.shuffle(photos)
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