import cv2
import os
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Настройки
MODEL_PATH = "meter_ocr/runs/yolo/digits_detect_v4/weights/best.pt"
INPUT_DIR = "database/meter_ocr_data/gas_meter_gold/images"
OUTPUT_DIR = "database/meter_ocr_data/digit_ocr/images"
CONF_THRESH = 0.6
CROP_W = 0  # 0.3
CROP_H = 0  # 0.7
ANGLE_RANGE = 10.0
ANGLE_STEP = 0.5
MAX_PER_CLASS = 5  # <--- Максимальное количество объектов для сохранения на один класс

# Загружаем модель
model = YOLO(MODEL_PATH)
class_names = model.names

# Создаём папки под классы
for name in class_names.values():
    Path(OUTPUT_DIR, name).mkdir(parents=True, exist_ok=True)

def find_best_angle(binary_img):
    """Крутим картинку туда-сюда и смотрим, где лучше"""
    h, w = binary_img.shape
    center = (w // 2, h // 2)
    angles = np.arange(-ANGLE_RANGE, ANGLE_RANGE + ANGLE_STEP, ANGLE_STEP)
    best_angle = 0
    best_score = -1

    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary_img, M, (w, h), flags=cv2.INTER_NEAREST)
        proj = np.sum(rotated, axis=1).astype(np.float32)
        score = np.var(proj)
        
        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle

def straighten_crop(crop_img):
    """Выпрямляем кроп"""
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    angle = find_best_angle(binary)

    h, w = crop_img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a = abs(np.cos(np.radians(angle)))
    sin_a = abs(np.sin(np.radians(angle)))

    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    straightened = cv2.warpAffine(crop_img, M, (new_w, new_h), 
                                  flags=cv2.INTER_CUBIC, 
                                  borderMode=cv2.BORDER_REPLICATE)

    # Обрезаем лишнее
    safe_w = int((w * cos_a - h * sin_a) * CROP_W + w * (1 - CROP_W))
    safe_h = int((h * cos_a - w * sin_a) * CROP_H + h * (1 - CROP_H))

    if safe_w >= 10 and safe_h >= 10:
        cx, cy = new_w // 2, new_h // 2
        x1 = cx - safe_w // 2
        y1 = cy - safe_h // 2
        straightened = straightened[y1:y1 + safe_h, x1:x1 + safe_w]

    return straightened, angle

def save_crop(img, original_name, class_name, index):
    """Сохраняем вырезанный кусок"""
    base_name = Path(original_name).stem
    ext = Path(original_name).suffix
    new_name = f"{base_name}__{class_name}_{index}{ext}"
    full_path = Path(OUTPUT_DIR, class_name, new_name)
    cv2.imwrite(str(full_path), img)
    return str(full_path)

def process_one_image(filename):
    """Обработка одной картинки"""
    img_path = Path(INPUT_DIR, filename)
    print(f"\n--- {filename} ---")
    
    # 1. СНАЧАЛА проверяем, читается ли файл как изображение
    img = cv2.imread(str(img_path))
    if img is None:
        print("  ⚠️ Ошибка чтения изображения (файл поврежден или пуст). Пропускаем.")
        return {}

    # 2. Детектим на исходном изображении
    results = model(str(img_path), conf=CONF_THRESH)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print("  ничего не нашлось :( ")
        return {}

    # Собираем все детекции по классам
    all_detections = {}

    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        conf_val = conf.item()
        cls_val = int(cls.item())
        class_name = class_names.get(cls_val, str(cls_val))
        
        if class_name not in all_detections:
            all_detections[class_name] = []
            
        all_detections[class_name].append({
            'box': (x1, y1, x2, y2),
            'conf': conf_val,
            'cls': cls_val
        })
    
    # Оставляем только MAX_PER_CLASS самых уверенных для каждого класса
    best_detections = {}
    for class_name, detections in all_detections.items():
        # Шаг 1: Сортируем по убыванию уверенности, чтобы отфильтровать "лучшие"
        detections.sort(key=lambda x: x['conf'], reverse=True)
        
        # Шаг 2: Оставляем максимум MAX_PER_CLASS
        best = detections[:MAX_PER_CLASS]
        
        # Шаг 3: Сортируем оставшиеся слева направо по координате x1 (левая граница бокса)
        best.sort(key=lambda x: x['box'][0])
        
        best_detections[class_name] = best

    stats = {}

    # 3. Обрабатываем и сохраняем лучшие кропы
    for class_name, detections in best_detections.items():
        for idx, data in enumerate(detections):
            x1, y1, x2, y2 = data['box']
            conf_val = data['conf']
            
            # Вырезаем кроп
            crop = img[y1:y2, x1:x2]
            
            # Выпрямляем кроп
            straightened_crop, angle = straighten_crop(crop)
            
            # Сохраняем (idx + 1 теперь соответствует порядку слева направо)
            out_path = save_crop(straightened_crop, filename, class_name, idx + 1)
            
            stats[class_name] = stats.get(class_name, 0) + 1
            print(f"  {class_name} (#{idx + 1}) | уверенность: {conf_val:.3f} | угол: {angle:.2f}° -> {out_path}")

    return stats

def main():
    photos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Нашёл {len(photos)} фоток\n")
    print("=" * 50)

    total = {}

    for photo in photos:
        stats = process_one_image(photo)
        for cls_name, count in stats.items():
            total[cls_name] = total.get(cls_name, 0) + count

    print("\n" + "=" * 50)
    print(f"Обработано файлов: {len(photos)}")
    print("Нарезано кропов:")
    for cls_name in sorted(total.keys()):
        print(f"  - {cls_name}: {total[cls_name]} шт")
    print(f"\nВсего: {sum(total.values())} кропов")
    print("=" * 50)

if __name__ == "__main__":
    main()