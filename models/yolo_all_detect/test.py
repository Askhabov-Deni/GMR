import cv2
import os
import shutil
import colorsys
import random
from ultralytics import YOLO

# ── Настройки ─────────────────────────────────────────────────────────────────
IMG_FOLDER_PATH = 'database/meter_ocr_data/gas_meter_gold/images'
SHOW_LABELS = False  # Включить/выключить надписи над объектами
SHOW_IMAGES = False  # <<< Показывать ли изображения в окне (True/False)

MODEL_PATHS = ["runs/detect/meter_ocr/runs/yolo/digits_detect_v1/weights/best.pt"]

# ── НАСТРОЙКИ АВТОСОХРАНЕНИЯ "ПЛОХИХ" ФОТО ────────────────────────────────────
SAVE_MISSED_IMAGES = True   # <<< Вкл/Выкл функционал (True/False)
BAD_IMAGES_FOLDER = 'database/meter_ocr_data/gas_meter_gold/images_bad'

# Ожидаемое количество объектов ПО КАЖДОМУ КЛАССУ
EXPECTED_COUNTS = {
    "digit": 5,
}
# ──────────────────────────────────────────────────────────────────────────────


def generate_distinct_colors(n: int) -> list[tuple[int, int, int]]:
    """Генерирует n визуально различимых цветов через равномерное распределение по HSV."""
    colors = []
    hue_start = random.random()
    for i in range(n):
        hue = (hue_start + i / n) % 1.0
        saturation = 0.85 + random.uniform(-0.1, 0.1)
        value = 0.95 + random.uniform(-0.05, 0.05)
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors


def load_models(paths: list[str]) -> dict:
    """Загружает модели и назначает каждой уникальный цвет."""
    colors = generate_distinct_colors(len(paths))
    models = {}
    for path, color in zip(paths, colors):
        name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        model = YOLO(path)
        class_names = model.names
        models[name] = {
            "model": model,
            "color": color,
            "class_names": class_names,
        }
        print(f"  Загружена модель '{name}'  цвет BGR: {color}")
    return models


def detect_all(model, path: str, conf_thresh: float = 0.8) -> list[dict]:
    """Возвращает ВСЕ найденные объекты на изображении."""
    results = model(path, conf=conf_thresh)
    result = results[0]
    detections = []

    if result.boxes is None or len(result.boxes) == 0:
        return detections

    for box, conf, cls in zip(
        result.boxes.xyxy,
        result.boxes.conf,
        result.boxes.cls,
    ):
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        detections.append({
            "coords": (x1, y1, x2, y2),
            "conf": conf.item(),
            "class_id": int(cls.item()),
        })

    return detections


def draw_detections(img, detections: list[dict], color: tuple,
                    class_names: dict, show_labels: bool = True) -> None:
    for det in detections:
        x1, y1, x2, y2 = det["coords"]
        conf = det["conf"]
        cls_id = det["class_id"]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if show_labels:
            label = f"{class_names.get(cls_id, str(cls_id))} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            cv2.rectangle(
                img,
                (x1, y1 - th - baseline - 4),
                (x1 + tw + 4, y1),
                color, -1,
            )
            cv2.putText(
                img, label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA,
            )


def draw_legend(img, models: dict, padding: int = 8, line_height: int = 28) -> None:
    """Рисует полупрозрачную легенду в левом верхнем углу."""
    n = len(models)
    box_w = 300
    box_h = padding * 2 + n * line_height

    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (5 + box_w, 5 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    for idx, (name, info) in enumerate(models.items()):
        y = 5 + padding + idx * line_height + line_height // 2
        color = info["color"]
        cv2.rectangle(img, (12, y - 8), (32, y + 8), color, -1)
        cv2.putText(
            img, name,
            (40, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            color, 1, cv2.LINE_AA,
        )


print("Загружаем модели...")
models = load_models(MODEL_PATHS)

if SAVE_MISSED_IMAGES:
    os.makedirs(BAD_IMAGES_FOLDER, exist_ok=True)
    print(f"Папка для проблемных фото: {os.path.abspath(BAD_IMAGES_FOLDER)}")
    print(f"Проверяемые классы и ожидаемое количество: {EXPECTED_COUNTS}")

photos = [
    f for f in os.listdir(IMG_FOLDER_PATH)
    if f.lower().endswith(('.jpeg', '.jpg', '.png'))
]
len_photos = len(photos)
print(f"\nНайдено {len_photos} фотографий. Тестируем начиная с 1000-й.\n")
print(f"Показ изображений: {'ВКЛ' if SHOW_IMAGES else 'ВЫКЛ'}")
if not SHOW_IMAGES:
    print("Для прерывания нажмите Ctrl+C\n")

# ── Главный цикл ──────────────────────────────────────────────────────────────
try:
    for i in range(len_photos):
        photo = photos[i]
        path = os.path.join(IMG_FOLDER_PATH, photo)
        img = cv2.imread(path)
        
        # Счетчик объектов по именам классов
        class_counts = {}

        print(f"\nФото {i + 1}/{len_photos}: {photo}")

        for name, info in models.items():
            detections = detect_all(info["model"], path)

            if detections:
                print(f"  [{name}] найдено объектов: {len(detections)}")
                for det in detections:
                    cls_name = info["class_names"].get(det["class_id"], str(det["class_id"]))
                    
                    # Считаем количество по классам
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    
                    print(f"    - {cls_name}: conf={det['conf']:.3f}  coords={det['coords']}")
            else:
                print(f"  [{name}] ничего не найдено")

            # Рисуем детекты только если показываем изображения
            if SHOW_IMAGES:
                draw_detections(img, detections, info["color"], info["class_names"], SHOW_LABELS)

        # ── ЛОГИКА АВТОСОХРАНЕНИЯ ПО КЛАССАМ ──────────────────────────────────────
        if SAVE_MISSED_IMAGES:
            should_save = False
            
            # Проверяем каждый класс из EXPECTED_COUNTS
            for cls_name, expected_count in EXPECTED_COUNTS.items():
                actual_count = class_counts.get(cls_name, 0)
                if actual_count != expected_count:
                    should_save = True
                    print(f"  [!] Класс '{cls_name}': найдено {actual_count}, ожидалось {expected_count}")
            
            # Если хотя бы для одного класса есть расхождение - сохраняем
            if should_save:
                save_path = os.path.join(BAD_IMAGES_FOLDER, photo)
                shutil.copy2(path, save_path)
                print(f"  [!] СОХРАНЕНО В 'BAD'")
        # ──────────────────────────────────────────────────────────────────────────

        # ── ПОКАЗ ИЗОБРАЖЕНИЯ ─────────────────────────────────────────────────────
        if SHOW_IMAGES:
            #draw_legend(img, models)

            cv2.putText(
                img, "ESC - exit  | any key - next",
                (5, img.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (220, 220, 220), 1, cv2.LINE_AA,
            )
            resized_img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            cv2.imshow(photo, resized_img)

            key = cv2.waitKey(0)
            cv2.destroyAllWindows()

            if key == 27:  # ESC
                print("\nВыход.")
                break
        # ──────────────────────────────────────────────────────────────────────────

except KeyboardInterrupt:
    print("\n\nПрервано пользователем (Ctrl+C)")