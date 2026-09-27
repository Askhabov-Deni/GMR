import cv2
import os
import numpy as np
from ultralytics import YOLO
from pathlib import Path


class YOLOInferer:
    def __init__(self, model, output_dir=None, conf_thresh=0.8, straighten=True):
        """
        model        — путь к .pt или уже загруженный YOLO объект
        output_dir   — папка для сохранения кропов (опционально)
        conf_thresh  — порог уверенности детектора
        straighten   — выравнивать ли кропы по углу (глобальный дефолт).
                       Для детектора цифр рекомендуется False — выпрямление
                       маленьких кропов цифр часто даёт артефакты/переворот.
        """
        self.model = YOLO(model) if isinstance(model, str) else model
        self.class_names = self.model.names
        self.output_dir = output_dir
        self.conf_thresh = conf_thresh
        self.straighten = straighten

        if output_dir:
            for name in self.class_names.values():
                Path(output_dir, name).mkdir(parents=True, exist_ok=True)

    def find_best_angle(self, binary_img):
        h, w = binary_img.shape
        center = (w // 2, h // 2)
        angles = np.arange(-25.0, 25.5, 0.5)

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

    def straighten_crop(self, crop_img):
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        angle = self.find_best_angle(binary)

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

        safe_w = int((w * cos_a - h * sin_a) * 0.3 + w * 0.7)
        safe_h = int((h * cos_a - w * sin_a) * 0.7 + h * 0.3)

        if safe_w >= 10 and safe_h >= 10:
            cx, cy = new_w // 2, new_h // 2
            x1 = cx - safe_w // 2
            y1 = cy - safe_h // 2
            straightened = straightened[y1:y1 + safe_h, x1:x1 + safe_w]

        return straightened, angle

    def process_image(self, img_path, save_crops=False, max_per_class=1, straighten=None):
        """
        straighten — переопределяет self.straighten для этого конкретного вызова.
                     None = использовать значение из __init__.
        """
        img = cv2.imread(img_path)
        if img is None:
            print("⚠️ Пустое изображение!")
            return None

        base_name = Path(img_path).stem
        ext = Path(img_path).suffix

        return self._process_img(img, save_crops=save_crops, max_per_class=max_per_class,
                                 straighten=straighten, base_name=base_name, ext=ext)

    def process_array(self, img: np.ndarray, save_crops=False, max_per_class=1, straighten=None):
        """
        Принимает ndarray (BGR, как из cv2.imread) напрямую — без I/O на диск.
        Удобно когда изображение уже в памяти (например, кроп счётчика).
        save_crops игнорируется (нет имени файла для сохранения).
        """
        if img is None or img.size == 0:
            print("⚠️ Пустой массив!")
            return None

        return self._process_img(img, save_crops=False, max_per_class=max_per_class,
                                 straighten=straighten, base_name=None, ext=None)

    def _process_img(self, img: np.ndarray, save_crops=False, max_per_class=1,
                     straighten=None, base_name=None, ext=None):
        """Общая логика детекции. Вызывается из process_image и process_array."""
        from collections import defaultdict

        do_straighten = self.straighten if straighten is None else straighten

        results = self.model(img, conf=self.conf_thresh, verbose=False)
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            print("⚠️ Детектор ничего не нашел")
            return None

        by_class = defaultdict(list)
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            conf_val = conf.item()
            class_name = self.class_names.get(int(cls.item()), str(int(cls.item())))
            by_class[class_name].append({'box': (x1, y1, x2, y2), 'conf': conf_val})

        crops = []
        for class_name, detections in by_class.items():
            detections.sort(key=lambda d: d['conf'], reverse=True)
            top = detections[:max_per_class]

            for data in top:
                x1, y1, x2, y2 = data['box']
                crop = img[y1:y2, x1:x2]

                if do_straighten:
                    final_crop, angle = self.straighten_crop(crop)
                else:
                    final_crop, angle = crop, 0.0

                saved_path = None
                if save_crops and self.output_dir and base_name and ext:
                    new_name = f"{base_name}__{class_name}_1{ext}"
                    full_path = Path(self.output_dir, class_name, new_name)
                    cv2.imwrite(str(full_path), final_crop)
                    saved_path = str(full_path)

                crops.append({
                    'class':  class_name,
                    'conf':   data['conf'],
                    'angle':  angle,
                    'bbox':   (x1, y1, x2, y2),
                    'crop':   final_crop,
                    'path':   saved_path,
                })

        return crops

    def process_directory(self, input_dir, save_crops=False):
        photos = [f for f in os.listdir(input_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"Нашёл {len(photos)} фоток\n")
        total = {}

        for i, photo in enumerate(photos, 1):
            img_path = str(Path(input_dir, photo))
            print(f"[{i}/{len(photos)}] {photo}")

            crops = self.process_image(img_path, save_crops=save_crops)

            if crops:
                for c in crops:
                    total[c['class']] = total.get(c['class'], 0) + 1
                    print(f"  -> {c['class']} | conf: {c['conf']:.3f} | angle: {c['angle']:.2f}°")
            else:
                print("  -> ничего не нашлось")

        print("\n" + "=" * 50)
        print("Итого:")
        for cls in sorted(total.keys()):
            print(f"  {cls}: {total[cls]}")
        print(f"Всего: {sum(total.values())}")

        return total


# Запуск как отдельного скрипта
if __name__ == "__main__":
    inferer = YOLOInferer(
        model="meter_detect/runs/detect/gas_meter_all_classes_s_v1/weights/best.pt",
        output_dir="database/crops0_8",
        conf_thresh=0.8,
        straighten=True,   # для счётчика и серийника — включено
    )
    inferer.process_directory("database/raw_photos2", save_crops=True)