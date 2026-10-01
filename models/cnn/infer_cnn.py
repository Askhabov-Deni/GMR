"""
Инференс (предсказание) для CNN-классификатора цифр.
Поддерживает предсказание одиночного изображения или целой папки.

ИСПОЛЬЗОВАНИЕ:
1. Предсказание одного кропа:
   python infer_cnn.py --checkpoint runs/cnn/best.pth --image path/to/crop.jpg

2. Обработка папки с сохранением подозрительных случаев (debug):
   python infer_cnn.py --checkpoint runs/cnn/best.pth --image_dir path/to/crops/ --min_conf 0.9 --debug
"""
import os
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from .config_cnn import IMG_SIZE, IMG_WIDTH, NORM_MEAN, NORM_STD, MIN_CONFIDENCE, DEVICE
    from .model_cnn import DigitCNN
except ImportError:  # запуск как отдельный скрипт: python models/cnn/infer_cnn.py
    from config_cnn import IMG_SIZE, IMG_WIDTH, NORM_MEAN, NORM_STD, MIN_CONFIDENCE, DEVICE
    from model_cnn import DigitCNN


class CNNInferer:
    def __init__(self, checkpoint_path: str, device: torch.device | None = None):
        self.device = device or DEVICE
        self.checkpoint_path = checkpoint_path
        
        # Трансформы для инференса. 
        # Должны в точности повторять базовые трансформы из dataset_cnn.py (без аугментаций!).
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=IMG_SIZE),
            A.PadIfNeeded(
                min_height=IMG_SIZE, min_width=IMG_WIDTH,
                border_mode=cv2.BORDER_CONSTANT, fill=0, # Паддинг черным (0)
            ),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ToTensorV2(),
        ])
                
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self) -> torch.nn.Module:
        """Загружает модель: либо TorchScript (.pt), либо обычный чекпоинт (.pth)."""
        path = Path(self.checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Чекпоинт не найден: {self.checkpoint_path}")
            
        # 1. Попытка загрузить как TorchScript (если файл имеет расширение .pt)
        if path.suffix.lower() == ".pt":
            try:
                model = torch.jit.load(self.checkpoint_path, map_location=self.device)
                print(f"✅ Загружена TorchScript модель: {self.checkpoint_path}")
                return model
            except Exception as e:
                print(f"⚠️ Не удалось загрузить как TorchScript ({e}), пробую как state_dict...")
            
        # 2. Загрузка как обычного чекпоинта (.pth или .pt) через наш новый метод
        print("🔄 Загрузка весов через DigitCNN.from_pretrained...")
        model = DigitCNN.from_pretrained(self.checkpoint_path, device=self.device)
        return model

    def _preprocess_image(self, image_input: str | np.ndarray) -> torch.Tensor:
        """
        Предобработка изображения.
        Повторяет базовую логику из dataset_cnn.py (без случайных аугментаций).
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"Не удалось прочитать изображение: {image_input}")
        elif image_input.ndim == 2:
            # Уже grayscale — ветка ниже (BGR2GRAY) упала бы на 2D-массиве
            img = image_input
        else:
            img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)

        # Инверсия НЕ нужна, так как модель обучалась на обоих вариантах (p=0.5)
        # и уже инвариантна к этому. Подаем изображение в исходном виде.

        # Конвертация в 3 канала для совместимости с трансформами (Conv2d ожидает 3 канала)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Применение трансформов
        augmented = self.transform(image=img)
        return augmented["image"].unsqueeze(0).to(self.device)

    def predict(self, image_input: str | np.ndarray) -> dict:
        """
        Делает предсказание для одного изображения.
        Возвращает: {"digit": int, "confidence": float}
        """
        tensor = self._preprocess_image(image_input)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)
            
        return {
            "digit": int(pred_class.item()),
            "confidence": float(confidence.item())
        }

    def process_directory(
        self,
        image_dir: str,
        min_confidence: float = MIN_CONFIDENCE,
        debug: bool = False,
        debug_dir: str = "debug_bad_crops_cnn",
    ) -> dict:
        """
        Прогоняет все изображения из папки и выводит статистику.
        """
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = sorted(
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in exts
        )
        print(f"📁 Найдено изображений: {len(paths)}")
        if debug:
            print(f"🐞 Режим DEBUG включен. Плохие кейсы будут сохранены в '{debug_dir}'\n")
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
        else:
            print()

        # "bad" — низкая уверенность (валидное предсказание, просто неуверенное);
        # "errors" — файл вообще не удалось обработать (битый файл и т.п.).
        # Раньше эти два случая смешивались в одну корзину "bad", из-за чего
        # итоговая статистика "good vs bad" была не тем, чем казалась.
        stats = {"good": 0, "bad": 0, "errors": 0}

        for path in paths:
            try:
                res = self.predict(path)
            except Exception as e:
                print(f"⚠️  Ошибка обработки {os.path.basename(path)}: {e}")
                stats["errors"] += 1
                continue

            digit = res["digit"]
            conf = res["confidence"]
            is_good = conf >= min_confidence

            if is_good:
                print(f"✅ {os.path.basename(path):35s} → Цифра: {digit}  (Уверенность: {conf:.4f})")
                stats["good"] += 1
            else:
                print(f"❌ {os.path.basename(path):35s} → Цифра: {digit}  (Уверенность: {conf:.4f}) [НИЗКАЯ]")
                stats["bad"] += 1

                if debug and debug_dir:
                    shutil.copy(path, os.path.join(debug_dir, os.path.basename(path)))

        print("\n" + "=" * 65)
        print(
            f"📊 Итог: Хороших (>= {min_confidence}): {stats['good']} | "
            f"Низкая уверенность: {stats['bad']} | Ошибки чтения: {stats['errors']}"
        )
        if debug and debug_dir:
            print(f"💾 Плохие кропы сохранены в: {os.path.abspath(debug_dir)}")
        print("=" * 65)

        return stats


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Инференс CNN классификатора цифр")
    p.add_argument("--checkpoint", type=str, default="runs/cnn/best.pth", help="Путь к .pt (TorchScript) или .pth чекпоинту")
    p.add_argument("--image", type=str, default=None, help="Путь к одному изображению для предсказания")
    p.add_argument("--image_dir", type=str, default=None, help="Путь к папке с изображениями для пакетной обработки")
    p.add_argument("--min_conf", type=float, default=MIN_CONFIDENCE, help="Минимальный порог уверенности (0.0 - 1.0)")
    p.add_argument("-d", "--debug", action="store_true", help="Сохранять кропы с низкой уверенностью для ручного разбора")
    p.add_argument("--debug_dir", type=str, default="debug_bad_crops_cnn", help="Папка для сохранения плохих кропов")
    
    args = p.parse_args()

    if not args.image and not args.image_dir:
        p.error("Необходимо указать либо --image, либо --image_dir")

    print(f"🖥️  Устройство: {DEVICE}")
    inferer = CNNInferer(args.checkpoint, device=DEVICE)

    if args.image:
        res = inferer.predict(args.image)
        print("\n" + "=" * 45)
        print(f"🖼️  Файл: {os.path.basename(args.image)}")
        print(f"🔢 Предсказанная цифра : {res['digit']}")
        print(f"🎯 Уверенность модели  : {res['confidence']:.4f} ({res['confidence']*100:.2f}%)")
        print("=" * 45)

    elif args.image_dir:
        inferer.process_directory(
            args.image_dir,
            min_confidence=args.min_conf,
            debug=args.debug,
            debug_dir=args.debug_dir,
        )