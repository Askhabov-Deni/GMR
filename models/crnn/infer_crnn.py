import os
import argparse
import shutil
from pathlib import Path

import cv2
import torch
from PIL import Image

from dataset_crnn import val_transform
from model_crnn import CRNN, predict_with_confidence


class CRNNInferer:
    def __init__(self, model_or_path: str | CRNN, device: torch.device | None = None):
        self.device    = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = val_transform()

        if isinstance(model_or_path, str):
            self.model = CRNN.from_pretrained(model_or_path, device=str(self.device))
        else:
            self.model = model_or_path.to(self.device)

        self.model.eval()

    def predict_with_details(self, image_input: str | object) -> dict:
        """Распознаёт изображение, возвращает текст и уверенность по каждому символу."""
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        else:
            img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        result = predict_with_confidence(self.model, tensor)[0]

        details  = result["details"]
        avg_conf = sum(d["conf"] for d in details) / len(details) if details else 0.0

        return {
            "text":           result["text"],
            "avg_confidence": avg_conf,
            "details":        details,
        }

    def process_directory(
        self,
        image_dir: str,
        min_confidence: float = 0.8,
        expected_length: int  = 5,
        debug: bool           = False,
        debug_dir: str        = "debug_bad_crops",
    ) -> dict:
        """
        Прогоняет все изображения из папки и выводит статистику.

        Args:
            debug: если True — копирует плохие кейсы в debug_dir.
        """
        exts  = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = sorted(
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in exts
        )
        print(f"Найдено изображений: {len(paths)}")
        if debug:
            print(f"Режим DEBUG включен. Плохие кейсы будут сохранены в '{debug_dir}'\n")
        else:
            print()

        if debug and debug_dir:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)

        stats = {"good": 0, "bad": 0}

        for path in paths:
            res          = self.predict_with_details(path)
            text         = res["text"]
            conf         = res["avg_confidence"]
            details      = res["details"]
            is_good_conf = conf >= min_confidence
            is_good_len  = len(text) == expected_length

            details_str = (
                " ".join(f"{d['char']}({d['conf']:.2f})" for d in details)
                if details else "пусто"
            )

            if is_good_conf and is_good_len:
                print(f"✅ {os.path.basename(path):30s} → '{text}' [{details_str}] (avg: {conf:.3f})")
                stats["good"] += 1
            else:
                reasons = []
                if not is_good_conf:
                    reasons.append("low conf")
                if not is_good_len:
                    reasons.append(f"bad len {len(text)}")
                print(f"❌ {os.path.basename(path):30s} → '{text}' [{details_str}] ({', '.join(reasons)})")
                stats["bad"] += 1

                if debug and debug_dir:
                    shutil.copy(path, os.path.join(debug_dir, os.path.basename(path)))

        print("\n" + "=" * 60)
        print(f"Итог: Хороших {stats['good']}, Плохих {stats['bad']}")
        if debug and debug_dir:
            print(f"Плохие кропы сохранены в: {os.path.abspath(debug_dir)}")

        return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/crnn/best.pt")
    p.add_argument("--image",      default=None)
    p.add_argument("--image_dir",  default=None)
    p.add_argument("--min_conf",   type=float, default=0.8)
    p.add_argument("-d", "--debug", action="store_true", help="Сохранять плохие кропы для разбора")
    p.add_argument("--debug_dir",  default="debug_bad_crops")
    args = p.parse_args()

    inferer = CRNNInferer(args.checkpoint)

    if args.image:
        res = inferer.predict_with_details(args.image)
        print(f"Текст: {res['text']}")
        print(f"Средняя уверенность: {res['avg_confidence']:.4f}")
        print(f"По символам: {res['details']}")

    elif args.image_dir:
        inferer.process_directory(
            args.image_dir,
            min_confidence=args.min_conf,
            debug=args.debug,
            debug_dir=args.debug_dir,
        )
