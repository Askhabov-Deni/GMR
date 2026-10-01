"""
Вычисляет реальные mean и std по всему датасету изображений.
Использует онлайн-алгоритм Уэлфорда — не грузит всё в память сразу.

Запуск:
    python utils/compute_mean_std.py --images_dir database/serial_id_ocr_data/serial_id_gold/images
"""

import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

IMG_W, IMG_H = 160, 40

to_tensor = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),  # [0, 1], shape (3, H, W)
])


def compute(images_dir: str):
    paths = sorted({
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        for p in Path(images_dir).glob(ext)
    })

    if not paths:
        print("❌ Изображения не найдены")
        return

    print(f"📂 Найдено изображений: {len(paths)}")

    # Онлайн-подсчёт: mean и M2 по каналам
    n       = 0
    mean    = torch.zeros(3)
    M2      = torch.zeros(3)

    for p in tqdm(paths, desc="Считаю"):
        img = to_tensor(Image.open(p).convert("RGB"))  # (3, H, W)
        pixels = img.permute(1, 2, 0).reshape(-1, 3)   # (H*W, 3)

        for px in pixels:
            n += 1
            delta  = px - mean
            mean  += delta / n
            delta2 = px - mean
            M2    += delta * delta2

    std = (M2 / n).sqrt()

    print(f"\n✅ Результат ({n:,} пикселей из {len(paths)} фото):")
    print(f"   mean = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"   std  = [{std[0]:.4f},  {std[1]:.4f},  {std[2]:.4f}]")
    print(f"\nВставь в dataset_crnn.py:")
    print(f"_MEAN = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"_STD  = [{std[0]:.4f},  {std[1]:.4f},  {std[2]:.4f}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True)
    compute(p.parse_args().images_dir)
