import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image, ImageOps

from config_crnn import IMG_W, IMG_H, _MEAN, _STD, MIN_LABEL_LENGTH, MAX_LABEL_LENGTH


# ── Загрузка аннотаций ────────────────────────────────────────────────

def _parse_label_txt(label_path: Path) -> str | None:
    """
    Читает .txt и возвращает строку из цифр.
    Авто-детекция формата: если в строках есть пробелы → YOLO, иначе → plain-text.
    Возвращает None, если длина не попадает в [MIN_LABEL_LENGTH, MAX_LABEL_LENGTH].
    """
    raw = label_path.read_text(encoding="utf-8").strip()
    if not raw:
        return None

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    is_yolo = any(" " in line for line in lines)

    if is_yolo:
        boxes = []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                cx  = float(parts[1])
                if 0 <= cls <= 9:
                    boxes.append((cx, cls))
            except ValueError:
                continue
        boxes.sort(key=lambda b: b[0])
        label = "".join(str(c) for _, c in boxes)
    else:
        label = "".join(lines)
        if not label.isdigit():
            return None

    if MIN_LABEL_LENGTH <= len(label) <= MAX_LABEL_LENGTH:
        return label
    return None


def load_dataset(images_dir: str, labels_dir: str) -> list[dict]:
    """
    Возвращает список {"file": Path, "label": "12345"}.
    Поддерживает оба формата разметки: YOLO и plain-text.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    all_images = sorted({
        p
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        for p in images_dir.glob(ext)
    })

    metadata = []
    skipped_no_label = skipped_bad = 0

    for img_path in all_images:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            skipped_no_label += 1
            continue
        label = _parse_label_txt(label_path)
        if label is None:
            skipped_bad += 1
            continue
        metadata.append({"file": img_path, "label": label})

    print(
        f"Загружено: {len(metadata)}"
        f"  |  без разметки: {skipped_no_label}"
        f"  |  некорректная длина/плохой формат: {skipped_bad}"
    )
    return metadata


# ── Train / val / test split ──────────────────────────────────────────

def split(metadata: list, train: float = 0.7, val: float = 0.15, seed: int = 42) -> tuple:
    data = metadata.copy()
    random.seed(seed)
    random.shuffle(data)
    n = len(data)
    a = int(n * train)
    b = int(n * (train + val))
    return data[:a], data[a:b], data[b:]


# ── Transforms ───────────────────────────────────────────────────────

def _add_noise(img: torch.Tensor) -> torch.Tensor:
    return (img + 0.03 * torch.randn_like(img)).clamp(-1, 1)

def _autocontrast(img: Image.Image) -> Image.Image:
    return ImageOps.autocontrast(img)


def train_transform() -> T.Compose:
    return T.Compose([
        T.Resize((IMG_H + 8, IMG_W + 16)),
        T.RandomCrop((IMG_H, IMG_W)),
        T.RandomRotation(degrees=5),
        T.RandomPerspective(distortion_scale=0.2, p=0.4),
        T.RandomAutocontrast(p=0.5),
        T.RandomEqualize(p=0.3),
        T.ColorJitter(brightness=0.4, contrast=0.4),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
        T.Lambda(_add_noise),
    ])


def val_transform() -> T.Compose:
    return T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.Lambda(_autocontrast),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])


# ── Dataset ──────────────────────────────────────────────────────────

class MeterDataset(Dataset):
    def __init__(self, metadata: list, transform=None):
        self.metadata  = metadata
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple:
        item = self.metadata[idx]
        img  = Image.open(item["file"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label   = item["label"]
        indices = torch.tensor([int(c) for c in label], dtype=torch.long)
        length  = torch.tensor(len(label), dtype=torch.long)
        return img, indices, length


# ── Collate ──────────────────────────────────────────────────────────

def collate(batch: list) -> tuple:
    imgs, indices_list, lengths = zip(*batch)
    return torch.stack(imgs), torch.cat(indices_list), torch.stack(list(lengths))
