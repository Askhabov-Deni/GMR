import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

try:
    from .config_crnn import IMG_W, IMG_H, _MEAN, _STD, MIN_LABEL_LENGTH, MAX_LABEL_LENGTH
except ImportError:  # запуск как отдельный скрипт (python models/crnn/dataset_crnn.py)
    from config_crnn import IMG_W, IMG_H, _MEAN, _STD, MIN_LABEL_LENGTH, MAX_LABEL_LENGTH
try:
    from ..datasets import split_by_photo
except ImportError:  # запуск как отдельный скрипт: нужна папка проекта в sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from models.datasets import split_by_photo


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
    """Деление по фото (этап 5, models/datasets.py): кропы одного фото
    (`<фото>__…`) — в одной части; доля фото — по отпечатку имени, новые
    файлы не перетасовывают старые."""
    parts = split_by_photo(metadata, lambda m: m["file"].name, seed, train, val)
    return parts["train"], parts["val"], parts["test"]


# ── Transforms ───────────────────────────────────────────────────────
# Как кроп приводится к IMG_W×IMG_H (этап 6c, docs/MODELS_REVIEW.md, раздел 3).
# Режим пишется в чекпоинт (input_mode): модель читается так же, как училась.
#   stretch     — растянуть (как учились модели до 6c; чекпоинт без input_mode);
#   keep_aspect — с сохранением пропорций, остаток — фоном (новое обучение).
STRETCH, KEEP_ASPECT = "stretch", "keep_aspect"
INPUT_MODES = (KEEP_ASPECT, STRETCH)


class FitToBox:
    """Кроп → IMG_W×IMG_H без растяжения: высота → IMG_H (длинный номер —
    по ширине), справа и сверху/снизу — фон (медианный цвет кропа).

    augment=True (обучение): поле вокруг кропа, поворот с расширением
    холста и перспектива с фоном — край номера не срезается (раньше
    RandomCrop срезал до 16 пикселей при той же метке)."""

    def __init__(self, w: int, h: int, augment: bool = False):
        self.w, self.h, self.augment = w, h, augment

    def __call__(self, img: Image.Image) -> Image.Image:
        bg = tuple(int(v) for v in np.median(np.asarray(img).reshape(-1, 3), axis=0))
        if self.augment:
            img = self._augment(img, bg)
        w, h = img.size
        scale = min(self.h / h, self.w / w)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        out = Image.new("RGB", (self.w, self.h), bg)
        out.paste(img.resize((nw, nh), Image.BILINEAR), (0, (self.h - nh) // 2))
        return out

    @staticmethod
    def _augment(img: Image.Image, bg: tuple) -> Image.Image:
        w, h = img.size
        left, right = random.randint(0, w // 12), random.randint(0, w // 12)
        top, bottom = random.randint(0, h // 6), random.randint(0, h // 6)
        canvas = Image.new("RGB", (w + left + right, h + top + bottom), bg)
        canvas.paste(img, (left, top))
        img = canvas.rotate(random.uniform(-5, 5), resample=Image.BILINEAR, expand=True, fillcolor=bg)
        if random.random() < 0.4:
            start, end = T.RandomPerspective.get_params(img.size[0], img.size[1], 0.2)
            img = TF.perspective(img, start, end, fill=list(bg))      # сжимает внутрь, края целы
        return img


def _add_noise(img: torch.Tensor) -> torch.Tensor:
    return (img + 0.03 * torch.randn_like(img)).clamp(-1, 1)

def _autocontrast(img: Image.Image) -> Image.Image:
    return ImageOps.autocontrast(img)


def train_transform(mode: str) -> T.Compose:
    if mode == KEEP_ASPECT:
        return T.Compose([
            FitToBox(IMG_W, IMG_H, augment=True),
            T.ColorJitter(brightness=0.4, contrast=0.4),
            T.RandomEqualize(p=0.3),
            T.Lambda(_autocontrast),        # как при чтении (val_transform): всегда
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            T.ToTensor(),
            T.Normalize(_MEAN, _STD),
            T.Lambda(_add_noise),
        ])
    return T.Compose([                      # STRETCH — как до этапа 6c
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


def val_transform(mode: str = STRETCH) -> T.Compose:
    return T.Compose([
        FitToBox(IMG_W, IMG_H) if mode == KEEP_ASPECT else T.Resize((IMG_H, IMG_W)),
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
