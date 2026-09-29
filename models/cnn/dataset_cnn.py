"""
Загрузка данных для CNN-классификатора цифр.
Структура папки crops_dir (создаётся build_dataset_cnn.py):
crops_dir/
0/  *.jpg
...
9/  *.jpg
"""
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from .config_cnn import (
        BATCH_SIZE,
        IMG_SIZE,
        IMG_WIDTH,
        NORM_MEAN,
        NORM_STD,
        NUM_CLASSES,
        SEED,
        TRAIN_RATIO,
        VAL_RATIO,
    )
except ImportError:  # запуск как отдельный скрипт: python models/cnn/dataset_cnn.py
    from config_cnn import (
        BATCH_SIZE,
        IMG_SIZE,
        IMG_WIDTH,
        NORM_MEAN,
        NORM_STD,
        NUM_CLASSES,
        SEED,
        TRAIN_RATIO,
        VAL_RATIO,
    )

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class DigitDataset(Dataset):
    def __init__(self, samples, transform: A.Compose):
        self.samples = samples
        self.transform = transform

    @property
    def targets(self):
        return [y for _, y in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot read: {img_path}")

        # Инверсия убрана — A.InvertImg(p=0.5) в трейн-трансформе
        # покрывает оба варианта (белые на чёрном / чёрные на белом)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = self.transform(image=img)["image"]

        return img, label


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────
def _train_transform() -> A.Compose:
    return A.Compose([
        # 1. Приводим длинную сторону к IMG_SIZE, сохраняя пропорции,
        #    затем паддим до прямоугольного канваса IMG_SIZE x IMG_WIDTH
        #    (а не квадрата — цифра существенно выше, чем шире).
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(
            min_height=IMG_SIZE, min_width=IMG_WIDTH,
            border_mode=cv2.BORDER_CONSTANT, fill=0,
        ),
        # 2. Инверсия: модель видит оба варианта полярности
        A.InvertImg(p=0.5),
        # 3. Фотометрические аугментации
        A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=0.5),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
        A.GaussNoise(std_range=(0.02, 0.06), p=0.2),
        # 4. Геометрические аугментации.
        # shift/scale_limit чуть уже, чем были при квадратном канвасе:
        # раньше вокруг узкой цифры был большой пустой запас по ширине,
        # теперь канвас по ширине почти впритык к цифре (IMG_WIDTH=48
        # против типичной ширины цифры ~40-44px), так что прежние лимиты
        # (0.02 / 0.05) чаще рисковали бы обрезать край цифры по ширине.
        A.ShiftScaleRotate(
            shift_limit=0.015, scale_limit=0.03, rotate_limit=2,
            border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.3,
        ),
        # 5. Нормализация
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])


def _val_transform() -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(
            min_height=IMG_SIZE, min_width=IMG_WIDTH,
            border_mode=cv2.BORDER_CONSTANT, fill=0,
        ),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────
# Split
# ─────────────────────────────────────────────
def _collect_samples(root: Path):
    samples = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            label = int(class_dir.name)
        except ValueError:
            continue
        for p in class_dir.iterdir():
            if p.suffix.lower() in _IMAGE_EXTENSIONS:
                samples.append((p, label))

    if not samples:
        raise FileNotFoundError(f"No images in {root}")
    return samples


def _train_val_test_split(samples, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED):
    rng = np.random.default_rng(seed)
    by_class: dict = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)

    train, val, test = [], [], []
    for items in by_class.values():
        items = np.array(items, dtype=object)
        rng.shuffle(items)

        n = len(items)
        n_val   = int(n * val_ratio)
        n_test  = int(n * (1.0 - train_ratio - val_ratio))
        n_train = n - n_val - n_test

        train.extend(items[:n_train].tolist())
        val.extend(items[n_train: n_train + n_val].tolist())
        test.extend(items[n_train + n_val:].tolist())

    return train, val, test


# ─────────────────────────────────────────────
# Collate (страховка — после трансформов все IMG_SIZE×IMG_SIZE)
# ─────────────────────────────────────────────
def dynamic_collate_fn(batch):
    images, labels = zip(*batch)
    max_h = max(i.shape[1] for i in images)
    max_w = max(i.shape[2] for i in images)

    out = []
    for img in images:
        dh = max_h - img.shape[1]
        dw = max_w - img.shape[2]
        img = F.pad(img, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2), value=0.0)
        out.append(img)

    return torch.stack(out), torch.tensor(labels)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def make_loaders(
    dataset_dir: Path,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    batch_size и seed — параметры, а не жёстко зашитые константы: раньше
    train_cnn.py принимал --batch_size/--seed из CLI, но эта функция их
    игнорировала и всегда использовала значения из config_cnn.py.
    """
    all_samples = _collect_samples(dataset_dir)
    train_samples, val_samples, test_samples = _train_val_test_split(all_samples, seed=seed)

    train_ds = DigitDataset(train_samples, _train_transform())
    val_ds   = DigitDataset(val_samples,   _val_transform())
    test_ds  = DigitDataset(test_samples,  _val_transform())

    # Class balancing
    class_counts = np.bincount(train_ds.targets, minlength=NUM_CLASSES)
    weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    sample_weights = torch.tensor(
        [weights[y] for _, y in train_samples], dtype=torch.float
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights),
        replacement=True, generator=generator,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=0, collate_fn=dynamic_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=dynamic_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=dynamic_collate_fn,
    )

    print(f"✅ Split → Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"📊 Class counts (Train): {class_counts.tolist()}")

    return train_loader, val_loader, test_loader