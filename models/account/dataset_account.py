"""
Данные для модели лицевого счёта по надписи маркером.

Датасет: <датасет>/images/<счёт>__<что угодно>.jpg — кроп надписи (его
вырезает YOLO-детектор), имя начинается с лицевого счёта:
1300000093__marker_id_1.jpeg. Разметки нет и не нужно (решение владельца
2026-10-12): что именно написано — 93, 0093, 13-0093, … — заранее не
известно, поэтому модель учится на «любом из допустимых написаний счёта»
(src/gmr/domain/account_match.account_variants, model_account.multi_ctc_loss):
какое из них на картинке, модель видит сама — после разминки на
синтетических надписях (synthetic.py). Неверный кроп (не тот счёт, не
надпись) — переименовать или удалить.

prepare_input — ЕДИНСТВЕННОЕ место, где картинка превращается во вход
модели; его вызывают и обучение, и инференс (infer_account.py), поэтому
расхождения «обучали так, читаем иначе» быть не может.
"""
import re
import sys
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .config_account import ALPHABET
except ImportError:  # запуск как отдельный скрипт
    from config_account import ALPHABET
try:
    from ..datasets import images_in, photo_key, split_by_photo
except ImportError:  # запуск как отдельный скрипт: нужна папка проекта в sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from models.datasets import images_in, photo_key, split_by_photo
from src.gmr.domain.account_match import account_variants

_LEADING_NUMBER = re.compile(r"\d+")


# ── Чтение ───────────────────────────────────────────────────────────

def read_gray(path) -> np.ndarray:
    """Серое изображение; путь может быть с кириллицей (cv2.imread на
    Windows такие не открывает)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None
    if img is None:
        raise IOError(f"Не удалось прочитать изображение: {path}")
    return img


def account_of(name: str) -> Optional[str]:
    """Лицевой счёт из имени кропа: «1300000093__marker_id_1.jpeg» →
    «1300000093»; None — имя не начинается с номера хотя бы из 3 цифр."""
    m = _LEADING_NUMBER.match(photo_key(name))
    return m.group() if m is not None and len(m.group()) >= 3 else None


def load_dataset(images_dir, verbose: bool = True) -> list[dict]:
    """[{"file": Path, "account": "1300000093", "variants": ("93", "093", …)}, ...]
    — кропы, в имени которых есть лицевой счёт."""
    items, no_account = [], 0
    for img in images_in(Path(images_dir)):
        account = account_of(img.name)
        variants = account_variants(account) if account else ()
        if not variants:
            no_account += 1
            continue
        items.append({"file": img, "account": account, "variants": variants})
    if verbose:
        print(f"Загружено: {len(items)}  |  в имени нет лицевого счёта: {no_account}")
    return items


def split(items: list[dict], train: float, val: float, seed: int) -> tuple[list, list, list]:
    """Деление по счёту: все кропы одного счёта — в одной части. Часть
    счёта определяется его отпечатком (models/datasets.py), поэтому новые
    кропы не перетасовывают старые."""
    parts = split_by_photo(items, lambda m: m["account"], seed, train, val)
    return parts["train"], parts["val"], parts["test"]


# ── Вход модели ──────────────────────────────────────────────────────

def prepare_input(img: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """Кроп (серый или BGR) → float32 (1, img_h, img_w).

    Пропорции сохраняются: высота → img_h (если надпись не слишком длинная),
    справа — фон. Нормировка по самой картинке: фон (медиана) → 0, разброс
    яркости → ~1, поэтому не нужны средние по датасету и не важна
    освещённость фото."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    if h < 2 or w < 2:
        raise ValueError(f"слишком маленький кроп: {w}×{h}")
    scale = min(img_h / h, img_w / w)
    nh = max(1, min(img_h, round(h * scale)))
    nw = max(1, min(img_w, round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    small = cv2.resize(img, (nw, nh), interpolation=interp).astype(np.float32)
    small = (small - float(np.median(small))) / max(float(small.std()), 8.0)
    out = np.zeros((1, img_h, img_w), dtype=np.float32)
    top = (img_h - nh) // 2
    out[0, top:top + nh, :nw] = small
    return out


# ── Аугментации (только обучение) ────────────────────────────────────
# Подобраны под фото надписи маркером из WhatsApp: наклон и разный почерк,
# толщина маркера, смаз, пересжатие JPEG, свет. Ни одна не отрезает край
# надписи: сначала добавляется поле, геометрия — с повтором края
# (у CRNN серийника RandomCrop срезал до 6% ширины при той же метке —
# см. docs/MODELS_REVIEW.md).

def train_augment() -> A.Compose:
    return A.Compose([
        A.Affine(scale=(0.85, 1.0), rotate=(-5, 5), shear={"x": (-15, 15), "y": (-3, 3)},
                 translate_percent={"x": (-0.02, 0.02), "y": (-0.05, 0.05)},
                 border_mode=cv2.BORDER_REPLICATE, p=0.8),
        A.Perspective(scale=(0.01, 0.05), border_mode=cv2.BORDER_REPLICATE, p=0.3),
        A.OneOf([A.Morphological(scale=(2, 3), operation="dilation", p=1.0),
                 A.Morphological(scale=(2, 3), operation="erosion", p=1.0)], p=0.25),
        A.OneOf([A.MotionBlur(blur_limit=(3, 5), p=1.0),
                 A.GaussianBlur(blur_limit=(3, 5), p=1.0)], p=0.3),
        A.Downscale(scale_range=(0.4, 0.9), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.RandomGamma(p=0.3),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.ImageCompression(quality_range=(30, 90), p=0.5),
    ])


def random_margin(img: np.ndarray, max_share: float = 0.10) -> np.ndarray:
    """Поле 0..max_share размера с каждой стороны (повтор края): рамка YOLO
    бывает и впритык, и с запасом. Случайность — от torch (своя у каждого
    процесса загрузки данных)."""
    h, w = img.shape[:2]
    t, b, l, r = (torch.rand(4) * max_share).tolist()
    return cv2.copyMakeBorder(img, int(t * h), int(b * h), int(l * w), int(r * w), cv2.BORDER_REPLICATE)


class AccountDataset(Dataset):
    def __init__(self, items: list[dict], img_h: int, img_w: int, augment: bool = False,
                 alphabet: str = ALPHABET):
        self.items = items
        self.img_h, self.img_w = img_h, img_w
        self.augment = train_augment() if augment else None
        self.index = {c: i for i, c in enumerate(alphabet)}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        item = self.items[i]
        img = item["image"] if "image" in item else read_gray(item["file"])     # синтетика — в памяти
        if self.augment is not None:
            img = random_margin(img)
            img = self.augment(image=img)["image"]
        x = torch.from_numpy(prepare_input(img, self.img_h, self.img_w))
        targets = [torch.tensor([self.index[c] for c in v], dtype=torch.long) for v in item["variants"]]
        return x, targets


def collate(batch):
    """(картинки, все варианты подряд, их длины, чей вариант — номер картинки в пачке)."""
    xs, variants = zip(*batch)
    flat = [t for vs in variants for t in vs]
    owners = [i for i, vs in enumerate(variants) for _ in vs]
    return (torch.stack(xs), torch.cat(flat), torch.tensor([len(t) for t in flat], dtype=torch.long),
            torch.tensor(owners, dtype=torch.long))
