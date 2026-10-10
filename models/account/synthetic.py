"""
Синтетические надписи для «разминки» модели надписи маркером.

Зачем: на своих кропах модель учится без разметки — на «любом из написаний
счёта» (model_account.multi_ctc_loss). С нуля так обучение не сдвигается:
у необученной модели самое короткое написание («93») всегда вероятнее, а
научиться «не видеть» нули трудно. Модель, которая уже умеет читать цифры,
сама выбирает написание, которое видит на картинке (проверено 2026-10-12:
разминка на одних шрифтах → свои кропы другим шрифтом: 99% «точно как
написано», счёт по словарю — 100%). Разминку дают надписи с точной меткой:
цифры шрифтами OpenCV, как пишут счёт (93, 0093, 1300000093, 13-0093),
с наклоном, толщиной маркера, смазом и шумом.
"""
import cv2
import numpy as np

FONTS = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN,
         cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX,
         cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX)


def written_text(rng: np.random.Generator) -> str:
    """Как контролёр мог написать случайный счёт «13 + 8 цифр» (черта — как на корпусе)."""
    core = str(int(rng.integers(1, 10 ** int(rng.integers(1, 7)))))
    rest = core.zfill(8)
    style = rng.random()
    if style < 0.35:
        return core                                                     # 93
    if style < 0.75:
        return rest[-int(rng.integers(len(core) + 1, min(8, len(core) + 3) + 1)):]   # 0093
    if style < 0.85:
        return "13" + rest                                              # 1300000093
    return "13-" + rest[-int(rng.integers(len(core), min(8, len(core) + 2) + 1)):]   # 13-0093


def render(text: str, rng: np.random.Generator) -> np.ndarray:
    """Серая картинка надписи: тёмный «маркер» на светлом корпусе."""
    font = FONTS[int(rng.integers(len(FONTS)))]
    scale, thick = float(rng.uniform(1.2, 2.4)), int(rng.integers(2, 7))
    (w, h), base = cv2.getTextSize(text, font, scale, thick)
    pad = int(rng.integers(6, 20))
    img = np.full((h + base + 2 * pad, w + 2 * pad), int(rng.integers(150, 245)), np.uint8)
    cv2.putText(img, text, (pad, pad + h), font, scale, int(rng.integers(0, 70)), thick, cv2.LINE_AA)
    rows, cols = img.shape
    shear = float(rng.uniform(-0.3, 0.3))                              # наклон почерка
    m = np.float32([[1, shear, -shear * rows / 2], [0, 1, 0]])
    img = cv2.warpAffine(img, m, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), float(rng.uniform(-6, 6)), 1.0)
    img = cv2.warpAffine(img, rot, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    noise = rng.normal(0, float(rng.uniform(0, 12)), img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def synthetic_items(n: int, seed: int = 0) -> list[dict]:
    """n надписей: {"image": серая картинка, "variants": (что написано без черты,)}."""
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n):
        text = written_text(rng)
        items.append({"image": render(text, rng), "variants": (text.replace("-", ""),)})
    return items
