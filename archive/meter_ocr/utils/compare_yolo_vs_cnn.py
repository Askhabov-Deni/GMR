"""
Универсальный инструмент тестирования: YOLO vs CNN vs CRNN.
Все модели опциональны. Загрузка через TorchScript (архитектура + веса).

Управление: ← → навигация, S — сохранить, Esc/Q — выход
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import tkinter as tk
from PIL import ImageTk
from abc import ABC, abstractmethod


# ═══════════════════════════════════════════════════════════════════════════════
#  Настройки и ФЛАГИ ВКЛЮЧЕНИЯ
# ═══════════════════════════════════════════════════════════════════════════════

YOLO_ENABLED   = True
CNN_ENABLED    = True   # Включаем/выключаем CNN целиком
CRNN_ENABLED   = True

YOLO_MODEL_PATH   = "meter_ocr/yolo/runs/yolov8n_gas_meter_digits_v1/weights/best.pt"
CNN_MODEL_PATH    = "meter_ocr/cnn/runs/v1/best_model_cnn.pt"  # TorchScript
CRNN_MODEL_PATH   = "meter_ocr/crnn/runs/best_model.pt"      # TorchScript
UNLABELED_DIR     = "database/meter_ocr_data/test_data/images"

YOLO_CONF     = 0.25
CNN_N_DIGITS  = 5
CROP_SIZE     = (32, 64)

AUTO_PROJ_THRESHOLD = 0.05
AUTO_MIN_WIDTH_FRAC = 0.04
AUTO_SMOOTH_KERNEL  = 1

SAVE_DIR   = Path("compare_saved")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ═══════════════════════════════════════════════════════════════════════════════
#  Базовый интерфейс
# ═══════════════════════════════════════════════════════════════════════════════

class Approach(ABC):
    name: str = "Base"

    @abstractmethod
    def predict(self, img_bgr: np.ndarray) -> str:
        ...


# ═══════════════════════════════════════════════════════════════════════════════
#  Предобработка CNN (проекции + тензоры)
# ═══════════════════════════════════════════════════════════════════════════════

_cnn_transform = A.Compose([
    A.Resize(CROP_SIZE[1], CROP_SIZE[0]),
    A.CLAHE(clip_limit=3.0, tile_grid_size=(4, 8), p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


def _find_digit_boxes(gray_crop):
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = np.sum(binary, axis=0).astype(float)
    if AUTO_SMOOTH_KERNEL > 1:
        kernel = np.ones(AUTO_SMOOTH_KERNEL) / AUTO_SMOOTH_KERNEL
        proj = np.convolve(proj, kernel, mode="same")
    threshold = proj.max() * AUTO_PROJ_THRESHOLD
    in_digit, groups, start = False, [], 0
    for x, val in enumerate(proj):
        if val > threshold and not in_digit: start = x; in_digit = True
        elif val <= threshold and in_digit: groups.append((start, x)); in_digit = False
    if in_digit: groups.append((start, len(proj) - 1))
    min_w = gray_crop.shape[1] * AUTO_MIN_WIDTH_FRAC
    return [(x1, x2) for x1, x2 in groups if (x2 - x1) >= min_w]


def cnn_preprocess(img_bgr, n=CNN_N_DIGITS):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    boxes = _find_digit_boxes(gray)
    mode = "проекция" if len(boxes) == n else f"равном.(→{len(boxes)})"
    if len(boxes) != n:
        step = img_bgr.shape[1] / n
        boxes = [(int(i * step), int((i + 1) * step)) for i in range(n)]
    return [img_bgr[:, x1:x2] for x1, x2 in boxes], mode


def _crop_to_tensor(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return _cnn_transform(image=rgb)["image"].unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Постобработка YOLO
# ═══════════════════════════════════════════════════════════════════════════════

def yolo_postprocess(yolo_results, conf_threshold=YOLO_CONF):
    dets = []
    for box in yolo_results[0].boxes:
        if float(box.conf[0]) < conf_threshold: continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        dets.append((x1, x2, int(box.cls[0]), float(box.conf[0])))
    dets.sort(key=lambda d: d[0])
    return "".join(str(d[2]) for d in dets), dets


# ═══════════════════════════════════════════════════════════════════════════════
#  Декодер CRNN (CTC Greedy)
# ═══════════════════════════════════════════════════════════════════════════════

_BLANK = 10

def _crnn_decode(log_probs):
    """(T, 1, C) → str"""
    indices = log_probs.argmax(2).squeeze(1)  # (T,)
    result, prev = [], -1
    for idx in indices.tolist():
        if idx != prev: result.append(idx)
        prev = idx
    return "".join(str(i) for i in result if i != _BLANK)


# ═══════════════════════════════════════════════════════════════════════════════
#  Подходы (Approach)
# ═══════════════════════════════════════════════════════════════════════════════

class YOLOApproach(Approach):
    name = "YOLO"
    def __init__(self, model): self.model = model
    def predict(self, img_bgr):
        res = self.model(img_bgr, conf=YOLO_CONF, iou=0.4, verbose=False)
        s, _ = yolo_postprocess(res)
        return s


class CNNApproach(Approach):
    name = "CNN"
    def __init__(self, model, to_tensor_fn):
        self.model, self.to_tensor_fn = model, to_tensor_fn
    def predict(self, img_bgr):
        crops, _ = cnn_preprocess(img_bgr, CNN_N_DIGITS)
        self.model.eval()
        digits = []
        with torch.no_grad():
            for crop in crops:
                if crop.shape[1] == 0: digits.append(0); continue
                logits = self.model(self.to_tensor_fn(crop))[0]
                digits.append(int(torch.softmax(logits, 0).argmax()))
        return "".join(str(d) for d in digits)


class CRNNApproach(Approach):
    name = "CRNN"
    def __init__(self, model, transform):
        self.model, self.transform = model, transform
    def predict(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0)
        with torch.no_grad():
            out = self.model(tensor)  # (T, 1, 11)
        return _crnn_decode(out)


# ═══════════════════════════════════════════════════════════════════════════════
#  Безопасные загрузчики
# ═══════════════════════════════════════════════════════════════════════════════

def _check(path, label):
    if not path or not Path(path).exists():
        print(f"  ⚠️  {label}: не найден — пропущено"); return False
    return True

def load_yolo(path):
    if not _check(path, "YOLO"): return None
    try:
        from ultralytics import YOLO
        return YOLOApproach(YOLO(path))
    except Exception as e:
        print(f"  ❌ YOLO: {e}"); return None

def load_cnn(path):
    if not _check(path, "CNN"): return None
    try:
        model = torch.jit.load(path, map_location="cpu")
        model.eval()
        print(f"  ✓ CNN: TorchScript")
        return CNNApproach(model, _crop_to_tensor)
    except Exception as e:
        print(f"  ❌ CNN: {e}"); return None

def load_crnn(path):
    if not _check(path, "CRNN"): return None
    try:
        model = torch.jit.load(path, map_location="cpu")
        model.eval()
        transform = transforms.Compose([
            transforms.Resize((40, 160)),  # H, W (как в твоем CRNN)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"  ✓ CRNN: TorchScript")
        return CRNNApproach(model, transform)
    except Exception as e:
        print(f"  ❌ CRNN: {e}"); return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Отрисовка
# ═══════════════════════════════════════════════════════════════════════════════

try:
    _font_big   = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    _font_small = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
except Exception:
    _font_big = _font_small = ImageFont.load_default()


def render_comparison(img_bgr, predictions, filename):
    crop = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)

    preds = [p for p in predictions.values() if p]
    unique = set(preds)
    n = len(predictions)

    if n == 0: status, sc = "нет моделей", (80,0,0)
    elif n == 1: status, sc = f"предсказание: {preds[0]}", (40,80,140)
    elif len(unique) == 1: status, sc = f"✓ согласованы: {unique.pop()}", (0,120,0)
    else: status, sc = "⚠ расхождение", (180,120,0)

    line_h, pad = 26, 10
    pw, ph = crop.width, line_h * (n + 1) + pad * 2
    panel = Image.new("RGB", (pw, ph), (14, 14, 20))
    draw = ImageDraw.Draw(panel)
    draw.text((pad, pad), f"[{status}]  •  {filename}", fill=sc, font=_font_small)

    y = pad + line_h
    agreed = (len(unique) == 1 and n > 1)
    for name, pred in predictions.items():
        col = (120,220,120) if (agreed and pred in unique) else ((160,200,255) if n==1 else (220,180,80))
        draw.text((pad, y), f"{name:>8} :  {pred or '—'}", fill=col, font=_font_big)
        y += line_h

    total = Image.new("RGB", (pw, crop.height + ph), (14, 14, 20))
    total.paste(crop, (0, 0))
    total.paste(panel, (0, crop.height))
    return total


# ═══════════════════════════════════════════════════════════════════════════════
#  Пайплайн + UI
# ═══════════════════════════════════════════════════════════════════════════════

def process_image(approaches, img_path):
    img = cv2.imread(str(img_path))
    if img is None: return None
    preds = {}
    for app in approaches:
        try: preds[app.name] = app.predict(img)
        except Exception as e: preds[app.name] = f"ERR: {e}"
    return render_comparison(img, preds, img_path.name)


def main():
    files = sorted([f for f in Path(UNLABELED_DIR).iterdir() if f.suffix.lower() in EXTENSIONS])
    if not files: print("❌ Изображений не найдено."); return

    print("Загружаю подходы (каждый опционален)...")
    approaches = []
    if YOLO_ENABLED:
        app = load_yolo(YOLO_MODEL_PATH)
        if app is not None:
            approaches.append(app)

    if CNN_ENABLED:
        app = load_cnn(CNN_MODEL_PATH)
        if app is not None:
            approaches.append(app)

    if CRNN_ENABLED:
        app = load_crnn(CRNN_MODEL_PATH)
        if app is not None:
            approaches.append(app)

    if not approaches:
        print("❌ Ни одна модель не загружена."); return
    print(f"✅ Активных: {len(approaches)} — {', '.join(a.name for a in approaches)}")

    cache = {}
    def get(i):
        if i not in cache: cache[i] = process_image(approaches, files[i])
        return cache[i]

    root = tk.Tk(); root.title("Universal Test Tool"); root.configure(bg="#0e0e14")
    info = tk.StringVar()
    tk.Label(root, textvariable=info, bg="#0e0e14", fg="#aaaacc", font=("Courier New", 9)).pack(side="top", anchor="w", padx=6, pady=2)
    canvas = tk.Label(root, bg="#0e0e14"); canvas.pack(padx=8, pady=4)
    st = {"idx": 0, "photo": None}

    def show(i):
        i = max(0, min(i, len(files)-1)); st["idx"] = i
        img = get(i)
        if not img: return
        img.thumbnail((1400, 700), Image.LANCZOS)
        st["photo"] = ImageTk.PhotoImage(img); canvas.config(image=st["photo"])
        info.set(f"{i+1}/{len(files)}  {files[i].name}   [← →  |  S save  |  Esc quit]")

    def save():
        img = get(st["idx"])
        if not img: return
        SAVE_DIR.mkdir(exist_ok=True); p = SAVE_DIR / files[st["idx"]].name; img.save(p); info.set(f"✅ {p}")

    root.bind("<Key>", lambda e: (
        show(st["idx"]+1) if e.keysym in ("Right","space") else
        show(st["idx"]-1) if e.keysym in ("Left","BackSpace") else
        save() if e.keysym in ("s","S") else
        root.destroy() if e.keysym in ("Escape","q","Q") else None
    ))
    show(0); root.mainloop()

if __name__ == "__main__":
    main()