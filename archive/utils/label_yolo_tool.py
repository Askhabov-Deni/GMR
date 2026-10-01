"""
Полуавтоматический инструмент разметки цифр на кропах газового счётчика.
Поддерживает три класса: gas_meter, serial_id, marker_id

Режимы bbox:
  - Авто (проекция)  — для gas_meter
  - Равномерный      — запасной вариант
  - Клики            — для serial_id / marker_id: кликаешь границы между цифрами
  - Боксы (2 клика)  — для marker_id: каждые 2 клика = один бокс (можно перекрывать)

Горячие клавиши ввода цифр:
  q=7  w=8  e=9
  a=4  s=5  d=6
  z=1  x=2  c=3  space=0

Предсказание (serial_id / marker_id):
  - serial_id  : EasyOCR запускается в фоне при загрузке изображения.
                 Как только боксов = len(предсказания) → поле заполняется автоматически.
  - marker_id  : кандидаты извлекаются из имени файла (последние ненулевые цифры +
                 варианты с ведущими нулями). Совпадение по длине → автозаполнение.
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import re
import threading
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
#  НАСТРОЙКИ — все магические числа здесь, больше нигде
# ═══════════════════════════════════════════════════════════════════════════════

# ── Пути ──────────────────────────────────────────────────────────────────────
BASE_DIR = "database/model_ocr"

# ── Окно приложения ───────────────────────────────────────────────────────────
WINDOW_W       = 1100
WINDOW_H       = 680
CTRL_PANEL_W   = 300

# ── Цвета интерфейса ──────────────────────────────────────────────────────────
COLOR_BG         = "#0e0e14"
COLOR_BG_DARK    = "#080810"
COLOR_BG_TAB     = "#14141e"
COLOR_BG_TAB_ACT = "#1a1a2e"
COLOR_FG_DIM     = "#555577"
COLOR_FG_DIMMER  = "#333355"
COLOR_FG_HINT    = "#444466"
COLOR_FG_CTRL    = "#aaaacc"
COLOR_SEPARATOR  = "#1a1a2e"
COLOR_ENTRY_BG   = "#080810"
COLOR_ENTRY_ERR  = "#2a0808"
COLOR_BTN_SAVE   = "#0d3320"
COLOR_BTN_SKIP   = "#14141e"
COLOR_BTN_DEL    = "#1a0808"
COLOR_BTN_CLEAR  = "#1a1408"
COLOR_KEYHINT    = "#252538"
COLOR_PREDICT    = "#1a2a1a"   # фон строки предсказания

# ── Шрифты ────────────────────────────────────────────────────────────────────
FONT_MONO        = "Courier New"
FONT_ENTRY_SIZE  = 32
FONT_TITLE_SIZE  = 26
FONT_TAB_SIZE    = 10
FONT_CTRL_SIZE   = 9
FONT_SMALL_SIZE  = 8

# ── Отрисовка боксов ──────────────────────────────────────────────────────────
BOX_OUTLINE_WIDTH  = 2
BOX_FILL_ALPHA     = 55
DIGIT_LABEL_FONT_SIZE = 36
DIGIT_LABEL_COLOR     = "#ffffff"
DIGIT_BADGE_COLOR     = "#000000"

# ── Отрисовка разделителей (режим «клики») ────────────────────────────────────
SPLIT_LINE_WIDTH   = 2
SPLIT_DOT_RADIUS   = 5

# ── Призрак мыши ──────────────────────────────────────────────────────────────
GHOST_STEP_CLICK   = 6
GHOST_GAP_CLICK    = 3
GHOST_STEP_PAIRS   = 10
GHOST_GAP_PAIRS    = 6
GHOST_DIAMOND_SIZE = 5

# ── Алгоритм авто-сегментации ─────────────────────────────────────────────────
AUTO_PROJ_THRESHOLD = 0.05
AUTO_MIN_WIDTH_FRAC = 0.04
AUTO_SMOOTH_KERNEL  = 1

# ── YOLO-разметка ─────────────────────────────────────────────────────────────
YOLO_BOX_HEIGHT    = 0.85

# ── Цвета боксов ──────────────────────────────────────────────────────────────
BOX_COLORS_LINE = [
    "#00ff88", "#ff6688", "#ffcc00", "#44eeff",
    "#aa88ff", "#ff8844", "#88ff44", "#ff44cc",
]
BOX_COLORS_RGBA = [
    (0, 255, 136, BOX_FILL_ALPHA),  (255, 102, 136, BOX_FILL_ALPHA),
    (255, 204, 0,  BOX_FILL_ALPHA), (68,  238, 255, BOX_FILL_ALPHA),
    (170, 136, 255,BOX_FILL_ALPHA), (255, 136, 68,  BOX_FILL_ALPHA),
    (136, 255, 68, BOX_FILL_ALPHA), (255, 68,  204, BOX_FILL_ALPHA),
]

# ── Классы разметки ───────────────────────────────────────────────────────────
CLASSES = {
    "gas_meter": {
        "label":        "Gas Meter",
        "color":        "#00ff88",
        "crops_dir":    f"{BASE_DIR}/gas_meter/images",
        "labels_dir":   f"{BASE_DIR}/gas_meter/labels",
        "fixed_len":    5,
        "hint":         "5 цифр (показания дисплея)",
        "default_mode": "auto",
    },
    "serial_id": {
        "label":        "Serial ID",
        "color":        "#00cfff",
        "crops_dir":    f"{BASE_DIR}/serial_id/images",
        "labels_dir":   f"{BASE_DIR}/serial_id/labels",
        "fixed_len":    None,
        "hint":         "Заводской номер — кликай границы между цифрами",
        "default_mode": "click",
    },
    "marker_id": {
        "label":        "Marker ID",
        "color":        "#ffaa00",
        "crops_dir":    f"{BASE_DIR}/marker_id/images",
        "labels_dir":   f"{BASE_DIR}/marker_id/labels",
        "fixed_len":    None,
        "hint":         "Написано маркером — кликай границы между цифрами\n"
                        "Или используй режим «Боксы» для перекрывающихся цифр",
        "default_mode": "click",
    },
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

KEY_TO_DIGIT = {
    "q": "7", "w": "8", "e": "9",
    "a": "4", "s": "5", "d": "6",
    "z": "1", "x": "2", "c": "3",
    " ": "0",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Ленивая загрузка EasyOCR
# ═══════════════════════════════════════════════════════════════════════════════

_easy_ocr = None
_easyocr_lock = threading.Lock()


def get_easyocr():
    global _easy_ocr
    with _easyocr_lock:
        if _easy_ocr is None:
            import easyocr
            _easy_ocr = easyocr.Reader(['en'], gpu=False)
    return _easy_ocr


# ═══════════════════════════════════════════════════════════════════════════════
#  Предсказание
# ═══════════════════════════════════════════════════════════════════════════════

def predict_serial_id(img_bgr: np.ndarray) -> str:
    """
    Запускает EasyOCR на кропе serial_id.
    Возвращает строку только из цифр, или '' если ничего не нашёл.
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    reader  = get_easyocr()
    results = reader.readtext(
        img_rgb,
        detail=1,
        allowlist='0123456789',
        paragraph=False,
    )
    texts = [text for (_, text, conf) in results if conf > 0.3]
    digits = re.sub(r'[^0-9]', '', ''.join(texts))
    return digits


def marker_id_core(filename: str) -> str:
    """
    Из имени файла вида '1300000013__marker_id_1.jpeg' извлекает core —
    последние ненулевые цифры числа перед '__'.
      1300000013  → '13'
      1300000726  → '726'
      1300000075  → '75'
    Возвращает строку или '' если не нашёл.
    """
    stem = Path(filename).stem
    m    = re.match(r'^(\d+)', stem)
    if not m:
        return ''
    number_str = m.group(1)
    zm = re.search(r'0+([^0].*)$', number_str)
    if zm:
        return zm.group(1)
    return number_str.lstrip('0') or '0'


def marker_id_candidates(filename: str) -> dict:
    """Оставлено для совместимости — не используется."""
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
#  Вспомогательные функции (bbox / yolo)
# ═══════════════════════════════════════════════════════════════════════════════

def _pil_font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except Exception:
            return ImageFont.load_default()


def find_digit_boxes_by_projection(gray_crop):
    _, binary = cv2.threshold(gray_crop, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = np.sum(binary, axis=0).astype(float)

    if AUTO_SMOOTH_KERNEL > 1:
        kernel = np.ones(AUTO_SMOOTH_KERNEL) / AUTO_SMOOTH_KERNEL
        proj   = np.convolve(proj, kernel, mode="same")

    threshold = proj.max() * AUTO_PROJ_THRESHOLD
    in_digit  = False
    groups    = []
    start     = 0
    for x, val in enumerate(proj):
        if val > threshold and not in_digit:
            start    = x
            in_digit = True
        elif val <= threshold and in_digit:
            groups.append((start, x))
            in_digit = False
    if in_digit:
        groups.append((start, len(proj) - 1))

    min_w  = gray_crop.shape[1] * AUTO_MIN_WIDTH_FRAC
    groups = [(x1, x2) for x1, x2 in groups if (x2 - x1) >= min_w]
    return groups


def uniform_split(width, n):
    step = width / n
    return [(int(i * step), int((i + 1) * step)) for i in range(n)]


def splits_to_boxes(splits, img_w):
    boundaries = [0] + sorted(splits) + [img_w]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def pair_clicks_to_boxes(clicks):
    boxes = []
    for i in range(0, len(clicks) - 1, 2):
        x1, x2 = sorted((clicks[i], clicks[i + 1]))
        boxes.append((x1, x2))
    return boxes


def boxes_to_yolo(boxes, img_w, img_h, digit_str):
    lines = []
    for (x1, x2), digit_char in zip(boxes, digit_str):
        cx       = (x1 + x2) / 2 / img_w
        bw       = (x2 - x1) / img_w
        class_id = int(digit_char)
        lines.append(f"{class_id} {cx:.6f} 0.500000 {bw:.6f} {YOLO_BOX_HEIGHT:.6f}")
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
#  Функции отрисовки
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_digit_centered(draw, x1, x2, y_center, ch, font):
    bbox = draw.textbbox((0, 0), ch, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]
    cx   = (x1 + x2) // 2
    tx   = cx - tw // 2
    ty   = y_center - th // 2
    # Толстая чёрная обводка — рисуем символ со смещением во все стороны
    for dx, dy in [(-2,-2),(-2,0),(-2,2),(0,-2),(0,2),(2,-2),(2,0),(2,2),
                   (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        draw.text((tx + dx, ty + dy), ch, fill="#000000", font=font)
    # Основной белый текст поверх
    draw.text((tx, ty), ch, fill=DIGIT_LABEL_COLOR, font=font)


def draw_preview(pil_img, boxes, digit_str, accent_color):
    draw = ImageDraw.Draw(pil_img)
    font = _pil_font(DIGIT_LABEL_FONT_SIZE)
    nw, nh = pil_img.size
    for i, ((x1, x2), ch) in enumerate(zip(boxes, digit_str)):
        color = BOX_COLORS_LINE[i % len(BOX_COLORS_LINE)]
        draw.rectangle([x1, 2, x2, nh - 2], outline=color, width=BOX_OUTLINE_WIDTH)
        _draw_digit_centered(draw, x1, x2, nh // 2, ch, font)
    return pil_img


def draw_splits(pil_img, splits_scaled, accent_color):
    draw   = ImageDraw.Draw(pil_img)
    nw, nh = pil_img.size
    r      = SPLIT_DOT_RADIUS
    for x in splits_scaled:
        draw.line([(x, 0), (x, nh)], fill=accent_color, width=SPLIT_LINE_WIDTH)
        draw.ellipse([x - r, 0, x + r, r * 2], fill=accent_color)
    return pil_img


def draw_ghost_line(pil_img, ghost_x, accent_color, mode):
    if ghost_x is None:
        return pil_img
    draw   = ImageDraw.Draw(pil_img)
    nw, nh = pil_img.size
    step   = GHOST_STEP_CLICK if mode == "click" else GHOST_STEP_PAIRS
    gap    = GHOST_GAP_CLICK  if mode == "click" else GHOST_GAP_PAIRS
    d      = GHOST_DIAMOND_SIZE
    for y in range(0, nh, step):
        draw.line([(ghost_x, y), (ghost_x, min(y + gap, nh))],
                  fill=accent_color, width=1)
    draw.polygon([(ghost_x, 0),   (ghost_x + d, d * 1.2),
                  (ghost_x, d * 2.4), (ghost_x - d, d * 1.2)],
                 fill=accent_color)
    return pil_img


def draw_pair_boxes(pil_img, all_boxes_scaled, digit_str, pending_x, accent_color):
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw_o  = ImageDraw.Draw(overlay)
    draw    = ImageDraw.Draw(pil_img)
    nw, nh  = pil_img.size

    font_big   = _pil_font(DIGIT_LABEL_FONT_SIZE)
    font_small = _pil_font(max(8, DIGIT_LABEL_FONT_SIZE - 6))

    digits = list(digit_str) if digit_str else []

    for i, (x1, x2) in enumerate(all_boxes_scaled):
        c_rgba = BOX_COLORS_RGBA[i % len(BOX_COLORS_RGBA)]
        c_line = BOX_COLORS_LINE[i % len(BOX_COLORS_LINE)]

        draw_o.rectangle([x1, 2, x2, nh - 2], fill=c_rgba)
        draw.rectangle([x1, 2, x2, nh - 2], outline=c_line, width=BOX_OUTLINE_WIDTH)

        if i < len(digits):
            _draw_digit_centered(draw, x1, x2, nh // 2, digits[i], font_big)
        else:
            draw.text((x1 + 3, 3), f"#{i+1}", fill=c_line, font=font_small)

    if pending_x is not None:
        for y in range(0, nh, 8):
            draw.line([(pending_x, y), (pending_x, min(y + 4, nh))],
                      fill=accent_color, width=2)
        draw.ellipse([pending_x - SPLIT_DOT_RADIUS, 0,
                      pending_x + SPLIT_DOT_RADIUS, SPLIT_DOT_RADIUS * 2],
                     fill=accent_color)

    pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")
    return pil_img


# ═══════════════════════════════════════════════════════════════════════════════
#  Приложение
# ═══════════════════════════════════════════════════════════════════════════════

class DigitLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Meter Digit Labeler")
        self.root.configure(bg=COLOR_BG)
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.root.resizable(True, True)

        self.current_class_key = tk.StringVar(value="gas_meter")
        self.image_paths  = []
        self.current_idx  = 0
        self.current_boxes= []
        self.current_img  = None
        self.photo        = None

        self.bbox_mode    = tk.StringVar(value="auto")
        self.click_splits = []
        self.pair_clicks  = []
        self._ghost_x     = None

        self._img_scale    = 1.0
        self._img_offset_x = 0
        self._img_offset_y = 0

        # ── Предсказание ──────────────────────────────────────────────────────
        # serial_id: результат фонового потока
        self._serial_prediction      = ""   # строка цифр или ''
        self._serial_pred_loading    = False
        # marker_id: словарь кандидатов {длина: строка}
        self._marker_candidates      = {}   # не используется
        self._marker_core            = ""
        # токен текущего изображения — чтобы не применять устаревший результат
        self._prediction_token       = 0
        # gas_meter OCR: включается вручную, не сбрасывается при смене фото
        self.ocr_enabled             = tk.BooleanVar(value=False)

        for cfg in CLASSES.values():
            Path(cfg["crops_dir"]).mkdir(parents=True, exist_ok=True)
            Path(cfg["labels_dir"]).mkdir(parents=True, exist_ok=True)

        self._build_ui()
        self._on_class_change()

    @property
    def cls(self):    return CLASSES[self.current_class_key.get()]
    @property
    def accent(self): return self.cls["color"]

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        top = tk.Frame(self.root, bg=COLOR_BG_DARK, pady=6)
        top.pack(fill="x")

        tk.Label(top, text="METER LABELER",
                 bg=COLOR_BG_DARK, fg="#ffffff",
                 font=(FONT_MONO, FONT_TITLE_SIZE, "bold")).pack(side="left", padx=16)

        self.progress_var = tk.StringVar(value="")
        tk.Label(top, textvariable=self.progress_var,
                 bg=COLOR_BG_DARK, fg=COLOR_FG_DIM,
                 font=(FONT_MONO, FONT_TAB_SIZE)).pack(side="left", padx=12)

        tk.Button(top, text="📂 Сменить папку",
                  command=self._choose_folder,
                  bg=COLOR_BG_TAB_ACT, fg=COLOR_FG_CTRL,
                  relief="flat", padx=10, pady=3,
                  cursor="hand2").pack(side="right", padx=12)

        tabs = tk.Frame(self.root, bg=COLOR_BG)
        tabs.pack(fill="x")
        self.tab_buttons = {}
        for key, cfg in CLASSES.items():
            btn = tk.Button(tabs, text=cfg["label"],
                            command=lambda k=key: self._switch_class(k),
                            bg=COLOR_BG_TAB, fg=COLOR_FG_DIMMER,
                            relief="flat", padx=18, pady=8,
                            cursor="hand2",
                            font=(FONT_MONO, FONT_TAB_SIZE, "bold"),
                            borderwidth=0)
            btn.pack(side="left")
            self.tab_buttons[key] = btn

        tk.Frame(self.root, bg=COLOR_SEPARATOR, height=1).pack(fill="x")

        center = tk.Frame(self.root, bg=COLOR_BG)
        center.pack(fill="both", expand=True, padx=12, pady=8)

        img_frame = tk.Frame(center, bg=COLOR_BG_DARK)
        img_frame.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(img_frame, bg=COLOR_BG_DARK, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)
        self.canvas.bind("<Motion>",   self._on_canvas_motion)
        self.canvas.bind("<Leave>",    self._on_canvas_leave)

        tk.Label(img_frame, text="", bg=COLOR_BG_DARK,
                 fg=COLOR_FG_DIMMER,
                 font=(FONT_MONO, FONT_SMALL_SIZE)).pack(pady=2)
        self.filename_label = img_frame.winfo_children()[-1]

        ctrl = tk.Frame(center, bg=COLOR_BG, width=CTRL_PANEL_W)
        ctrl.pack(side="right", fill="y", padx=(12, 0))
        ctrl.pack_propagate(False)

        wrap = CTRL_PANEL_W - 15

        self.hint_label = tk.Label(ctrl, text="", bg=COLOR_BG, fg=COLOR_FG_HINT,
                                   font=(FONT_MONO, FONT_SMALL_SIZE),
                                   wraplength=wrap)
        self.hint_label.pack(anchor="w", pady=(10, 4))

        # ── Строка предсказания ───────────────────────────────────────────────
        pred_frame = tk.Frame(ctrl, bg=COLOR_PREDICT,
                              highlightbackground="#223322", highlightthickness=1)
        pred_frame.pack(fill="x", pady=(0, 4))
        tk.Label(pred_frame, text="OCR:", bg=COLOR_PREDICT,
                 fg=COLOR_FG_DIM, font=(FONT_MONO, FONT_SMALL_SIZE)).pack(side="left", padx=6)
        self.pred_var = tk.StringVar(value="—")
        self.pred_label = tk.Label(pred_frame, textvariable=self.pred_var,
                                   bg=COLOR_PREDICT, fg="#44cc88",
                                   font=(FONT_MONO, FONT_CTRL_SIZE, "bold"))
        self.pred_label.pack(side="left", padx=4)
        self.pred_status = tk.Label(pred_frame, text="", bg=COLOR_PREDICT,
                                    fg=COLOR_FG_DIMMER,
                                    font=(FONT_MONO, FONT_SMALL_SIZE))
        self.pred_status.pack(side="left", padx=4)

        # ── OCR чекбокс для gas_meter ─────────────────────────────────────────
        self.ocr_cb = tk.Checkbutton(
            ctrl, text="OCR предсказание (gas meter)",
            variable=self.ocr_enabled,
            command=self._on_ocr_toggle,
            bg=COLOR_BG, fg=COLOR_FG_CTRL,
            selectcolor=COLOR_ENTRY_BG,
            activebackground=COLOR_BG,
            font=(FONT_MONO, FONT_SMALL_SIZE))
        self.ocr_cb.pack(anchor="w", pady=(0, 4))

        # ── Поле ввода ────────────────────────────────────────────────────────
        ef = tk.Frame(ctrl, bg=COLOR_ENTRY_BG,
                      highlightbackground="#222233", highlightthickness=1)
        ef.pack(fill="x", pady=4)
        self.digit_var   = tk.StringVar()
        self.digit_entry = tk.Entry(ef, textvariable=self.digit_var,
                                    font=(FONT_MONO, FONT_ENTRY_SIZE, "bold"),
                                    bg=COLOR_ENTRY_BG, fg="#00ff88",
                                    insertbackground="#00ff88",
                                    relief="flat", width=8, justify="center")
        self.digit_entry.pack(padx=8, pady=10)
        self.digit_entry.bind("<KeyRelease>", self._on_digit_change)
        self.digit_entry.bind("<Return>",     lambda e: self._save_and_next())
        self.digit_entry.bind("<KP_Enter>",   lambda e: self._save_and_next())
        self.digit_entry.bind("<KeyPress>",   self._on_key_press)

        self.len_label = tk.Label(ctrl, text="", bg=COLOR_BG, fg=COLOR_FG_HINT,
                                  font=(FONT_MONO, FONT_CTRL_SIZE))
        self.len_label.pack()

        tk.Label(ctrl, text="q=7 w=8 e=9 | a=4 s=5 d=6 | z=1 x=2 c=3 | ␣=0",
                 bg=COLOR_BG, fg=COLOR_KEYHINT,
                 font=(FONT_MONO, FONT_SMALL_SIZE),
                 justify="left").pack(anchor="w", padx=4, pady=(0, 6))

        # ── Режим bbox ────────────────────────────────────────────────────────
        mf = tk.Frame(ctrl, bg=COLOR_BG)
        mf.pack(fill="x", pady=4)
        tk.Label(mf, text="BBOX РЕЖИМ", bg=COLOR_BG, fg=COLOR_FG_DIMMER,
                 font=(FONT_MONO, FONT_SMALL_SIZE, "bold")).pack(anchor="w")
        for val, txt in [
            ("auto",      "Авто (проекция)"),
            ("uniform",   "Равномерный"),
            ("click",     "Клики  — ЛКМ граница, ПКМ удалить"),
            ("box_pairs", "Боксы  — 2 клика = 1 цифра, overlap OK"),
        ]:
            tk.Radiobutton(mf, text=txt, variable=self.bbox_mode, value=val,
                           bg=COLOR_BG, fg=COLOR_FG_CTRL,
                           selectcolor=COLOR_ENTRY_BG,
                           activebackground=COLOR_BG,
                           font=(FONT_MONO, FONT_CTRL_SIZE),
                           wraplength=wrap, justify="left",
                           command=self._on_mode_change).pack(anchor="w", pady=1)

        self.bbox_status = tk.Label(ctrl, text="", bg=COLOR_BG, fg=COLOR_FG_DIM,
                                    font=(FONT_MONO, FONT_SMALL_SIZE),
                                    wraplength=wrap, justify="left")
        self.bbox_status.pack(anchor="w", pady=2)

        tk.Button(ctrl, text="🗑  Сбросить клики  [C]",
                  command=self._clear_clicks,
                  bg=COLOR_BTN_CLEAR, fg="#ffaa44",
                  relief="flat", pady=5, cursor="hand2",
                  font=(FONT_MONO, FONT_CTRL_SIZE)).pack(fill="x", pady=2)

        self.btn_save = tk.Button(ctrl, text="✅  Сохранить  [Enter]",
                                  command=self._save_and_next,
                                  bg=COLOR_BTN_SAVE, fg="#00ff88",
                                  relief="flat", pady=9, cursor="hand2",
                                  font=(FONT_MONO, FONT_TAB_SIZE, "bold"))
        self.btn_save.pack(fill="x", pady=(10, 3))

        tk.Button(ctrl, text="⏭  Пропустить  [→]",
                  command=self._skip, bg=COLOR_BTN_SKIP, fg=COLOR_FG_CTRL,
                  relief="flat", pady=7, cursor="hand2",
                  font=(FONT_MONO, FONT_CTRL_SIZE)).pack(fill="x", pady=2)

        tk.Button(ctrl, text="⏮  Назад  [←]",
                  command=self._prev, bg=COLOR_BTN_SKIP, fg=COLOR_FG_CTRL,
                  relief="flat", pady=7, cursor="hand2",
                  font=(FONT_MONO, FONT_CTRL_SIZE)).pack(fill="x", pady=2)

        tk.Button(ctrl, text="🗑  Удалить кроп",
                  command=self._delete_current,
                  bg=COLOR_BTN_DEL, fg="#ff6666",
                  relief="flat", pady=6, cursor="hand2",
                  font=(FONT_MONO, FONT_CTRL_SIZE)).pack(fill="x", pady=(10, 2))

        tk.Label(self.root,
                 text="Enter — сохранить    ← → — навигация    Tab — класс    C — сбросить клики",
                 bg=COLOR_BG, fg=COLOR_FG_DIMMER,
                 font=(FONT_MONO, FONT_SMALL_SIZE)).pack(side="bottom", pady=3)

        self.root.bind("<Right>", lambda e: self._skip())
        self.root.bind("<Left>",  lambda e: self._prev())
        self.root.bind("<Tab>",   lambda e: self._cycle_class())
        self.root.bind("c",       lambda e: self._on_global_key("c"))
        self.root.bind("C",       lambda e: self._clear_clicks())

        self._update_tab_styles()

    # ── Клавиши-цифры ────────────────────────────────────────────────────────

    def _on_key_press(self, event):
        ch = event.char
        if ch in KEY_TO_DIGIT:
            digit = KEY_TO_DIGIT[ch]
            try:
                s = self.digit_entry.index(tk.SEL_FIRST)
                e = self.digit_entry.index(tk.SEL_LAST)
                self.digit_entry.delete(s, e)
                self.digit_entry.insert(s, digit)
            except tk.TclError:
                self.digit_entry.insert(self.digit_entry.index(tk.INSERT), digit)
            self._on_digit_change()
            return "break"

    def _on_global_key(self, ch):
        if self.root.focus_get() != self.digit_entry:
            self._clear_clicks()

    # ── Переключение классов ──────────────────────────────────────────────────

    def _switch_class(self, key):
        self.current_class_key.set(key)
        self._on_class_change()

    def _cycle_class(self):
        keys = list(CLASSES.keys())
        cur  = self.current_class_key.get()
        self._switch_class(keys[(keys.index(cur) + 1) % len(keys)])

    def _on_class_change(self):
        self._update_tab_styles()
        self.hint_label.config(text=self.cls["hint"])
        self.digit_entry.config(fg=self.accent, insertbackground=self.accent)
        self.btn_save.config(fg=self.accent)
        self.bbox_mode.set(self.cls["default_mode"])
        self._clear_clicks(render=False)
        self._reset_prediction()
        self._load_image_list()
        if self.image_paths:
            self._show_image(0)
        else:
            self._show_empty()
        self.digit_entry.focus()

    def _update_tab_styles(self):
        cur = self.current_class_key.get()
        for key, btn in self.tab_buttons.items():
            btn.config(
                bg=COLOR_BG_TAB_ACT if key == cur else COLOR_BG,
                fg=CLASSES[key]["color"] if key == cur else COLOR_FG_DIMMER,
            )

    # ── Предсказание ─────────────────────────────────────────────────────────

    def _reset_prediction(self):
        self._serial_prediction   = ""
        self._serial_pred_loading = False
        self._marker_candidates   = {}
        self._marker_core         = ""
        self._prediction_token   += 1
        self.pred_var.set("—")
        self.pred_status.config(text="")

    def _start_prediction(self, img_path: str, img_bgr: np.ndarray):
        """Запускает нужный вид предсказания для текущего класса."""
        cls_key = self.current_class_key.get()
        self._reset_prediction()
        token = self._prediction_token   # сохраняем токен

        if cls_key == "serial_id":
            self._serial_pred_loading = True
            self.pred_var.set("…")
            self.pred_status.config(text="OCR...")

            def _run():
                result = predict_serial_id(img_bgr)
                self.root.after(0, lambda: self._on_serial_prediction(result, token))

            threading.Thread(target=_run, daemon=True).start()

        elif cls_key == "marker_id":
            core = marker_id_core(Path(img_path).name)
            self._marker_core = core
            if core:
                self.pred_var.set(core)
                self.pred_status.config(text="из имени файла")
            else:
                self.pred_var.set("?")
                self.pred_status.config(text="нет в имени")
            self._try_apply_marker_prediction()

        elif cls_key == "gas_meter" and self.ocr_enabled.get():
            self._launch_gas_meter_ocr(img_bgr, token)

        else:
            self.pred_var.set("—")
            self.pred_status.config(text="")

    def _launch_gas_meter_ocr(self, img_bgr: np.ndarray, token: int):
        """Запускает EasyOCR для gas_meter в фоне."""
        self._serial_pred_loading = True
        self.pred_var.set("…")
        self.pred_status.config(text="OCR...")

        def _run():
            result = predict_serial_id(img_bgr)   # та же функция — CLAHE + EasyOCR
            self.root.after(0, lambda: self._on_gas_meter_prediction(result, token))

        threading.Thread(target=_run, daemon=True).start()

    def _on_gas_meter_prediction(self, result: str, token: int):
        """Результат OCR для gas_meter."""
        if token != self._prediction_token:
            return
        self._serial_pred_loading = False
        self._serial_prediction   = result
        if result:
            self.pred_var.set(result)
            self.pred_status.config(text=f"({len(result)} цифр)")
        else:
            self.pred_var.set("?")
            self.pred_status.config(text="не распознано")
        # gas_meter всегда 5 цифр, боксы авто — просто подставляем
        if result and len(result) == 5 and not self.digit_var.get():
            self._apply_prediction(result)

    def _on_ocr_toggle(self):
        """Чекбокс OCR для gas_meter: при включении сразу запускаем для текущего фото."""
        if self.ocr_enabled.get() and self.current_img is not None:
            if self.current_class_key.get() == "gas_meter":
                token = self._prediction_token
                self._launch_gas_meter_ocr(self.current_img, token)
        elif not self.ocr_enabled.get():
            self.pred_var.set("—")
            self.pred_status.config(text="")

    def _on_serial_prediction(self, result: str, token: int):
        """Вызывается из главного потока когда EasyOCR закончил."""
        if token != self._prediction_token:
            return   # изображение уже сменилось
        self._serial_pred_loading = False
        self._serial_prediction   = result
        if result:
            self.pred_var.set(result)
            self.pred_status.config(text=f"({len(result)} цифр)")
        else:
            self.pred_var.set("?")
            self.pred_status.config(text="не распознано")
        self._try_apply_serial_prediction()

    def _try_apply_serial_prediction(self):
        """Подставляет предсказание serial_id если длина совпадает с боксами."""
        if self.current_class_key.get() != "serial_id":
            return
        pred = self._serial_prediction
        if not pred:
            return
        n_boxes = len(self.current_boxes)
        if n_boxes == 0:
            return
        if len(pred) == n_boxes and not self.digit_var.get():
            self._apply_prediction(pred)

    def _try_apply_marker_prediction(self):
        """Подставляет core с нужным количеством ведущих нулей по числу боксов.
        Перезаписывает поле если там уже стоит предыдущий вариант предсказания."""
        if self.current_class_key.get() != "marker_id":
            return
        core = getattr(self, '_marker_core', '')
        if not core:
            return
        n_boxes = len(self.current_boxes)
        if n_boxes == 0 or n_boxes < len(core):
            return
        value   = core.zfill(n_boxes)
        current = self.digit_var.get()
        # Перезаписываем если поле пустое ИЛИ там стоит предыдущий вариант того же core
        if not current or current.lstrip('0') == core.lstrip('0'):
            self.digit_var.set(value)
            self._update_len_label()
            self._render_canvas()

    def _apply_prediction(self, value: str):
        """Вставляет значение в поле ввода (только если поле пустое)."""
        if self.digit_var.get():
            return
        self.digit_var.set(value)
        self._update_len_label()
        self._render_canvas()

    # ── Призрак мыши ─────────────────────────────────────────────────────────

    def _on_canvas_motion(self, event):
        mode = self.bbox_mode.get()
        if mode not in ("click", "box_pairs") or self.current_img is None:
            self._ghost_x = None
            return
        h, w = self.current_img.shape[:2]
        self._ghost_x = max(0, min(int(self._cx2img(event.x)), w - 1))
        self._render_canvas()

    def _on_canvas_leave(self, event):
        self._ghost_x = None
        if self.current_img is not None:
            self._render_canvas()

    # ── Клики ────────────────────────────────────────────────────────────────

    def _cx2img(self, canvas_x):
        return (canvas_x - self._img_offset_x) / self._img_scale

    def _on_canvas_click(self, event):
        mode = self.bbox_mode.get()
        if self.current_img is None or mode not in ("click", "box_pairs"):
            return
        h, w  = self.current_img.shape[:2]
        img_x = max(1, min(int(self._cx2img(event.x)), w - 1))
        if mode == "click":
            self.click_splits.append(img_x)
            self._update_boxes_from_clicks()
        else:
            self.pair_clicks.append(img_x)
            self._update_boxes_from_pairs()
        # После изменения боксов — пробуем подставить предсказание
        self._try_apply_prediction_for_class()
        self._render_canvas()

    def _on_canvas_right_click(self, event):
        mode  = self.bbox_mode.get()
        if self.current_img is None:
            return
        img_x = self._cx2img(event.x)
        if mode == "click" and self.click_splits:
            idx = min(range(len(self.click_splits)),
                      key=lambda i: abs(self.click_splits[i] - img_x))
            self.click_splits.pop(idx)
            self._update_boxes_from_clicks()
            self._render_canvas()
        elif mode == "box_pairs" and self.pair_clicks:
            idx  = min(range(len(self.pair_clicks)),
                       key=lambda i: abs(self.pair_clicks[i] - img_x))
            pi   = idx // 2
            del self.pair_clicks[pi * 2: pi * 2 + 2]
            self._update_boxes_from_pairs()
            self._render_canvas()

    def _try_apply_prediction_for_class(self):
        """Диспетчер: пробует подставить предсказание после изменения боксов."""
        cls_key = self.current_class_key.get()
        if cls_key == "serial_id":
            self._try_apply_serial_prediction()
        elif cls_key == "marker_id":
            self._try_apply_marker_prediction()

    def _clear_clicks(self, render=True):
        self.click_splits = []
        self.pair_clicks  = []
        self._update_boxes_from_clicks()
        if render:
            self._render_canvas()

    def _update_boxes_from_clicks(self):
        if self.current_img is None:
            return
        h, w  = self.current_img.shape[:2]
        n     = len(self.click_splits)
        self.current_boxes = splits_to_boxes(self.click_splits, w)
        self.bbox_status.config(
            text=f"🖱 Разделителей: {n}  →  {n+1} цифр\nЛКМ — добавить, ПКМ — удалить",
            fg="#ffaa44" if n == 0 else "#44ff88")

    def _update_boxes_from_pairs(self):
        if self.current_img is None:
            return
        n  = len(self.pair_clicks)
        nc = n // 2
        self.current_boxes = pair_clicks_to_boxes(self.pair_clicks)
        pend = "⚡ ожидаю 2-й клик..." if n % 2 == 1 else ""
        self.bbox_status.config(
            text=f"🖱 Боксов: {nc}  (кликов: {n})\n"
                 f"ЛКМ — левый/правый край, ПКМ — удалить пару\n{pend}",
            fg="#ffaa44" if nc == 0 else "#44ff88")

    def _on_mode_change(self):
        self._clear_clicks(render=False)
        self._recompute_boxes()

    # ── Файлы ────────────────────────────────────────────────────────────────

    def _choose_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку с кропами (или базовую папку)")
        if folder:
            p = Path(folder)
            cls_key = self.current_class_key.get()
            
            # Умное определение папки с лейблами
            if (p / "images").is_dir() and (p / "labels").is_dir():
                # Выбрана базовая папка, внутри которой есть и images, и labels
                CLASSES[cls_key]["crops_dir"] = str(p / "images")
                CLASSES[cls_key]["labels_dir"] = str(p / "labels")
            elif p.name.lower() in ("images", "crops", "train", "val", "test"):
                # Выбрана папка images (или аналогичная), лейблы ищем рядом
                CLASSES[cls_key]["crops_dir"] = str(p)
                CLASSES[cls_key]["labels_dir"] = str(p.parent / "labels")
            else:
                # Во всех остальных случаях сохраняем/ищем лейблы прямо в выбранной папке
                CLASSES[cls_key]["crops_dir"] = str(p)
                CLASSES[cls_key]["labels_dir"] = str(p)
                
            # Гарантируем, что папка для лейблов существует, чтобы не было ошибок
            Path(CLASSES[cls_key]["labels_dir"]).mkdir(parents=True, exist_ok=True)
            
            self._load_image_list()
            if self.image_paths:
                self._show_image(0)

    def _load_image_list(self):
        p = Path(self.cls["crops_dir"])
        if not p.exists():
            self.image_paths = []
            self._update_progress()
            return
            
        # Безопасно проверяем наличие папки с лейблами
        labels_p = Path(self.cls["labels_dir"])
        if labels_p.exists():
            done = {Path(f).stem for f in labels_p.glob("*.txt")}
        else:
            done = set()
            
        self.image_paths = sorted([
            str(f) for f in p.iterdir()
            if f.suffix.lower() in EXTENSIONS and f.stem not in done
        ])
        self.current_idx = 0
        self._update_progress()

    def _update_progress(self):
        # Безопасно считаем количество готовых лейблов
        labels_p = Path(self.cls["labels_dir"])
        if labels_p.exists():
            done = len(list(labels_p.glob("*.txt")))
        else:
            done = 0
            
        self.progress_var.set(
            f"{self.cls['label']}   осталось: {len(self.image_paths)}  |  готово: {done}")

    # ── Отображение ───────────────────────────────────────────────────────────

    def _show_image(self, idx):
        if not self.image_paths:
            self._show_empty(); return
        self.current_idx = max(0, min(idx, len(self.image_paths) - 1))
        img_path = self.image_paths[self.current_idx]
        img_bgr  = cv2.imread(img_path)
        if img_bgr is None:
            self._skip(); return
        self.current_img = img_bgr
        self.digit_var.set("")
        self._clear_clicks(render=False)
        self._recompute_boxes()
        self.filename_label.config(text=Path(img_path).name)
        self._update_len_label()
        # Запускаем предсказание для нужных классов
        self._start_prediction(img_path, img_bgr)
        self.digit_entry.focus()

    def _recompute_boxes(self):
        if self.current_img is None:
            return
        mode  = self.bbox_mode.get()
        h, w  = self.current_img.shape[:2]

        if mode == "click":
            self._update_boxes_from_clicks()
            self._render_canvas(); return
        if mode == "box_pairs":
            self._update_boxes_from_pairs()
            self._render_canvas(); return

        digit_str = self.digit_var.get().strip()
        n = len(digit_str) if digit_str else (self.cls["fixed_len"] or 5)

        if mode == "auto":
            gray  = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
            boxes = find_digit_boxes_by_projection(gray)
            if len(boxes) == n:
                self.current_boxes = boxes
                self.bbox_status.config(text=f"✅ Авто: {len(boxes)} групп", fg="#44ff88")
            else:
                self.current_boxes = uniform_split(w, n)
                self.bbox_status.config(
                    text=f"⚠️ Авто: {len(boxes)} → равномерный ({n})", fg="#ffaa44")
        else:
            self.current_boxes = uniform_split(w, n)
            self.bbox_status.config(text=f"Равномерно: {n} частей", fg="#888aaa")

        self._render_canvas()

    def _render_canvas(self):
        if self.current_img is None:
            return
        digit_str = self.digit_var.get().strip()
        pil_img   = Image.fromarray(cv2.cvtColor(self.current_img, cv2.COLOR_BGR2RGB))
        h, w      = self.current_img.shape[:2]

        self.canvas.update_idletasks()
        cw    = max(self.canvas.winfo_width(),  400)
        ch    = max(self.canvas.winfo_height(), 200)
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((nw, nh), Image.LANCZOS)

        self._img_scale    = scale
        self._img_offset_x = (cw - nw) // 2
        self._img_offset_y = (ch - nh) // 2

        mode = self.bbox_mode.get()

        if self._ghost_x is not None and mode in ("click", "box_pairs"):
            pil_img = draw_ghost_line(
                pil_img, int(self._ghost_x * scale), self.accent, mode)

        if mode == "click":
            splits_sc = [int(x * scale) for x in self.click_splits]
            draw_splits(pil_img, splits_sc, self.accent)
            if digit_str.isdigit() and len(digit_str) == len(self.current_boxes):
                sc_boxes = [(int(x1*scale), int(x2*scale)) for x1,x2 in self.current_boxes]
                draw_preview(pil_img, sc_boxes, digit_str, self.accent)

        elif mode == "box_pairs":
            sc_boxes  = [(int(x1*scale), int(x2*scale)) for x1,x2 in self.current_boxes]
            pending_x = int(self.pair_clicks[-1] * scale) if len(self.pair_clicks) % 2 == 1 else None
            pil_img   = draw_pair_boxes(pil_img, sc_boxes, digit_str, pending_x, self.accent)

        else:
            sc_boxes = [(int(x1*scale), int(x2*scale)) for x1,x2 in self.current_boxes]
            valid = (digit_str.isdigit() and len(digit_str) > 0 and
                     len(digit_str) == len(self.current_boxes) and
                     (self.cls["fixed_len"] is None or len(digit_str) == self.cls["fixed_len"]))
            if valid:
                draw_preview(pil_img, sc_boxes, digit_str, self.accent)

        self.photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(self._img_offset_x, self._img_offset_y,
                                 anchor="nw", image=self.photo)

    def _show_empty(self):
        self.current_img = None
        self.canvas.delete("all")
        self.canvas.create_text(300, 150,
                                text="Нет изображений\nВыберите папку с кропами",
                                fill="#222244", font=(FONT_MONO, 12), justify="center")
        self.progress_var.set(f"{self.cls['label']}   — всё размечено 🎉")

    # ── Ввод ─────────────────────────────────────────────────────────────────

    def _update_len_label(self):
        val   = self.digit_var.get()
        fixed = self.cls["fixed_len"]
        mode  = self.bbox_mode.get()
        if fixed:
            self.len_label.config(text=f"{len(val)} / {fixed} цифр")
        elif mode in ("click", "box_pairs"):
            self.len_label.config(text=f"{len(val)} цифр  (боксов: {len(self.current_boxes)})")
        else:
            self.len_label.config(text=f"{len(val)} цифр")

    def _on_digit_change(self, event=None):
        val     = self.digit_var.get()
        fixed   = self.cls["fixed_len"]
        max_len = fixed if fixed else 20
        clean   = "".join(c for c in val if c.isdigit())[:max_len]
        if clean != val:
            self.digit_var.set(clean)
            self.digit_entry.icursor(len(clean))
        self._update_len_label()
        mode = self.bbox_mode.get()
        if mode not in ("click", "box_pairs"):
            if self.cls["fixed_len"] is None:
                self._recompute_boxes()
            else:
                self._render_canvas()
        else:
            self._render_canvas()

    # ── Действия ─────────────────────────────────────────────────────────────

    def _save_and_next(self):
        digit_str = self.digit_var.get().strip()
        fixed     = self.cls["fixed_len"]
        if not digit_str.isdigit() or len(digit_str) == 0:
            self._flash_entry(); return
        if fixed and len(digit_str) != fixed:
            self._flash_entry(); return
        if self.bbox_mode.get() in ("click", "box_pairs"):
            if len(self.current_boxes) != len(digit_str):
                self.bbox_status.config(
                    text=f"⚠️ Боксов {len(self.current_boxes)}, цифр {len(digit_str)}",
                    fg="#ff4444")
                self._flash_entry(); return

        path = self.image_paths[self.current_idx]
        stem = Path(path).stem
        h, w = self.current_img.shape[:2]
        if len(self.current_boxes) != len(digit_str):
            self.current_boxes = uniform_split(w, len(digit_str))

        lines = boxes_to_yolo(self.current_boxes, w, h, digit_str)
        with open(Path(self.cls["labels_dir"]) / f"{stem}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")

        self.image_paths.pop(self.current_idx)
        self._update_progress()
        if not self.image_paths:
            self._show_empty()
        else:
            self._show_image(min(self.current_idx, len(self.image_paths) - 1))

    def _flash_entry(self):
        self.digit_entry.config(bg=COLOR_ENTRY_ERR)
        self.root.after(300, lambda: self.digit_entry.config(bg=COLOR_ENTRY_BG))

    def _skip(self):
        if self.image_paths:
            self._show_image((self.current_idx + 1) % len(self.image_paths))

    def _prev(self):
        if self.image_paths:
            self._show_image((self.current_idx - 1) % len(self.image_paths))

    def _delete_current(self):
        if not self.image_paths:
            return
        path = self.image_paths[self.current_idx]
        if messagebox.askyesno("Удалить?", f"Удалить файл?\n{Path(path).name}"):
            os.remove(path)
            self.image_paths.pop(self.current_idx)
            self._update_progress()
            if not self.image_paths:
                self._show_empty()
            else:
                self._show_image(min(self.current_idx, len(self.image_paths) - 1))


def main():
    root = tk.Tk()
    app  = DigitLabelerApp(root)
    root.bind("<Configure>",
              lambda e: app._render_canvas() if e.widget == root else None)
    root.mainloop()


if __name__ == "__main__":
    main()