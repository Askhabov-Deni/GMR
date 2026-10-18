"""
src/gmr/overlay.py — надпись контролёра поверх фото: «24317», «пок51234»,
«лс17777», набранные в редакторе телефона (решение владельца 2026-10-21:
найти и прочитать без нейросети; docs/BACKLOG.md).

Найти. Буквы надписи залиты одним ярким цветом: на живом фото такого почти
не бывает. Для каждого яркого цвета на фото — пятна этого цвета; «буква» —
пятно разумного размера, в котором почти весь цвет — один; строка — не меньше
3 «букв» одной высоты подряд. Пломбы, бирки, изолента строку не образуют.
Белый и чёрный текст, текст на плашке не ищутся (владелец: запретить).

Прочитать. Каждая буква сравнивается с образцами символов 12 свободных
шрифтов телефонных редакторов (Android, iPhone, WhatsApp — models/overlay/
glyphs.npz, make_bank). Ответ — самый похожий образец; запас до лучшего
другого символа — уверенность (меньше MARGIN — «не уверен»).

Разобрать. «пок…», «пк…» — показание, «лс…» — лицевой счёт, без приписки —
не известно (решает таблица: число без приписки обычно лицевой счёт).
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

BANK_PATH = Path(__file__).resolve().parents[2] / "models" / "overlay" / "glyphs.npz"
CHARS = "0123456789покслПОКЛС"
GLYPH_H, GLYPH_W = 48, 36
MAX_SIDE = 1280             # ищется на уменьшенном фото: надпись крупная
MARGIN = 0.05               # запас уверенности символа (проверено на синтетике, 2026-10-21)

_HUE_BINS, _MIN_COLOR_PX = 30, 200
_NEAR, _CORE = 35, 18       # расстояние в Lab: пиксель того же цвета / «чистый» цвет надписи


@dataclass
class Line:
    text: str               # прочитано, буквы — строчные: «пок51234»
    margin: float           # наименьший запас уверенности среди символов
    box: tuple              # x, y, w, h на уменьшенном фото
    crop: np.ndarray = field(repr=False)   # строка: чёрный текст на белом

    @property
    def sure(self) -> bool:
        return self.margin >= MARGIN


def _norm(mask: np.ndarray) -> np.ndarray:
    """Глиф (True — штрих) → GLYPH_H×GLYPH_W, по рамке, пропорции сохранены."""
    ys, xs = np.nonzero(mask)
    g = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    h, w = g.shape
    s = min(GLYPH_H / h, GLYPH_W / w)
    g = cv2.resize(g, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((GLYPH_H, GLYPH_W), np.float32)
    y0, x0 = (GLYPH_H - g.shape[0]) // 2, (GLYPH_W - g.shape[1]) // 2
    out[y0:y0 + g.shape[0], x0:x0 + g.shape[1]] = g / 255
    return out


def _vector(glyph: np.ndarray) -> np.ndarray:
    """Размытый, без среднего, единичной длины: сравнение — скалярное произведение.
    Размытие прощает толщину штриха и мелкие отличия шрифтов."""
    v = cv2.GaussianBlur(glyph.astype(np.float32), (0, 0), 3).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n else v


class GlyphBank:
    """Образцы символов: вектор, пропорции (высота/ширина) и высота
    относительно цифры того же шрифта (строчные буквы ниже цифр)."""

    def __init__(self, chars, glyphs, aspect, rel_h):
        self.chars = np.asarray(chars)
        self.vectors = np.stack([_vector(g) for g in glyphs])
        self.aspect = np.asarray(aspect, np.float32)
        self.rel_h = np.asarray(rel_h, np.float32)

    @classmethod
    def load(cls, path: Path = BANK_PATH) -> "GlyphBank":
        d = np.load(path)
        return cls(d["chars"], d["glyphs"].astype(np.float32) / 255, d["aspect"], d["rel_h"])

    def read(self, mask: np.ndarray, rel_h: float) -> tuple[str, float]:
        """(символ, запас до лучшего другого символа)."""
        ys, xs = np.nonzero(mask)
        aspect = (np.ptp(ys) + 1) / (np.ptp(xs) + 1)
        score = (self.vectors @ _vector(_norm(mask))
                 - 0.3 * np.abs(np.log(aspect / self.aspect)) - 0.8 * np.abs(rel_h - self.rel_h))
        order = np.argsort(score)[::-1]
        best = self.chars[order[0]]
        other = next(i for i in order if self.chars[i] != best)
        return str(best), float(score[order[0]] - score[other])


def make_bank(fonts, path: Path = BANK_PATH, size: int = 120) -> int:
    """Образцы символов CHARS из файлов шрифтов (.ttf) → path. Символа нет в
    шрифте — пропускается (телефон нарисует его системным шрифтом). Нужны
    Pillow и fontTools (ставятся с ultralytics). Возвращает число образцов."""
    from fontTools.ttLib import TTFont
    from PIL import Image, ImageDraw, ImageFont
    chars, glyphs, aspect, rel_h = [], [], [], []
    for f in fonts:
        cmap = TTFont(str(f)).getBestCmap()
        font = ImageFont.truetype(str(f), size)
        digit_h = None
        for c in CHARS:
            if ord(c) not in cmap:
                continue
            im = Image.new("L", (size * 2, size * 2), 0)
            ImageDraw.Draw(im).text((size // 4, size // 6), c, font=font, fill=255)
            m = np.array(im) > 127
            ys, xs = np.nonzero(m)
            h = np.ptp(ys) + 1
            digit_h = digit_h or h                       # первый символ CHARS — «0»
            chars.append(c.lower())
            glyphs.append((_norm(m) * 255).astype(np.uint8))
            aspect.append(h / (np.ptp(xs) + 1))
            rel_h.append(h / digit_h)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, chars=np.array(chars), glyphs=np.stack(glyphs),
                        aspect=np.array(aspect, np.float32), rel_h=np.array(rel_h, np.float32))
    return len(chars)


def _lines(chars: list) -> list[list]:
    """«Буквы» → строки: одна высота, один уровень, рядом друг с другом, ≥3."""
    groups: list[list] = []
    for b in sorted(chars, key=lambda b: b[0]):
        cy, h = b[1] + b[3] / 2, b[3]
        for g in groups:
            gy, gh = np.median([x[1] + x[3] / 2 for x in g]), max(x[3] for x in g)
            if abs(cy - gy) < 0.35 * gh and 0.45 < h / gh < 1.6:
                g.append(b)
                break
        else:
            groups.append([b])
    out = []
    for g in groups:
        run = [g[0]]
        for b in g[1:]:
            if b[0] - (run[-1][0] + run[-1][2]) <= 1.5 * max(x[3] for x in run):
                run.append(b)
                continue
            if len(run) >= 3:
                out.append(run)
            run = [b]
        if len(run) >= 3:
            out.append(run)
    return out


def find(img: np.ndarray) -> Optional[tuple[np.ndarray, list[list]]]:
    """(номера пятен, строки — списки (x, y, w, h, номер пятна) слева направо)
    для цвета, у которого больше всего «букв» в строках; None — надписи нет.
    img — BGR, уже уменьшенное до MAX_SIDE."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bright = (hsv[..., 1] >= 110) & (hsv[..., 2] >= 90)
    hue = hsv[..., 0].astype(int) // (180 // _HUE_BINS)
    height = img.shape[0]
    best = None
    for b in range(_HUE_BINS):
        sel = bright & (hue == b)
        if sel.sum() < _MIN_COLOR_PX:
            continue
        dist = np.linalg.norm(lab - np.median(lab[sel], axis=0), axis=2)
        mask = cv2.morphologyEx((dist < _NEAR).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lbl, st, _ = cv2.connectedComponentsWithStats(mask)
        chars = []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a < 60 or not 0.025 * height <= h <= 0.35 * height or w > 1.5 * h + 10:
                continue
            if not 0.12 < a / (w * h) < 0.8:                              # штрих, а не сплошное пятно
                continue
            comp = lbl[y:y + h, x:x + w] == i
            if (dist[y:y + h, x:x + w][comp] < _CORE).mean() < 0.35:       # почти весь цвет — один
                continue
            chars.append((x, y, w, h, i))
        lines = _lines(chars)
        score = sum(len(g) for g in lines)
        if lines and (best is None or score > best[0]):
            best = (score, lbl, lines)
    if best is None:
        return None
    return best[1], sorted(best[2], key=lambda g: min(b[1] for b in g))


def read(img: np.ndarray, bank: GlyphBank) -> list[Line]:
    """Строки надписи на фото (BGR), сверху вниз. Пусто — надписи нет."""
    s = MAX_SIDE / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    found = find(img)
    if found is None:
        return []
    lbl, lines = found
    out = []
    for g in lines:
        digit_h = float(max(b[3] for b in g))                          # цифры — самые высокие
        text, margin = "", 1.0
        for x, y, w, h, i in g:
            c, m = bank.read(lbl[y:y + h, x:x + w] == i, h / digit_h)
            text += c
            margin = min(margin, m)
        x0, y0 = min(b[0] for b in g), min(b[1] for b in g)
        x1, y1 = max(b[0] + b[2] for b in g), max(b[1] + b[3] for b in g)
        ink = np.isin(lbl[y0:y1, x0:x1], [b[4] for b in g])
        out.append(Line(text, margin, (x0, y0, x1 - x0, y1 - y0), np.where(ink, 0, 255).astype(np.uint8)))
    return out


ACCOUNT, READING = "account", "reading"
_TOKEN = re.compile(r"(\D*)(\d+)")


def parse(text: str) -> list[tuple[str, str]]:
    """Строка → [(что, число)]: что — ACCOUNT («лс»), READING («пок», «пк»)
    или "" (приписки нет: решает таблица)."""
    out = []
    for prefix, number in _TOKEN.findall(text.lower()):
        kind = ACCOUNT if set(prefix) & set("лс") else READING if set(prefix) & set("пок") else ""
        out.append((kind, number))
    return out
