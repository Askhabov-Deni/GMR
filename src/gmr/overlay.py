"""
src/gmr/overlay.py — надпись контролёра поверх фото: «24317», «пок51234»,
«лс17777», набранные в редакторе телефона (решение владельца 2026-10-21:
найти и прочитать без нейросети; docs/BACKLOG.md).

Найти. Буквы надписи залиты одним ярким цветом: на живом фото такого почти
не бывает. Для каждого яркого цвета на фото — пятна этого цвета; «буква» —
пятно разумного размера, в котором почти весь цвет — один и штрих ровной
толщины; строка — не меньше 3 «букв» одной высоты подряд, и каждая похожа
на какой-то символ (SIM_MIN). Пломбы, бирки, изолента, тени и трубы строку
не образуют (замер на эталоне архива 2026-10-21: лишние находки — пятна
неровной толщины, ни на какой символ не похожие).
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
# Буква или пятно (вырезки замера 2026-10-21): у набранной буквы толщина
# штриха 0.10–0.25 высоты и ровная (разброс ≤ 0.4), похожесть на образец ≥ 0.84
# (на синтетике с незнакомыми шрифтами — ≥ 0.66 у 99% строк);
# у теней, труб, стен — толщина 0.45–0.9 или разброс 0.7–1.1, похожесть 0.24–0.79.
STROKE_W_MAX, STROKE_CV_MAX, SIM_MIN = 0.35, 0.6, 0.6
# Строка в целом (второй замер 2026-10-22): набранный текст похож на образцы
# в среднем на ≥ 0.87 (синтетика, 99.5% строк; ваши фото — 0.95–0.99), пропорции
# каждой буквы — как у образца её символа (отклонение ≤ 0.21). Рукопись пальцем
# («16 65» → «7666») — средняя похожесть 0.79; решётка («11111…») — «единицы»
# в 1.3–2 раза уже настоящей (отклонение 0.26–0.7).
MEAN_SIM_MIN, ASPECT_DEV_MAX = 0.85, 0.4

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

    def read(self, mask: np.ndarray, rel_h: float) -> tuple[str, float, float, float]:
        """(символ, запас до лучшего другого символа, похожесть формы на
        самый похожий образец — 1.0 совпадает, у пятна-мусора меньше SIM_MIN,
        отклонение пропорций от ближайшего образца этого символа — |ln|)."""
        ys, xs = np.nonzero(mask)
        aspect = (np.ptp(ys) + 1) / (np.ptp(xs) + 1)
        sim = self.vectors @ _vector(_norm(mask))
        score = sim - 0.3 * np.abs(np.log(aspect / self.aspect)) - 0.8 * np.abs(rel_h - self.rel_h)
        order = np.argsort(score)[::-1]
        best = self.chars[order[0]]
        other = next(i for i in order if self.chars[i] != best)
        aspect_dev = float(np.min(np.abs(np.log(aspect / self.aspect[self.chars == best]))))
        return str(best), float(score[order[0]] - score[other]), float(sim.max()), aspect_dev


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


def _stroke(mask: np.ndarray) -> tuple[float, float]:
    """(толщина штриха / высота, разброс толщины) — по середине штриха."""
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = dt[(dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt >= 1)]
    if not ridge.size:                                  # слишком мелко, чтобы судить
        return 0.0, 0.0
    return float(2 * ridge.mean() / mask.shape[0]), float(ridge.std() / ridge.mean())


def _even_strokes(line: list) -> bool:
    """Штрихи всех «букв» строки — ровной толщины и не толще STROKE_W_MAX
    высоты строки (строчные буквы ниже — толщина меряется по строке). Хоть
    одна не такая — строка целиком не надпись: обрезанная строка опаснее."""
    top = max(b[3] for b in line)
    return all(width * h / top <= STROKE_W_MAX and spread <= STROKE_CV_MAX
               for _, _, _, h, _, (width, spread) in line)


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
    """(номера пятен, строки — списки (x, y, w, h, номер пятна, (толщина, разброс) штриха) слева направо)
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
            chars.append((x, y, w, h, i, _stroke(comp)))
        lines = [g for g in _lines(chars) if _even_strokes(g)]
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
        text, margin, sims, dev = "", 1.0, [], 0.0
        for x, y, w, h, i, _ in g:
            c, m, s, d = bank.read(lbl[y:y + h, x:x + w] == i, h / digit_h)
            text += c
            margin, dev = min(margin, m), max(dev, d)
            sims.append(s)
        # какая-то «буква» ни на что не похожа, строка в целом — не набранный
        # текст (рукопись) или пропорции не те (решётка) — не надпись
        if min(sims) < SIM_MIN or np.mean(sims) < MEAN_SIM_MIN or dev > ASPECT_DEV_MAX:
            continue
        x0, y0 = min(b[0] for b in g), min(b[1] for b in g)
        x1, y1 = max(b[0] + b[2] for b in g), max(b[1] + b[3] for b in g)
        ink = np.isin(lbl[y0:y1, x0:x1], [b[4] for b in g])
        out.append(Line(text, margin, (x0, y0, x1 - x0, y1 - y0), np.where(ink, 0, 255).astype(np.uint8)))
    return out


ACCOUNT, READING = "account", "reading"
_RUNS = re.compile(r"\d+|\D+")


def wrapped(upper: str, lower: str) -> bool:
    """Число перенесено на следующую строку: верхняя кончается цифрой, нижняя
    начинается цифрой («лс04290» / «12345» — счёт 0429012345)."""
    return upper[-1:].isdigit() and lower[:1].isdigit()


_ZE = re.compile(r"(?<=[пк])3")


def spellings(text: str) -> list[str]:
    """Как ещё могла быть написана строка: «З» и «3» в шрифте почти одинаковы,
    и «ПКЗ» (показания) читается как «пк3». Вариант с «з» вместо «3» сразу
    после «п»/«к» — какой верный, решает таблица."""
    alt = _ZE.sub("з", text)
    return [text] if alt == text else [text, alt]


def _kind(letters: str) -> str:
    return ACCOUNT if set(letters) & set("лс") else READING if set(letters) & set("пок") else ""


def parse(text: str) -> list[tuple[str, str]]:
    """Строка → [(что, число)]: что — ACCOUNT («лс»), READING («пок», «пк»)
    или "" (приписки нет: решает таблица). Приписка обычно перед числом;
    в конце строки после числа без своей приписки — его («17743пок»)."""
    runs = _RUNS.findall(text.lower())
    out = []
    for n, run in enumerate(runs):
        if not run.isdigit():
            continue
        before = runs[n - 1] if n else ""
        after = runs[n + 1] if n + 2 == len(runs) else ""
        out.append((_kind(before) or (_kind(after) if not before else ""), run))
    return out
