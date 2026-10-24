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

Обрывок или испорченная строка опаснее, чем ничего. Строка не принимается,
если рядом с ней на её уровне — пятно того же цвета ростом с букву, но не
буква, или она касается края фото (обрывок: _loose_ink); если в «букве»
толстое место (слилась с пятном того же цвета: THICK_MAX); если цифры
разной высоты (DIGIT_H_DEV). Такая строка похожа на часть надписи
(Dropped.text_like): надпись тогда понята не целиком и не используется.

Прочитать. Каждая буква сравнивается с образцами символов 12 свободных
шрифтов телефонных редакторов (Android, iPhone, WhatsApp — models/overlay/
glyphs.npz, make_bank). Ответ — самый похожий образец; запас до лучшего
другого символа — уверенность: хоть у одного символа меньше MARGIN —
строку не разобрать (угадывать по таблице опасно: для показания подходит
слишком многое).

Что значит надпись (счёт, показание) — src/gmr/domain/note_match.py.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

BANK_PATH = Path(__file__).resolve().parents[2] / "models" / "overlay" / "glyphs.npz"
CHARS = "0123456789покслцПОКЛСЦ/"       # «/» — в «л/с», «ц» — в «лц»
GLYPH_H, GLYPH_W = 48, 36
MAX_SIDE = 1280             # ищется на уменьшенном фото: надпись крупная
MARGIN = 0.05               # запас уверенности символа (проверено на синтетике, 2026-10-21)
# Буква или пятно (вырезки замера 2026-10-21): у набранной буквы толщина
# штриха 0.10–0.25 высоты и ровная (разброс ≤ 0.4), похожесть на образец ≥ 0.84
# (на синтетике с незнакомыми шрифтами — ≥ 0.66 у 99% строк);
# у теней, труб, стен — толщина 0.45–0.9 или разброс 0.7–1.1, похожесть 0.24–0.79.
STROKE_W_MAX, STROKE_CV_MAX, SIM_MIN = 0.35, 0.6, 0.6
# Самое толстое место буквы — до 0.33 высоты строки (17 шрифтов, и жирный),
# в набранной строке после JPEG — до 0.47 (синтетика, 380 строк); у буквы,
# слипшейся с пятном того же цвета, — 0.65: уже не буква.
THICK_MAX = 0.5
# Строка в целом (второй замер 2026-10-22): набранный текст похож на образцы
# в среднем на ≥ 0.87 (синтетика, 99.5% строк; ваши фото — 0.95–0.99), пропорции
# каждой буквы — как у образца её символа (отклонение ≤ 0.21). Рукопись пальцем
# («16 65» → «7666») — средняя похожесть 0.79; решётка («11111…») — «единицы»
# в 1.3–2 раза уже настоящей (отклонение 0.26–0.7).
MEAN_SIM_MIN, ASPECT_DEV_MAX = 0.85, 0.4
WIDE = 0.9                  # «буква» шире 0.9 высоты строки — может быть двумя слипшимися
WIDEST = 8                  # пятно шире 8 своих высот — не слипшиеся буквы
# Цифры одного шрифта — одной высоты: в прочитанных строках отличаются до 5%
# (синтетика, 194 строки). Больше 15% — цифра слилась с фоном того же цвета
# или строчная буква прочитана цифрой («к» как «4»): строка испорчена.
DIGIT_H_DEV = 0.15
EDGE = 0.3                  # у края фото кусок того же цвета от 0.3 высоты строки — срезанная буква
EDGE_PX = 3                 # «касается края» — до 3 точек: при строгом цвете (_CORE) буква на 1–2 точки уже

FRAGMENT, UNSURE, DIGITS = "обрывок", "не разобрать символы", "цифры разной высоты"   # Dropped.why
SAME_PAINT = "цвет надписи"   # приписка к причине: строка той же краски, что прочитанная

_HUE_BINS, _MIN_COLOR_PX = 30, 200
_NEAR, _CORE = 35, 18       # расстояние в Lab: пиксель того же цвета / «чистый» цвет надписи


@dataclass
class Line:
    text: str               # прочитано, буквы — строчные: «пок51234»
    margin: float           # наименьший запас уверенности среди символов
    box: tuple              # x, y, w, h на уменьшенном фото
    crop: np.ndarray = field(repr=False)   # строка: чёрный текст на белом


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

    def read(self, mask: np.ndarray, rel_h: float) -> tuple[str, float, float, float, str]:
        """(символ, запас до лучшего другого символа, похожесть формы на
        самый похожий образец — 1.0 совпадает, у пятна-мусора меньше SIM_MIN,
        отклонение пропорций от ближайшего образца этого символа — |ln|,
        лучший другой символ)."""
        ys, xs = np.nonzero(mask)
        aspect = (np.ptp(ys) + 1) / (np.ptp(xs) + 1)
        sim = self.vectors @ _vector(_norm(mask))
        score = sim - 0.3 * np.abs(np.log(aspect / self.aspect)) - 0.8 * np.abs(rel_h - self.rel_h)
        order = np.argsort(score)[::-1]
        best = self.chars[order[0]]
        other = next(i for i in order if self.chars[i] != best)
        aspect_dev = float(np.min(np.abs(np.log(aspect / self.aspect[self.chars == best]))))
        return str(best), float(score[order[0]] - score[other]), float(sim.max()), aspect_dev, str(self.chars[other])


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


def _stroke(mask: np.ndarray) -> tuple[float, float, float]:
    """(толщина штриха / высота, разброс толщины, самое толстое место / высота)
    — по середине штриха."""
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = dt[(dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt >= 1)]
    if not ridge.size:                                  # слишком мелко, чтобы судить
        return 0.0, 0.0, 0.0
    h = mask.shape[0]
    return float(2 * ridge.mean() / h), float(ridge.std() / ridge.mean()), float(2 * ridge.max() / h)


@dataclass
class _Glyph:
    """«Буква»: пятно одного цвета на уменьшенном фото."""
    x: int
    y: int
    w: int
    h: int
    mask: np.ndarray = field(repr=False)     # h×w, True — штрих
    stroke: tuple = (0.0, 0.0, 0.0)          # _stroke(mask)


def _even_strokes(line: list) -> bool:
    """Штрихи всех «букв» строки — ровной толщины, не толще STROKE_W_MAX
    высоты строки и без толстых мест (THICK_MAX: буква на пятне того же
    цвета); строчные буквы ниже — толщина меряется по строке. Хоть одна не
    такая — строка целиком не надпись: обрезанная строка опаснее."""
    top = max(g.h for g in line)
    return all(g.stroke[0] * g.h / top <= STROKE_W_MAX and g.stroke[1] <= STROKE_CV_MAX
               and g.stroke[2] * g.h / top <= THICK_MAX for g in line)


def _count(run: list) -> int:
    """Сколько букв в строке: слипшиеся — по ширине."""
    return sum(max(1, round(g.w / (0.62 * g.h))) for g in run)


def _runs(glyphs: list, min_len: int) -> list[list]:
    """«Буквы» → строки: одна высота, один уровень, рядом друг с другом; не
    меньше min_len букв (слипшиеся — по ширине, но пятен — хотя бы два)."""
    groups: list[list] = []
    for g in sorted(glyphs, key=lambda g: g.x):
        cy = g.y + g.h / 2
        for grp in groups:
            gy, gh = np.median([x.y + x.h / 2 for x in grp]), max(x.h for x in grp)
            if abs(cy - gy) < 0.35 * gh and 0.45 < g.h / gh < 1.6:
                grp.append(g)
                break
        else:
            groups.append([g])
    runs = []
    for grp in groups:
        run = [grp[0]]
        for g in grp[1:]:
            if g.x - (run[-1].x + run[-1].w) <= 1.5 * max(x.h for x in run):
                run.append(g)
            else:
                runs.append(run)
                run = [g]
        runs.append(run)
    return [r for r in runs if _count(r) >= min_len and len(r) >= min(2, min_len)]


def _tails(lines: list, glyphs: list) -> list[list]:
    """Короткая строка (1–2 знака) под строкой надписи того же цвета,
    начинается в её пределах: хвост перенесённого числа («57») или приписка
    («ЛЦ», «лс» — строчные ниже цифр, «Ц» с хвостиком выше)."""
    used = {id(g) for ln in lines for g in ln}
    rest = [g for g in glyphs if id(g) not in used]
    out = []
    for ln in lines:
        top = max(g.h for g in ln)
        x0, x1, bottom = min(g.x for g in ln), max(g.x + g.w for g in ln), max(g.y + g.h for g in ln)
        near = [g for g in rest if 0.6 <= g.h / top <= 1.3 and bottom < g.y < bottom + 1.2 * top
                and x0 - top <= g.x <= x1]
        out += [r for r in _runs(near, 1) if _count(r) <= 2]
    return out


def _box(line: list) -> tuple:
    x0, y0 = min(g.x for g in line), min(g.y for g in line)
    return x0, y0, max(g.x + g.w for g in line) - x0, max(g.y + g.h for g in line) - y0


def _overlap(a: tuple, b: tuple) -> float:
    """Доля меньшей рамки, закрытая другой."""
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h) / max(1, min(a[2] * a[3], b[2] * b[3]))


def _glyphs(lab: np.ndarray, sel: np.ndarray, near: float) -> tuple[list[_Glyph], np.ndarray, np.ndarray]:
    """(«буквы», разметка пятен, их рамки) одного цвета — среднего у пикселей
    sel; свой цвет — ближе near."""
    dist = np.linalg.norm(lab - np.median(lab[sel], axis=0), axis=2)
    mask = cv2.morphologyEx((dist < near).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask)
    height, out = lab.shape[0], []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 60 or not 0.025 * height <= h <= 0.35 * height or w > WIDEST * h + 10:
            continue
        if not 0.12 < a / (w * h) < 0.8:                              # штрих, а не сплошное пятно
            continue
        comp = lbl[y:y + h, x:x + w] == i
        if (dist[y:y + h, x:x + w][comp] < _CORE).mean() < 0.35:       # почти весь цвет — один
            continue
        out.append(_Glyph(int(x), int(y), int(w), int(h), comp, _stroke(comp)))
    return out, lbl, st


def _loose_ink(line: list, lbl: np.ndarray, st: np.ndarray) -> bool:
    """Строка — обрывок: на её уровне, между буквами или до высоты строки
    слева и справа — пятно того же цвета ростом с букву, не вошедшее в
    строку (слипшиеся буквы, буква на пятне того же цвета), или срезанный
    краем фото кусок буквы; или сама строка касается края (остальное — за
    ним). Точки, двоеточие, черта ниже половины строки — не в счёт."""
    top = max(g.h for g in line)
    x0, y0, w, h = _box(line)
    height, width = lbl.shape
    if min(x0, y0) <= EDGE_PX or x0 + w >= width - EDGE_PX or y0 + h >= height - EDGE_PX:
        return True
    band = lbl[y0 + h // 5:y0 + h - h // 5, max(0, x0 - top):x0 + w + top]
    mine = {int(i) for g in line for i in np.unique(lbl[g.y:g.y + g.h, g.x:g.x + g.w][g.mask])}
    area = float(np.median([g.mask.sum() for g in line]))
    ids, counts = np.unique(band[band > 0], return_counts=True)
    for i, c in zip(ids, counts):
        x, _, cw, ch, _ = st[i]
        at_edge = x <= EDGE_PX or x + cw >= width - EDGE_PX
        if i not in mine and (ch >= 0.5 * top and c >= 0.1 * area or at_edge and ch >= EDGE * top):
            return True
    return False


def find(img: np.ndarray) -> list[tuple]:
    """Строки-кандидаты [(проход, строка, почему отброшена или "")] — строка
    — список «букв» слева направо. Каждый яркий цвет — отдельно: надпись
    бывает в два цвета («лс» синим, «пок» красным). Одна и та же строка
    бывает несколько раз (соседние цвета, два прохода): какую взять, решает
    read() по тому, как она прочиталась. Отброшена — штрих неровный или
    строка — обрывок (_loose_ink). img — BGR, уже уменьшенное до MAX_SIDE."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bright = (hsv[..., 1] >= 110) & (hsv[..., 2] >= 90)
    hue = hsv[..., 0].astype(int) // (180 // _HUE_BINS)
    out = []
    for b in range(_HUE_BINS):
        sel = bright & (hue == b)
        if sel.sum() < _MIN_COLOR_PX:
            continue
        # Бледный фон того же оттенка при _NEAR сливается с буквами в одно
        # пятно (строка — «обрывок»); буквы из редактора залиты ровно — при
        # _CORE они отдельно. Строка второго прохода — обрывок и тогда, когда
        # рядом при _NEAR цельная буква (а не фон, слитый с её же буквами).
        loose = None
        for strict, near in enumerate((_NEAR, _CORE)):
            glyphs, lbl, st = _glyphs(lab, sel, near)
            loose = loose or (lbl, st)
            lines = _runs(glyphs, 3)
            for ln in lines + _tails(lines, glyphs):
                cut = _loose_ink(ln, lbl, st) or (strict and _loose_ink(ln, *loose))
                why = "неровный штрих" if not _even_strokes(ln) else FRAGMENT if cut else ""
                out.append((strict, ln, why))
    return out


def _parts(mask: np.ndarray, top: float) -> list[np.ndarray]:
    """Слипшиеся буквы («00» одним пятном): разрезать там, где штриха меньше всего."""
    h, w = mask.shape
    n = round(w / (0.62 * top))
    if n < 2:
        return [mask]
    ink = mask.sum(axis=0).astype(float)
    cuts = [0]
    for i in range(1, n):
        c, r = int(w * i / n), max(2, int(0.18 * w / n))
        lo, hi = max(cuts[-1] + 2, c - r), min(w - 2, c + r)
        cuts.append(lo + int(np.argmin(ink[lo:hi])) if hi > lo else c)
    cuts.append(w)
    return [p for p in (mask[:, a:b] for a, b in zip(cuts, cuts[1:])) if p.any()]


def _read_glyph(bank: GlyphBank, g: _Glyph, top: float) -> list[tuple]:
    """[(символ, запас, похожесть, отклонение пропорций, другой символ,
    высота)] — обычно один; широкое пятно, которое лучше читается кусками, —
    по куску на символ."""
    one = bank.read(g.mask, g.h / top) + (g.h,)
    if g.w <= WIDE * top:
        return [one]
    heights = [np.ptp(np.nonzero(p)[0]) + 1 for p in _parts(g.mask, top)]
    reads = [bank.read(p, h / top) + (h,) for p, h in zip(_parts(g.mask, top), heights)]
    if len(reads) > 1 and all(r[2] >= SIM_MIN and r[3] <= ASPECT_DEV_MAX for r in reads) \
            and np.mean([r[2] for r in reads]) > one[2]:
        return reads
    return [one]


def _zero_or_o(reads: list) -> list:
    """«0» и «О» в некоторых шрифтах почти одинаковы. Буква «о» бывает только
    в «пок» — после «п» или перед «к»; иначе это 0. Так решённый символ —
    уверенный."""
    out = []
    for i, r in enumerate(reads):
        if r[1] < MARGIN and {r[0], r[4]} == {"0", "о"}:
            letter = (i > 0 and reads[i - 1][0] == "п") or (i + 1 < len(reads) and reads[i + 1][0] == "к")
            r = ("о" if letter else "0", MARGIN) + r[2:]
        out.append(r)
    return out


@dataclass
class Dropped:
    """Строка, которую поиск нашёл, но не принял (для разбора замера)."""
    text: str
    why: str
    box: tuple
    crop: np.ndarray = field(repr=False)

    @property
    def text_like(self) -> bool:
        """Похоже на часть надписи, но не прочитано (обрывок, неуверенные
        символы, цифры разной высоты, той же краски, что прочитанная строка)
        — надпись тогда понята не целиком. Остальное — не текст."""
        return self.why in (FRAGMENT, UNSURE, DIGITS) or self.why.endswith(SAME_PAINT)


def _paint(line: list, lab: np.ndarray) -> np.ndarray:
    """Цвет строки (Lab) — средний у её штрихов."""
    return np.median(np.concatenate([lab[g.y:g.y + g.h, g.x:g.x + g.w][g.mask] for g in line]), axis=0)


def _crop(line: list) -> np.ndarray:
    x0, y0, w, h = _box(line)
    out = np.full((h, w), 255, np.uint8)
    for g in line:
        out[g.y - y0:g.y - y0 + g.h, g.x - x0:g.x - x0 + g.w][g.mask] = 0
    return out


def _twin(a: tuple, b: tuple) -> bool:
    """Два прочтения одной строки равны по всему, кроме похожести букв: оба
    приняты (или оба нет), один проход, ширина — в пределах трети высоты."""
    (_, _, wa, ha), (_, _, wb, _) = _box(a[2]), _box(b[2])
    return bool(a[0]) == bool(b[0]) and a[1] == b[1] and abs(wa - wb) <= ha / 3


def read(img: np.ndarray, bank: GlyphBank, dropped: Optional[list] = None) -> list[Line]:
    """Строки надписи на фото (BGR), сверху вниз. Пусто — надписи нет.
    dropped — список: туда — найденные, но не принятые строки и почему."""
    s = MAX_SIDE / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    results = []
    for strict, ln, why in find(img):
        top = float(max(g.h for g in ln))                             # цифры — самые высокие
        reads = _zero_or_o([r for g in ln for r in _read_glyph(bank, g, top)])
        text = "".join(r[0] for r in reads)
        sims, margin, dev = [r[2] for r in reads], min(r[1] for r in reads), max(r[3] for r in reads)
        # какая-то «буква» ни на что не похожа, строка в целом — не набранный
        # текст (рукопись) или пропорции не те (решётка) — не надпись; обрывок
        # — не надпись, только если и большинство букв ни на что не похожи
        # (срезанная краем буква сама по себе «не похожа»)
        unlike = ("не похоже на буквы" if min(sims) < SIM_MIN else "как рукопись" if np.mean(sims) < MEAN_SIM_MIN
                  else "пропорции не те" if dev > ASPECT_DEV_MAX else "")
        if why == FRAGMENT and unlike and np.median(sims) < MEAN_SIM_MIN:
            why = unlike
        digits = [r[5] for r in reads if r[0].isdigit()]
        uneven = digits and max(abs(h / np.median(digits) - 1) for h in digits) > DIGIT_H_DEV
        why = why or unlike or (DIGITS if uneven else "") or (UNSURE if margin < MARGIN else "")
        results.append((why, strict, ln, text, margin, float(np.mean(sims))))
    # одно место — одна строка: прочитанная важнее непрочитанной (при
    # обычном допуске цвета буквы бывают слиты с фоном, при строгом — целы),
    # дальше — первый проход и широкая строка (обрывок — кусок широкой; число
    # пятен не годится — буква бывает расколота); той же ширины — у которой
    # буквы больше похожи на образцы (соседний оттенок мог съесть край буквы)
    chosen: list = []
    for r in sorted(results, key=lambda r: (bool(r[0]), r[1], -_box(r[2])[2])):
        rival = next((i for i, c in enumerate(chosen) if _overlap(_box(r[2]), _box(c[2])) >= 0.5), None)
        if rival is None:
            chosen.append(r)
        elif _twin(r, chosen[rival]) and r[5] > chosen[rival][5]:
            chosen[rival] = r
    out, kept, refused = [], [], []
    for why, _, ln, text, margin, _ in sorted(chosen, key=lambda r: (_box(r[2])[1], _box(r[2])[0])):
        if why:
            refused.append((Dropped(text, why, _box(ln), _crop(ln)), _paint(ln, lab)))
        else:
            out.append(Line(text, margin, _box(ln), _crop(ln)))
            kept.append(_paint(ln, lab))
    # не прочитана строка той же краски, что прочитанная, — это часть той же
    # надписи (буквы на пёстром фоне), а не мусор: надпись понята не целиком
    for d, paint in refused:
        if not d.text_like and any(np.linalg.norm(paint - k) < _CORE for k in kept):
            d.why = f"{d.why}, {SAME_PAINT}"
        if dropped is not None:
            dropped.append(d)
    return out
