"""
tools/make_icon.py — значок программы GMR (2026-11-22, просьба владельца:
вместо пёрышка Tk и значка PyInstaller — свой).

  python tools/make_icon.py

Газовое пламя над счётным механизмом счётчика (последний барабан — красный,
как на настоящих счётчиках) на тёмно-синем квадрате — цвет шапки окна.
Рисуется крупно и уменьшается — края гладкие и на 16×16.
Пишет packaging/gmr.ico (exe, ярлык) и src/gmr/operator/gmr.png (окно).
Файлы — в git; заново — только если меняется рисунок.
"""
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ICO = ROOT / "packaging" / "gmr.ico"
PNG = ROOT / "src" / "gmr" / "operator" / "gmr.png"
S = 1024                                     # рисуем крупно, потом уменьшаем

NAVY_TOP, NAVY_BOTTOM = (42, 64, 89), (20, 31, 44)
FLAME_OUT, FLAME_MID, FLAME_IN = (31, 111, 235), (77, 160, 255), (196, 230, 255)
DRUM, DRUM_LAST, DIGIT = (250, 251, 253), (214, 48, 49), (31, 41, 51)


def _font(size):
    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _flame(cx, base, height, width, lean=0.0):
    """Язык пламени остриём вверх: низ круглый, кончик острый и чуть загнут
    (lean — насколько, в долях ширины), бока с волной — не капля воды."""
    pts = []
    for i in range(361):
        t = math.radians(i)
        x = math.sin(t) * math.sin(t / 2) ** 2.2
        y = math.cos(t)                      # 1 — острие, -1 — низ
        up = (y + 1) / 2                     # 0 — низ, 1 — кончик
        x *= 1 + 0.10 * math.sin(up * math.pi * 2.2)
        x += lean * up ** 2.5
        pts.append((cx + x * width / 2, base - up * height))
    return pts


def draw(digits: bool = True) -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    # фон: скруглённый квадрат с вертикальным градиентом
    grad = Image.new("RGBA", (S, S))
    gd = ImageDraw.Draw(grad)
    for y in range(S):
        k = y / (S - 1)
        gd.line([(0, y), (S, y)], fill=tuple(round(a + (b - a) * k) for a, b in zip(NAVY_TOP, NAVY_BOTTOM)))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([24, 24, S - 24, S - 24], radius=210, fill=255)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)

    # пламя: три слоя, внутри светлее
    cx, base = S / 2, 610
    d.polygon(_flame(cx, base, 500, 420, lean=0.35), fill=FLAME_OUT)
    d.polygon(_flame(cx + 10, base - 8, 350, 280, lean=-0.30), fill=FLAME_MID)
    d.polygon(_flame(cx + 4, base - 16, 190, 150, lean=0.2), fill=FLAME_IN)

    # счётный механизм: окошко с барабанами
    left, right, top, bottom = 150, S - 150, 660, 860
    d.rounded_rectangle([left - 14, top - 14, right + 14, bottom + 14], radius=48, fill=(12, 19, 28))
    n, gap = 4, 16
    w = (right - left - gap * (n - 1)) / n
    font = _font(150)
    for i, ch in enumerate("0472"):
        x0 = left + i * (w + gap)
        last = i == n - 1
        d.rounded_rectangle([x0, top, x0 + w, bottom], radius=26, fill=DRUM_LAST if last else DRUM)
        if digits:
            d.text((x0 + w / 2, (top + bottom) / 2 + 4), ch, font=font, anchor="mm",
                   fill=(255, 255, 255) if last else DIGIT)
    return img


def main() -> int:
    big = draw(digits=True)
    small = draw(digits=False)               # на 16–32 цифры — каша: только барабаны
    sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
    frames = [(small if s <= 32 else big).resize((s, s), Image.LANCZOS) for s in sizes]
    ICO.parent.mkdir(parents=True, exist_ok=True)
    frames[-1].save(ICO, format="ICO", sizes=[(s, s) for s in sizes], append_images=frames[:-1])
    big.resize((128, 128), Image.LANCZOS).save(PNG)      # окно и экран входа
    print(f"{ICO}\n{PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
