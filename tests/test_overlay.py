"""
Надписи контролёров поверх фото: найти и прочитать (src/gmr/overlay.py),
замер gmr.py notes (tools/notes.py). Надпись рисуется образцами символов из
models/overlay/glyphs.npz поверх «фото»: шум, градиент, цветные пломбы и
бирки; потом JPEG. Что значит надпись — tests/test_note_match.py.
"""
import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

import gmr
from src.gmr import overlay
from src.gmr.render.image_io import write_image
from tools import notes

BANK = overlay.GlyphBank.load()
_RAW = np.load(overlay.BANK_PATH)


def _glyph(c: str, font: int, height: int) -> np.ndarray:
    """Образец символа font-го шрифта набора высотой height·rel_h (строчные ниже цифр)."""
    idx = [i for i, ch in enumerate(_RAW["chars"]) if ch == c]
    i = idx[font % len(idx)]
    g = _RAW["glyphs"][i] > 127
    ys, xs = np.nonzero(g)
    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    h = max(4, round(height * float(_RAW["rel_h"][i])))
    w = max(2, round(h / float(_RAW["aspect"][i])))
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST) > 127


def _background(seed: int = 0, h: int = 1280, w: int = 960) -> np.ndarray:
    rnd = np.random.default_rng(seed)
    img = np.clip(np.linspace(60, 200, w)[None, :, None] + rnd.normal(0, 12, (h, w, 3)), 0, 255).astype(np.uint8)
    cv2.circle(img, (150, 150), 60, (0, 120, 255), -1)                 # оранжевая пломба
    cv2.rectangle(img, (700, 100), (820, 160), (60, 180, 60), -1)      # зелёная бирка
    cv2.rectangle(img, (500, 1100), (900, 1150), (200, 80, 20), -1)    # синяя изолента
    for x in (100, 260, 420):                                          # три одинаковых сплошных пятна в ряд
        cv2.rectangle(img, (x, 1180), (x + 80, 1260), (40, 40, 220), -1)
    return img


def _paint(img, text: str, color, x: int, y: int, height: int, font: int = 0, gap=None) -> int:
    """Текст — образцами набора, по одной линии низа, с промежутком gap
    (по умолчанию 0.12 высоты). Возвращает x правого края."""
    gap = round(0.12 * height) if gap is None else gap
    for c in text:
        g = _glyph(c, font, height)
        top = y + height - g.shape[0]
        img[top:top + g.shape[0], x:x + g.shape[1]][g] = color
        x += g.shape[1] + gap
    return x - gap


def _jpeg(img, q: int = 70) -> np.ndarray:
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def _photo(lines, seed=0, q=70):
    img = _background(seed)
    for text, color, y, font in lines:
        _paint(img, text, color, 60, y, 110, font)
    return _jpeg(img, q)


# ─── Поиск и чтение ──────────────────────────────────────────────────────────

BLUE, RED = (250, 199, 52), (40, 40, 245)


def _found(img):
    dropped = []
    return [ln.text for ln in overlay.read(img, BANK, dropped)], [(d.text, d.why, d.text_like) for d in dropped]


@pytest.mark.parametrize("font", [0, 3, 6])
def test_reads_two_lines_in_blue(font):
    img = _photo([("пок51234", BLUE, 500, font), ("лс17777", BLUE, 700, font)])
    got = overlay.read(img, BANK)
    assert [ln.text for ln in got] == ["пок51234", "лс17777"]                 # сверху вниз
    assert all(ln.margin >= overlay.MARGIN for ln in got)
    assert got[0].crop.dtype == np.uint8 and set(np.unique(got[0].crop)) <= {0, 255}   # чёрное на белом


def test_reads_red_text():
    img = _photo([("9005357", RED, 600, 0)], seed=1)
    assert [ln.text for ln in overlay.read(img, BANK)] == ["9005357"]


def test_two_colors():
    """«лс» одним цветом, «пок» — другим: обе строки."""
    assert _found(_photo([("лс17777", BLUE, 500, 0), ("пок51234", RED, 700, 0)])) == (["лс17777", "пок51234"], [])


def test_short_tail_under_line():
    """Конец числа (1–2 цифры) перенесён под строку — тоже строка; такие же 2
    цифры не под строкой — не надпись."""
    img = _background(9)
    _paint(img, "лс04290", BLUE, 60, 450, 110)
    _paint(img, "57", BLUE, 60, 600, 110)
    _paint(img, "85", BLUE, 600, 1000, 110)
    assert _found(_jpeg(img)) == (["лс04290", "57"], [])


@pytest.mark.parametrize("font", [0, 3])
def test_labels_on_own_lines(font):
    """Приписка своей строкой под числом: «ПКЗ» (3 знака) и «ЛЦ» (2 — как
    короткий хвост под строкой)."""
    img = _photo([("51234", BLUE, 300, font), ("пк3", BLUE, 450, font), ("17777", BLUE, 600, font),
                  ("лц", BLUE, 750, font)])
    assert _found(img) == (["51234", "пк3", "17777", "лц"], [])


def test_unread_line_of_note_paint_is_part_of_note():
    """Не прочитанная строка той же краски, что прочитанная, — часть надписи
    (буквы на пёстром фоне), а не мусор; другого цвета — мусор."""
    wedge = np.zeros((120, 90), np.uint8)
    cv2.fillPoly(wedge, [np.array([[5, 115], [85, 115], [45, 5]])], 1)
    cv2.fillPoly(wedge, [np.array([[30, 115], [60, 115], [45, 60]])], 0)
    for color, part in ((BLUE, True), (RED, False)):
        img = _background(12)
        _paint(img, "24317", BLUE, 60, 300, 110)
        _row(img, wedge > 0, color, y=700)
        texts, dropped = _found(_jpeg(img, 90))
        why = "не похоже на буквы, " + overlay.SAME_PAINT if part else "не похоже на буквы"
        assert (texts, dropped) == (["24317"], [("4444", why, part)])


def test_slash_in_ls():
    assert _found(_photo([("л/с17777", BLUE, 600, 3)])) == (["л/с17777"], [])


@pytest.mark.parametrize("font", [3, 6])
@pytest.mark.parametrize("text", ["24317", "9005357"])
def test_touching_letters(font, text):
    """Буквы слиплись (тесный шрифт) — пятно режется на буквы; и длинное,
    шире 2.5 своих высот."""
    img = _background(9)
    _paint(img, text, BLUE, 60, 500, 110, font, gap=-4)
    assert _found(_jpeg(img)) == ([text], [])


def test_big_photo_is_scaled_down():
    img = _photo([("24317", BLUE, 600, 1)], seed=2)
    big = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3))
    got = overlay.read(big, BANK)
    assert [ln.text for ln in got] == ["24317"] and max(got[0].box) <= overlay.MAX_SIDE


def test_lines_top_to_bottom_and_gap_splits_line():
    img = _background(9)
    _paint(img, "2431", BLUE, 500, 400, 110)
    _paint(img, "лс5", BLUE, 20, 700, 110)                       # нижняя строка левее верхней
    _paint(img, "85", BLUE, 60, 400, 110)                        # «85» далеко слева от «2431» — не та же строка
    assert [ln.text for ln in overlay.read(_jpeg(img), BANK)] == ["2431", "лс5"]


def test_not_one_color_is_not_text():
    """Пёстрый цветной фон (ржавчина, кирпич): пятен «буквами в ряд» много, но
    цвет внутри каждого — разный. Надписи нет."""
    img = _background(0)
    n = cv2.GaussianBlur(np.random.default_rng(0).normal(0, 1, (700, 960, 3)).astype(np.float32), (0, 0), 4)
    img[300:1000] = np.clip(np.array([40, 90, 200], np.float32) + 60 * n / n.std(), 0, 255).astype(np.uint8)
    assert overlay.read(_jpeg(img, 80), BANK) == []


def test_no_overlay_nothing_found():
    assert _found(_jpeg(_background(3))) == ([], [])                      # пломбы, бирки, пятна в ряд — не текст


def test_two_glyphs_are_not_a_line():
    img = _photo([("85", BLUE, 600, 0)], seed=4)
    assert overlay.read(img, BANK) == []


def test_height_tells_letter_from_digit():
    zero = _glyph("0", 0, 100)                               # «0» Roboto: по форме — почти «о»
    assert BANK.read(zero, 1.0)[0] == "0" and BANK.read(zero, 0.7)[0] == "о"


def test_bank_margin_and_similarity():
    zero = _glyph("0", 0, 100)
    c, m, sim, dev, other = BANK.read(zero, 1.0)
    assert c == "0" and m >= overlay.MARGIN and sim > 0.95 and dev < 0.05 and other == "о"
    thin = cv2.resize(zero.astype(np.uint8), (zero.shape[1] // 2, zero.shape[0])) > 0
    assert BANK.read(thin, 1.0)[3] > overlay.ASPECT_DEV_MAX                # «0» вдвое уже любого образца
    assert BANK.read(np.ones((100, 70), bool), 1.0)[1] < 0.2              # сплошное пятно — без запаса


def test_zero_or_letter_o_by_neighbours():
    """«0» и «О» почти одинаковы: «о» — только в «пок» (после «п», перед «к»)."""
    def r(c, other, margin=0.01):
        return (c, margin, 0.9, 0.0, other, 100)
    got = overlay._zero_or_o([r("п", "л", 0.3), r("0", "о"), r("к", "л", 0.3), r("о", "0"), r("1", "7", 0.3)])
    assert [x[0] for x in got] == ["п", "о", "к", "0", "1"]
    assert [x[1] for x in got] == [0.3, overlay.MARGIN, 0.3, overlay.MARGIN, 0.3]
    assert [x[0] for x in overlay._zero_or_o([r("0", "о"), r("к", "л", 0.3)])] == ["о", "к"]
    assert [x[0] for x in overlay._zero_or_o([r("п", "л", 0.3), r("0", "о"), r("1", "7", 0.3)])] == ["п", "о", "1"]
    assert overlay._zero_or_o([r("о", "с")]) == [r("о", "с")]              # не та пара — не трогаем
    assert overlay._zero_or_o([r("о", "0", 0.2)]) == [r("о", "0", 0.2)]    # уверенный — тоже


def test_unsure_symbol_line_is_not_read(monkeypatch):
    """Хоть один символ неуверенный — строка не прочитана, но похожа на часть
    надписи (надпись тогда понята не целиком)."""
    monkeypatch.setattr(overlay, "MARGIN", 1.0)
    assert _found(_photo([("24317", BLUE, 600, 0)])) == ([], [("24317", overlay.UNSURE, True)])


def test_line_next_to_same_color_spot_is_fragment():
    """Рядом со строкой, на её уровне, — пятно того же цвета ростом с букву
    (буква не прочиталась, слиплась): строка — обрывок. Точка и черта — нет."""
    img = _background(9)
    x = _paint(img, "24317", BLUE, 60, 500, 110)
    cv2.rectangle(img, (x + 30, 520), (x + 90, 600), BLUE, -1)
    assert _found(_jpeg(img)) == ([], [("24317", overlay.FRAGMENT, True)])
    img = _background(9)                                                  # чуть другой краски — тоже
    x = _paint(img, "24317", BLUE, 60, 500, 110)
    cv2.rectangle(img, (x + 30, 520), (x + 90, 600), (240, 215, 140), -1)
    assert _found(_jpeg(img)) == ([], [("24317", overlay.FRAGMENT, True)])
    img = _background(9)
    x = _paint(img, "24317", BLUE, 60, 500, 110)
    cv2.rectangle(img, (x + 15, 572), (x + 45, 610), BLUE, -1)            # жирная точка — треть строки
    cv2.rectangle(img, (40, 545), (55, 553), BLUE, -1)                    # черта
    assert _found(_jpeg(img)) == (["24317"], [])


def test_letter_on_same_color_spot_is_not_text():
    """Пятно того же цвета касается буквы: у «буквы» толстое место — не буква."""
    img = _background(9)
    x = _paint(img, "24317", BLUE, 60, 500, 110)
    cv2.ellipse(img, (x + 25, 555), (35, 50), 0, 0, 360, BLUE, -1)
    assert _found(_jpeg(img))[0] == []


def test_pale_background_of_same_hue():
    """Бледный фон того же оттенка (серый корпус с голубым отливом) за концом
    строки: при обычном допуске цвета буквы сливаются с ним — начало строки
    и обрывок; при строгом (буквы из редактора залиты ровно) — вся строка."""
    img = _background(9)
    cv2.rectangle(img, (300, 420), (959, 720), (240, 215, 140), -1)
    _paint(img, "24317", BLUE, 60, 500, 110)
    assert _found(_jpeg(img)) == (["24317"], [])


def test_letters_merged_with_pale_band():
    """Бледная полоса того же оттенка (кронштейн) через буквы: при обычном
    допуске цвета строка читается «04294» (конец слит с полосой) — берётся
    прочтение строгого прохода; из двух его прочтений (соседние оттенки,
    в одном «0» расколот на «с1») — у которого буквы больше похожи на образцы."""
    img = _background(9)
    cv2.rectangle(img, (380, 540), (900, 556), (240, 215, 140), -1)
    _paint(img, "0429005357", BLUE, 60, 500, 90)
    assert _found(_jpeg(img)) == (["0429005357"], [])


def test_twin_readings_of_same_width_only():
    def result(width, why="", strict=1, sim=0.9):
        return (why, strict, [overlay._Glyph(10, 10, width, 90, np.ones((1, 1), bool))], "", 0.2, sim)
    assert overlay._twin(result(600), result(629))
    assert not overlay._twin(result(600), result(631))               # уже на треть высоты — обрывок, не двойник
    assert not overlay._twin(result(600), result(600, strict=0))
    assert not overlay._twin(result(600), result(600, why="обрывок"))


def test_digits_of_different_height_spoil_line():
    """Цифры одного шрифта — одной высоты; одна выше на треть (слилась с
    фоном того же цвета) — строка испорчена, часть надписи не прочитана."""
    img = _background(9)
    x = _paint(img, "2431", BLUE, 60, 500, 110)
    _paint(img, "7", BLUE, x + 13, 460, 150)
    assert _found(_jpeg(img)) == ([], [("24317", overlay.DIGITS, True)])


def _paint_cut(img, text: str, keep: int) -> None:
    """Текст у правого края: от последней буквы видно keep пикселей."""
    widths = [_glyph(c, 0, 110).shape[1] for c in text]
    x = img.shape[1] - (sum(widths) + 13 * (len(text) - 1) - widths[-1] + keep)
    for c, w in zip(text, widths):
        g = _glyph(c, 0, 110)
        visible = max(0, min(w, img.shape[1] - x))
        img[610 - g.shape[0]:610, x:x + visible][g[:, :visible]] = BLUE
        x += w + 13


def test_line_at_photo_edge_is_fragment():
    img = _background(9)
    _paint(img, "24317", BLUE, 0, 500, 110)                              # дальше — за краем фото
    assert _found(_jpeg(img)) == ([], [("24317", overlay.FRAGMENT, True)])
    img = _background(9)
    _paint_cut(img, "243170", 8)                                          # от «0» — узкая дуга у края
    assert _found(_jpeg(img))[1] == [("243171", overlay.FRAGMENT, True)]
    img = _background(9)
    _paint(img, "24317", BLUE, 555, 500, 110)                             # до края 21 пиксель…
    cv2.rectangle(img, (954, 535), (959, 575), BLUE, -1)                  # …а там срезанный кусок буквы
    assert _found(_jpeg(img)) == ([], [("24317", overlay.FRAGMENT, True)])
    img = _background(9)
    _paint(img, "24317", BLUE, 20, 500, 110)                             # близко к краю — можно
    assert _found(_jpeg(img)) == (["24317"], [])
    width = sum(_glyph(c, 0, 110).shape[1] for c in "24317") + 13 * 4
    for gap, cut in ((3, True), (5, False)):          # до правого края 3 точки — ещё «касается»
        img = _background(9)
        _paint(img, "24317", BLUE, 960 - width - gap, 500, 110)
        assert _found(_jpeg(img)) == (([], [("24317", overlay.FRAGMENT, True)]) if cut else (["24317"], []))


def test_dropped_text_like():
    crop = np.zeros((1, 1), np.uint8)
    assert overlay.Dropped("1", overlay.FRAGMENT, (0, 0, 1, 1), crop).text_like
    assert overlay.Dropped("1", overlay.UNSURE, (0, 0, 1, 1), crop).text_like
    assert overlay.Dropped("1", overlay.DIGITS, (0, 0, 1, 1), crop).text_like
    assert overlay.Dropped("1", "как рукопись, " + overlay.SAME_PAINT, (0, 0, 1, 1), crop).text_like
    assert not overlay.Dropped("1", "неровный штрих", (0, 0, 1, 1), crop).text_like


def _row(img, shape, color, y=500, n=4):
    for k in range(n):
        x0 = 80 + k * 150
        img[y:y + shape.shape[0], x0:x0 + shape.shape[1]][shape] = color


def _uneven():
    shape = np.zeros((120, 90), np.uint8)
    cv2.circle(shape, (45, 85), 30, 1, -1)
    cv2.line(shape, (45, 5), (45, 60), 1, 4)
    return shape > 0


def test_uneven_stroke_line_is_not_text():
    """Пятно с тонким хвостом (тень, трос у пломбы): одного цвета, в ряд, но
    толщина «штриха» скачет — не буквы и не часть надписи."""
    assert overlay._stroke(_uneven())[1] > overlay.STROKE_CV_MAX
    img = _background(11)
    _row(img, _uneven(), BLUE)
    texts, dropped = _found(_jpeg(img, 90))
    assert texts == [] and dropped and all(d[1] == "неровный штрих" and not d[2] for d in dropped)


def test_even_strokes_rule():
    def line(*strokes, h=100):
        return [overlay._Glyph(0, 0, 50, h, np.ones((1, 1), bool), st) for st in strokes]
    ok, thick, uneven, blob = (0.15, 0.2, 0.3), (0.4, 0.2, 0.45), (0.15, 0.8, 0.3), (0.2, 0.5, 0.6)
    assert overlay._even_strokes(line(ok, ok, ok))
    assert not overlay._even_strokes(line(ok, ok, uneven))          # одна неровная — вся строка не надпись
    assert not overlay._even_strokes(line(ok, thick, ok))
    assert not overlay._even_strokes(line(ok, blob, ok))            # толстое место — пятно, не буква
    low = line((0.4, 0.2, 0.6), h=70)                               # строчная: 0.4 своей высоты = 0.28 строки
    assert overlay._even_strokes(line(ok, ok) + low)
    assert overlay._stroke(np.eye(3, dtype=bool)) == (0.0, 0.0, 0.0)  # тоньше пикселя — не судим


def test_runs_count_wide_spots_as_letters():
    def g(x, w, h=100):
        return overlay._Glyph(x, 0, w, h, np.ones((1, 1), bool))
    assert len(overlay._runs([g(0, 60), g(80, 130)], 3)) == 1      # вторая — две слипшиеся буквы
    assert overlay._runs([g(0, 60), g(80, 60)], 3) == []
    assert overlay._runs([g(0, 190)], 3) == []                     # одно пятно — не строка
    assert overlay._runs([g(0, 190), g(800, 60)], 3) == []


def test_grille_is_not_ones():
    """Решётка, радиатор: ровные наклонные палки одного цвета в ряд — похожи на
    «1», но вдвое уже любой «1» из образцов."""
    img = _background(15)
    for k in range(7):
        cv2.line(img, (100 + k * 40, 500), (110 + k * 40, 640), BLUE, 7)
    assert overlay.read(_jpeg(img, 85), BANK) == []


def test_finger_handwriting_is_not_text():
    """Рукопись (пальцем в редакторе): буквы на образцы шрифтов похожи плохо —
    в среднем меньше MEAN_SIM_MIN; иначе «16 65» читалось бы уверенно как «7665»."""
    img = _background(14)
    cv2.putText(img, "16 65", (60, 650), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 5, BLUE, 9, cv2.LINE_AA)
    assert overlay.read(_jpeg(img, 85), BANK) == []


def test_line_of_non_characters_is_not_text():
    """Ровные штрихи, но фигура ни на один символ не похожа (клин с вырезом)."""
    wedge = np.zeros((120, 90), np.uint8)
    cv2.fillPoly(wedge, [np.array([[5, 115], [85, 115], [45, 5]])], 1)
    cv2.fillPoly(wedge, [np.array([[30, 115], [60, 115], [45, 60]])], 0)
    assert BANK.read(wedge > 0, 1.0)[2] < overlay.SIM_MIN
    img = _background(12)
    _row(img, wedge > 0, BLUE)
    assert overlay.read(_jpeg(img, 90), BANK) == []
    ys, xs = np.nonzero(wedge)
    wedge = wedge[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0
    img = _background(12)
    for k in range(4):                                                    # касается края фото — всё равно не текст
        img[500:500 + wedge.shape[0], k * 150:k * 150 + wedge.shape[1]][wedge] = BLUE
    assert _found(_jpeg(img, 90)) == ([], [("4444", "не похоже на буквы", False)])


def test_thick_letters_measured_by_line_height():
    """Строчные буквы ниже цифр: их штрих относительно своей высоты толще, но
    относительно строки — обычный; такая строка — надпись."""
    img = _photo([("лсок52", BLUE, 600, 3)], seed=13)
    assert [ln.text for ln in overlay.read(img, BANK)] == ["лсок52"]


def test_make_bank(tmp_path):
    pytest.importorskip("fontTools")
    mpl = pytest.importorskip("matplotlib")
    font = Path(mpl.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
    n = overlay.make_bank([font], tmp_path / "b.npz")
    d = np.load(tmp_path / "b.npz")
    assert n == len(overlay.CHARS) and "".join(d["chars"]) == overlay.CHARS.lower()
    assert d["glyphs"].shape == (n, overlay.GLYPH_H, overlay.GLYPH_W)
    assert d["rel_h"][0] == 1.0 and d["rel_h"][list(d["chars"]).index("о")] < 0.85   # строчная ниже цифры
    assert BANK.chars.size >= 200 and "/" in BANK.chars                       # набор проекта: 12 шрифтов


# ─── Замер: gmr.py notes ─────────────────────────────────────────────────────

def _table(tmp_path) -> Path:
    t = tmp_path / "реестр.csv"
    t.write_text("Лицевой счет;Номер счетчика;Последние показания;Текущие показания\n"
                 "1300017777;095011111;51000;51234\n1300024317;1260001;4300;5100\n"
                 "0429005357;2250001;1;2\n1300000001;111111;1;2\n1300055555;3330001;51000;51300\n"
                 "1300014317;4440001;9000;\n",
                 encoding="utf-8")
    return t


def _write(path: Path, img) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_image(path, img)
    return path


def _report(out: Path) -> list[dict]:
    with open(out / notes.REPORT, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=";"))


def test_notes_run_on_archive_folder(tmp_path, capsys):
    photos = tmp_path / "фото"
    _write(photos / "плюс" / "1300017777.jpg", _photo([("пок51234", BLUE, 500, 0), ("лс17777", BLUE, 700, 0)]))
    _write(photos / "минус" / "1300024317.jpg", _photo([("24317", BLUE, 600, 2)], seed=5))
    _write(photos / "плюс" / "0429005357.jpg", _photo([("9005358", RED, 600, 4)], seed=6))       # нет такого счёта
    # чужой счёт: показание фото (51300) подходит и ему — без таблички это опасно
    _write(photos / "плюс" / "1300055555.jpg", _photo([("лс17777", BLUE, 600, 1)], seed=9))
    _write(photos / "минус" / "1300024317 (2).jpg", _photo([("лс17777", BLUE, 600, 1)], seed=10))  # спор
    _write(photos / "плюс" / "1300000001.jpg", _jpeg(_background(7)))                            # без надписи
    shadows = _background(11)
    _row(shadows, _uneven(), BLUE)
    _write(photos / "плюс" / "1300000001 (2).jpg", _jpeg(shadows, 90))                            # не текст
    _write(photos / "плюс" / "1399999999.jpg", _photo([("12345", BLUE, 600, 0)], seed=8))         # счёта нет в таблице
    cut = _background(12)
    _paint(cut, "24317", BLUE, 0, 500, 110)
    _write(photos / "минус" / "1300024317 (3).jpg", _jpeg(cut))                                  # обрывок у края
    _write(photos / "минус" / "1300024317 (4).jpg", _photo([("4317", BLUE, 600, 0)], seed=12))    # и 1300014317
    _write(photos / "плюс" / "1300014317.jpg", _photo([("4317", BLUE, 600, 3)], seed=13))      # показания нет
    (photos / "битое.jpg").write_bytes(b"x")
    out = tmp_path / "фото_надписи"
    (out / "не_совпало").mkdir(parents=True)                                     # папка прошлой версии замера
    text = notes.run(photos, _table(tmp_path))
    assert text.splitlines()[:5] == [
        f"Надписи: {photos} — фото 12",
        "  надпись есть: 9 (75%)",
        "  по фото с ответом в таблице (8): верно 3, опасно 1, спор (к оператору) 1, подтверждает счёт 1, "
        "не понять (надпись не используется) 2",
        "    счёт по надписи (как без таблички): верно 3, опасно 1, спор 0",
        "    показание по надписи (как без барабана): верно 1, опасно 0, спор 2"]
    assert "  надпись есть, но ответа в таблице нет: 1" in text and "  не открылись как картинка: 1" in text
    assert "  найдено только не-текст (тени, трубы — отброшено): 1" in text
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.jpg")) == [
        "верно/минус__1300024317 (4).jpg", "верно/минус__1300024317.jpg", "верно/плюс__1300017777.jpg",
        "не_понять/минус__1300024317 (3).jpg", "не_понять/плюс__0429005357.jpg",
        "не_текст/плюс__1300000001 (2).jpg", "нет_ответа/плюс__1399999999.jpg",
        "опасно/плюс__1300055555.jpg", "подтверждает/плюс__1300014317.jpg", "спор/минус__1300024317 (2).jpg"]
    assert not (out / "не_совпало").exists()
    rows = {r["фото"]: r for r in _report(out)}
    assert rows["плюс/1300017777.jpg"]["показание по надписи"] == "51234"
    assert rows["плюс/1300055555.jpg"]["счёт по надписи"] == "1300017777"
    assert rows["минус/1300024317 (2).jpg"]["итог"] == "спор"
    assert rows["плюс/0429005357.jpg"]["почему"].startswith("без таблички: не запись счёта: 9005358")
    assert rows["минус/1300024317 (3).jpg"]["отброшено"] == "24317 (обрывок)"
    assert rows["минус/1300024317 (4).jpg"]["счёт по надписи"] == "1300024317"   # из двух — по показанию
    assert rows["плюс/1300014317.jpg"]["почему"] == (
        "без таблички: л/с 1300014317? показание — ему не подходит (прошлое 9000); л/с 1300024317? показание "
        "— ему не подходит (прошлое 4300); с табличкой: подтверждает л/с 1300014317")
    assert "Фото 12 из 12" in capsys.readouterr().out
    # повторный замер — прошлые картинки заменяются
    (photos / "плюс" / "1300055555.jpg").unlink()
    notes.run(photos, _table(tmp_path))
    assert not (out / "опасно").exists()


def test_notes_number_wrapped_to_next_line(tmp_path):
    photos = tmp_path / "ф"
    _write(photos / "0429005357.jpg", _photo([("лс04290", BLUE, 450, 0), ("05357", BLUE, 650, 0)]))
    notes.run(photos, _table(tmp_path), tmp_path / "o")
    row, = _report(tmp_path / "o")
    assert (row["строки"], row["итог"], row["счёт по надписи"]) == ("лс04290 / 05357", "верно", "0429005357")


def test_notes_on_archive_etalon(tmp_path):
    et = tmp_path / "etalon_archive"
    _write(et / "photos" / "aaa.jpg", _photo([("24317", BLUE, 600, 0)]))             # счёт — по номеру 1260001
    _write(et / "photos" / "bbb.jpg", _photo([("лс17777", BLUE, 600, 0)], seed=2))   # в таблице номер с нулём
    _write(et / "photos" / "ccc.jpg", _photo([("пок99999", BLUE, 600, 2)], seed=3))  # не подходит к прошлому
    _write(et / "photos" / "ddd.jpg", _photo([("лс17777", BLUE, 600, 0)], seed=4))   # номера нет в таблице
    with open(et / "answers.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["photo_hash", "file", "serial", "reading"])
        w.writeheader()
        w.writerows([{"photo_hash": "a", "file": "aaa.jpg", "serial": "1260001", "reading": "5100"},
                     {"photo_hash": "b", "file": "bbb.jpg", "serial": "095011111", "reading": "51234"},
                     {"photo_hash": "c", "file": "ccc.jpg", "serial": "95011111", "reading": "51234"},
                     {"photo_hash": "d", "file": "ddd.jpg", "serial": "999", "reading": "51234"}])
    text = notes.run(et, _table(tmp_path))
    assert "фото 4" in text and "верно 2, опасно 0, спор (к оператору) 0, подтверждает счёт 0, " \
        "не понять (надпись не используется) 1" in text
    assert "ответа в таблице нет: 1" in text
    out = tmp_path / "etalon_archive_надписи"
    assert (out / "не_понять" / "ccc.jpg").is_file() and (out / "нет_ответа" / "ddd.jpg").is_file()


def test_picture_shows_lines_and_answer():
    img = _photo([("лс17777", BLUE, 600, 0)])
    dropped = []
    lines = overlay.read(img, BANK, dropped)
    dropped = [overlay.Dropped("4", "неровный штрих", (10, 10, 30, 40), None),
               overlay.Dropped("5", overlay.FRAGMENT, (100, 10, 30, 40), None)]
    pic = notes.picture(img, lines, dropped, {"итог": "верно", "почему": "л/с 1300017777 по надписи"})
    x, y, w, h = lines[0].box
    assert tuple(pic[y - 3, x + w // 2]) == notes.GREEN and tuple(pic[7, 20]) == notes.RED
    assert tuple(pic[7, 110]) == notes.ORANGE
    assert pic.shape[0] > img.shape[0] if notes._font(22) else pic.shape == img.shape


def test_notes_command(tmp_path, capsys):
    photos = tmp_path / "ф"
    _write(photos / "1300024317.jpg", _photo([("24317", BLUE, 600, 0)]))
    assert gmr.main(["notes", str(photos), "--table", str(_table(tmp_path)), "--out", str(tmp_path / "o")]) == 0
    assert "по фото с ответом в таблице (1): верно 1, опасно 0" in capsys.readouterr().out
    assert (tmp_path / "o" / "верно" / "1300024317.jpg").is_file()
    assert gmr.main(["notes", str(tmp_path / "нет"), "--table", str(_table(tmp_path))]) == 1
    assert "ОШИБКА: нет папки" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        gmr.main(["notes", str(photos)])
