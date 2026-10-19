"""
Надписи контролёров поверх фото (src/gmr/overlay.py, gmr.py notes —
tools/notes.py). Надпись рисуется образцами символов из models/overlay/
glyphs.npz поверх «фото»: шум, градиент, цветные пломбы и бирки; потом JPEG.
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


def _paint(img, text: str, color, x: int, y: int, height: int, font: int = 0) -> None:
    """Текст — образцами набора, по одной линии низа, с промежутком 0.12 высоты."""
    for c in text:
        g = _glyph(c, font, height)
        top = y + height - g.shape[0]
        img[top:top + g.shape[0], x:x + g.shape[1]][g] = color
        x += g.shape[1] + round(0.12 * height)


def _jpeg(img, q: int = 70) -> np.ndarray:
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def _photo(lines, seed=0, q=70):
    img = _background(seed)
    for text, color, y, font in lines:
        _paint(img, text, color, 60, y, 110, font)
    return _jpeg(img, q)


# ─── Поиск и чтение ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("font", [0, 3, 6])
def test_reads_two_lines_in_blue(font):
    img = _photo([("пок51234", (250, 199, 52), 500, font), ("лс17777", (250, 199, 52), 700, font)])
    got = overlay.read(img, BANK)
    assert [ln.text for ln in got] == ["пок51234", "лс17777"]                 # сверху вниз
    assert all(ln.sure for ln in got)
    assert got[0].crop.dtype == np.uint8 and set(np.unique(got[0].crop)) <= {0, 255}   # чёрное на белом


def test_reads_red_text():
    img = _photo([("9005357", (40, 40, 245), 600, 0)], seed=1)
    assert [ln.text for ln in overlay.read(img, BANK)] == ["9005357"]


def test_big_photo_is_scaled_down():
    img = _photo([("24317", (250, 199, 52), 600, 1)], seed=2)
    big = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3))
    got = overlay.read(big, BANK)
    assert [ln.text for ln in got] == ["24317"] and max(got[0].box) <= overlay.MAX_SIDE


def test_lines_top_to_bottom_and_gap_splits_line():
    img = _background(9)
    _paint(img, "2431", (250, 199, 52), 500, 400, 110)
    _paint(img, "лс5", (250, 199, 52), 20, 700, 110)             # нижняя строка левее верхней
    _paint(img, "85", (250, 199, 52), 60, 400, 110)              # «85» далеко слева от «2431» — не та же строка
    assert [ln.text for ln in overlay.read(_jpeg(img), BANK)] == ["2431", "лс5"]


def test_not_one_color_is_not_text():
    """Пёстрый цветной фон (ржавчина, кирпич): пятен «буквами в ряд» много, но
    цвет внутри каждого — разный. Надписи нет."""
    img = _background(0)
    n = cv2.GaussianBlur(np.random.default_rng(0).normal(0, 1, (700, 960, 3)).astype(np.float32), (0, 0), 4)
    img[300:1000] = np.clip(np.array([40, 90, 200], np.float32) + 60 * n / n.std(), 0, 255).astype(np.uint8)
    assert overlay.read(_jpeg(img, 80), BANK) == []


def test_no_overlay_nothing_found():
    assert overlay.read(_jpeg(_background(3)), BANK) == []                  # пломбы, бирки, пятна в ряд — не текст


def test_two_glyphs_are_not_a_line():
    img = _photo([("85", (250, 199, 52), 600, 0)], seed=4)
    assert overlay.read(img, BANK) == []


def test_height_tells_letter_from_digit():
    zero = _glyph("0", 0, 100)                               # «0» Roboto: по форме — почти «о»
    assert BANK.read(zero, 1.0)[0] == "0" and BANK.read(zero, 0.7)[0] == "о"


def test_line_sure_by_margin():
    crop = np.zeros((2, 2), np.uint8)
    assert overlay.Line("1", overlay.MARGIN, (0, 0, 1, 1), crop).sure
    assert not overlay.Line("1", overlay.MARGIN - 0.001, (0, 0, 1, 1), crop).sure


def test_bank_margin_and_similarity():
    zero = _glyph("0", 0, 100)
    c, m, sim, dev = BANK.read(zero, 1.0)
    assert c == "0" and m >= overlay.MARGIN and sim > 0.95 and dev < 0.05
    thin = cv2.resize(zero.astype(np.uint8), (zero.shape[1] // 2, zero.shape[0])) > 0
    assert BANK.read(thin, 1.0)[3] > overlay.ASPECT_DEV_MAX                # «0» вдвое уже любого образца
    assert BANK.read(np.ones((100, 70), bool), 1.0)[1] < 0.2              # сплошное пятно — без запаса


def _row(img, shape, color, y=500, n=4):
    for k in range(n):
        x0 = 80 + k * 150
        img[y:y + shape.shape[0], x0:x0 + shape.shape[1]][shape] = color


def test_uneven_stroke_line_is_not_text():
    """Пятно с тонким хвостом (тень, трос у пломбы): одного цвета, в ряд, но
    толщина «штриха» скачет — не буквы."""
    shape = np.zeros((120, 90), np.uint8)
    cv2.circle(shape, (45, 85), 30, 1, -1)
    cv2.line(shape, (45, 5), (45, 60), 1, 4)
    a, b = overlay._stroke(shape > 0)
    assert b > overlay.STROKE_CV_MAX
    img = _background(11)
    _row(img, shape > 0, (250, 199, 52))
    assert overlay.read(_jpeg(img, 90), BANK) == []


def test_even_strokes_rule():
    def line(*strokes, h=100):
        return [(0, 0, 50, h, i, st) for i, st in enumerate(strokes)]
    ok, thick, uneven = (0.15, 0.2), (0.4, 0.2), (0.15, 0.8)
    assert overlay._even_strokes(line(ok, ok, ok))
    assert not overlay._even_strokes(line(ok, ok, uneven))          # одна неровная — вся строка не надпись
    assert not overlay._even_strokes(line(ok, thick, ok))
    low = [(0, 0, 50, 70, 9, (0.4, 0.2))]                            # строчная: 0.4 своей высоты = 0.28 строки
    assert overlay._even_strokes(line(ok, ok) + low)
    assert overlay._stroke(np.eye(3, dtype=bool)) == (0.0, 0.0)      # тоньше пикселя — не судим


def test_grille_is_not_ones():
    """Решётка, радиатор: ровные наклонные палки одного цвета в ряд — похожи на
    «1», но вдвое уже любой «1» из образцов."""
    img = _background(15)
    for k in range(7):
        cv2.line(img, (100 + k * 40, 500), (110 + k * 40, 640), (250, 199, 52), 7)
    assert overlay.read(_jpeg(img, 85), BANK) == []


def test_finger_handwriting_is_not_text():
    """Рукопись (пальцем в редакторе): буквы на образцы шрифтов похожи плохо —
    в среднем меньше MEAN_SIM_MIN; иначе «16 65» читалось бы уверенно как «7665»."""
    img = _background(14)
    cv2.putText(img, "16 65", (60, 650), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 5, (250, 199, 52), 9, cv2.LINE_AA)
    assert overlay.read(_jpeg(img, 85), BANK) == []


def test_line_of_non_characters_is_not_text():
    """Ровные штрихи, но фигура ни на один символ не похожа (клин с вырезом)."""
    wedge = np.zeros((120, 90), np.uint8)
    cv2.fillPoly(wedge, [np.array([[5, 115], [85, 115], [45, 5]])], 1)
    cv2.fillPoly(wedge, [np.array([[30, 115], [60, 115], [45, 60]])], 0)
    assert BANK.read(wedge > 0, 1.0)[2] < overlay.SIM_MIN
    img = _background(12)
    _row(img, wedge > 0, (250, 199, 52))
    assert overlay.read(_jpeg(img, 90), BANK) == []


def test_thick_letters_measured_by_line_height():
    """Строчные буквы ниже цифр: их штрих относительно своей высоты толще, но
    относительно строки — обычный; такая строка — надпись."""
    img = _photo([("лсок52", (250, 199, 52), 600, 3)], seed=13)
    assert [ln.text for ln in overlay.read(img, BANK)] == ["лсок52"]


def test_wrapped():
    assert overlay.wrapped("лс04290", "05357") and overlay.wrapped("пк3", "4567")
    assert not overlay.wrapped("лс17777", "пок51234") and not overlay.wrapped("24317", "")
    assert not overlay.wrapped("пок", "123")
    assert notes.readings_of(["лс04290", "05357", "пок7"], 0) == ["лс04290", "лс0429005357"]
    assert notes.readings_of(["лс04290", "05357", "пок7"], 1) == ["05357", "лс0429005357"]
    assert notes.readings_of(["лс04290", "05357", "пок7"], 2) == ["пок7"]


def test_ze_read_as_three():
    """«ПКЗ12345» читается как «пк312345»: проверяется и вариант с «З»."""
    assert overlay.spellings("пк312345") == ["пк312345", "пкз12345"]
    assert overlay.spellings("24317") == ["24317"] and overlay.spellings("лс3") == ["лс3"]
    assert overlay.parse("пкз12345") == [("reading", "12345")]
    assert notes.readings_of(["пк3", "12345"], 0) == ["пк3", "пкз", "пк312345", "пкз12345"]
    acc = {"1300017777"}
    assert any(notes.agrees(overlay.parse(t), acc, "12345") for t in notes.readings_of(["пк3", "12345"], 0))


@pytest.mark.parametrize("text, tokens", [
    ("пок51234", [("reading", "51234")]), ("лс17777", [("account", "17777")]),
    ("24317", [("", "24317")]), ("пк123лс45", [("reading", "123"), ("account", "45")]),
    ("ПОК7", [("reading", "7")]), ("лсок", []), ("с0042", [("account", "0042")]),
    ("17743пок", [("reading", "17743")]), ("17743лс", [("account", "17743")]),        # приписка после числа
    ("123пок456", [("", "123"), ("reading", "456")]),                                 # «пок» — к числу после
    ("лс5пок", [("account", "5")]),
])
def test_parse(text, tokens):
    assert overlay.parse(text) == tokens


def test_make_bank(tmp_path):
    pytest.importorskip("fontTools")
    mpl = pytest.importorskip("matplotlib")
    font = Path(mpl.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
    n = overlay.make_bank([font], tmp_path / "b.npz")
    d = np.load(tmp_path / "b.npz")
    assert n == len(overlay.CHARS) and "".join(d["chars"]) == overlay.CHARS.lower()
    assert d["glyphs"].shape == (n, overlay.GLYPH_H, overlay.GLYPH_W)
    assert d["rel_h"][0] == 1.0 and d["rel_h"][list(d["chars"]).index("о")] < 0.85   # строчная ниже цифры
    assert BANK.chars.size >= 200                                             # набор проекта: 12 шрифтов


# ─── Замер: gmr.py notes ─────────────────────────────────────────────────────

def test_account_forms():
    assert notes.is_account_form("24317", "1300024317") and notes.is_account_form("9005357", "0429005357")
    assert notes.is_account_form("0024317", "1300024317")
    assert not notes.is_account_form("314", "1300024317")                     # меньше 4 цифр хвоста
    assert not notes.is_account_form("24315", "1300024317")
    assert notes.is_account_form("042900013", "0429000013")                   # «0429» + «00013»
    assert notes.is_account_form("13000017", "1300000017") and notes.is_account_form("130017", "1300000017")
    assert not notes.is_account_form("042900014", "0429000013")
    assert not notes.is_account_form("0013", "0429000013")                    # 2 значащие цифры — мало
    assert not notes.is_account_form("04290", "0429000000")                   # остаток счёта — нули


def test_agrees():
    acc = {"1300017777"}
    assert notes.agrees([("account", "17777"), ("reading", "51234")], acc, "51234")
    assert notes.agrees([("", "17777")], acc, None) and notes.agrees([("", "51234")], acc, "51234,0")
    assert not notes.agrees([("account", "51234")], acc, "51234")             # «лс» — только счёт
    assert not notes.agrees([("reading", "17777")], acc, "51234")
    assert not notes.agrees([], acc, "1") and not notes.agrees([("", "777")], acc, "51234")


def _table(tmp_path) -> Path:
    t = tmp_path / "реестр.csv"
    t.write_text("Лицевой счет;Номер счетчика;Последние показания;Текущие показания\n"
                 "1300017777;095011111;51000;51234\n1300024317;1260001;4300;5100\n"
                 "0429005357;2250001;1;2\n1300000001;111111;1;2\n", encoding="utf-8")
    return t


def _write(path: Path, img) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_image(path, img)
    return path


def test_notes_run_on_archive_folder(tmp_path, capsys):
    blue = (250, 199, 52)
    photos = tmp_path / "фото"
    _write(photos / "плюс" / "1300017777.jpg", _photo([("пок51234", blue, 500, 0), ("лс17777", blue, 700, 0)]))
    _write(photos / "минус" / "1300024317.jpg", _photo([("24317", blue, 600, 2)], seed=5))
    _write(photos / "плюс" / "0429005357.jpg", _photo([("9005358", (40, 40, 245), 600, 4)], seed=6))   # не тот счёт
    _write(photos / "плюс" / "1300000001.jpg", _jpeg(_background(7)))                                  # без надписи
    _write(photos / "плюс" / "1399999999.jpg", _photo([("12345", blue, 600, 0)], seed=8))           # счёта нет в таблице
    (photos / "битое.jpg").write_bytes(b"x")
    text = notes.run(photos, _table(tmp_path))
    out = tmp_path / "фото_надписи"
    assert text.splitlines()[:4] == [
        f"Надписи: {photos} — фото 6",
        "  с надписью: 4 (67%), строк: 5",
        "  уверенно прочитано и сверено с таблицей: 4 — верно 3, не совпало 1",
        "  не уверен (пошло бы оператору): 0"]
    assert "  уверенно, но ответа в таблице нет: 1" in text and "  не открылись как картинка: 1" in text
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.png")) == [
        "верно/1300017777__1.png", "верно/1300017777__2.png", "верно/1300024317__1.png",
        "не_совпало/0429005357__1.png", "нет_ответа/1399999999__1.png"]
    with open(out / notes.REPORT, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    assert rows[0]["фото"] == "минус/1300024317.jpg" and rows[0]["текст"] == "24317" and rows[0]["итог"] == "верно"
    assert "Фото 6 из 6" in capsys.readouterr().out
    # повторный замер — прошлые вырезки заменяются
    (photos / "плюс" / "0429005357.jpg").unlink()
    notes.run(photos, _table(tmp_path))
    assert not (out / "не_совпало").exists()


def test_notes_number_wrapped_to_next_line(tmp_path):
    photos = tmp_path / "ф"
    blue = (250, 199, 52)
    _write(photos / "0429005357.jpg", _photo([("лс04290", blue, 450, 0), ("05357", blue, 650, 0)]))
    notes.run(photos, _table(tmp_path), tmp_path / "o")
    with open(tmp_path / "o" / notes.REPORT, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    assert [(r["текст"], r["итог"]) for r in rows] == [("лс04290", "верно"), ("05357", "верно")]


def test_notes_on_archive_etalon(tmp_path):
    et = tmp_path / "etalon_archive"
    blue = (250, 199, 52)
    _write(et / "photos" / "aaa.jpg", _photo([("24317", blue, 600, 0)]))             # счёт — по номеру 1260001
    _write(et / "photos" / "bbb.jpg", _photo([("лс17777", blue, 600, 0)], seed=2))   # в таблице номер с нулём
    _write(et / "photos" / "ccc.jpg", _photo([("пок99999", blue, 600, 2)], seed=3))
    with open(et / "answers.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["photo_hash", "file", "serial", "reading"])
        w.writeheader()
        w.writerows([{"photo_hash": "a", "file": "aaa.jpg", "serial": "1260001", "reading": "5100"},
                     {"photo_hash": "b", "file": "bbb.jpg", "serial": "095011111", "reading": "51234"},
                     {"photo_hash": "c", "file": "ccc.jpg", "serial": "95011111", "reading": "51234"}])
    text = notes.run(et, _table(tmp_path))
    assert "фото 3" in text and "верно 2, не совпало 1" in text
    assert (tmp_path / "etalon_archive_надписи" / "не_совпало" / "ccc__1.png").is_file()


def test_notes_command(tmp_path, capsys):
    photos = tmp_path / "ф"
    _write(photos / "1300024317.jpg", _photo([("24317", (250, 199, 52), 600, 0)]))
    assert gmr.main(["notes", str(photos), "--table", str(_table(tmp_path)), "--out", str(tmp_path / "o")]) == 0
    assert "верно 1, не совпало 0" in capsys.readouterr().out
    assert (tmp_path / "o" / "верно" / "1300024317__1.png").is_file()
    assert gmr.main(["notes", str(tmp_path / "нет"), "--table", str(_table(tmp_path))]) == 1
    assert "ОШИБКА: нет папки" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        gmr.main(["notes", str(photos)])
