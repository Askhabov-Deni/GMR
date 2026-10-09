"""
Этап 7a: замер наклона (tools/angles.py, `gmr.py angles`). Модели —
подделки в интерфейсе YOLOInferer / CNNInferer / CRNNInferer, обёрнутые
настоящими адаптерами. Фото узнаётся по метке в левом верхнем углу: если
фото повёрнуто, метки там нет — «детектор» ничего не находит.
"""
import csv

import cv2
import numpy as np
import pytest

import gmr
from src.gmr.application.recognition import RecognitionModels
from src.gmr.domain.preset import RecognitionPreset
from src.gmr.ml import CnnDigitRecognizer, CrnnSerialRecognizer, YoloDigitDetector, YoloMeterDetector
from src.gmr.render.image_io import write_image
from tests._month import make_month
from tools import angles

UP = 200      # метка «кроп стоит правильно» в левом верхнем углу кропа


def _crop(kind="up"):
    c = np.zeros((20, 60, 3), np.uint8)
    if kind != "blank":                       # blank — не читается никак
        c[0:2, 0:2] = UP
    return cv2.rotate(c, cv2.ROTATE_180) if kind == "down" else c


# метка фото → (угол показаний или None — показаний нет, рамка, какой кроп, угол таблички или None)
SPEC = {
    1: (1.0, (10, 10, 70, 30), "up", -0.5),
    2: (-12.0, (10, 10, 70, 30), "up", 25.0),
    3: (3.0, (10, 10, 30, 70), "down", 7.5),      # боком и вверх ногами
    4: (0.0, (10, 10, 70, 30), "up", None),
    5: (None, None, "up", 3.0),                   # только табличка
    6: (5.0, (10, 10, 70, 30), "blank", None),    # показания не читаются и перевёрнутые
}


class FakeMeterYolo:
    def __init__(self):
        self.calls = 0

    def process_array(self, img, save_crops=False, max_per_class=1, straighten=None):
        self.calls += 1
        tag = int(img[0, 0, 0])
        if tag not in SPEC:
            print("⚠️ Детектор ничего не нашел")
            return None
        angle, bbox, kind, s_angle = SPEC[tag]
        out = []
        if angle is not None:
            out.append({"class": "gas_meter", "conf": 0.9, "angle": angle, "bbox": bbox, "crop": _crop(kind)})
        if s_angle is not None:
            out.append({"class": "serial_number", "conf": 0.9, "angle": s_angle, "bbox": (0, 40, 50, 50),
                        "crop": _crop(kind)})
        return out


class FakeDigitsYolo:
    def process_array(self, crop, save_crops=False, max_per_class=5, straighten=None):
        if int(crop[0, 0, 0]) != UP:
            return None
        return [{"class": "digit", "conf": 0.9, "angle": 0.0, "bbox": (i * 12, 0, i * 12 + 10, 20),
                 "crop": crop[:, i * 12:i * 12 + 10]} for i in range(5)]


class FakeCnn:
    def predict(self, crop):
        return {"digit": 1, "confidence": 0.95}


class FakeCrnn:
    def predict_with_details(self, crop):
        return {"text": "12345", "avg_confidence": 0.95 if int(crop[0, 0, 0]) == UP else 0.3, "details": []}


def _models():
    return RecognitionModels(YoloMeterDetector(FakeMeterYolo()), YoloDigitDetector(FakeDigitsYolo()),
                             CnnDigitRecognizer(FakeCnn()), CrnnSerialRecognizer(FakeCrnn()))


def _photo(path, tag, turn=None, corners=False):
    img = np.zeros((60, 90, 3), np.uint8)
    img[0:6, 0:6] = tag
    if corners:                               # метка во всех углах: и повёрнутое фото «узнаётся»
        img[-6:, :6] = img[:6, -6:] = img[-6:, -6:] = tag
    write_image(path, cv2.rotate(img, turn) if turn is not None else img)


def _folder(root):
    root.mkdir(parents=True, exist_ok=True)
    _photo(root / "a.png", 1)
    _photo(root / "b.png", 2)
    _photo(root / "c.png", 3)
    (root / "sub").mkdir()
    _photo(root / "sub" / "d.png", 4)
    _photo(root / "turned.png", 1, cv2.ROTATE_90_COUNTERCLOCKWISE)   # вернуть — повернуть по часовой (90°)
    _photo(root / "upside.png", 1, cv2.ROTATE_180)
    _photo(root / "turned270.png", 1, cv2.ROTATE_90_CLOCKWISE)
    _photo(root / "only_serial.png", 5, corners=True)
    _photo(root / "blank.png", 6)
    _photo(root / "empty.png", 0)                                    # счётчика нет вообще
    (root / "broken.jpg").write_bytes(b"not an image")
    return root


def _csv(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return {r["фото"]: r for r in csv.DictReader(fh, delimiter=";")}


def test_measure_folder(tmp_path, capsys):
    folder = _folder(tmp_path / "photos")
    text = angles.run(folder, models=_models())
    assert "Детектор ничего не нашел" not in capsys.readouterr().out    # повороты не шумят
    lines = text.splitlines()
    assert lines[0] == f"Замер наклона: 11 фото ({folder})"
    assert "Показания найдены: 5 (45%)" in lines
    assert "  наклон: до 2°: 2  2–5°: 1  5–10°: 1  10–15°: 1  15–25°: 0  25° и больше: 0" in lines
    assert "  счётчик боком (рамка выше, чем шире): 1" in lines
    assert "  не прочитаны: 2 — прочитались бы, если перевернуть на 180°: 1" in lines
    assert "Табличка с номером найдена: 4 (36%)" in lines
    assert "  наклон: до 2°: 1  2–5°: 1  5–10°: 1  10–15°: 0  15–25°: 0  25° и больше: 1" in lines
    assert "  номер неуверенный: 1 — уверенный, если перевернуть на 180°: 1" in lines
    assert "Счётчик не найден: 5 (45%)" in lines
    assert "  нашёлся бы, если повернуть фото: на 90° — 1, на 180° — 1, на 270° — 1; никак — 2" in lines
    assert "Не открылись как картинка: 1" in lines
    out = tmp_path / "наклон_photos.csv"                               # рядом с папкой, не в ней
    assert text.endswith(f"По каждому фото: {out}")
    rows = _csv(out)
    assert set(rows) == {"a.png", "b.png", "c.png", "sub/d.png", "turned.png", "upside.png", "empty.png",
                         "broken.jpg", "turned270.png", "only_serial.png", "blank.png"}
    b, c = rows["b.png"], rows["c.png"]
    assert (b["наклон показаний"], b["наклон таблички"], b["прочитаны"], b["перевёрнутые прочитаны"]) == (
        "-12,0", "25,0", "да", "")
    assert (c["боком"], c["прочитаны"], c["перевёрнутые прочитаны"], c["номер уверенный"],
            c["перевёрнутый уверенный"]) == ("да", "нет", "да", "нет", "да")
    assert rows["turned.png"]["найден после поворота фото"] == "90" and rows["turned.png"]["показания найдены"] == "нет"
    assert rows["upside.png"]["найден после поворота фото"] == "180"
    assert rows["turned270.png"]["найден после поворота фото"] == "270"
    assert rows["only_serial.png"]["найден после поворота фото"] == ""     # табличка — ещё не счётчик
    assert (rows["blank.png"]["прочитаны"], rows["blank.png"]["перевёрнутые прочитаны"]) == ("нет", "нет")
    assert rows["empty.png"]["найден после поворота фото"] == "" and rows["broken.jpg"]["открылось"] == "нет"
    assert rows["sub/d.png"]["табличка найдена"] == "нет" and rows["sub/d.png"]["наклон таблички"] == ""


def test_month_folder_uses_its_photos_preset_and_csv(tmp_path):
    from src.gmr.application import month as month_app
    f = make_month(tmp_path, [("100000", "A1", "1", "")])
    month_app.set_month_preset(f.root, RecognitionPreset(serial_pad=0.05, forgive_last=0))
    (f.photos / "Аюб").mkdir(parents=True)
    _photo(f.photos / "Аюб" / "a.png", 1)
    folder, csv_path, cfg = angles._source(f.root)
    assert (folder, csv_path) == (f.photos, f.root / "наклон.csv")
    assert (cfg.serial_crop_pad, cfg.ignore_last_digits) == (0.05, 0)
    text = angles.run(f.root, models=_models())
    assert "Замер наклона: 1 фото" in text and set(_csv(f.root / "наклон.csv")) == {"Аюб/a.png"}


def test_limit_and_own_csv(tmp_path):
    folder = _folder(tmp_path / "photos")
    text = angles.run(folder, tmp_path / "x.csv", limit=2, models=_models())
    assert text.startswith("Замер наклона: 2 фото") and len(_csv(tmp_path / "x.csv")) == 2


@pytest.mark.parametrize("a, name", [(0.0, "до 2°"), (-1.5, "до 2°"), (2.0, "2–5°"), (-9.5, "5–10°"),
                                     (14.5, "10–15°"), (24.5, "15–25°"), (25.0, "25° и больше"),
                                     (-25.0, "25° и больше")])
def test_angle_bins(a, name):
    assert f"{name}: 1" in angles.angle_bins([a])


def test_command(tmp_path, monkeypatch, capsys):
    folder = _folder(tmp_path / "photos")
    from src.gmr.ml import loader
    seen = []
    monkeypatch.setattr(loader, "load_models", lambda cfg: seen.append(cfg) or _models())
    assert gmr.main(["angles", str(folder), "--limit", "3"]) == 0
    out = capsys.readouterr().out
    assert "Фото 3 из 3" in out and "Замер наклона: 3 фото" in out and len(seen) == 1
    (tmp_path / "none").mkdir()
    assert gmr.main(["angles", str(tmp_path / "none")]) == 1
    assert "ОШИБКА: в " in capsys.readouterr().out
    assert gmr.main(["angles", str(tmp_path / "nothing")]) == 1


def test_detect_array_adapter():
    seen = []

    class Inferer:
        def process_array(self, img, save_crops=True, max_per_class=1):
            seen.append(save_crops)
            return []
    assert YoloMeterDetector(Inferer()).detect_array(np.zeros((2, 2, 3), np.uint8)) == [] and seen == [False]
