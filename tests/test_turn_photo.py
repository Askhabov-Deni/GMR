"""
Этап 7b (решение владельца 2026-10-12): детектор на фото не нашёл вообще
ничего → то же фото, повёрнутое на 90, 180, 270° (настройка месяца
«на фото ничего не найдено — повернуть», по умолчанию выключена). Замер 7a:
повёрнутые кропы показаний и номера читались ложно (много девяток) —
поэтому поворачивается только фото, где не найдено ничего.
"""
import logging
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import gmr
import reader
from src.gmr.application import detect_meter, recognize_photo
from src.gmr.application.recognition import RecognitionModels
from src.gmr.domain import Outcome, PipelineConfig
from src.gmr.domain.preset import RecognitionPreset
from src.gmr.ml import YoloMeterDetector
from src.gmr.render import read_image, turn_image, write_image
from tests._fixtures import (
    FakeDigitDetector, FakeDigitOCR, FakeSerialOCR, make_df, make_digit_crops_with_centers,
    make_meter_crops, process_photo,
)
from tests._month import make_month
from tests.test_account_model import TABLE

MARK = 9      # метка в левом верхнем углу фото, когда оно стоит прямо


def _photo(path, turn=None, size=(40, 60)):
    """Фото (высота, ширина); turn — как оно лежит (код cv2.rotate), None — прямо."""
    img = np.zeros((*size, 3), np.uint8)
    img[0:4, 0:4] = MARK
    img[-4:, -4:] = 7                         # правый нижний угол (левый верхний закроет подпись)
    write_image(path, cv2.rotate(img, turn) if turn is not None else img)
    return str(path)


class TurnedYolo:
    """YOLOInferer: на исходном фото не находит ничего (process_image), на
    картинке в памяти — счётчик, если метка в левом верхнем углу; «only_serial»
    — только табличку (это ещё не счётчик)."""
    def __init__(self, crops, only_serial=False):
        self.crops, self.only_serial, self.seen = crops, only_serial, []

    def process_image(self, path, save_crops=False, max_per_class=1, straighten=None):
        return []

    def process_array(self, img, save_crops=False, max_per_class=1, straighten=None):
        self.seen.append(img.shape[:2])
        if self.only_serial:
            return [c for c in self.crops if c["class"] == "serial_number"]
        return self.crops if int(img[0, 0, 0]) == MARK else None


def test_turn_image():
    img = np.arange(6, dtype=np.uint8).reshape(2, 3)
    assert turn_image(img, 0) is img
    assert turn_image(img, 90).tolist() == [[3, 0], [4, 1], [5, 2]]          # по часовой
    assert turn_image(img, 180).tolist() == [[5, 4, 3], [2, 1, 0]]
    assert turn_image(img, 270).tolist() == [[2, 5], [1, 4], [0, 3]]


@pytest.mark.parametrize("lies, turn", [(cv2.ROTATE_90_COUNTERCLOCKWISE, 90), (cv2.ROTATE_180, 180),
                                        (cv2.ROTATE_90_CLOCKWISE, 270)])
def test_detect_turned_finds_first_turn_with_meter(tmp_path, lies, turn):
    yolo = TurnedYolo(make_meter_crops())
    found, degrees = YoloMeterDetector(yolo).detect_turned(_photo(tmp_path / "p.png", lies))
    assert degrees == turn and found == yolo.crops
    assert len(yolo.seen) == [90, 180, 270].index(turn) + 1                   # дальше не крутит


def test_detect_turned_nothing(tmp_path):
    yolo = TurnedYolo(make_meter_crops())
    det = YoloMeterDetector(yolo)
    assert det.detect_turned(str(tmp_path / "нет.png")) == (None, 0)         # не открылось
    (tmp_path / "bad.jpg").write_bytes(b"x")
    assert det.detect_turned(str(tmp_path / "bad.jpg")) == (None, 0)
    blank = np.zeros((40, 60, 3), np.uint8)
    write_image(tmp_path / "blank.png", blank)
    assert det.detect_turned(str(tmp_path / "blank.png")) == (None, 0) and len(yolo.seen) == 3
    only = YoloMeterDetector(TurnedYolo(make_meter_crops(), only_serial=True))
    assert only.detect_turned(_photo(tmp_path / "s.png")) == (None, 0)      # табличка без счётчика — нет


class _Det:
    def __init__(self, found, turned=None):
        self.found, self.turned, self.calls = found, turned, 0

    def detect(self, path):
        return self.found

    def detect_turned(self, path):
        self.calls += 1
        return self.turned


def test_detect_meter_turns_only_when_nothing_found_and_switched_on():
    on, off = PipelineConfig(turn_if_nothing=True), PipelineConfig()
    nothing = _Det([], ([{"class": "gas_meter"}], 180))
    assert detect_meter(nothing, "p", off) == ([], 0) and nothing.calls == 0
    assert detect_meter(nothing, "p", on) == ([{"class": "gas_meter"}], 180) and nothing.calls == 1
    something = _Det([{"class": "serial_number"}], ([{"class": "gas_meter"}], 90))
    assert detect_meter(something, "p", on) == ([{"class": "serial_number"}], 0) and something.calls == 0
    plain = SimpleNamespace(detect=lambda p: None)                                # нет detect_turned
    assert detect_meter(plain, "p", on) == (None, 0)


def _reader_photo(config, path):
    digit_crops, arrays = make_digit_crops_with_centers([i * 40 for i in range(5)])
    docr = FakeDigitOCR({id(a): (d, 0.95) for a, d in zip(arrays, "01200")})
    return process_photo(path, make_df(TABLE), config, TurnedYolo(make_meter_crops()),
                         FakeDigitDetector(digit_crops), docr, FakeSerialOCR("123456", 0.95))


def test_reader_turns_photo_where_nothing_found(base_config, tmp_path):
    path = _photo(tmp_path / "p.png", cv2.ROTATE_90_COUNTERCLOCKWISE)
    r = _reader_photo(base_config, path)
    assert (r.outcome, r.error_detail, r.photo_turn) == (Outcome.NO_METER, "YOLO returned no crops", 0)
    base_config.turn_if_nothing = True
    r = _reader_photo(base_config, path)
    assert (r.outcome, r.account_id, r.reading, r.photo_turn) == (Outcome.PLUS, "1300000065", 1200, 90)
    notes = reader._make_log_row("p.png", r, base_config)["notes"]
    assert notes.startswith("фото повёрнуто на 90°: на исходном детектор ничего не нашёл")
    assert reader._make_log_row("p.png", SimpleNamespace(**{**vars(r), "photo_turn": 0}),
                                base_config)["notes"] == ""


def test_result_photo_saved_turned(tmp_path):
    from src.gmr.domain.models import PhotoResult
    src = _photo(tmp_path / "p.png", cv2.ROTATE_90_COUNTERCLOCKWISE, (400, 600))   # лежит боком
    r = PhotoResult(photo_path=src, outcome=Outcome.PLUS, photo_turn=90)
    reader._save_annotated(src, str(tmp_path / "out"), "a.png", r, move=False)
    saved = read_image(tmp_path / "out" / "a.png")
    assert saved.shape[:2] == (400, 600) and int(saved[399, 599, 0]) == 7    # стоит прямо
    r.photo_turn = 0
    reader._save_annotated(src, str(tmp_path / "out"), "b.png", r, move=False)
    assert read_image(tmp_path / "out" / "b.png").shape[:2] == (600, 400)


def test_recognize_photo_turns_too():
    det = _Det(None, ([{"class": "gas_meter", "crop": None}], 270))
    models = RecognitionModels(det, None, None, None)
    import src.gmr.application.recognition as rec_mod
    old = rec_mod.read_meter_digits_for_config
    rec_mod.read_meter_digits_for_config = lambda *a, **k: "digits"
    try:
        rec = recognize_photo(models, "p", PipelineConfig(turn_if_nothing=True))
        assert (rec.turn, rec.meter is not None, rec.digits) == (270, True, "digits")
        rec = recognize_photo(models, "p", PipelineConfig())
        assert (rec.turn, rec.detections) == (0, None)
    finally:
        rec_mod.read_meter_digits_for_config = old


def test_preset_switch(tmp_path):
    p = RecognitionPreset(turn_photo=True)
    assert RecognitionPreset.from_json(p.to_json()) == p
    assert RecognitionPreset.from_json('{"turn_photo": 1}') == RecognitionPreset()
    assert p.apply(PipelineConfig()).turn_if_nothing and not RecognitionPreset().apply(
        PipelineConfig(turn_if_nothing=True)).turn_if_nothing
    assert p.describe().endswith("дополнительно — фото без находок — повернуть")
    from src.gmr.application import month as month_app
    f = make_month(tmp_path, [("100000", "A1", "1", "")])
    month_app.set_month_preset(f.root, p)
    cfg = reader.PipelineConfig(month_dir=str(f.root))
    reader._apply_month_preset(cfg, f, logging.getLogger("t"))
    assert cfg.turn_if_nothing


def test_etalon_flag(tmp_path, monkeypatch):
    from tests.test_etalon import _etalon_with, _rec
    et = _etalon_with(tmp_path, [("a", "111111", "1234", "NO_METER")])
    seen = []
    from src.gmr.ml import loader
    monkeypatch.setattr(loader, "load_models", lambda cfg: seen.append(cfg))
    import src.gmr.application as app
    monkeypatch.setattr(app, "recognize_photo", lambda models, path, cfg, **kw: _rec("111111", number=1234))
    assert gmr.main(["etalon", "check", "--etalon", str(et), "--turn-photo"]) == 0
    assert gmr.main(["etalon", "check", "--etalon", str(et)]) == 0
    assert [c.turn_if_nothing for c in seen] == [True, False]
