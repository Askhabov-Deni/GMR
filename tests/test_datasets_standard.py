"""
Стандарт датасетов (docs/DATASETS.md, решение владельца 2026-10-15): имя кропа
<фото>__<что><№>[__<ответ>].jpg, ответ серийника — в имени (без labels/),
цифры — в папке; забор разметки месяца без повторов имён; классы детектора
одним словом. (Разовый переход старых имён сделан и удалён, 2026-10-18.)
"""
from pathlib import Path

import pytest

from models import datasets as ds
from models.crnn import dataset_crnn
from src.gmr.domain.datasets import (
    CropName, crop_name, free_crop_name, parse_crop_name, photo_id,
)
from tests._month import make_month
from tools import datasets as check_ds


def touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ─── Имена ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name, crop", [
    ("1300000375__serial1__0070351.jpg", CropName("1300000375", "serial", 1, "0070351")),
    ("1300000375-2__account1.jpg", CropName("1300000375-2", "account", 1)),
    ("IMG_12__digit3.png", CropName("IMG_12", "digit", 3)),
    ("IMG_1__account2__1300000082", CropName("IMG_1", "account", 2, "1300000082")),
    ("1300000143__serial_number_1_set1.jpeg", None),
    ("1300000375__serial1__.jpg", None),
    ("__serial1__1.jpg", None),
    ("a__b__serial1__1.jpg", None),
    ("1300000375__meter1.jpg", None),
])
def test_parse_crop_name(name, crop):
    assert parse_crop_name(name) == crop


def test_source_and_account():
    assert parse_crop_name("1300000375-2__serial1__1.jpg").source == "1300000375"
    assert parse_crop_name("1300000375-12__digit1.jpg").source == "1300000375"
    assert parse_crop_name("IMG-20241010__digit1.jpg").source == "IMG-20241010"    # не счёт — как есть
    assert parse_crop_name("1300000375-2__account1.jpg").account == "1300000375"
    assert parse_crop_name("IMG_1__account1__1300000082.jpg").account == "1300000082"


def test_make_names():
    assert crop_name("1300000375", "serial", 1, "0070351") == "1300000375__serial1__0070351.jpg"
    assert crop_name("1300000375", "digit", 3, ext=".png") == "1300000375__digit3.png"
    assert photo_id("IMG 12(1)") == "IMG_12_1" and photo_id("a__b") == "a_b" and photo_id("  ") == "photo"
    assert photo_id("Фото-1.2") == "Фото-1.2"
    with pytest.raises(ValueError):
        crop_name("1", "meter", 1)
    taken = ["1300__serial1__1.jpg", "1300-2__SERIAL1__1.JPG"]
    assert free_crop_name(taken, "1300", "serial", 1, "1") == "1300-3__serial1__1.jpg"
    assert free_crop_name(taken, "1300", "serial", 2, "1") == "1300__serial2__1.jpg"


def test_photo_key_keeps_one_meter_together():
    assert ds.photo_key("1300000375-2__serial1__0070351.jpg") == "1300000375"
    assert ds.photo_key("1300000375__digit3.jpg") == "1300000375"
    assert ds.photo_key("IMG-20241010-2.jpg") == "IMG-20241010-2"                 # фото для YOLO — как было
    assert ds.photo_key("1300000013__gas_meter_1__digit_3.jpg") == "1300000013"


def test_serial_label_from_name(tmp_path, capsys):
    for n in ("1300000375__serial1__0070351.jpg", "1300000375-2__serial1__0070351.JPG",
              "1300000376__serial1__123.jpg", "1300000377__serial1__12345678901.jpg",
              "1300000378__account1.jpg", "1300000380__account1__1234567.jpg", "старый.jpg"):
        touch(tmp_path / n)
    touch(tmp_path / "1300000379__serial1__1234567.txt")
    meta = dataset_crnn.load_dataset(tmp_path)
    assert [(m["file"].name, m["label"]) for m in meta] == [
        ("1300000375-2__serial1__0070351.JPG", "0070351"), ("1300000375__serial1__0070351.jpg", "0070351")]
    assert "Загружено: 2  |  имя не по стандарту или номер не из 4–10 цифр: 5" in capsys.readouterr().out
    assert dataset_crnn.label_of("1300000375__serial1__1234") == "1234"
    assert dataset_crnn.label_of("1300000375__serial1__1234567890") == "1234567890"


# ─── Забор разметки месяца: имена без повторов, повторный запуск — ничего ───

def test_collect_crops_unique_and_idempotent(tmp_path):
    from src.gmr.storage.month import MonthDB
    f = make_month(tmp_path, [("100000", "A1", "1", "")], name="Октябрь_2026")
    with MonthDB(f.db) as db, db.transaction():
        db.set_meta("created_at", "2026-10-01 09:00:00")
    root = tmp_path / "ds"
    touch(root / "serials_crnn" / "images" / "1300000001__serial1__0234567.jpg", b"old")   # уже в датасете
    touch(root / "digits_cnn" / "7" / "1300000001__digit3.jpg", b"old7")
    m = f.markup
    touch(m / "serials_crnn" / "images" / "1300000001__serial1__0234567.jpg", b"new")
    touch(m / "serials_crnn" / "images" / "1300000002__serial1__0234568.jpg", b"new2")
    touch(m / "digits_cnn" / "7" / "1300000001__digit3.jpg", b"new7")
    out = check_ds.collect(f.root, root)
    assert out == "Разметка месяца Октябрь_2026 (2026-10): digits_cnn — 1, serials_crnn — 2"
    new = root / "serials_crnn" / "new" / "2026-10" / "images"
    assert sorted(p.name for p in new.iterdir()) == [
        "1300000001-2__serial1__0234567.jpg", "1300000002__serial1__0234568.jpg"]
    assert (new / "1300000001-2__serial1__0234567.jpg").read_bytes() == b"new"
    assert (root / "digits_cnn" / "new" / "2026-10" / "7" / "1300000001-2__digit3.jpg").read_bytes() == b"new7"
    assert check_ds.collect(f.root, root) == "Разметка месяца Октябрь_2026 (2026-10): новых нет; уже были: 3"
    touch(m / "serials_crnn" / "images" / "1300000002-2__serial1__0234568.jpg", b"new3")    # ещё один кроп
    assert check_ds.collect(f.root, root).endswith("serials_crnn — 1; уже были: 3")
    assert (new / "1300000002-2__serial1__0234568.jpg").read_bytes() == b"new3"


# ─── Классы детектора — одним словом ─────────────────────────────────────────

def test_class_names_one_word():
    from src.gmr.application.recognition import find_detection
    from src.gmr.domain.datasets import standard_class
    assert [standard_class(c) for c in ("gas_meter", "serial_number", "marker_id", "digit", "meter")] == [
        "meter", "serial", "account", "digit", "meter"]
    dets = [{"class": "serial_number", "n": 1}, {"class": "gas_meter", "n": 2}, {"class": "account", "n": 3}]
    assert find_detection(dets, "meter")["n"] == 2 and find_detection(dets, "gas_meter")["n"] == 2
    assert find_detection(dets, "serial")["n"] == 1 and find_detection(dets, "marker_id")["n"] == 3
    assert find_detection(dets, "digit") is None and find_detection(None, "meter") is None


def test_turn_photo_finds_meter_by_new_class_name(tmp_path):
    import numpy as np

    from src.gmr.ml import YoloMeterDetector
    from src.gmr.render import write_image

    class Yolo:                                      # детектор, обученный уже с новыми именами
        def process_image(self, path, save_crops=False, max_per_class=1, straighten=None):
            return []

        def process_array(self, img, save_crops=False, max_per_class=1, straighten=None):
            return [{"class": "meter"}]
    write_image(tmp_path / "p.png", np.zeros((20, 30, 3), np.uint8))
    assert YoloMeterDetector(Yolo()).detect_turned(str(tmp_path / "p.png")) == ([{"class": "meter"}], 90)
