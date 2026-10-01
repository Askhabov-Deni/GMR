"""
Модели и инструменты по ним: единые настройки моделей, учёт весов,
tools.inspect_model (появились в Фазе 4, docs/MIGRATION_TZ.md; до 2026-10-01
файл назывался test_phase4_models.py).

  1. Пороги standalone-инструментов = прод-пороги PipelineConfig (находка 1).
  2. CRNN-инференс: «хорошая» длина серийника — 4..10, а не ровно 5 (находка 2).
  3. Все torch.load в models/ — с weights_only=True; чекпоинт формата
     train_crnn.py грузится, чужой pickle — нет (находка 3).
  4. extract_digit_crops.py в архиве (находка 4).
  5. train_val_split.py: одинаковый split при повторном запуске (находка 5).
  6. Нет абсолютных путей C:\\AD\\... в models/ (находка 6).
  7. tools.inspect_model — все режимы на fake-моделях.
  8. tools.weights_manifest — отпечатки и запись в docs/models.md.
"""
import ast
import csv
import hashlib
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.gmr.domain import DigitPrediction, PipelineConfig, SerialPrediction
from tools import inspect_model, weights_manifest

ROOT = Path(__file__).resolve().parent.parent


# ─── 1. Пороги ───────────────────────────────────────────────────────────────

def test_cnn_cli_threshold_equals_prod():
    from models.cnn import config_cnn, infer_cnn
    assert config_cnn.MIN_CONFIDENCE == PipelineConfig().digit_conf_thresh
    default = inspect.signature(infer_cnn.CNNInferer.process_directory).parameters["min_confidence"].default
    assert default == PipelineConfig().digit_conf_thresh


def test_crnn_cli_threshold_equals_prod():
    from models.crnn import config_crnn, infer_crnn
    assert config_crnn.MIN_CONFIDENCE == PipelineConfig().serial_conf_thresh
    default = inspect.signature(infer_crnn.CRNNInferer.process_directory).parameters["min_confidence"].default
    assert default == PipelineConfig().serial_conf_thresh
    # и в argparse — не захардкоженное число
    src = (ROOT / "models/crnn/infer_crnn.py").read_text(encoding="utf-8")
    assert 'default=MIN_CONFIDENCE' in src and "default=0.8" not in src


# ─── 2. Длина серийника в CRNN-инструменте ──────────────────────────────────

def _crnn_stats(tmp_path, texts, **kwargs):
    from models.crnn.infer_crnn import CRNNInferer
    for i in range(len(texts)):
        (tmp_path / f"{i}.jpg").write_bytes(b"x")
    it = iter(texts)
    inferer = CRNNInferer.__new__(CRNNInferer)
    inferer.predict_with_details = lambda path: {
        "text": next(it), "avg_confidence": 0.99, "details": []}
    return inferer.process_directory(str(tmp_path), **kwargs)


def test_crnn_good_length_is_label_range(tmp_path):
    # 4..10 — MIN/MAX_LABEL_LENGTH из config_crnn.py; 3 и 11 — плохие
    stats = _crnn_stats(tmp_path, ["123", "1234", "1234567", "1234567890", "12345678901"])
    assert stats == {"good": 3, "bad": 2}


def test_crnn_expected_length_still_available(tmp_path):
    stats = _crnn_stats(tmp_path, ["12345", "1234567"], expected_length=5)
    assert stats == {"good": 1, "bad": 1}


# ─── 3. weights_only ─────────────────────────────────────────────────────────

def test_all_torch_load_in_models_use_weights_only():
    missing = []
    for path in (ROOT / "models").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"):
                kw = {k.arg: k.value for k in node.keywords}
                ok = "weights_only" in kw and getattr(kw["weights_only"], "value", None) is True
                if not ok:
                    missing.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert missing == []


def test_crnn_loads_checkpoint_in_train_format(tmp_path):
    # формат torch.save из models/crnn/train_crnn.py
    from models.crnn.model_crnn import CRNN
    model = CRNN()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    ckpt = tmp_path / "best.pt"
    torch.save({"epoch": 3, "model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                "val_acc": 0.9266, "val_cer": 0.02}, ckpt)
    loaded = CRNN.from_pretrained(str(ckpt))
    for k, v in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v)


class _NotWeights:
    pass


def test_crnn_refuses_arbitrary_pickle(tmp_path):
    from models.crnn.model_crnn import CRNN
    ckpt = tmp_path / "evil.pt"
    torch.save({"model_state": CRNN().state_dict(), "extra": _NotWeights()}, ckpt)
    with pytest.raises(Exception):
        CRNN.from_pretrained(str(ckpt))


# ─── 4. Архив ────────────────────────────────────────────────────────────────

def test_extract_digit_crops_archived():
    assert not (ROOT / "models/cnn/extract_digit_crops.py").exists()
    assert (ROOT / "archive/models/cnn/extract_digit_crops.py").exists()


# ─── 5. Воспроизводимый split ────────────────────────────────────────────────

def _run_split(root: Path, out_name: str, create_order) -> tuple[list, list]:
    images, labels = root / "images", root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(exist_ok=True)
    for name in create_order:
        (images / f"{name}.jpg").write_bytes(b"x")
        (labels / f"{name}.txt").write_text("0 0.5 0.5 0.1 0.1")
    out = root / out_name
    subprocess.run(
        [sys.executable, str(ROOT / "models/yolo_all_detect/train_val_split.py"),
         "--images_dir", str(images), "--labels_dir", str(labels), "--output_dir", str(out)],
        check=True, capture_output=True, cwd=root,
    )
    return (sorted(p.name for p in (out / "train/images").iterdir()),
            sorted(p.name for p in (out / "val/images").iterdir()))


def test_train_val_split_is_reproducible(tmp_path):
    names = [f"p{i:03d}" for i in range(50)]
    a = _run_split(tmp_path / "a", "out", names)
    b = _run_split(tmp_path / "b", "out", list(reversed(names)))  # другой порядок на диске
    assert a == b
    assert len(a[0]) == 40 and len(a[1]) == 10


# ─── 6. Абсолютные пути ─────────────────────────────────────────────────────

def test_no_absolute_windows_paths_in_models_code():
    bad = []
    for path in (ROOT / "models").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and ("C:\\AD" in node.value or "D:\\" in node.value):
                bad.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert bad == []


def test_build_dataset_cnn_has_cli():
    r = subprocess.run([sys.executable, str(ROOT / "models/cnn/build_dataset_cnn.py"), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "--digits_dir" in r.stdout


# ─── 7. tools.inspect_model ──────────────────────────────────────────────────

def _img(h, w, value):
    return np.full((h, w, 3), value, dtype=np.uint8)


class FakeMeter:
    def __init__(self, dets):
        self.dets, self.calls = dets, []

    def detect(self, photo_path):
        self.calls.append(photo_path)
        return self.dets


class FakeDigits:
    """5 цифр слева направо; кроп цифры i залит значением i+1."""
    def __init__(self, n=5):
        self.n = n

    def detect(self, crop):
        return [{"class": "digit", "conf": 0.9, "angle": 0.0, "bbox": (x * 20, 0, x * 20 + 15, 30),
                 "crop": _img(30, 15, x + 1), "path": None}
                for x in reversed(range(self.n))]   # нарочно не по порядку


class FakeDigit:
    def recognize(self, crop):
        v = int(crop[0, 0, 0])
        return DigitPrediction(digit=str(v), confidence=0.5 if v == 3 else 0.95)


class FakeSerial:
    def recognize(self, crop):
        return SerialPrediction(text="0071234", confidence=0.55,
                                details=[{"char": "0", "conf": 0.5}])


def _meter_dets():
    return [
        {"class": "gas_meter", "conf": 0.93, "angle": 1.5, "bbox": (10, 10, 110, 50),
         "crop": _img(40, 100, 50), "path": None},
        {"class": "serial_number", "conf": 0.88, "angle": 0.0, "bbox": (10, 60, 90, 80),
         "crop": _img(20, 80, 70), "path": None},
    ]


def _fake_loaders(meter_dets=None, n_digits=5):
    def factory(cfg):
        factory.cfg = cfg
        return {"meter": lambda: FakeMeter(_meter_dets() if meter_dets is None else meter_dets),
                "digits": lambda: FakeDigits(n_digits),
                "serial": lambda: FakeSerial(),
                "digit": lambda: FakeDigit()}
    return factory


@pytest.fixture
def photos(tmp_path):
    # подпапка с кириллицей — как папки операторов
    d = tmp_path / "in" / "Сулиман С"
    d.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        inspect_model.write_image(d / name, _img(100, 200, 90))
    (d / "notes.txt").write_text("не картинка")
    return tmp_path / "in"


def _summary(out):
    with open(out / "summary.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";"))


def test_inspect_meter(photos, tmp_path):
    out = tmp_path / "out"
    rows = inspect_model.main(["meter", str(photos), "--out", str(out)], _fake_loaders())
    assert len(rows) == 4   # 2 фото × 2 детекции, txt пропущен
    assert {r["class"] for r in rows} == {"gas_meter", "serial_number"}
    assert (out / "Сулиман С__a.jpg").exists()
    assert (out / "Сулиман С__a__gas_meter_0.jpg").exists()
    assert len(_summary(out)) == 4


def test_inspect_meter_nothing_found(photos, tmp_path):
    rows = inspect_model.main(["meter", str(photos), "--out", str(tmp_path / "o")], _fake_loaders([]))
    assert [r["note"] for r in rows] == ["ничего не найдено"] * 2


def test_inspect_digits_sorted_left_to_right(photos, tmp_path):
    out = tmp_path / "out"
    rows = inspect_model.main(["digits", str(photos / "Сулиман С" / "a.jpg"), "--out", str(out)],
                              _fake_loaders())
    assert [r["position"] for r in rows] == [0, 1, 2, 3, 4]
    assert [int(r["bbox"].split()[0]) for r in rows] == [0, 20, 40, 60, 80]
    assert all(r["n_digits"] == 5 and r["expected"] == 5 for r in rows)
    assert (out / "a__d4.jpg").exists() and (out / "a.jpg").exists()


def test_inspect_digits_from_meter_crop_skips_meter_detector(tmp_path):
    crop = tmp_path / "meter_crop.jpg"
    inspect_model.write_image(crop, _img(40, 100, 50))
    factory = _fake_loaders(meter_dets=[])   # детектор счётчика ничего бы не нашёл
    rows = inspect_model.main(["digits", str(crop), "--from", "crop", "--out", str(tmp_path / "o")], factory)
    assert len(rows) == 5


def test_inspect_serial_uses_prod_threshold(photos, tmp_path):
    rows = inspect_model.main(["serial", str(photos), "--out", str(tmp_path / "o")], _fake_loaders())
    assert rows[0]["threshold"] == PipelineConfig().serial_conf_thresh
    assert rows[0]["text"] == "0071234" and rows[0]["ok"] is False   # 0.55 < 0.6
    rows = inspect_model.main(["serial", str(photos), "--conf", "0.5", "--out", str(tmp_path / "o2")],
                              _fake_loaders())
    assert rows[0]["threshold"] == 0.5 and rows[0]["ok"] is True


def test_inspect_digit_from_crops_and_from_photo(photos, tmp_path):
    crops = tmp_path / "crops"
    crops.mkdir()
    inspect_model.write_image(crops / "c.png", _img(30, 15, 7))   # png — без потерь
    rows = inspect_model.main(["digit", str(crops), "--out", str(tmp_path / "o")], _fake_loaders())
    assert [(r["digit"], r["ok"]) for r in rows] == [("7", True)]

    rows = inspect_model.main(["digit", str(photos / "Сулиман С" / "a.jpg"), "--from", "photo",
                               "--out", str(tmp_path / "o2")], _fake_loaders())
    assert [r["digit"] for r in rows] == ["1", "2", "3", "4", "5"]
    assert [r["ok"] for r in rows] == [True, True, False, True, True]


def test_inspect_photo_matches_reader_rules(photos, tmp_path):
    out = tmp_path / "out"
    rows = inspect_model.main(["photo", str(photos / "Сулиман С" / "a.jpg"), "--out", str(out)],
                              _fake_loaders())
    (row,) = rows
    assert row["serial_text"] == "0071234" and row["serial_ok"] is False
    # цифра 3 (позиция 2) неуверенна и не в прощённых хвостовых → ошибка, как в reader.py
    assert row["reading_str"] == "12?45" and row["reading"] is None
    assert row["error"].startswith("low conf digits")
    assert (out / "a__meter.jpg").exists()


def test_inspect_photo_rejects_conf_and_weights(photos):
    with pytest.raises(SystemExit):
        inspect_model.main(["photo", str(photos), "--conf", "0.1"], _fake_loaders())


def test_inspect_weights_override_goes_to_selected_model_only(photos, tmp_path):
    factory = _fake_loaders()
    inspect_model.main(["digit", str(photos), "--from", "photo", "--weights", "new.pth",
                        "--out", str(tmp_path / "o")], factory)
    default = PipelineConfig()
    assert factory.cfg.digit_ocr_model == "new.pth"
    assert factory.cfg.digit_detect_model == default.digit_detect_model
    assert factory.cfg.meter_detect_model == default.meter_detect_model


def test_inspect_one_broken_image_does_not_stop_folder(photos, tmp_path):
    (photos / "Сулиман С" / "broken.jpg").write_bytes(b"not an image")
    rows = inspect_model.main(["serial", str(photos), "--from", "crop", "--out", str(tmp_path / "o")],
                              _fake_loaders())
    assert sum(r.get("note") == "не удалось открыть" for r in rows) == 1
    assert sum("text" in r for r in rows) == 2


def test_default_loaders_use_prod_loader(monkeypatch):
    from src.gmr.ml import loader
    seen = {}
    monkeypatch.setattr(loader, "load_meter_detector", lambda cfg, resolve: seen.setdefault("m", resolve("x.pt")))
    loaders = inspect_model.default_loaders(PipelineConfig())
    assert loaders["meter"]() == str(ROOT / "x.pt")


# ─── 8. tools.weights_manifest ───────────────────────────────────────────────

def test_weights_manifest(tmp_path):
    cfg = PipelineConfig(meter_detect_model="w/m.pt", digit_detect_model="w/d.pt",
                         digit_ocr_model="w/c.pth", serial_ocr_model="w/missing.pt")
    (tmp_path / "w").mkdir()
    for name, data in (("m.pt", b"meter"), ("d.pt", b"digits"), ("c.pth", b"cnn")):
        (tmp_path / "w" / name).write_bytes(data)
    rows = weights_manifest.manifest(cfg, root=tmp_path)
    assert rows[0]["sha256"] == hashlib.sha256(b"meter").hexdigest()[:16]
    assert rows[3]["sha256"] == "ФАЙЛ НЕ НАЙДЕН"

    md = tmp_path / "models.md"
    md.write_text(f"до\n{weights_manifest.BEGIN}\nстарое\n{weights_manifest.END}\nпосле\n", encoding="utf-8")
    weights_manifest.write_into(md, weights_manifest.to_markdown(rows))
    text = md.read_text(encoding="utf-8")
    assert "старое" not in text and text.startswith("до\n") and text.endswith("после\n")
    assert hashlib.sha256(b"cnn").hexdigest()[:16] in text


def test_weights_manifest_keeps_line_endings(tmp_path):
    for nl in ("\n", "\r\n"):
        md = tmp_path / "m.md"
        md.write_bytes(f"a{nl}{weights_manifest.BEGIN}{nl}{weights_manifest.END}{nl}b{nl}".encode())
        weights_manifest.write_into(md, "x\ny")
        data = md.read_bytes().decode()
        assert data == f"a{nl}{weights_manifest.BEGIN}{nl}x{nl}y{nl}{weights_manifest.END}{nl}b{nl}"


def test_cnn_transforms_use_fill_not_value():
    # albumentations 2.x: аргумент value игнорируется (UserWarning), цвет
    # паддинга задаётся fill. По умолчанию fill=0 — поведение то же, чёрный.
    import warnings
    from models.cnn import dataset_cnn
    for path in ("models/cnn/dataset_cnn.py", "models/cnn/infer_cnn.py"):
        assert "BORDER_CONSTANT, value=" not in (ROOT / path).read_text(encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dataset_cnn._train_transform()
        val = dataset_cnn._val_transform()
    assert not [w for w in caught if "'value'" in str(w.message)]
    # паддинг чёрный: узкая белая картинка → по краям после Normalize значение (0-mean)/std
    out = val(image=np.full((96, 10, 3), 255, dtype=np.uint8))["image"]
    expected = (0 - dataset_cnn.NORM_MEAN[0]) / dataset_cnn.NORM_STD[0]
    assert abs(float(out[0, 0, 0]) - expected) < 1e-4


def test_models_md_has_manifest_markers():
    text = (ROOT / "docs/models.md").read_text(encoding="utf-8")
    assert weights_manifest.BEGIN in text and weights_manifest.END in text
