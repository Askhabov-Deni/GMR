"""
Фаза 3 (ML Interfaces, docs/MIGRATION_TZ.md): контракты моделей, адаптеры,
загрузчик и общий сервис распознавания для reader.py и program2.py.

  1. Адаптеры выполняют контракты и зовут модели с теми же аргументами,
     что reader.py до Фазы 3 (включая то, что straighten не переопределяется).
  2. load_models создаёт модели с теми же параметрами: выпрямление включено
     для детектора счётчика и выключено для детектора цифр.
  3. program2.ModelBundle.run_on_photo выдаёт тот же dict, что и раньше, и
     читает цифры тем же сервисом, что и reader.process_photo.
  4. Границы слоёв: program2.py не импортирует приватные _read_meter_digits /
     _find_crop_entry; domain и application не импортируют torch/cv2/pandas.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

from src.gmr.application import RecognitionModels, read_meter_digits
from src.gmr.domain import (
    DigitDetector, DigitPrediction, DigitRecognizer, MeterDetector,
    PipelineConfig, SerialPrediction, SerialRecognizer,
)
from src.gmr.ml import (
    CnnDigitRecognizer, CrnnSerialRecognizer, YoloDigitDetector, YoloMeterDetector,
)
from src.gmr.ml import loader
from tests._fixtures import (
    FakeDigitDetector, FakeDigitOCR, FakeMeterDetector, FakeSerialOCR,
    make_digit_crops_with_centers, make_meter_crops,
)

ROOT = Path(__file__).resolve().parent.parent


# ─── 1. Адаптеры ─────────────────────────────────────────────────────────────

class _Recorder:
    """Модель-шпион: запоминает вызовы, отдаёт заданный ответ."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def process_image(self, *args, **kwargs):
        self.calls.append(("process_image", args, kwargs))
        return self.answer

    def process_array(self, *args, **kwargs):
        self.calls.append(("process_array", args, kwargs))
        return self.answer

    def predict(self, *args, **kwargs):
        self.calls.append(("predict", args, kwargs))
        return self.answer

    def predict_with_details(self, *args, **kwargs):
        self.calls.append(("predict_with_details", args, kwargs))
        return self.answer


def test_adapters_satisfy_contracts():
    assert isinstance(YoloMeterDetector(None), MeterDetector)
    assert isinstance(YoloDigitDetector(None), DigitDetector)
    assert isinstance(CrnnSerialRecognizer(None), SerialRecognizer)
    assert isinstance(CnnDigitRecognizer(None), DigitRecognizer)


def test_meter_detector_calls_model_like_before():
    crops = make_meter_crops()
    model = _Recorder(crops)
    assert YoloMeterDetector(model).detect("p.jpg") is crops
    # как reader.py до Фазы 3: straighten не передаётся → действует настройка
    # YOLOInferer из __init__ (для счётчика — включено)
    assert model.calls == [("process_image", ("p.jpg",), {"save_crops": False})]


def test_meter_detector_passes_none_through():
    # YOLOInferer.process_image возвращает None на нечитаемом файле
    assert YoloMeterDetector(_Recorder(None)).detect("broken.jpg") is None


def test_digit_detector_calls_model_like_before():
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    model = _Recorder([])
    assert YoloDigitDetector(model).detect(crop) == []
    (name, args, kwargs), = model.calls
    assert name == "process_array" and args[0] is crop
    assert kwargs == {"save_crops": False, "max_per_class": 5}


def test_serial_recognizer_converts_answer():
    details = [{"char": "1", "conf": 0.9}]
    model = _Recorder({"text": "0071", "avg_confidence": 0.55, "details": details})
    crop = object()
    pred = CrnnSerialRecognizer(model).recognize(crop)
    assert pred == SerialPrediction(text="0071", confidence=0.55, details=details)
    assert model.calls == [("predict_with_details", (crop,), {})]


def test_digit_recognizer_converts_digit_to_str():
    model = _Recorder({"digit": 7, "confidence": 0.61})
    crop = object()
    assert CnnDigitRecognizer(model).recognize(crop) == DigitPrediction(digit="7", confidence=0.61)
    assert model.calls == [("predict", (crop,), {})]


# ─── 2. Загрузчик ────────────────────────────────────────────────────────────

def test_load_models_keeps_construction_params(monkeypatch):
    made = []

    def fake(kind):
        def make(*args, **kwargs):
            made.append((kind, args, kwargs))
            return object()
        return make

    monkeypatch.setattr(loader, "YOLOInferer", fake("yolo"))
    monkeypatch.setattr(loader, "CNNInferer", fake("cnn"))
    monkeypatch.setattr(loader, "CRNNInferer", fake("crnn"))

    cfg = PipelineConfig()
    models = loader.load_models(cfg, device="DEV", resolve_path=lambda p: "/abs/" + p)

    assert made == [
        ("yolo", ("/abs/" + cfg.meter_detect_model,), {"conf_thresh": cfg.meter_conf_thresh}),
        ("yolo", ("/abs/" + cfg.digit_detect_model,),
         {"conf_thresh": cfg.digit_detect_conf_thresh, "straighten": False}),
        ("cnn", ("/abs/" + cfg.digit_ocr_model,), {"device": "DEV"}),
        ("crnn", ("/abs/" + cfg.serial_ocr_model,), {"device": "DEV"}),
    ]
    assert isinstance(models.meter_detector, YoloMeterDetector)
    assert isinstance(models.digit_detector, YoloDigitDetector)
    assert isinstance(models.digit_recognizer, CnnDigitRecognizer)
    assert isinstance(models.serial_recognizer, CrnnSerialRecognizer)


def test_straighten_defaults_of_real_yolo_inferer():
    # Выпрямление счётчика держится на дефолте YOLOInferer(straighten=True):
    # load_models его не передаёт. Если дефолт поменяют — поведение изменится.
    import inspect
    from models.yolo_all_detect.infer_yolo import YOLOInferer
    assert inspect.signature(YOLOInferer.__init__).parameters["straighten"].default is True


# ─── 3. program2 и reader читают цифры одним сервисом ───────────────────────

def _digits(values, conf=0.95):
    crops, arrays = make_digit_crops_with_centers([i * 40 for i in range(len(values))])
    ocr = FakeDigitOCR({id(a): (v, conf) for a, v in zip(arrays, values)})
    return FakeDigitDetector(crops), ocr


def _bundle(meter_crops, digit_values, serial=("12345", 0.9)):
    import program2
    dd, docr = _digits(digit_values)
    bundle = program2.ModelBundle.__new__(program2.ModelBundle)   # без загрузки весов
    bundle.config = PipelineConfig()
    bundle.models = RecognitionModels(
        meter_detector=YoloMeterDetector(FakeMeterDetector(meter_crops)),
        digit_detector=YoloDigitDetector(dd),
        digit_recognizer=CnnDigitRecognizer(docr),
        serial_recognizer=CrnnSerialRecognizer(FakeSerialOCR(*serial)),
    )
    return bundle, dd, docr


def test_program2_run_on_photo_full():
    meter_crops = make_meter_crops()
    meter_crops[0]["crop"] = np.arange(100 * 300 * 3, dtype=np.uint8).reshape(100, 300, 3)
    bundle, dd, docr = _bundle(meter_crops, ["0", "1", "2", "3", "4"])

    r = bundle.run_on_photo("p.jpg")

    assert r["error"] is None
    assert r["serial_text"] == "12345" and r["serial_conf"] == 0.9
    assert r["serial_crop"] is meter_crops[1]["crop"]
    assert r["reading_str"] == "01234"
    # тот же результат, что у общего сервиса чтения цифр (им же пользуется reader.py)
    same = read_meter_digits(YoloDigitDetector(dd), CnnDigitRecognizer(docr), meter_crops[0]["crop"],
                             0.6, 5, ignore_last_digits=PipelineConfig().ignore_last_digits)
    assert r["digit_preds"] == same.digit_results
    # кропы цифр вырезаны из кропа счётчика по bbox каждой позиции
    for crop, (x1, y1, x2, y2) in zip(r["digit_crops"], same.digit_bboxes):
        assert np.array_equal(crop, meter_crops[0]["crop"][y1:y2, x1:x2])


def test_program2_run_on_photo_nothing_found():
    bundle, _, _ = _bundle([], ["0"] * 5)
    r = bundle.run_on_photo("p.jpg")
    assert r["error"] == "YOLO: ничего не найдено"
    assert r["serial_text"] is None and r["reading_str"] is None


def test_program2_run_on_photo_meter_without_serial():
    meter_only = [make_meter_crops()[0]]
    bundle, _, _ = _bundle(meter_only, ["0", "0", "1", "0", "0"])
    r = bundle.run_on_photo("p.jpg")
    assert r["error"] is None
    assert r["serial_text"] is None and r["serial_crop"] is None
    assert r["reading_str"] == "00100"


def test_program2_run_on_photo_serial_without_meter():
    serial_only = [make_meter_crops()[1]]
    bundle, _, _ = _bundle(serial_only, ["0"] * 5)
    r = bundle.run_on_photo("p.jpg")
    assert r["serial_text"] == "12345"
    assert r["reading_str"] is None and r["digit_preds"] is None and r["digit_crops"] is None


def test_program2_run_on_photo_model_exception_goes_to_error():
    bundle, _, _ = _bundle(make_meter_crops(), ["0"] * 5)

    class Boom:
        def recognize(self, crop):
            raise RuntimeError("cuda died")
    bundle.models.serial_recognizer = Boom()
    r = bundle.run_on_photo("p.jpg")
    assert r["error"] == "cuda died"


def test_application_read_meter_digits_low_conf_and_forgiven():
    # до Фазы 7 сравнивался с legacy-обёрткой reader._read_meter_digits (удалена)
    dd, docr = _digits(["9", "8", "7", "6", "5"], conf=0.5)   # всё ниже порога
    crop = make_meter_crops()[0]["crop"]
    new = read_meter_digits(YoloDigitDetector(dd), CnnDigitRecognizer(docr), crop, 0.6, 5,
                            ignore_last_digits=2)
    assert new.reading_str == "???00" and new.number is None
    assert new.error.startswith("low conf digits: pos0(pred=9,conf=0.500)")


# ─── 4. Границы слоёв ────────────────────────────────────────────────────────

def _imports(path: Path) -> list[tuple[str, list[str]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name, []) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:   # не относительный
            out.append((node.module or "", [a.name for a in node.names]))
    return out


def test_program2_does_not_import_private_reader_functions():
    # критерий готовности Фазы 3 (ТЗ, раздел 4)
    from_reader = [names for mod, names in _imports(ROOT / "program2.py") if mod == "reader"]
    imported = {n for names in from_reader for n in names}
    assert "_read_meter_digits" not in imported
    assert "_find_crop_entry" not in imported
    # и не зовёт модели в обход контрактов
    mods = {mod for mod, _ in _imports(ROOT / "program2.py")}
    assert not any(m.startswith("models.") for m in mods)


@pytest.mark.parametrize("layer", ["domain", "application"])
def test_layer_does_not_import_heavy_libraries(layer):
    forbidden = {"torch", "cv2", "pandas", "tkinter", "ultralytics", "models"}
    for path in (ROOT / "src" / "gmr" / layer).glob("*.py"):
        for mod, _ in _imports(path):
            assert mod.split(".")[0] not in forbidden, f"{path.name} импортирует {mod}"


def test_ml_adapters_do_not_import_torch():
    # адаптеры тестируются и используются без torch; torch — только в loader.py
    for mod, _ in _imports(ROOT / "src" / "gmr" / "ml" / "adapters.py"):
        assert mod.split(".")[0] not in {"torch", "cv2", "ultralytics", "models"}
