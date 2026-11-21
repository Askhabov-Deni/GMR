"""
Фаза 3 (ML Interfaces, ТЗ миграции, история git): контракты моделей, адаптеры,
загрузчик и общий сервис распознавания для reader.py и program2.py.

  1. Адаптеры выполняют контракты и зовут модели с теми же аргументами,
     что reader.py до Фазы 3 (включая то, что straighten не переопределяется).
  2. load_models создаёт модели с теми же параметрами: выпрямление включено
     для детектора счётчика и выключено для детектора цифр.
  3. Что прочитали модели на фото оператора, пишет прогон (pipeline/seen.py,
     этап Е) тем же сервисом чтения цифр; окно оператора моделей не запускает.
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
    Outcome, PhotoResult, PipelineConfig, SerialPrediction, SerialRecognizer,
)
from src.gmr.ml import (
    CnnDigitRecognizer, CrnnSerialRecognizer, YoloDigitDetector, YoloMeterDetector,
)
from src.gmr.ml import loader
from src.gmr.pipeline.seen import safe_what_models_saw, what_models_saw
from tests._fixtures import (
    FakeDigitDetector, FakeDigitOCR, FakeSerialOCR,
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

    cfg = PipelineConfig(account_ocr_model="")          # без модели надписи
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


# ─── 3. Что прочитали модели — для окна оператора (этап Е) ──────────────────
# До этапа Е окно читало фото само (program2.ModelBundle.run_on_photo). Теперь
# это пишет прогон (src/gmr/pipeline/seen.py) тем же сервисом чтения цифр, а
# окно показывает. Прочитанное process_photo не перечитывается; недостающее
# (номер без счётчика, цифры без номера) — дочитывается по найденным рамкам.

def _digits(values, conf=0.95):
    crops, arrays = make_digit_crops_with_centers([i * 40 for i in range(len(values))])
    ocr = FakeDigitOCR({id(a): (v, conf) for a, v in zip(arrays, values)})
    return FakeDigitDetector(crops), ocr


def _models(digit_values, serial=("12345", 0.9)):
    dd, docr = _digits(digit_values)
    models = RecognitionModels(
        meter_detector=None,
        digit_detector=YoloDigitDetector(dd),
        digit_recognizer=CnnDigitRecognizer(docr),
        serial_recognizer=CrnnSerialRecognizer(FakeSerialOCR(*serial)),
    )
    return models, dd, docr


def _queued(detections, outcome=Outcome.NO_SERIAL, **kw):
    return PhotoResult(photo_path="p.jpg", outcome=outcome, meter_crops_raw=detections, **kw)


def _decode(jpeg):
    import cv2
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


class _Never:
    """Модель, которую звать нельзя: прочитанное прогоном не перечитывается."""

    def recognize(self, crop):
        raise AssertionError("прочитано повторно")

    def detect(self, crop):
        raise AssertionError("прочитано повторно")


def test_seen_reads_digits_when_run_did_not():
    # NO_SERIAL: прогон цифры не читает — для окна они дочитываются
    meter = make_meter_crops()[0]
    meter["crop"] = np.arange(100 * 300 * 3, dtype=np.uint8).reshape(100, 300, 3)
    models, dd, docr = _models(["0", "1", "2", "3", "4"])
    rec = what_models_saw(_queued([meter]), models, PipelineConfig())
    assert (rec.serial_read, rec.serial_conf, rec.reading_str) == ("", None, "01234")
    assert rec.digits_ok == [True] * 5
    # тот же результат, что у общего сервиса чтения цифр (им пользуется и process_photo)
    same = read_meter_digits(YoloDigitDetector(dd), CnnDigitRecognizer(docr), meter["crop"],
                             0.6, 5, ignore_last_digits=PipelineConfig().ignore_last_digits)
    assert rec.reading_str == same.reading_str
    # кропы цифр — из кропа счётчика по рамке каждой позиции (JPEG: размер тот же)
    assert sorted(rec.crops) == [f"digit{i}" for i in range(5)]
    for i, (x1, y1, x2, y2) in enumerate(same.digit_bboxes):
        assert _decode(rec.crops[f"digit{i}"]).shape == meter["crop"][y1:y2, x1:x2].shape


def test_seen_reads_serial_without_meter():
    # NO_METER: прогон номер не читает — для окна он дочитывается
    serial_only = [make_meter_crops()[1]]
    models, _, _ = _models(["0"] * 5)
    rec = what_models_saw(_queued(serial_only, Outcome.NO_METER), models, PipelineConfig())
    assert (rec.serial_read, rec.serial_conf, rec.reading_str, rec.digits_ok) == ("12345", 0.9, "", [])
    assert sorted(rec.crops) == ["serial"]
    assert _decode(rec.crops["serial"]).shape == serial_only[0]["crop"].shape


def test_seen_keeps_what_run_read():
    crops = make_meter_crops()
    models = RecognitionModels(None, _Never(), _Never(), _Never())
    result = _queued(crops, Outcome.DIGITS_ERROR, serial_text="0012345", serial_read="12345",
                     serial_conf=0.93, reading_str="01?45",
                     digit_results=[{"ok": True}, {"ok": True}, {"ok": False}, {"ok": True}, {"ok": True}],
                     digit_bboxes=[(0, 0, 20, 30), None, (80, 0, 100, 30), (120, 0, 140, 30), (160, 0, 180, 30)])
    rec = what_models_saw(result, models, PipelineConfig())
    assert (rec.serial_read, rec.serial_conf, rec.reading_str) == ("12345", 0.93, "01?45")
    assert rec.digits_ok == [True, True, False, True, True]
    assert sorted(rec.crops) == ["digit0", "digit2", "digit3", "digit4", "serial"]   # digit1 — заглушка


@pytest.mark.parametrize("outcome", [Outcome.PLUS, Outcome.MINUS, Outcome.REPEAT, Outcome.ERROR])
def test_seen_only_for_operator_queue(outcome):
    models = RecognitionModels(None, _Never(), _Never(), _Never())
    assert what_models_saw(_queued(make_meter_crops(), outcome), models, PipelineConfig()) is None


def test_seen_nothing_found():
    models = RecognitionModels(None, _Never(), _Never(), _Never())
    assert what_models_saw(_queued(None, Outcome.NO_METER), models, PipelineConfig()) is None


def test_seen_model_exception_does_not_stop_run():
    import logging

    class Boom:
        def recognize(self, crop):
            raise RuntimeError("cuda died")
    models, _, _ = _models(["0"] * 5)
    models.serial_recognizer = Boom()
    result = _queued([make_meter_crops()[1]], Outcome.NO_METER)
    assert safe_what_models_saw(result, models, PipelineConfig(), logging.getLogger("t")) is None


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
