"""
src/gmr/ml/loader.py — загрузка моделей по PipelineConfig (Фаза 3).
Единственное место, где создаются YOLOInferer / CNNInferer / CRNNInferer /
AccountInferer (reader.run_pipeline, program2.ModelBundle, tools/inspect_model).

Параметры создания моделей — ровно те, что были в reader.py до Фазы 3:
  - детектор счётчика: conf=meter_conf_thresh, straighten по умолчанию (True)
  - детектор цифр:     conf=digit_detect_conf_thresh, straighten=False
  - CNN / CRNN:        на cuda, если есть, иначе cpu
  - надпись маркером:  только если задан account_ocr_model (иначе None)
Выпрямление включено для счётчика/серийника и выключено для цифр намеренно —
это поведенческая деталь, а не случайность (ТЗ, Фаза 3).
"""
from typing import Callable, Optional

import torch

from models.cnn.infer_cnn import CNNInferer
from models.account.infer_account import AccountInferer
from models.crnn.infer_crnn import CRNNInferer
from models.yolo_all_detect.infer_yolo import YOLOInferer

from src.gmr.application.recognition import RecognitionModels
from src.gmr.domain.config import PipelineConfig
from src.gmr.ml.adapters import (
    CnnDigitRecognizer,
    CrnnAccountRecognizer,
    CrnnSerialRecognizer,
    YoloDigitDetector,
    YoloMeterDetector,
)


def default_device() -> "torch.device":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_meter_detector(config: PipelineConfig, resolve_path: Callable[[str], str] = str) -> YoloMeterDetector:
    # запас вокруг рамки серийника (этап 5b): при 0 YOLOInferer создаётся как раньше
    pad = {"pad": {"serial_number": config.serial_crop_pad}} if config.serial_crop_pad else {}
    return YoloMeterDetector(YOLOInferer(
        resolve_path(config.meter_detect_model),
        conf_thresh=config.meter_conf_thresh,
        **pad,
    ))


def load_digit_detector(config: PipelineConfig, resolve_path: Callable[[str], str] = str) -> YoloDigitDetector:
    return YoloDigitDetector(YOLOInferer(
        resolve_path(config.digit_detect_model),
        conf_thresh=config.digit_detect_conf_thresh,
        straighten=False,
    ))


def load_digit_recognizer(
    config: PipelineConfig, device: Optional["torch.device"] = None,
    resolve_path: Callable[[str], str] = str,
) -> CnnDigitRecognizer:
    return CnnDigitRecognizer(
        CNNInferer(resolve_path(config.digit_ocr_model), device=device or default_device())
    )


def load_serial_recognizer(
    config: PipelineConfig, device: Optional["torch.device"] = None,
    resolve_path: Callable[[str], str] = str,
) -> CrnnSerialRecognizer:
    return CrnnSerialRecognizer(
        CRNNInferer(resolve_path(config.serial_ocr_model), device=device or default_device())
    )


def load_account_recognizer(
    config: PipelineConfig, device: Optional["torch.device"] = None,
    resolve_path: Callable[[str], str] = str,
) -> Optional[CrnnAccountRecognizer]:
    """Модель надписи маркером; None, если account_ocr_model пуст."""
    if not config.account_ocr_model:
        return None
    return CrnnAccountRecognizer(
        AccountInferer(resolve_path(config.account_ocr_model), device=device or default_device())
    )


# классы детектора счётчика, которые программа ищет по имени
# (src/gmr/application/recognition.py, reader.py)
METER_CLASSES = ("gas_meter", "serial_number")


def _class_names(detector) -> Optional[list]:
    names = getattr(getattr(detector, "inferer", None), "class_names", None)
    if names is None:
        return None
    return list(names.values()) if isinstance(names, dict) else list(names)


def check_meter_classes(meter_detector) -> None:
    """Детектор переобучили с другими именами классов — все фото молча стали
    бы NO_METER / NO_SERIAL; лучше сразу сказать (этап 6c)."""
    names = _class_names(meter_detector)
    missing = [c for c in METER_CLASSES if names is not None and c not in names]
    if missing:
        raise ValueError(
            f"У детектора счётчика нет класса {', '.join(f'«{c}»' for c in missing)}; "
            f"его классы: {', '.join(map(str, names))}. Классы в classes.txt датасета должны "
            f"называться {', '.join(METER_CLASSES)} (порядок любой).")


def check_account_class(meter_detector, config: PipelineConfig) -> None:
    """Модель надписи без класса надписи у детектора счётчика бесполезна
    молча — лучше сразу сказать, какие классы у детектора есть."""
    names = _class_names(meter_detector)
    if names is None:
        return
    if config.account_class not in names:
        raise ValueError(
            f"У детектора счётчика нет класса «{config.account_class}» (PipelineConfig.account_class); "
            f"его классы: {', '.join(map(str, names))}. Укажите в account_class имя класса надписи маркером.")


def load_models(
    config: PipelineConfig,
    device: Optional["torch.device"] = None,
    resolve_path: Callable[[str], str] = str,
) -> RecognitionModels:
    """
    resolve_path — как превратить путь из конфига в путь к файлу весов
    (program2.py резолвит относительные пути от своей папки).
    """
    device = device or default_device()
    meter_detector = load_meter_detector(config, resolve_path)
    check_meter_classes(meter_detector)
    account_recognizer = load_account_recognizer(config, device, resolve_path)
    if account_recognizer is not None:
        check_account_class(meter_detector, config)
    return RecognitionModels(
        meter_detector=meter_detector,
        digit_detector=load_digit_detector(config, resolve_path),
        digit_recognizer=load_digit_recognizer(config, device, resolve_path),
        serial_recognizer=load_serial_recognizer(config, device, resolve_path),
        account_recognizer=account_recognizer,
    )
