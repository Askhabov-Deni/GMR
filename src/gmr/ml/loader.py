"""
src/gmr/ml/loader.py — загрузка всех четырёх моделей по PipelineConfig
(Фаза 3). Единственное место, где создаются YOLOInferer / CNNInferer /
CRNNInferer (reader.run_pipeline, program2.ModelBundle, tools/inspect_model).

Параметры создания моделей — ровно те, что были в reader.py до Фазы 3:
  - детектор счётчика: conf=meter_conf_thresh, straighten по умолчанию (True)
  - детектор цифр:     conf=digit_detect_conf_thresh, straighten=False
  - CNN / CRNN:        на cuda, если есть, иначе cpu
Выпрямление включено для счётчика/серийника и выключено для цифр намеренно —
это поведенческая деталь, а не случайность (ТЗ, Фаза 3).
"""
from typing import Callable, Optional

import torch

from models.cnn.infer_cnn import CNNInferer
from models.crnn.infer_crnn import CRNNInferer
from models.yolo_all_detect.infer_yolo import YOLOInferer

from src.gmr.application.recognition import RecognitionModels
from src.gmr.domain.config import PipelineConfig
from src.gmr.ml.adapters import (
    CnnDigitRecognizer,
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
    return RecognitionModels(
        meter_detector=load_meter_detector(config, resolve_path),
        digit_detector=load_digit_detector(config, resolve_path),
        digit_recognizer=load_digit_recognizer(config, device, resolve_path),
        serial_recognizer=load_serial_recognizer(config, device, resolve_path),
    )
