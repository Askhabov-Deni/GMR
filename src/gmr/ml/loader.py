"""
src/gmr/ml/loader.py — загрузка всех четырёх моделей по PipelineConfig
(Фаза 3). Единственное место, где создаются YOLOInferer / CNNInferer /
CRNNInferer; раньше этот код был продублирован в reader.run_pipeline,
reader.test_one и program2.ModelBundle.

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
        meter_detector=YoloMeterDetector(YOLOInferer(
            resolve_path(config.meter_detect_model),
            conf_thresh=config.meter_conf_thresh,
        )),
        digit_detector=YoloDigitDetector(YOLOInferer(
            resolve_path(config.digit_detect_model),
            conf_thresh=config.digit_detect_conf_thresh,
            straighten=False,
        )),
        digit_recognizer=CnnDigitRecognizer(
            CNNInferer(resolve_path(config.digit_ocr_model), device=device)
        ),
        serial_recognizer=CrnnSerialRecognizer(
            CRNNInferer(resolve_path(config.serial_ocr_model), device=device)
        ),
    )
