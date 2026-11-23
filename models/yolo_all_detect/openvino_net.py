"""
models/yolo_all_detect/openvino_net.py — детекторы YOLO через OpenVINO (1.6.0).

Та же сеть, выгруженная в формат OpenVINO (папка `<имя весов>_openvino_model`
рядом с best.pt: `python gmr.py openvino`), считается на процессоре в 1,5–2,5
раза быстрее PyTorch. Всё остальное — подготовка фото, отбор рамок — остаётся
у ultralytics как было: подменяется только сама сеть внутри YOLO, поэтому
вход сети тот же до точки.

Только процессор (CPU) и только точная арифметика (f32): видеокарта и
«быстрая неточная» арифметика на разных компьютерах дали бы разные ответы.
Работает на любом 64-битном процессоре Intel или AMD. Нет OpenVINO или
выгрузки — модель работает через PyTorch, как до 1.6.0 (load_openvino
возвращает причину).
"""
from pathlib import Path
from typing import Optional

CONFIG = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"}
SOURCE = "веса.txt"          # отпечаток best.pt, из которого выгружена сеть: переобучили — выгрузка не годится


def weights_print(weights) -> str:
    import hashlib
    return hashlib.sha256(Path(weights).read_bytes()).hexdigest()[:16]


def is_current(weights) -> bool:
    """Выгрузка есть и сделана из этих же весов."""
    f = openvino_dir(weights) / SOURCE
    return f.is_file() and f.read_text(encoding="utf-8").strip() == weights_print(weights)


def openvino_dir(weights) -> Path:
    """Куда выгружается сеть весов best.pt — как это делает ultralytics."""
    p = Path(weights)
    return p.with_name(f"{p.stem}_openvino_model")


class OpenVinoNet:
    """Вызывается как torch-сеть внутри YOLO: тензор (1, 3, H, W) → выход
    сети тензором. Под каждый размер входа — своя готовая модель (размеров
    у фото месяца — один-два, готовится за доли секунды)."""

    def __init__(self, folder):
        import openvino as ov
        xml = next(Path(folder).glob("*.xml"))
        self._core = ov.Core()
        self._model = self._core.read_model(str(xml), str(xml.with_suffix(".bin")))
        self._requests: dict = {}

    def _request(self, shape: tuple):
        req = self._requests.get(shape)
        if req is None:
            model = self._model.clone()
            model.reshape({model.input().get_any_name(): list(shape)})
            req = self._requests[shape] = self._core.compile_model(model, "CPU", CONFIG).create_infer_request()
        return req

    def __call__(self, im, **_):
        import torch
        req = self._request(tuple(im.shape))
        req.infer({0: im.detach().cpu().numpy()})
        return torch.from_numpy(req.get_output_tensor(0).data.copy())


def load_openvino(yolo, weights) -> Optional[str]:
    """Подменить сеть внутри ultralytics.YOLO на OpenVINO. None — подменена;
    иначе — почему нет (модель остаётся на PyTorch)."""
    folder = openvino_dir(weights)
    if not folder.is_dir():
        return f"нет выгрузки OpenVINO ({folder.name})"
    try:
        if not is_current(weights):
            return "выгрузка OpenVINO — от других весов (python gmr.py openvino)"
        net = OpenVinoNet(folder)
    except Exception as e:                              # noqa: BLE001 — нет пакета, старый процессор…
        return f"OpenVINO не запустился: {type(e).__name__}: {e}"
    import numpy as np
    yolo(np.zeros((64, 64, 3), np.uint8), verbose=False)      # ultralytics готовит обработку фото
    yolo.predictor.model.backend.model = net
    return None


def export_openvino(weights, imgsz: int) -> Path:
    """Выгрузить сеть best.pt в OpenVINO рядом с весами (размер входа —
    любой: модель под размер готовится при работе)."""
    import shutil
    from ultralytics import YOLO
    out = Path(YOLO(str(weights)).export(format="openvino", imgsz=imgsz, dynamic=True, half=False))
    dst = openvino_dir(weights)
    if out.resolve() != dst.resolve():
        shutil.rmtree(dst, ignore_errors=True)
        shutil.move(str(out), dst)
    (dst / SOURCE).write_text(weights_print(weights), encoding="utf-8")
    return dst
