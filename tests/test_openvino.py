"""1.6.0: детекторы через OpenVINO (models/yolo_all_detect/openvino_net.py)
— где лежит выгрузка, и что без неё всё как раньше (PyTorch)."""
from pathlib import Path

import pytest

from models.yolo_all_detect import openvino_net
from src.gmr.domain import PipelineConfig


def test_export_folder_next_to_weights():
    assert openvino_net.openvino_dir("a/b/best.pt") == Path("a/b/best_openvino_model")


def test_without_export_stays_on_pytorch(tmp_path):
    why = openvino_net.load_openvino(object(), tmp_path / "best.pt")
    assert why == "нет выгрузки OpenVINO (best_openvino_model)"


def test_broken_export_stays_on_pytorch(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"w")
    d = tmp_path / "best_openvino_model"
    d.mkdir()                                            # папка есть, модели нет
    (d / openvino_net.SOURCE).write_text(openvino_net.weights_print(tmp_path / "best.pt"), encoding="utf-8")
    why = openvino_net.load_openvino(object(), tmp_path / "best.pt")
    assert why.startswith("OpenVINO не запустился: ")


def test_loader_marks_backend(monkeypatch):
    from src.gmr.ml import loader

    class Inf:
        model = object()
    cfg = PipelineConfig()
    monkeypatch.setattr(openvino_net, "load_openvino", lambda yolo, w: None)
    assert loader._use_openvino(Inf(), "x.pt", cfg).backend == "OpenVINO"
    monkeypatch.setattr(openvino_net, "load_openvino", lambda yolo, w: "нет выгрузки")
    assert loader._use_openvino(Inf(), "x.pt", cfg).backend == "PyTorch: нет выгрузки"
    cfg.openvino = False
    assert loader._use_openvino(Inf(), "x.pt", cfg).backend == "PyTorch"


def test_ultralytics_does_not_phone_home():
    from ultralytics.utils import events
    from src.gmr.ml.loader import quiet_ultralytics
    quiet_ultralytics()
    assert events.events.enabled is False


@pytest.mark.skipif(not openvino_net.openvino_dir(PipelineConfig().digit_detect_model).is_dir(),
                    reason="нет выгрузки OpenVINO детектора цифр (python gmr.py openvino)")
def test_openvino_net_equals_pytorch():
    import torch
    from ultralytics import YOLO
    weights = PipelineConfig().digit_detect_model
    net = YOLO(weights).model.eval().float()
    ov = openvino_net.OpenVinoNet(openvino_net.openvino_dir(weights))
    x = torch.rand(1, 3, 160, 320)
    with torch.no_grad():
        want = net(x)
    want = want[0] if isinstance(want, (list, tuple)) else want
    got = ov(x)
    assert got.shape == want.shape and float((got - want).abs().max()) < 1e-2


def test_export_from_other_weights_is_not_used(tmp_path):
    w = tmp_path / "best.pt"
    w.write_bytes(b"new weights")
    d = openvino_net.openvino_dir(w)
    d.mkdir()
    (d / openvino_net.SOURCE).write_text("0123456789abcdef", encoding="utf-8")   # от прежних весов
    assert not openvino_net.is_current(w)
    assert openvino_net.load_openvino(object(), w) == "выгрузка OpenVINO — от других весов (python gmr.py openvino)"
    (d / openvino_net.SOURCE).write_text(openvino_net.weights_print(w), encoding="utf-8")
    assert openvino_net.is_current(w)


def test_gmr_sets_passive_openmp_before_torch():
    # потоки PyTorch не крутятся вхолостую — иначе отнимают ядра у OpenVINO
    import os
    import gmr  # noqa: F401
    assert os.environ.get("OMP_WAIT_POLICY") == "PASSIVE" or os.environ.get("OMP_WAIT_POLICY")
