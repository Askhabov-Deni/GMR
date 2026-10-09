"""
Этап 6c (docs/MODELS_REVIEW.md): обучение без изменения работы программы —
CRNN серийника без среза края и с пропорциями (режим входа в чекпоинте),
YOLO без зеркал (tests/test_datasets.py), проверка имён классов детектора
при загрузке, таблица порогов в `etalon check`.
"""
import random
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from models.crnn import dataset_crnn as ds
from models.crnn.infer_crnn import CRNNInferer
from models.crnn.model_crnn import CRNN
from src.gmr.domain import PipelineConfig
from src.gmr.ml import loader

# ─── CRNN серийника: кроп → 240×40 ───────────────────────────────────────────


def _crop(w=120, h=30, corners=False):
    a = np.full((h, w, 3), 230, np.uint8)
    a[h // 3: 2 * h // 3, 10:w - 10] = 20                     # «цифры»
    if corners:                                               # красные метки в углах содержимого
        for y in (0, h - 4):
            for x in (0, w - 4):
                a[y:y + 4, x:x + 4] = (255, 0, 0)
    return Image.fromarray(a)


def test_fit_to_box_keeps_proportions_and_pads_with_background():
    out = np.asarray(ds.FitToBox(240, 40)(_crop(120, 30)))
    assert out.shape == (40, 240, 3)
    assert (out[:, 165:] == 230).all()                        # 120×30 → 160×40, справа — фон
    assert out[20, 50, 0] < 60                                # «цифры» на месте, не растянуты
    long = np.asarray(ds.FitToBox(240, 40)(_crop(600, 40, corners=True))).astype(int)   # длинный — по ширине
    assert long.shape == (40, 240, 3) and (long[:12] == 230).all() and (long[-12:] == 230).all()
    red = (long[..., 0] > 140) & (long[..., 1] < 110)
    assert red[:, 230:].any() and red[:, :10].any()           # оба конца номера на месте


def test_fit_to_box_augmentation_never_cuts_the_edges():
    random.seed(1)
    torch.manual_seed(1)
    fit = ds.FitToBox(240, 40, augment=True)
    for _ in range(60):
        a = np.asarray(fit(_crop(200, 34, corners=True))).astype(int)
        ys, xs = np.nonzero((a[..., 0] > 140) & (a[..., 1] < 110) & (a[..., 2] < 110))
        assert len(xs), "метки пропали"
        cx, cy = xs.mean(), ys.mean()
        quads = {(x < cx, y < cy) for x, y in zip(xs, ys)}
        assert quads == {(True, True), (True, False), (False, True), (False, False)}   # все 4 угла целы


def test_old_stretch_crop_did_cut_the_edge():
    # для сравнения: прежний RandomCrop теряет угловые метки
    random.seed(0)
    torch.manual_seed(0)
    old = T.Compose([T.Resize((48, 256)), T.RandomCrop((40, 240))])
    lost = 0
    for _ in range(30):
        a = np.asarray(old(_crop(200, 34, corners=True))).astype(int)
        red = (a[..., 0] > 140) & (a[..., 1] < 110) & (a[..., 2] < 110)
        lost += red[:, :3].sum() == 0 or red[:, -3:].sum() == 0 or red[:3].sum() == 0 or red[-3:].sum() == 0
    assert lost > 0


def test_transforms_by_mode():
    img = _crop()
    names = [type(t).__name__ for t in ds.train_transform(ds.KEEP_ASPECT).transforms]
    assert names[0] == "FitToBox" and "RandomCrop" not in names and "Resize" not in names
    assert "RandomCrop" in [type(t).__name__ for t in ds.train_transform(ds.STRETCH).transforms]
    assert isinstance(ds.val_transform().transforms[0], T.Resize)          # по умолчанию — как раньше
    assert isinstance(ds.val_transform(ds.KEEP_ASPECT).transforms[0], ds.FitToBox)
    for mode in ds.INPUT_MODES:
        assert ds.train_transform(mode)(img).shape == ds.val_transform(mode)(img).shape == (3, 40, 240)


def _ckpt(path, **extra):
    torch.save({"model_state": CRNN().state_dict(), **extra}, path)
    return str(path)


def test_checkpoint_says_how_to_read(tmp_path, capsys):
    old = CRNN.from_pretrained(_ckpt(tmp_path / "old.pt"))                 # без val_acc — не падает
    assert old.input_mode == "stretch" and "val_acc=?" in capsys.readouterr().out
    new = CRNN.from_pretrained(_ckpt(tmp_path / "new.pt", val_acc=0.9, input_mode="keep_aspect"))
    assert new.input_mode == "keep_aspect"
    assert isinstance(CRNNInferer(str(tmp_path / "new.pt")).transform.transforms[0], ds.FitToBox)
    assert isinstance(CRNNInferer(str(tmp_path / "old.pt")).transform.transforms[0], T.Resize)
    assert isinstance(CRNNInferer(CRNN()).transform.transforms[0], T.Resize)
    res = CRNNInferer(str(tmp_path / "new.pt")).predict_with_details(np.zeros((30, 150, 3), np.uint8))
    assert set(res) == {"text", "avg_confidence", "details"}


def test_resume_keeps_mode_of_checkpoint(tmp_path):
    from models.crnn import train_crnn
    assert train_crnn.input_mode_for(Namespace(resume=None, input="keep_aspect")) == "keep_aspect"
    assert train_crnn.input_mode_for(Namespace(resume=_ckpt(tmp_path / "a.pt"), input="keep_aspect")) == "stretch"
    b = _ckpt(tmp_path / "b.pt", input_mode="keep_aspect")
    assert train_crnn.input_mode_for(Namespace(resume=b, input="stretch")) == "keep_aspect"


@pytest.mark.filterwarnings("ignore::FutureWarning", "ignore::torch.jit.TracerWarning")  # экспорт TorchScript
def test_train_crnn_writes_input_mode(tmp_path, monkeypatch):
    import cv2

    from models.crnn import train_crnn
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    rng = np.random.default_rng(0)
    for i in range(40):
        text = "".join(str(d) for d in rng.integers(0, 10, 6))
        img = np.full((30, 140, 3), 255, np.uint8)
        cv2.putText(img, text, (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite(str(images / f"p{i}__2026-10__serial.jpg"), img)
        (labels / f"p{i}__2026-10__serial.txt").write_text(text, encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["train_crnn.py", "--images_dir", str(images), "--labels_dir", str(labels),
                                     "--epochs", "1", "--batch_size", "8", "--save_dir", str(tmp_path / "runs")])
    monkeypatch.setattr(train_crnn, "save_plots", lambda *a: None)
    modes = []
    for name in ("train_transform", "val_transform"):
        real = getattr(train_crnn, name)
        monkeypatch.setattr(train_crnn, name, lambda mode, real=real: modes.append(mode) or real(mode))
    train_crnn.main()
    assert modes == ["keep_aspect"] * 3                       # обучение, проверка и тест — одинаково
    best = next((tmp_path / "runs").rglob("best.pt"))
    assert torch.load(best, map_location="cpu", weights_only=True)["input_mode"] == "keep_aspect"


def test_evaluate_reads_like_training(monkeypatch):
    from models.crnn import evaluate_crnn
    seen = []
    monkeypatch.setattr(evaluate_crnn, "MeterDataset", lambda meta, tr: seen.append(tr) or [])
    monkeypatch.setattr(evaluate_crnn, "DataLoader", lambda *a, **k: [])
    evaluate_crnn.collect_predictions(CRNN(), [], torch.device("cpu"), 2, 0, "keep_aspect")
    evaluate_crnn.collect_predictions(CRNN(), [], torch.device("cpu"), 2, 0)
    assert isinstance(seen[0].transforms[0], ds.FitToBox) and isinstance(seen[1].transforms[0], T.Resize)


# ─── Имена классов детектора счётчика ────────────────────────────────────────

def _det(names):
    return SimpleNamespace(inferer=SimpleNamespace(class_names=names))


def test_meter_detector_class_names_checked():
    loader.check_meter_classes(_det({0: "serial_number", 1: "marker_id", 2: "gas_meter"}))   # порядок любой
    loader.check_meter_classes(_det(["gas_meter", "serial_number"]))
    loader.check_meter_classes(SimpleNamespace())                                     # имён не узнать — молчим
    with pytest.raises(ValueError, match="нет класса «serial_number»; его классы: gas_meter, display"):
        loader.check_meter_classes(_det({0: "gas_meter", 1: "display"}))
    with pytest.raises(ValueError, match="«gas_meter», «serial_number»"):
        loader.check_meter_classes(_det({0: "display"}))


def test_load_models_refuses_detector_with_other_class_names(monkeypatch):
    monkeypatch.setattr(loader, "load_meter_detector", lambda cfg, rp: _det({0: "meter", 1: "serial"}))
    with pytest.raises(ValueError, match="gas_meter"):
        loader.load_models(PipelineConfig())


# ─── Таблица порогов в etalon check ──────────────────────────────────────────

def _pc(h, d_conf, d_right, s_conf, s_right):
    from tools import etalon
    return etalon.PhotoCheck(h, "DIGITS_ERROR", True, True, False, False, False, False, "", "",
                             s_conf, s_right, d_conf, d_right)


def test_threshold_table():
    from tools import etalon
    rs = [_pc("a", 0.95, True, 0.92, True), _pc("b", 0.55, False, 0.45, False),
          _pc("c", 0.65, True, 0.7, False), _pc("d", None, None, None, None)]
    lines = etalon.threshold_table(rs, PipelineConfig(digit_conf_thresh=0.7))
    assert "сейчас: цифры 0.7, серийник 0.6" in lines[1]
    rows = {ln.split()[0]: ln for ln in lines if ln[:3] in {f"0.{i}" for i in range(3, 10)}}
    assert list(rows) == ["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    assert "3 из 3, неверно 1" in rows["0.5"] and rows["0.5"].count("неверно 1") == 2
    assert "2 из 3, неверно 0  " in rows["0.6"] and rows["0.6"].endswith("2 из 3, неверно 1")
    assert rows["0.7"].count("неверно 0") == 1 and rows["0.7"].endswith("2 из 3, неверно 1")
    assert rows["0.9"].count("1 из 3, неверно 0") == 2
    assert etalon.threshold_table([_pc("d", None, None, None, None)], PipelineConfig())[3].split() == ["0.3", "—", "—"]


def test_digits_answer_without_substitutions():
    from tools.etalon import _digits_answer
    cfg = PipelineConfig()

    def reading(*confs, digits="12345"):
        return SimpleNamespace(digit_results=[{"digit": d, "confidence": c} for d, c in zip(digits, confs)])
    assert _digits_answer(reading(0.9, 0.8, 0.95, 0.4, 0.9), "12345", cfg) == (0.4, True)
    assert _digits_answer(reading(0.9, 0.8, 0.95, 0.4, 0.9, digits="12340"), "12345", cfg) == (0.4, False)
    assert _digits_answer(reading(0.9, 0.8, 0.9, 0.9, 0.9, digits="01234"), "1234", cfg) == (0.8, True)
    assert _digits_answer(reading(0.9, None, 0.9, 0.9, 0.9), "12345", cfg) == (None, None)   # «5» вместо цифры
    assert _digits_answer(reading(0.9, 0.9, 0.9, 0.9), "12345", cfg) == (None, None)
    assert _digits_answer(None, "12345", cfg) == (None, None)


def test_check_writes_confidences(tmp_path, monkeypatch):
    import csv

    from tests.test_etalon import _etalon_with
    from tools import etalon
    et = _etalon_with(tmp_path, [("a", "111111", "1234", "DIGITS_ERROR")])
    rec = SimpleNamespace(meter={}, serial={}, serial_prediction=SimpleNamespace(text="111112", confidence=0.7),
                          digits=SimpleNamespace(number=None, digit_results=[
                              {"digit": d, "confidence": c} for d, c in zip("01234", (0.9, 0.9, 0.5, 0.9, 0.9))]))
    import src.gmr.application as app
    monkeypatch.setattr(app, "recognize_photo", lambda models, path, cfg, **kw: rec)
    text, out = etalon.check(PipelineConfig(), et, models=object())
    row = next(csv.DictReader(open(out, encoding="utf-8-sig")))
    assert (row["serial_conf"], row["serial_right"], row["digits_conf"], row["digits_right"]) == (
        "0.7", "False", "0.5", "True")
    lines = [" ".join(ln.split()) for ln in text.splitlines()]
    assert "Пороги уверенности (сейчас: цифры 0.6, серийник 0.6):" in lines
    assert "0.5 1 из 1, неверно 0 1 из 1, неверно 1" in lines and "0.6 0 из 1, неверно 0 1 из 1, неверно 1" in lines
