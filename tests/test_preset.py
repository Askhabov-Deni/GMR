"""
Настройки распознавания месяца (src/gmr/domain/preset.py, решение владельца
2026-10-11, 2а) и варианты чтения цифр вместо заглушек «5» и «0».
"""
from pathlib import Path

import numpy as np
import pytest

import gmr
import reader
from src.gmr.application import month as month_app
from src.gmr.application.recognition import describe_substitutions, read_meter_digits
from src.gmr.domain import DigitPrediction, PipelineConfig
from src.gmr.domain.preset import RecognitionPreset
from src.gmr.storage.month import MonthDB, MonthFolder
from tests._month import changes, make_month

# ─── Пресет ──────────────────────────────────────────────────────────────────


def test_default_preset_is_old_behaviour():
    p, cfg = RecognitionPreset(), PipelineConfig()
    applied = p.apply(cfg)
    assert (applied.missing_digit_mode, applied.ignore_last_digits, applied.forgiven_digit_mode,
            applied.serial_crop_pad) == ("placeholder", cfg.ignore_last_digits, "placeholder", 0.0)
    assert p.describe() == ("пропущенная цифра — подставить «5»; неуверенные последние 2 → «0»; "
                            "запас рамки серийника 0%; лицевой счёт по надписи — нет")


def test_preset_json_roundtrip_and_bad_values():
    p = RecognitionPreset("model", 1, "model", 0.05, True)
    assert RecognitionPreset.from_json(p.to_json()) == p
    bad = '{"missing_digit": "guess", "forgive_last": 7, "forgive_with": 1, "serial_pad": 0.3, ' \
          '"account_marker": "yes", "new_key": 1}'
    assert RecognitionPreset.from_json(bad) == RecognitionPreset()
    assert RecognitionPreset.from_json("не json") == RecognitionPreset()
    assert RecognitionPreset.from_json("[1, 2]") == RecognitionPreset()
    assert RecognitionPreset.from_json(None) == RecognitionPreset()


def test_preset_apply_and_describe():
    cfg = PipelineConfig(account_ocr_model="acc.pt")
    on = RecognitionPreset("operator", 0, "model", 0.1, True).apply(cfg)
    assert (on.missing_digit_mode, on.ignore_last_digits, on.forgiven_digit_mode, on.serial_crop_pad,
            on.account_ocr_model) == ("operator", 0, "model", 0.1, "acc.pt")
    assert RecognitionPreset(account_marker=False).apply(cfg).account_ocr_model == ""   # надпись выключена
    assert "последние цифры не прощаются" in RecognitionPreset(forgive_last=0).describe()
    assert "неуверенные последние 1 → ответом модели" in RecognitionPreset(
        forgive_last=1, forgive_with="model").describe()


# ─── Чтение цифр: пропущенная цифра и прощённые ──────────────────────────────

# центры цифр по ширине: с пропуском сотен разрыв заметен (правило восстановления —
# разрыв больше 1.6 среднего шага); X3 — то же для пропуска десятков
X = [20, 50, 95, 140, 170]
X3 = [20, 50, 80, 140, 170]


class _Digits:
    """Детектор цифр: рамки найденных позиций; цифра «написана» в пикселе."""
    def __init__(self, positions, xs=X):
        self.positions, self.xs = positions, xs

    def detect(self, crop):
        return [{"bbox": (self.xs[i] - 12, 5, self.xs[i] + 12, 35),
                 "crop": crop[5:35, self.xs[i] - 12:self.xs[i] + 12]} for i in self.positions]


class _Cnn:
    """Цифра — значение пикселя // 10; уверенность — из таблицы по цифре."""
    def __init__(self, conf=None):
        self.conf = conf or {}
        self.seen = []

    def recognize(self, crop):
        d = int(crop[crop.shape[0] // 2, crop.shape[1] // 2, 0]) // 10
        self.seen.append(d)
        return DigitPrediction(digit=str(d), confidence=self.conf.get(d, 0.95))


def _meter(xs=X, digits="12345"):
    img = np.zeros((40, 200, 3), np.uint8)
    for x, d in zip(xs, digits):
        img[:, x - 12:x + 12] = int(d) * 10 + 5
    return img


def _read(positions, cnn, xs=X, **kw):
    return read_meter_digits(_Digits(positions, xs), cnn, _meter(xs), 0.6, 5, ignore_last_digits=2, **kw)


def test_missing_digit_placeholder_as_before():
    r = _read([0, 1, 3, 4], _Cnn())                     # сотни не найдены
    assert r.reading_str == "12545" and r.number == 12545
    assert describe_substitutions(r.digit_results) == "подставлено: pos2='5' (цифра не найдена)"


def test_missing_digit_read_by_model_on_its_place():
    cnn = _Cnn()
    r = _read([0, 1, 3, 4], cnn, missing_mode="model")
    assert r.reading_str == "12345" and r.number == 12345
    assert r.digit_bboxes[2] == (83, 5, 107, 35)        # между соседями, ширина как у них
    assert describe_substitutions(r.digit_results) == (
        "подставлено: pos2='3' (цифра не найдена) — прочитано моделью на её месте, conf=0.950")
    edge = _read([0, 1, 2, 3], _Cnn(), missing_mode="model")   # пропала последняя
    assert edge.reading_str == "12345" and edge.digit_bboxes[4] == (168, 5, 192, 35)


def test_missing_digit_unsure_model_goes_to_operator_or_forgiven():
    r = _read([0, 1, 3, 4], _Cnn({3: 0.3}), missing_mode="model")
    assert r.number is None and r.reading_str == "12?45"
    r = _read([0, 1, 2, 4], _Cnn({4: 0.3}), xs=X3, missing_mode="model")   # десятки: прощаются
    assert r.reading_str == "12305" and r.number == 12305


def test_missing_digit_model_without_place_goes_to_operator(monkeypatch):
    from src.gmr.application import recognition
    monkeypatch.setattr(recognition, "_gap_bbox", lambda *a: None)   # места для цифры нет
    r = _read([0, 1, 3, 4], _Cnn(), missing_mode="model")
    assert r.number is None and r.reading_str == "12?45"


def test_gap_bbox_too_small():
    from src.gmr.application.recognition import _gap_bbox
    assert _gap_bbox([(0, 5, 10, 6), None, (20, 5, 30, 6)], 1, (40, 200)) is None   # высота 1 пиксель
    assert _gap_bbox([(0, 5, 10, 35), None, (20, 5, 30, 35)], 1, (40, 200)) == (10, 5, 20, 35)


def test_missing_digit_to_operator():
    r = _read([0, 1, 3, 4], _Cnn(), missing_mode="operator")
    assert r.number is None and r.reading_str == "12?45"
    assert r.error == "low conf digits: pos2(цифра не найдена)  →  '12?45'"


def test_forgiven_digit_by_model():
    zero = _read([0, 1, 2, 3, 4], _Cnn({5: 0.4}))
    assert zero.reading_str == "12340"                  # как было: «0»
    model = _read([0, 1, 2, 3, 4], _Cnn({5: 0.4}), forgiven_mode="model")
    assert model.reading_str == "12345" and model.number == 12345
    note = describe_substitutions(model.digit_results)
    assert note == "подставлено: pos4='5' (прочитано 5, conf=0.400) — неуверенно, взято как есть"
    assert note.count("(прочитано ") == 1               # tools/analyze_log считает такие прощёнными


def test_config_modes_reach_reader(monkeypatch):
    from src.gmr.application import recognition
    seen = {}
    monkeypatch.setattr(recognition, "read_meter_digits", lambda *a, **k: seen.update(k))
    cfg = PipelineConfig(missing_digit_mode="model", forgiven_digit_mode="model")
    recognition.read_meter_digits_for_config(type("M", (), {"digit_detector": 1, "digit_recognizer": 2})(),
                                             None, cfg)
    assert seen["missing_mode"] == "model" and seen["forgiven_mode"] == "model"


# ─── Пресет месяца: хранение, итог, прогон ──────────────────────────────────

def test_month_preset_stored_and_changed(tmp_path):
    f = make_month(tmp_path, [("100000", "A1", "1", "")])
    assert month_app.month_preset(f.root) is None              # месяц до настроек
    assert "распознавание: по умолчанию" in month_app.month_summary(str(f.root))
    p = RecognitionPreset(missing_digit="model")
    month_app.set_month_preset(f.root, p, who="Мадина")
    assert month_app.month_preset(f.root) == p
    assert "распознавание: пропущенная цифра — прочитать моделью" in month_app.month_summary(str(f.root))
    month_app.set_month_preset(f.root, p, who="Мадина")       # то же — в журнал не пишется
    log = [c for c in changes(f) if c["action"] == month_app.PRESET_ACTION]
    assert len(log) == 1 and log[0]["who"] == "Мадина" and "прочитать моделью" in log[0]["note"]
    assert month_app.month_preset(tmp_path / "нет") is None


def test_new_month_with_preset(tmp_path):
    from tests.test_month import SAMPLE
    p = RecognitionPreset(forgive_last=1)
    month_app.load_table(str(tmp_path / "m"), str(SAMPLE), preset=p)
    assert month_app.month_preset(tmp_path / "m") == p
    month_app.load_table(str(tmp_path / "m"), str(SAMPLE), preset=RecognitionPreset())   # обновление не меняет
    assert month_app.month_preset(tmp_path / "m") == p


def test_run_applies_month_preset(tmp_path):
    import logging
    f = make_month(tmp_path, [("100000", "A1", "1", "")])
    cfg = PipelineConfig(month_dir=str(f.root))
    log = logging.getLogger("test-preset")
    reader._apply_month_preset(cfg, f, log)                    # без настроек — как есть
    assert (cfg.ignore_last_digits, cfg.missing_digit_mode) == (2, "placeholder")
    month_app.set_month_preset(f.root, RecognitionPreset("operator", 0, "model", 0.05))
    reader._apply_month_preset(cfg, f, log)
    assert (cfg.missing_digit_mode, cfg.ignore_last_digits, cfg.forgiven_digit_mode, cfg.serial_crop_pad) == (
        "operator", 0, "model", 0.05)


def test_run_journal_names_preset(tmp_path):
    f = make_month(tmp_path, [("100000", "A1", "1", "")])
    month_app.set_month_preset(f.root, RecognitionPreset(forgive_last=0))
    reader.run_pipeline(reader.PipelineConfig(month_dir=str(f.root)))
    journal = sorted((f.results / "run_logs").glob("run_*.txt"))[-1].read_text(encoding="utf-8")
    assert "Настройки распознавания месяца: пропущенная цифра — подставить «5»; " \
           "последние цифры не прощаются" in journal


# ─── Эталон: настройки месяца и варианты ────────────────────────────────────

def test_etalon_check_with_month_preset_and_flags(tmp_path, monkeypatch, capsys):
    from tests.test_etalon import _etalon_with, _rec
    et = _etalon_with(tmp_path, [("a", "111111", "1234", "DIGITS_ERROR")])
    f = make_month(tmp_path, [("100000", "A1", "1", "")], name="Ноябрь")
    month_app.set_month_preset(f.root, RecognitionPreset(missing_digit="model", serial_pad=0.1))
    seen = []
    from src.gmr.ml import loader
    monkeypatch.setattr(loader, "load_models", lambda cfg: seen.append(cfg))
    import src.gmr.application as app
    monkeypatch.setattr(app, "recognize_photo", lambda models, path, cfg: _rec("111111", number=1234))
    assert gmr.main(["etalon", "check", "--etalon", str(et), "--month", str(f.root),
                     "--forgive-last", "0", "--forgive-with", "model"]) == 0
    cfg = seen[0]
    assert (cfg.missing_digit_mode, cfg.serial_crop_pad, cfg.ignore_last_digits, cfg.forgiven_digit_mode) == (
        "model", 0.1, 0, "model")
    out = capsys.readouterr().out
    assert "настройки месяца Ноябрь: пропущенная цифра — прочитать моделью" in out
    assert "ignore_last_digits = 0" in out
    assert gmr.main(["etalon", "check", "--etalon", str(et), "--missing-digit", "operator"]) == 0
    assert seen[1].missing_digit_mode == "operator" and seen[1].ignore_last_digits == 2


@pytest.mark.parametrize("bad", [["--missing-digit", "guess"], ["--forgive-last", "3"]])
def test_etalon_check_rejects_unknown_values(tmp_path, bad):
    with pytest.raises(SystemExit):
        gmr.main(["etalon", "check", "--etalon", str(tmp_path), *bad])


def test_month_folder_unchanged_by_preset_reads(tmp_path):
    f = make_month(tmp_path, [("100000", "A1", "1", "")])
    with MonthDB(MonthFolder(Path(f.root)).db) as db:
        before = db.meta("recognition")
    month_app.month_preset(f.root)
    with MonthDB(f.db) as db:
        assert db.meta("recognition") == before is None
