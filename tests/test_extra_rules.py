"""
Этап 6b (MODELS_REVIEW.md, решение владельца 2026-10-11 — каждое правило как
настройка месяца, по умолчанию выключено): правило барабана, старшая цифра
по прошлому показанию, серийник по таблице. Модели — подделки.
"""
import csv
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import gmr
import reader
from src.gmr.application import month as month_app
from src.gmr.application.recognition import (
    _drum_digit, _last_digits, describe_substitutions, read_meter_digits, read_meter_digits_for_config,
)
from src.gmr.domain import DigitPrediction, Outcome, PipelineConfig, SerialTableMatch
from src.gmr.domain.preset import RecognitionPreset
from src.gmr.domain.serial_match import build_serial_groups
from src.gmr.ml import CnnDigitRecognizer, CrnnSerialRecognizer
from tests._fixtures import (
    FakeDigitDetector, FakeDigitOCR, FakeMeterDetector, FakeSerialOCR, make_df,
    make_digit_crops_with_centers, make_meter_crops, process_photo,
)
from tests._month import make_month
from tests._pipeline import fake_models  # noqa: F401 (фикстура)
from tests.test_account_model import TABLE, peaky
from tests.test_preset import _Digits, _meter

# ─── Правило барабана ────────────────────────────────────────────────────────


@pytest.mark.parametrize("top, want", [
    ([("3", 0.5), ("4", 0.45)], "3"),
    ([("4", 0.5), ("3", 0.45)], "3"),        # младшая из двух, кто бы ни был первым
    ([("8", 0.3), ("9", 0.3)], "8"),         # ровно порог — достаточно
    ([("0", 0.5), ("1", 0.2)], "0"),
    ([("9", 0.5), ("0", 0.45)], "9"),        # между 9 и 0 — ещё 9
    ([("0", 0.5), ("9", 0.45)], "9"),
    ([("3", 0.5), ("5", 0.45)], None),       # не соседние
    ([("1", 0.5), ("9", 0.45)], None),
    ([("3", 0.3), ("4", 0.29)], None),       # вместе неуверенно
    ([("3", 0.5)], None),
    ([], None),
])
def test_drum_digit(top, want):
    assert _drum_digit(top, 0.6) == want


class _TopCnn:
    """Цифра — пиксель кропа; для цифр из tops — свой ответ и две лучшие."""
    def __init__(self, tops=None):
        self.tops = tops or {}

    def recognize(self, crop):
        d = int(crop[crop.shape[0] // 2, crop.shape[1] // 2, 0]) // 10
        if d in self.tops:
            top = self.tops[d]
            return DigitPrediction(digit=top[0][0], confidence=top[0][1], top=top)
        return DigitPrediction(digit=str(d), confidence=0.95, top=[(str(d), 0.95), ("0", 0.01)])


def _read(cnn, **kw):
    return read_meter_digits(_Digits([0, 1, 2, 3, 4]), cnn, _meter(), 0.6, 5, ignore_last_digits=2, **kw)


def test_drum_rule_in_reading():
    cnn = _TopCnn({3: [("4", 0.5), ("3", 0.45)]})            # сотни между 3 и 4
    assert _read(cnn).reading_str == "12?45"                  # правило выключено — как было
    r = _read(cnn, drum_rule=True)
    assert r.number == 12345 and r.error is None
    assert describe_substitutions(r.digit_results) == "подставлено: pos2='3' (барабан между 4 и 3, conf=0.500/0.450)"
    assert r.digit_results[2]["digit"] == "4"                 # ответ модели в логе остаётся


def test_drum_rule_before_forgiveness_and_only_for_neighbours():
    r = _read(_TopCnn({5: [("5", 0.5), ("6", 0.4)]}), drum_rule=True)
    assert r.reading_str == "12345"                           # последняя — барабан, а не «0»
    r = _read(_TopCnn({5: [("5", 0.5), ("7", 0.4)]}), drum_rule=True)
    assert r.reading_str == "12340"                           # не соседние — прощается как было
    r = _read(_TopCnn({2: [("2", 0.5), ("8", 0.4)]}), drum_rule=True)
    assert r.number is None and r.reading_str == "1?345"
    sure = _read(_TopCnn({2: [("3", 0.7), ("2", 0.2)]}), drum_rule=True)
    assert sure.reading_str == "13345"                        # уверенную цифру правило не трогает


# ─── Старшая цифра по прошлому показанию ─────────────────────────────────────

def test_last_digits():
    assert _last_digits("12000", 5) == "12000"
    assert _last_digits("1234.0", 5) == "01234" and _last_digits(1234, 5) == "01234"
    assert _last_digits("1 234", 5) is None and _last_digits(None, 5) is None
    assert _last_digits("123456", 5) is None                  # длиннее счётчика


def test_first_digit_from_last_reading():
    cnn = _TopCnn({1: [("7", 0.5), ("1", 0.3)]})              # старшая: 7 или 1
    assert _read(cnn).reading_str == "?2345"
    r = _read(cnn, last_reading="12000")
    assert r.number == 12345 and r.error is None
    assert describe_substitutions(r.digit_results) == (
        "подставлено: pos0='1' (по прошлому показанию; модель: 7 0.500 / 1 0.300)")
    assert _read(cnn, last_reading="32000").number is None    # прошлой цифры нет среди двух лучших
    assert _read(cnn, last_reading="мусор").number is None


def test_first_digit_only_when_it_is_the_only_unsure():
    both = _TopCnn({1: [("7", 0.5), ("1", 0.3)], 2: [("2", 0.5), ("8", 0.3)]})
    assert _read(both, last_reading="12000").reading_str == "??345"
    second = _TopCnn({2: [("1", 0.5), ("2", 0.3)]})           # неуверенна вторая, не старшая
    assert _read(second, last_reading="12000").reading_str == "1?345"
    tail = _TopCnn({1: [("7", 0.5), ("1", 0.3)], 5: [("5", 0.3), ("8", 0.2)]})
    r = _read(tail, last_reading="12000")                     # прощённая последняя не мешает
    assert r.reading_str == "12340" and r.number == 12340


def test_config_switches_reach_reading(monkeypatch):
    from src.gmr.application import recognition
    seen = []
    monkeypatch.setattr(recognition, "read_meter_digits", lambda *a, **k: seen.append(k))
    models = SimpleNamespace(digit_detector=1, digit_recognizer=2)
    read_meter_digits_for_config(models, None, PipelineConfig(), last_reading="1000")
    read_meter_digits_for_config(models, None, PipelineConfig(drum_rule=True, first_digit_from_last=True),
                                 last_reading="1000")
    assert (seen[0]["drum_rule"], seen[0]["last_reading"]) == (False, None)
    assert (seen[1]["drum_rule"], seen[1]["last_reading"]) == (True, "1000")


def test_recognize_photo_passes_last_reading(monkeypatch):
    from src.gmr.application import recognition
    seen = []
    monkeypatch.setattr(recognition, "read_meter_digits_for_config",
                        lambda models, crop, cfg, last_reading=None: seen.append(last_reading))
    models = SimpleNamespace(meter_detector=SimpleNamespace(detect=lambda p: [{"class": "gas_meter", "crop": 1}]))
    recognition.recognize_photo(models, "p.jpg", PipelineConfig(), last_reading="1000")
    recognition.recognize_photo(models, "p.jpg", PipelineConfig())
    assert seen == ["1000", None]


def _digits_photo(config, preds, serial="123456"):
    digit_crops, arrays = make_digit_crops_with_centers([i * 40 for i in range(5)])
    docr = FakeDigitOCR({id(a): p for a, p in zip(arrays, preds)})
    docr.predict = lambda image: {"digit": int(docr._predictions[id(image)][0]),
                                  "confidence": docr._predictions[id(image)][1],
                                  "top": docr._predictions[id(image)][2]}
    return process_photo("p.jpg", make_df(TABLE), config, FakeMeterDetector(make_meter_crops()),
                         FakeDigitDetector(digit_crops), docr, FakeSerialOCR(serial, 0.95))


def test_reader_gives_last_reading_of_the_account(base_config):
    preds = [("7", 0.5, [(7, 0.5), (0, 0.3)])] + [(d, 0.95, [(int(d), 0.95)]) for d in "1200"]
    r = _digits_photo(base_config, preds)
    assert r.outcome == Outcome.DIGITS_ERROR
    base_config.first_digit_from_last = True                  # прошлое 1000 → «01000»
    r = _digits_photo(base_config, preds)
    assert (r.outcome, r.reading) == (Outcome.PLUS, 1200)
    assert "по прошлому показанию" in r.digit_notes


# ─── Серийник по таблице ─────────────────────────────────────────────────────

def test_serial_groups_with_leading_zeros():
    assert build_serial_groups(["001234", " 5678 ", "5678", "AB12", "", "0099", 777]) == {
        "001234": ("001234", "01234", "1234"), "5678": ("5678",),
        "0099": ("0099", "099", "99"), "777": ("777",)}
    df = make_df([{"serial": "0123"}, {"serial": "4"}])
    assert reader.serial_groups_of(df, PipelineConfig()) == {"0123": ("0123", "123"), "4": ("4",)}


class _TableOCR(FakeSerialOCR):
    """CRNN с серийником по таблице: что прочитано, какой номер ближе всех."""
    def __init__(self, text, conf, serial, share):
        super().__init__(text, conf)
        self.serial, self.share, self.calls = serial, share, []

    def predict_in_table(self, image, serials):
        self.calls.append(serials)
        return {"text": self._text, "serial": self.serial, "confidence": self.share}


def _serial_photo(config, ocr, table=TABLE, **kw):
    digit_crops, arrays = make_digit_crops_with_centers([i * 40 for i in range(5)])
    docr = FakeDigitOCR({id(a): (d, 0.95) for a, d in zip(arrays, "01200")})
    return process_photo("p.jpg", make_df(table), config, FakeMeterDetector(make_meter_crops()),
                         FakeDigitDetector(digit_crops), docr, ocr, **kw)


def test_serial_by_table_off_changes_nothing(base_config):
    ocr = _TableOCR("123465", 0.95, "123456", 0.99)
    r = _serial_photo(base_config, ocr)
    assert r.outcome == Outcome.SERIAL_NOT_FOUND and r.serial_notes is None and ocr.calls == []


def test_serial_by_table_takes_sure_table_number(base_config):
    base_config.serial_by_table = True
    ocr = _TableOCR("123465", 0.95, "123456", 0.97)
    r = _serial_photo(base_config, ocr)
    assert (r.outcome, r.account_id, r.reading, r.serial_text) == (Outcome.PLUS, "1300000065", 1200, "123456")
    assert r.serial_notes == "серийник по таблице: прочитано 123465, взят 123456 (доля 0.97)"
    assert set(ocr.calls[0]) == {"123456", "654321"}                     # словарь — вся таблица
    assert "серийник по таблице" in reader._make_log_row("p.jpg", r, base_config)["notes"]
    low = _serial_photo(base_config, _TableOCR("123456", 0.4, "123456", 0.95))
    assert (low.outcome, low.account_id) == (Outcome.PLUS, "1300000065")  # неуверенный номер — тоже


def test_serial_by_table_unsure_goes_to_operator_with_hint(base_config):
    base_config.serial_by_table = True
    r = _serial_photo(base_config, _TableOCR("123465", 0.95, "123456", 0.89))
    assert r.outcome == Outcome.SERIAL_NOT_FOUND and r.account_id is None and r.serial_text == "123465"
    assert r.serial_notes == "серийник по таблице: не уверен — ближе всех 123456 (доля 0.89)"
    none = _serial_photo(base_config, _TableOCR("123465", 0.95, None, 0.0))
    assert none.outcome == Outcome.SERIAL_NOT_FOUND and none.serial_notes is None


def test_serial_by_table_not_used_when_number_found(base_config):
    base_config.serial_by_table = True
    ocr = _TableOCR("123456", 0.95, "654321", 0.99)
    r = _serial_photo(base_config, ocr)
    assert (r.outcome, r.account_id) == (Outcome.PLUS, "1300000065") and ocr.calls == []


def test_serial_by_table_ambiguous(base_config):
    base_config.serial_by_table = True
    table = TABLE + [{"serial": "123456", "account_id": "1300000099", "last_reading": "1100"}]
    r = _serial_photo(base_config, _TableOCR("123465", 0.95, "123456", 0.97), table=table)
    assert r.outcome == Outcome.SERIAL_AMBIGUOUS
    assert "1300000065" in r.error_detail and "1300000099" in r.error_detail


def test_serial_by_table_uses_given_dictionary(base_config):
    base_config.serial_by_table = True
    ocr, groups = _TableOCR("123465", 0.95, "123456", 0.97), build_serial_groups(["123456"])
    _serial_photo(base_config, ocr, serial_groups=groups)
    assert ocr.calls == [groups] and ocr.calls[0] is groups             # один словарь на прогон


def test_serial_by_table_in_month_run(tmp_path, fake_models, monkeypatch):  # noqa: F811
    from tests import _pipeline as pl
    seen, given = [], []
    pl.SerialOCR.predict_in_table = lambda self, image, serials: (
        seen.append(serials) or {"text": "99999", "serial": "33333", "confidence": 0.95})
    real = reader.process_photo
    monkeypatch.setattr(reader, "process_photo", lambda *a, **k: given.append(k["serial_groups"]) or real(*a, **k))
    try:
        folder = pl.setup_month(tmp_path)
        month_app.set_month_preset(folder.root, RecognitionPreset(serial_by_table=True))
        pl.run(folder)
    finally:
        del pl.SerialOCR.predict_in_table
    row = pl.log_by_name(folder)["p2.jpg"]
    assert row["outcome"] != "SERIAL_NOT_FOUND" and row["account_id"] == "A-3"
    assert "серийник по таблице: прочитано 99999, взят 33333" in row["notes"]
    assert len(seen) == 1 and set(seen[0]) == {"11111", "22222", "33333"}
    assert given and all(g is seen[0] for g in given)         # словарь собран один раз на прогон


# ─── Модели: две лучшие цифры, номер по таблице ──────────────────────────────

def test_cnn_gives_two_best_digits():
    from models.cnn.infer_cnn import CNNInferer
    inferer = CNNInferer.__new__(CNNInferer)
    inferer._preprocess_image = lambda image: image
    inferer.model = lambda t: torch.tensor([[0.0, 3.0, 2.9, -5, -5, -5, -5, -5, -5, -5]])
    res = inferer.predict(None)
    assert res["digit"] == 1 and [d for d, _ in res["top"]] == [1, 2]
    assert res["top"][0][1] == pytest.approx(res["confidence"]) and res["top"][1][1] < res["confidence"]
    pred = CnnDigitRecognizer(SimpleNamespace(predict=lambda c: res)).recognize(None)
    assert [d for d, _ in pred.top] == ["1", "2"]
    old = CnnDigitRecognizer(SimpleNamespace(predict=lambda c: {"digit": 3, "confidence": 0.9})).recognize(None)
    assert old.top == []                                      # старый ответ без top — правила молчат


def test_crnn_number_by_table():
    from models.crnn.infer_crnn import CRNNInferer
    inferer = CRNNInferer.__new__(CRNNInferer)
    inferer.log_probs = lambda image: peaky("123465")
    groups = build_serial_groups(["123456", "777777"])
    res = inferer.predict_in_table(None, groups)
    assert res["text"] == "123465" and res["confidence"] < 0.1            # видно другой номер — не натягивается
    inferer.log_probs = lambda image: peaky("12345", alt={4: ("6", 0.4)})
    res = inferer.predict_in_table(None, build_serial_groups(["12345", "12346", "777777"]))
    assert res["serial"] == "12345" and 0.5 < res["confidence"] < 0.9     # 6 тоже похоже — не уверен
    lex = inferer._lexicon
    inferer.predict_in_table(None, inferer._lexicon_for)
    assert inferer._lexicon is lex                            # тот же словарь — не пересобирается
    m = CrnnSerialRecognizer(inferer).match_table(None, inferer._lexicon_for)
    assert isinstance(m, SerialTableMatch) and m.serial == "12345" and m.text == "12345"


def test_crnn_log_probs_shape():
    from models.crnn.infer_crnn import CRNNInferer
    from models.crnn.model_crnn import CRNN
    lp = CRNNInferer(CRNN(), device=torch.device("cpu")).log_probs(np.zeros((30, 150, 3), np.uint8))
    assert lp.ndim == 2 and lp.shape[0] > 5 and lp.shape[1] == 11
    assert torch.allclose(lp.exp().sum(1), torch.ones(lp.shape[0]), atol=1e-4)


# ─── Пресет и эталон ─────────────────────────────────────────────────────────

def test_preset_switches():
    p = RecognitionPreset(drum_rule=True, first_from_last=True, serial_by_table=True)
    assert RecognitionPreset.from_json(p.to_json()) == p
    cfg = p.apply(PipelineConfig())
    assert (cfg.drum_rule, cfg.first_digit_from_last, cfg.serial_by_table) == (True, True, True)
    assert p.describe().endswith("дополнительно — правило барабана, старшая цифра по прошлому показанию, "
                                 "серийник по таблице")
    assert RecognitionPreset.from_json('{"drum_rule": 1, "serial_by_table": "yes"}') == RecognitionPreset()
    cfg = RecognitionPreset().apply(PipelineConfig(drum_rule=True))
    assert not cfg.drum_rule                                  # пресет решает за месяц


def test_run_applies_new_switches(tmp_path, monkeypatch):
    f = make_month(tmp_path, [("100000", "A1", "1", "")])
    month_app.set_month_preset(f.root, RecognitionPreset(drum_rule=True, first_from_last=True))
    cfg = reader.PipelineConfig(month_dir=str(f.root))
    reader._apply_month_preset(cfg, f, logging.getLogger("t"))
    assert (cfg.drum_rule, cfg.first_digit_from_last, cfg.serial_by_table) == (True, True, False)


def test_etalon_add_keeps_last_reading_and_fills_old_rows(tmp_path):
    from tests.test_etalon import _answers, _hard_month
    from tools import etalon
    f, rows = _hard_month(tmp_path)
    et = tmp_path / "etalon"
    etalon.add(f.root, etalon=et)
    first = _answers(et)
    assert first and all(a["last_reading"] == "" for a in first)   # в логе нет прошлого показания
    with open(et / "answers.csv", "w", encoding="utf-8-sig", newline="") as fh:   # эталон до этапа 6b
        w = csv.DictWriter(fh, fieldnames=etalon.ANSWER_COLUMNS[:-1], extrasaction="ignore")
        w.writeheader()
        w.writerows(first)
    from src.gmr.storage.month import MonthDB
    with MonthDB(f.db) as db, db.transaction():
        db.conn.execute("UPDATE processing_log SET last_reading='900' WHERE source='manual'")
    rep = etalon.add(f.root, etalon=et)
    assert rep.filled == len(first) and rep.added == 0
    assert f"дописано прошлое показание: {len(first)}" in rep.text()
    again = _answers(et)
    assert [a["photo_hash"] for a in again] == [a["photo_hash"] for a in first]
    assert all(a["last_reading"] == "900" for a in again)
    assert etalon.add(f.root, etalon=et).filled == 0


def test_hard_answers_take_last_reading():
    from tests.test_etalon import row
    from tools import etalon
    got = etalon.hard_answers([
        row("DIGITS_ERROR", h="h1", last_reading="1000.0"),
        row("PLUS", "manual", h="h1", serial_id="1", reading="1200", last_reading="1000.0"),
        row("NO_SERIAL", h="h2", last_reading=""),
        row("PLUS", "manual", h="h2", serial_id="2", reading="5", last_reading="4"),
        row("DIGITS_ERROR", h="h3", last_reading="7"),
        row("PLUS", "manual", h="h3", serial_id="3", reading="9", last_reading=""),
    ])
    assert [a.last_reading for a in got] == ["1000", "4", "7"]


def _check_rec(serial, conf, number=1234):
    return SimpleNamespace(meter={}, serial={"crop": "c"}, digits=SimpleNamespace(number=number),
                           serial_prediction=SimpleNamespace(text=serial, confidence=conf))


def test_etalon_check_new_switches(tmp_path, monkeypatch, capsys):
    from tests.test_etalon import _etalon_with
    from tools import etalon
    et = _etalon_with(tmp_path, [("a", "111111", "1234", "SERIAL_NOT_FOUND"),
                                 ("b", "222222", "1234", "SERIAL_LOW_CONF"),
                                 ("c", "333333", "1234", "DIGITS_ERROR"),
                                 ("d", "444444", "1234", "SERIAL_NOT_FOUND")])
    rows = etalon.read_answers(et)
    rows[2]["last_reading"] = "1000"
    with open(et / "answers.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=etalon.ANSWER_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    f = make_month(tmp_path, [("111111", "A1", "1", ""), ("222222", "A2", "1", ""),
                              ("333333", "A3", "1", ""), ("444444", "A4", "1", "")], name="Ноябрь")
    recs = {"a": _check_rec("111112", 0.95), "b": _check_rec("222223", 0.3), "c": _check_rec("333333", 0.95),
            "d": _check_rec("444445", 0.95)}
    table = {"111112": ("111111", 0.95), "222223": ("444444", 0.95),   # b — уверенно, но чужой номер
             "444445": ("444444", 0.85)}                                # d — доля ниже порога 0.9
    seen_last, seen_groups = {}, []

    class Serial:
        def match_table(self, crop, groups):
            seen_groups.append(groups)
            serial, share = table[crop_text[0]]
            return SerialTableMatch(crop_text[0], serial, share)
    crop_text = [""]
    models = SimpleNamespace(serial_recognizer=Serial())
    from src.gmr.ml import loader
    monkeypatch.setattr(loader, "load_models", lambda cfg: models)

    def recognize(models, path, cfg, last_reading=None):
        name = Path(path).stem
        seen_last[name] = last_reading
        crop_text[0] = recs[name].serial_prediction.text
        return recs[name]
    import src.gmr.application as app
    monkeypatch.setattr(app, "recognize_photo", recognize)

    assert gmr.main(["etalon", "check", "--etalon", str(et), "--serial-by-table"]) == 1
    assert "укажите --month" in capsys.readouterr().out
    assert gmr.main(["etalon", "check", "--etalon", str(et), "--month", str(f.root),
                     "--serial-by-table", "--first-from-last", "--drum-rule"]) == 0
    out = capsys.readouterr().out
    assert "drum_rule = да" in out and "first_digit_from_last = да" in out and "serial_by_table = да" in out
    assert seen_last == {"a": None, "b": None, "c": "1000", "d": None}
    assert len(seen_groups) == 3 and set(seen_groups[0]) == {"111111", "222222", "333333", "444444"}
    res = {r["photo_hash"]: r for r in csv.DictReader(open(sorted((et / "checks").glob("*.csv"))[-1],
                                                           encoding="utf-8-sig"))}
    assert res["a"]["serial_ok"] == "True" and res["a"]["model_serial"] == "111111"
    assert res["b"]["serial_wrong_sure"] == "True" and res["b"]["model_serial"] == "444444"
    assert res["c"]["serial_ok"] == "True"                    # найден в таблице — словарь не нужен
    assert res["d"]["serial_ok"] == "False" and res["d"]["model_serial"] == "444445"
    assert gmr.main(["etalon", "check", "--etalon", str(et)]) == 0  # без переключателей — как было
    res = {r["photo_hash"]: r for r in csv.DictReader(open(sorted((et / "checks").glob("*.csv"))[-1],
                                                           encoding="utf-8-sig"))}
    assert res["a"]["serial_ok"] == "False" and len(seen_groups) == 3
