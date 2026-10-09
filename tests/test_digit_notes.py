"""
Пометка подставленных цифр в notes лога (решение владельца 2026-09-30,
вариант А). Исходы, показания и таблица не меняются — только текст notes.
"""
from src.gmr.application import SUBSTITUTED_PREFIX, describe_substitutions
from src.gmr.storage import LOG_COLUMNS
from tests._fixtures import (
    FakeDigitDetector, FakeDigitOCR, make_df, make_digit_crops_with_centers,
    make_meter_crops, process_photo, FakeMeterDetector, FakeSerialOCR,
)
from tools import analyze_log as al
import reader
from reader import Outcome

SERIAL, ACCOUNT = "12345", "A-0001"


def _run(cfg, xs, digits, confs, last_reading="1000"):
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": last_reading}])
    crops, arrays = make_digit_crops_with_centers(xs, width=20)
    ocr = FakeDigitOCR({id(a): (d, c) for a, d, c in zip(arrays, digits, confs)})
    return process_photo("p.jpg", df, cfg, FakeMeterDetector(make_meter_crops()),
                         FakeDigitDetector(crops), ocr, FakeSerialOCR(SERIAL, 0.95))


# ─── describe_substitutions ──────────────────────────────────────────────────

def test_describe_substitutions():
    results = [
        {"position": 0, "digit": "1", "confidence": 0.95, "ok": True},
        {"position": 2, "digit": "5", "confidence": None, "ok": True, "note": "inserted placeholder"},
        {"position": 4, "digit": "7", "confidence": 0.4123, "ok": False, "note": "forgiven→0"},
    ]
    assert describe_substitutions(results) == (
        "подставлено: pos2='5' (цифра не найдена), pos4='0' (прочитано 7, conf=0.412)")
    assert describe_substitutions(results[:1]) is None
    assert describe_substitutions(None) is None


# ─── process_photo: исход тот же, появляется пометка ─────────────────────────

def test_all_digits_read_no_note(base_config):
    r = _run(base_config, [i * 40 for i in range(5)], list("01200"), [0.95] * 5)
    assert r.outcome == Outcome.PLUS and r.digit_notes is None


def test_missing_digit_placeholder_noted(base_config):
    # как golden case 12: пропущена цифра на позиции 2
    r = _run(base_config, [0, 50, 200, 250], list("1200"), [0.95] * 4)
    assert r.reading_str == "12500"                  # поведение прежнее
    assert r.digit_notes == "подставлено: pos2='5' (цифра не найдена)"


def test_forgiven_digit_noted(base_config):
    # как golden case 14: последняя цифра неуверенная → '0'
    r = _run(base_config, [i * 40 for i in range(5)], list("01209"), [0.95] * 4 + [0.3])
    assert r.reading_str == "01200" and r.outcome == Outcome.PLUS
    assert r.digit_notes == "подставлено: pos4='0' (прочитано 9, conf=0.300)"


def test_note_also_on_digits_error(base_config):
    # первая цифра неуверенная (ошибка), последняя прощена — пометка про последнюю
    r = _run(base_config, [i * 40 for i in range(5)], list("01209"), [0.3] + [0.95] * 3 + [0.3])
    assert r.outcome == Outcome.DIGITS_ERROR
    assert r.digit_notes == "подставлено: pos4='0' (прочитано 9, conf=0.300)"


def test_note_on_serial_not_found(base_config):
    df = make_df([{"serial": "999", "account_id": ACCOUNT, "last_reading": "1000"}])
    crops, arrays = make_digit_crops_with_centers([i * 40 for i in range(5)], width=20)
    ocr = FakeDigitOCR({id(a): (d, c) for a, d, c in zip(arrays, "01209", [0.95] * 4 + [0.3])})
    r = process_photo("p.jpg", df, base_config, FakeMeterDetector(make_meter_crops()),
                      FakeDigitDetector(crops), ocr, FakeSerialOCR(SERIAL, 0.95))
    assert r.outcome == Outcome.SERIAL_NOT_FOUND
    assert r.digit_notes.startswith(SUBSTITUTED_PREFIX)


# ─── строка лога ─────────────────────────────────────────────────────────────

def test_log_row_notes(base_config):
    r = _run(base_config, [i * 40 for i in range(5)], list("01209"), [0.95] * 4 + [0.3])
    assert reader._make_log_row("p.jpg", r, base_config)["notes"] == r.digit_notes

    plain = _run(base_config, [i * 40 for i in range(5)], list("01200"), [0.95] * 5)
    assert reader._make_log_row("p.jpg", plain, base_config)["notes"] == ""   # как раньше

    # SUSPICIOUS: причина исхода и пометка вместе
    sus = _run(base_config, [i * 40 for i in range(5)], list("90009"), [0.95] * 4 + [0.3],
               last_reading="1000")
    assert sus.outcome == Outcome.SUSPICIOUS
    notes = reader._make_log_row("p.jpg", sus, base_config)["notes"]
    assert notes == f"{sus.error_detail} | {sus.digit_notes}"


# ─── analyze_log считает пометки ─────────────────────────────────────────────

def _row(name, outcome, **kw):
    r = {c: "" for c in LOG_COLUMNS}
    r.update(original_filename=name, outcome=outcome, source="auto", **kw)
    return r


def test_analyze_log_counts_substitutions(base_config):
    rows = [
        _row("a", "PLUS", model_reading_str="12500", notes="подставлено: pos2='5' (цифра не найдена)"),
        _row("b", "MINUS", model_reading_str="12300",
             notes="подставлено: pos3='0' (прочитано 7, conf=0.400), pos4='0' (прочитано 1, conf=0.300)"),
        _row("c", "SUSPICIOUS", model_reading_str="92300",
             notes="delta=+9000 > ±10000 | подставлено: pos4='0' (прочитано 1, conf=0.300)"),
        _row("d", "PLUS", model_reading_str="11111"),
    ]
    d = al.digit_stats(rows, base_config)
    assert d["substituted"] == {"photos": 3, "of": 4, "missing": 1, "forgiven": 3, "drum": 0, "last": 0}
    text = al.build_report(rows, base_config)
    assert "подставленными цифрами (PLUS/MINUS/SUSPICIOUS): 3 из 4" in text
    assert "барабана" not in text and "по прошлому показанию" not in text


def test_analyze_log_counts_new_rules(base_config):
    from src.gmr.application.recognition import describe_substitutions
    note = describe_substitutions([
        {"position": 0, "digit": "7", "confidence": 0.5, "ok": True, "note": "last→1",
         "top": [("7", 0.5), ("1", 0.3)]},
        {"position": 2, "digit": "4", "confidence": 0.5, "ok": True, "note": "drum→3",
         "top": [("4", 0.5), ("3", 0.45)]},
        {"position": 4, "digit": "8", "confidence": 0.3, "ok": False, "note": "forgiven→0"}])
    rows = [_row("a", "PLUS", model_reading_str="12300", notes=note)]
    assert al.digit_stats(rows, base_config)["substituted"] == {
        "photos": 1, "of": 1, "missing": 0, "forgiven": 1, "drum": 1, "last": 1}
    assert "по правилу барабана — 1, старших по прошлому показанию — 1" in al.build_report(rows, base_config)
