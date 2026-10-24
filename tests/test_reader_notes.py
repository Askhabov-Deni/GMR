"""
Надписи контролёров поверх фото в работе (настройка месяца overlay_notes,
решение владельца 2026-10-26): reader.process_photo и правило
note_match.rescue. Модели — поддельные, фото с надписью — настоящее (её
читает src/gmr/overlay.py); счета выдуманные.
"""
import pytest

import reader
from src.gmr.domain import Outcome
from src.gmr.domain.note_match import AccountIndex, NoteAnswer, rescue
from src.gmr.overlay import GlyphBank
from src.gmr.render.image_io import write_image
from tests._fixtures import (
    FakeDigitDetector, FakeDigitOCR, FakeMeterDetector, FakeSerialOCR, make_df,
    make_digit_crops_with_centers, make_meter_crops, process_photo,
)
from tests._pipeline import fake_models  # noqa: F401 (фикстура)
from tests.test_overlay import BLUE, _jpeg, _background, _photo

TABLE = [                                          # барабан на фото — 1200 (цифры «01200»)
    {"serial": "123456", "account_id": "1300000065", "last_reading": "1000"},
    {"serial": "654321", "account_id": "1300000082", "last_reading": "2000"},
]
BANK = GlyphBank.load()


def _notes(table=TABLE):
    accounts = [r["account_id"] for r in table]
    return reader.NoteContext(BANK, AccountIndex(accounts), {r["account_id"]: int(r["last_reading"]) for r in table})


def _run(tmp_path, config, serial_text, lines=(), with_serial=True, notes=True, table=TABLE, serial_conf=0.95):
    path = tmp_path / "p.jpg"
    img = _photo([(t, BLUE, 400 + 200 * i, 0) for i, t in enumerate(lines)]) if lines else _jpeg(_background(3))
    write_image(path, img)
    crops = make_meter_crops()
    if not with_serial:
        crops = [c for c in crops if c["class"] != "serial_number"]
    digit_crops, arrays = make_digit_crops_with_centers([i * 40 for i in range(5)])
    docr = FakeDigitOCR({id(a): (d, 0.95) for a, d in zip(arrays, "01200")})
    return process_photo(str(path), make_df(table), config, FakeMeterDetector(crops), FakeDigitDetector(digit_crops),
                         docr, FakeSerialOCR(serial_text, serial_conf), notes=_notes(table) if notes else None)


def test_no_serial_account_by_note(tmp_path, base_config):
    r = _run(tmp_path, base_config, None, ["лс82"], with_serial=False)
    assert (r.outcome, r.account_id, r.reading) == (Outcome.MINUS, "1300000082", 1200)
    assert r.serial_text == "654321" and r.account_notes == "надпись на фото «лс82» → л/с 1300000082"


def test_unsure_serial_account_by_note(tmp_path, base_config):
    r = _run(tmp_path, base_config, "999999", ["лс65"], serial_conf=0.3)
    assert (r.outcome, r.account_id) == (Outcome.PLUS, "1300000065")


def test_switch_off_is_as_before(tmp_path, base_config):
    r = _run(tmp_path, base_config, None, ["лс82"], with_serial=False, notes=False)
    assert r.outcome == Outcome.NO_SERIAL and r.account_id is None and r.account_notes is None


def test_serial_not_in_table_and_unlike_keeps_operator(tmp_path, base_config):
    r = _run(tmp_path, base_config, "777777", ["лс65"])
    assert r.outcome == Outcome.SERIAL_NOT_FOUND and r.account_id is None
    assert "надпись на фото «лс65» → л/с 1300000065, но серийник '777777' не похож" in r.account_notes
    assert r.reading == 1200                                          # показание — оператору


def test_note_not_understood_is_a_hint(tmp_path, base_config):
    r = _run(tmp_path, base_config, None, ["лс99999"], with_serial=False)
    assert r.outcome == Outcome.NO_SERIAL and r.account_id is None
    assert r.account_notes.startswith("надпись на фото «лс99999»: ")


def test_note_against_serial_is_suspicious(tmp_path, base_config):
    r = _run(tmp_path, base_config, "123456", ["лс82"])
    assert r.outcome == Outcome.SUSPICIOUS and r.account_id is None
    assert r.error_detail == "надпись на фото «лс82»: надпись указывает на л/с 1300000082, по табличке — 1300000065"


def test_note_agreeing_with_serial_changes_nothing(tmp_path, base_config):
    r = _run(tmp_path, base_config, "123456", ["лс65"])
    assert (r.outcome, r.account_id, r.error_detail) == (Outcome.PLUS, "1300000065", None)


def test_no_note_on_photo(tmp_path, base_config):
    r = _run(tmp_path, base_config, None, [], with_serial=False)
    assert r.outcome == Outcome.NO_SERIAL and r.account_notes is None


def test_note_picks_among_owners_of_number(tmp_path, base_config):
    table = TABLE + [{"serial": "123456", "account_id": "1300000099", "last_reading": "1100"}]
    r = _run(tmp_path, base_config, "123456", ["лс99"], table=table)
    assert (r.outcome, r.account_id, r.delta) == (Outcome.PLUS, "1300000099", 100)
    assert r.account_notes.startswith("номер у нескольких абонентов, выбран по надписи")


def test_note_context_of_table(base_config):
    df = make_df(TABLE + [{"serial": "1", "account_id": "1300000001", "last_reading": "нет"},
                          {"serial": "2", "account_id": "1300000002", "last_reading": "2353,5"}])
    ctx = reader.note_context_of(df, base_config)
    assert ctx.lasts == {"1300000065": 1000, "1300000082": 2000, "1300000002": 2353}
    assert ctx.index.find("65") == {"1300000065"}


@pytest.mark.parametrize("outcome, serial, account, expected", [
    (Outcome.NO_SERIAL, None, "1300000082", "1300000082"),
    (Outcome.SERIAL_LOW_CONF, "999", "1300000082", "1300000082"),
    (Outcome.SERIAL_NOT_FOUND, "65432", "1300000082", "1300000082"),      # номер — с точностью до символа
    (Outcome.SERIAL_NOT_FOUND, "777777", "1300000082", None),
    (Outcome.SERIAL_AMBIGUOUS, "654321", "1300000082", "1300000082"),
    (Outcome.SERIAL_AMBIGUOUS, "654321", "1300000065", None),
    (Outcome.NO_SERIAL, None, None, None),
])
def test_rescue_rules(outcome, serial, account, expected):
    d = rescue(outcome, ["лс82"], NoteAnswer(account=account, why="почему"), serial, ["654321"],
               ["1300000082", "1300000099"])
    assert d.account == expected and d.note.startswith("номер у нескольких" if expected and outcome ==
                                                       Outcome.SERIAL_AMBIGUOUS else "надпись на фото «лс82»")


def test_month_run_passes_notes_when_switched_on(tmp_path, monkeypatch, fake_models):  # noqa: F811
    from src.gmr.application import month as month_app
    from src.gmr.domain.preset import RecognitionPreset
    from tests import _pipeline
    seen = []
    original = reader.process_photo

    def spy(*args, **kw):
        seen.append(kw.get("notes"))
        return original(*args, **kw)
    monkeypatch.setattr(reader, "process_photo", spy)
    folder = _pipeline.setup_month(tmp_path / "выкл")
    _pipeline.run(folder)
    assert seen and all(n is None for n in seen)                     # по умолчанию — как без надписей
    seen.clear()
    folder = _pipeline.setup_month(tmp_path / "вкл")
    month_app.set_month_preset(folder.root, RecognitionPreset(overlay_notes=True))
    _pipeline.run(folder)
    assert seen and all(n is seen[0] for n in seen)                  # один на прогон
    assert seen[0].index.find("A-1") == set() and "A-1" in seen[0].lasts
