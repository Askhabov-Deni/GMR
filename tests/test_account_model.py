"""
Лицевой счёт по надписи маркером: модель (models/account), чтение по
словарю (models/ctc_lexicon.py), правила (src/gmr/domain/account_match.py)
и встраивание в reader.process_photo.

Без весов: модель — маленькая случайная или поддельная.
"""
import itertools
import math

import numpy as np
import pytest
import torch

from models.account import config_account
from models.account.dataset_account import prepare_input
from models.account.model_account import AccountCRNN, load_checkpoint, save_checkpoint
from models.ctc_lexicon import CompiledLexicon, greedy_decode, match, string_log_likelihoods
from src.gmr.domain import AccountPrediction, Outcome, PipelineConfig
from src.gmr.domain.account_match import (
    AccountMarkerPolicy, account_groups, account_variants, serial_agrees, within_one_edit,
)
from src.gmr.ml import CrnnAccountRecognizer
from tests._fixtures import (
    FakeDigitDetector, FakeDigitOCR, FakeMeterDetector, FakeSerialOCR, make_crop, make_df,
    make_digit_crops_with_centers, make_meter_crops, process_photo,
)

DIGITS = "0123456789"
BLANK = 10


def peaky(text: str, steps_per_char: int = 3, sure: float = 0.98, alt: dict = None) -> torch.Tensor:
    """log_probs (T, 11), в которых модель «видит» text: на каждый символ —
    steps_per_char шагов, между символами — blank. alt = {позиция: (цифра, доля)} —
    сомнение в символе."""
    rows = []

    def row(idx_probs):
        p = torch.full((11,), (1 - sum(idx_probs.values())) / (11 - len(idx_probs)))
        for i, v in idx_probs.items():
            p[i] = v
        return p

    rows.append(row({BLANK: sure}))
    for pos, ch in enumerate(text):
        probs = {int(ch): sure}
        if alt and pos in alt:
            d, share = alt[pos]
            probs = {int(ch): sure * (1 - share), int(d): sure * share}
        rows += [row(probs)] * steps_per_char
        rows.append(row({BLANK: sure}))
    return torch.log(torch.stack(rows))


# ─── CTC: вероятность строки ─────────────────────────────────────────────────

def _brute_force(log_probs: torch.Tensor, blank: int) -> dict:
    """P(строка) перебором всех путей CTC — эталон для маленьких T."""
    T, C = log_probs.shape
    probs = log_probs.exp()
    out = {}
    for path in itertools.product(range(C), repeat=T):
        p = math.prod(float(probs[t, c]) for t, c in enumerate(path))
        chars, prev = [], -1
        for c in path:
            if c != prev and c != blank:
                chars.append(c)
            prev = c
        key = tuple(chars)
        out[key] = out.get(key, 0.0) + p
    return out


def test_string_likelihood_equals_brute_force():
    torch.manual_seed(0)
    lp = torch.log_softmax(torch.randn(5, 3), dim=1)     # алфавит «01», blank = 2
    truth = _brute_force(lp, blank=2)
    for s in [(0,), (1,), (0, 1), (1, 1), (0, 0, 1), (1, 0, 1)]:
        got = string_log_likelihoods(lp, [torch.tensor(s)], blank=2)[0]
        assert math.isclose(math.exp(float(got)), truth.get(s, 0.0), rel_tol=1e-4)


def test_too_long_string_is_impossible_not_certain():
    lp = torch.log_softmax(torch.randn(3, 11), dim=1)
    ll = string_log_likelihoods(lp, [torch.tensor([1, 2, 3, 4, 5])], blank=BLANK)
    assert ll[0] == float("-inf")      # zero_infinity дал бы 0 → «вероятность 1»


def test_greedy_decode_collapses_repeats_and_blanks():
    assert greedy_decode(peaky("00065"), DIGITS, BLANK) == "00065"


# ─── Чтение по словарю ───────────────────────────────────────────────────────

GROUPS = {
    "1300000065": account_variants("1300000065"),
    "1300000066": account_variants("1300000066"),
    "1300000082": account_variants("1300000082"),
}


def test_lexicon_picks_written_account_confidently():
    m = match(peaky("00065"), CompiledLexicon(GROUPS, DIGITS), BLANK)
    assert (m.group, m.string, m.greedy) == ("1300000065", "00065", "00065")
    assert m.confidence > 0.99


def test_lexicon_fixes_one_wrong_digit():
    # жадно читается 00068 (такого счёта нет), но 5 — второй вариант модели
    m = match(peaky("00068", alt={4: ("5", 0.3)}), CompiledLexicon(GROUPS, DIGITS), BLANK)
    assert m.greedy == "00068"
    assert m.group == "1300000065"


def test_lexicon_is_unsure_between_two_similar_accounts():
    m = match(peaky("00065", alt={4: ("6", 0.5)}), CompiledLexicon(GROUPS, DIGITS), BLANK)
    assert m.group in ("1300000065", "1300000066")
    assert 0.4 < m.confidence < 0.6


def test_lexicon_does_not_force_number_missing_from_table():
    m = match(peaky("54321"), CompiledLexicon(GROUPS, DIGITS), BLANK)
    assert m.greedy == "54321"
    assert m.confidence < 0.1


def test_shared_tail_makes_both_accounts_unsure():
    groups = {"1300000065": account_variants("1300000065"),
              "1400000065": account_variants("1400000065")}
    m = match(peaky("00065"), CompiledLexicon(groups, DIGITS), BLANK)
    assert m.confidence < 0.6


# ─── Варианты записи и правила ───────────────────────────────────────────────

def test_account_variants():
    # как пишут контролёры (владелец, 2026-10-12): без нулей, с частью нулей,
    # реже — с кодом региона (13-0093: черту модель не пишет)
    v = account_variants("1300000093")
    assert v == ("93", "093", "0093", "00093", "000093", "0000093", "00000093",
                 "1393", "13093", "130093", "1300093", "13000093", "130000093", "1300000093")
    assert account_variants("1300010000") == ("10000", "010000", "0010000", "00010000",
                                               "1310000", "13010000", "130010000", "1300010000")
    assert account_variants("1312345678") == ("12345678", "1312345678")
    assert account_variants("1300000000")[:2] == ("0", "00")
    assert account_variants("65") == ("65",) and account_variants(" 1300000093 ") == v
    assert account_variants("A-1") == ()
    assert account_groups(["1300000065", " 1300000065 ", "", "A-1"]) == {
        "1300000065": account_variants("1300000065")}


def test_within_one_edit_and_serial_agrees():
    assert within_one_edit("12345", "12345")
    assert within_one_edit("12345", "12845")       # замена
    assert within_one_edit("1234", "12345")        # пропуск
    assert within_one_edit("123456", "12345")      # лишний
    assert not within_one_edit("12345", "12354")   # перестановка — два действия
    assert not within_one_edit("123", "12345")
    assert serial_agrees("3456", ["003456"])       # ведущие нули — как в reader.py
    assert serial_agrees("1234567", ["123456"])
    assert not serial_agrees("", ["123456"])
    assert not serial_agrees("999999", ["123456"])


def pred(account="1300000065", conf=0.97, text="00065"):
    return AccountPrediction(text=text, text_conf=0.9, account=account, confidence=conf)


THR = 0.9
policy = AccountMarkerPolicy()


def test_policy_conflict():
    assert policy.conflict("1300000065", pred(), THR) is None
    assert policy.conflict("1300000065", None, THR) is None
    assert policy.conflict("1300000082", pred(conf=0.5), THR) is None      # неуверенно — молчим
    note = policy.conflict("1300000082", pred(), THR)
    assert "1300000065" in note and "1300000082" in note


@pytest.mark.parametrize("outcome, serial, table_serials, ambiguous, expected", [
    (Outcome.NO_SERIAL, None, ["123456"], [], "1300000065"),
    (Outcome.SERIAL_NOT_FOUND, "12345", ["123456"], [], "1300000065"),     # серийник «в одном символе»
    (Outcome.SERIAL_LOW_CONF, "123456", ["123456"], [], "1300000065"),
    (Outcome.SERIAL_NOT_FOUND, "777", ["123456"], [], None),                # совсем не похож
    (Outcome.SERIAL_AMBIGUOUS, "123456", ["123456"], ["1300000065", "1300000082"], "1300000065"),
    (Outcome.SERIAL_AMBIGUOUS, "123456", ["123456"], ["1300000070", "1300000082"], None),
])
def test_policy_rescue(outcome, serial, table_serials, ambiguous, expected):
    d = policy.rescue(outcome, pred(), THR, serial, table_serials, ambiguous)
    assert d.account == expected
    assert d.note


def test_policy_rescue_needs_confidence():
    d = policy.rescue(Outcome.NO_SERIAL, pred(conf=0.6), THR, None, ["123456"])
    assert d.account is None and "?" in d.note                 # подсказка оператору
    assert policy.rescue(Outcome.NO_SERIAL, None, THR, None, []).account is None


# ─── Модель, вход, чекпоинт ──────────────────────────────────────────────────

def test_prepare_input_keeps_aspect_and_background_zero():
    crop = np.full((100, 330), 200, np.uint8)
    crop[30:70, 20:300] = 30                                   # «надпись»
    x = prepare_input(crop, 48, 224)
    assert x.shape == (1, 48, 224) and x.dtype == np.float32
    assert abs(float(np.median(x[0, :, :140]))) < 1e-6         # фон → 0
    assert np.all(x[0, :, 160:] == 0)                          # 330×100 → 158×48, дальше поле
    wide = prepare_input(np.full((20, 400), 100, np.uint8), 48, 224)
    assert wide.shape == (1, 48, 224)                          # длинная надпись влезает по ширине
    bgr = prepare_input(np.full((40, 120, 3), 100, np.uint8), 48, 224)
    assert bgr.shape == (1, 48, 224)


def test_model_output_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = AccountCRNN(img_h=48).eval()
    x = torch.randn(2, 1, 48, 224)
    out = model(x)
    assert out.shape == (56, 2, 11)                            # T = W / 4
    assert torch.allclose(out.exp().sum(2), torch.ones(56, 2), atol=1e-5)
    save_checkpoint(tmp_path / "best.pt", model, {"img_h": 48, "img_w": 224}, DIGITS, epoch=3)
    loaded, meta = load_checkpoint(tmp_path / "best.pt")
    assert meta["preprocess"] == {"img_h": 48, "img_w": 224} and meta["epoch"] == 3
    assert torch.allclose(loaded(x), out, atol=1e-6)


def test_foreign_checkpoint_is_refused(tmp_path):
    torch.save({"model_state": {}}, tmp_path / "x.pt")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "x.pt")


def test_inferer_end_to_end_with_random_model(tmp_path):
    from models.account.infer_account import AccountInferer
    save_checkpoint(tmp_path / "best.pt", AccountCRNN(img_h=48), {"img_h": 48, "img_w": 224}, DIGITS)
    inf = AccountInferer(str(tmp_path / "best.pt"))
    crop = np.random.default_rng(0).integers(0, 255, (60, 200, 3), dtype=np.uint8)
    res = inf.predict(crop, GROUPS)
    assert set(res) >= {"text", "text_conf", "group", "string", "confidence", "top"}
    assert res["group"] in GROUPS and 0 <= res["confidence"] <= 1
    assert inf.compile(GROUPS) is inf.compile(GROUPS)           # словарь готовится один раз
    p = CrnnAccountRecognizer(inf).recognize(crop, GROUPS)
    assert p.account == res["group"]


def test_cli_threshold_equals_prod():
    assert config_account.MIN_CONFIDENCE == PipelineConfig().account_conf_thresh


# ─── reader.process_photo с надписью ─────────────────────────────────────────

class FakeAccountInferer:
    """Как AccountInferer.predict: что «написано» и какой это счёт."""

    def __init__(self, account, confidence, text="00065"):
        self.account, self.confidence, self.text = account, confidence, text
        self.calls = []

    def predict(self, image, groups=None):
        self.calls.append(groups)
        return {"text": self.text, "text_conf": 0.9, "group": self.account, "string": self.text,
                "confidence": self.confidence, "top": [(self.account, self.confidence)]}


TABLE = [
    {"serial": "123456", "account_id": "1300000065", "last_reading": "1000"},
    {"serial": "654321", "account_id": "1300000082", "last_reading": "2000"},
]


def _photo(config, serial_text, serial_conf=0.95, marker=None, with_serial=True, with_account=True,
           table=TABLE):
    crops = make_meter_crops()
    if not with_serial:
        crops = [c for c in crops if c["class"] != "serial_number"]
    if with_account:
        crops.append({"class": config.account_class, "conf": 0.9, "angle": 0.0,
                      "bbox": (0, 150, 120, 190), "crop": make_crop(40, 120), "path": None})
    digit_crops, arrays = make_digit_crops_with_centers([i * 40 for i in range(5)])
    docr = FakeDigitOCR({id(a): (d, 0.95) for a, d in zip(arrays, "01200")})
    kw = {}
    if marker is not None:
        kw = {"account_recognizer": CrnnAccountRecognizer(marker)}
    return process_photo("p.jpg", make_df(table), config, FakeMeterDetector(crops),
                         FakeDigitDetector(digit_crops), docr, FakeSerialOCR(serial_text, serial_conf), **kw)


def test_without_account_model_nothing_changes(base_config):
    r = _photo(base_config, "999999")
    assert r.outcome == Outcome.SERIAL_NOT_FOUND and r.account_notes is None


def test_marker_agrees_with_serial(base_config):
    marker = FakeAccountInferer("1300000065", 0.99)
    r = _photo(base_config, "123456", marker=marker)
    assert (r.outcome, r.account_id, r.reading) == (Outcome.PLUS, "1300000065", 1200)
    assert r.account_notes is None
    assert set(marker.calls[0]) == {"1300000065", "1300000082"}     # словарь — вся таблица


def test_marker_conflict_is_suspicious(base_config):
    r = _photo(base_config, "123456", marker=FakeAccountInferer("1300000082", 0.99))
    assert r.outcome == Outcome.SUSPICIOUS
    assert r.reading == 1200                                            # показание — оператору
    assert "1300000082" in r.error_detail and "1300000065" in r.error_detail
    # счёт не присвоен: показание 1300000065 с другого фото не закроет это фото молча
    assert r.account_id is None


def test_unsure_marker_does_not_block_serial(base_config):
    r = _photo(base_config, "123456", marker=FakeAccountInferer("1300000082", 0.5))
    assert (r.outcome, r.account_id) == (Outcome.PLUS, "1300000065")


def test_marker_finds_account_when_serial_label_missing(base_config):
    r = _photo(base_config, None, with_serial=False, marker=FakeAccountInferer("1300000082", 0.97, "00082"))
    assert (r.outcome, r.account_id, r.reading) == (Outcome.MINUS, "1300000082", 1200)
    assert r.serial_text == "654321"                                    # номер из таблицы
    assert "00082" in r.account_notes


def test_marker_finds_account_when_serial_misread_by_one_char(base_config):
    r = _photo(base_config, "12356", marker=FakeAccountInferer("1300000065", 0.97))
    assert (r.outcome, r.account_id) == (Outcome.PLUS, "1300000065")
    assert "12356" in r.account_notes and "123456" in r.account_notes


def test_marker_does_not_override_unlike_serial(base_config):
    r = _photo(base_config, "777777", marker=FakeAccountInferer("1300000065", 0.97))
    assert r.outcome == Outcome.SERIAL_NOT_FOUND and r.account_id is None
    assert "1300000065" in r.account_notes                              # подсказка оператору


def test_marker_picks_among_ambiguous_accounts(base_config):
    table = TABLE + [{"serial": "123456", "account_id": "1300000099", "last_reading": "1100"}]
    r = _photo(base_config, "123456", marker=FakeAccountInferer("1300000099", 0.97), table=table)
    assert (r.outcome, r.account_id, r.delta) == (Outcome.PLUS, "1300000099", 100)


def test_no_marker_on_photo_is_as_before(base_config):
    r = _photo(base_config, "999999", marker=FakeAccountInferer("1300000065", 0.99), with_account=False)
    assert r.outcome == Outcome.SERIAL_NOT_FOUND and r.account_notes is None


def test_log_notes_carry_marker_note(base_config):
    import reader
    r = _photo(base_config, None, with_serial=False, marker=FakeAccountInferer("1300000082", 0.97, "00082"))
    row = reader._make_log_row("p.jpg", r, base_config)
    assert "00082" in row["notes"]


def test_loader_refuses_detector_without_account_class():
    from src.gmr.ml import loader

    class Det:
        class inferer:
            class_names = {0: "gas_meter", 1: "serial_number"}
    cfg = PipelineConfig(account_ocr_model="x.pt", account_class="account")
    with pytest.raises(ValueError, match="gas_meter, serial_number"):
        loader.check_account_class(Det(), cfg)
    loader.check_account_class(Det(), PipelineConfig(account_class="serial_number"))


# ─── inspect account ─────────────────────────────────────────────────────────

def test_inspect_account_mode(tmp_path):
    from tools import inspect_model

    class Meter:
        def detect(self, path):
            return [{"class": "marker_id", "conf": 0.9, "angle": 0.0, "bbox": (0, 0, 120, 40),
                     "crop": np.full((40, 120, 3), 128, np.uint8), "path": None}]

    class Account:
        def recognize(self, crop, accounts=None):
            if accounts is None:
                return AccountPrediction("00065", 0.93)
            return AccountPrediction("00065", 0.93, "1300000065", 0.97, [("1300000065", 0.97)])

    def factory(cfg):
        return {"meter": Meter, "account": Account}

    photo = tmp_path / "p.jpg"
    inspect_model.write_image(photo, np.full((100, 200, 3), 90, np.uint8))
    rows = inspect_model.main(["account", str(photo), "--weights", "x.pt", "--out", str(tmp_path / "o")], factory)
    assert rows[0]["text"] == "00065" and "account" not in rows[0]
    assert (tmp_path / "o" / "p__crop.jpg").exists()

    table = tmp_path / "t.csv"
    table.write_text("Лицевой счет;Номер счетчика\n1300000065;123456\n", encoding="utf-8")
    rows = inspect_model.main(["account", str(photo), "--weights", "x.pt", "--table", str(table),
                               "--out", str(tmp_path / "o2")], factory)
    assert rows[0]["account"] == "1300000065" and rows[0]["ok"] is True
    with pytest.raises(SystemExit):        # без весов — понятная ошибка, а не падение
        inspect_model.main(["account", str(photo), "--out", str(tmp_path / "o3")], factory)


# ─── Отчёт оценки ────────────────────────────────────────────────────────────

def test_evaluate_counts_lexicon_hits_and_threshold_table():
    from models.account.evaluate_account import evaluate, report_text
    lps = [peaky("0065"), peaky("00082"), peaky("00065", alt={4: ("6", 0.5)})]
    accounts = ["1300000065", "1300000083", "1300000066"]   # второй кроп — не тот счёт (на нём 00082)
    groups = account_groups(["1300000065", "1300000082", "1300000083", "1300000066"])
    res = evaluate(lps, accounts, DIGITS, BLANK, groups)
    assert res["exact_match"] == 1 / 3 and res["lexicon_size"] == 4      # 0065 — одно из написаний 65
    assert res["cer"] == pytest.approx(2 / 14)                            # 00082 → 00083, 00065 → 00066
    picks = res["picks"]
    assert picks[0]["ok"] and picks[0]["confidence"] > 0.99 and picks[0]["string"] == "0065"
    assert not picks[1]["ok"] and picks[1]["group"] == "1300000082" and picks[1]["confidence"] > 0.95
    own = evaluate(lps[:1], accounts[:1], DIGITS, BLANK)["picks"][0]          # словарь — написания самих счетов
    assert own["ok"] and own["confidence"] > 0.99
    assert 0.4 < picks[2]["confidence"] < 0.6                       # 5 или 6 — не решить
    by = {r["threshold"]: r for r in res["by_threshold"]}
    assert (by[0.9]["accepted_n"], by[0.9]["wrong_n"]) == (2, 1)
    assert (by[0.95]["accepted_n"], by[0.95]["wrong_n"]) == (2, 1)
    assert "порог" in report_text(res, "тест") and res["not_in_dictionary"] == 0
    assert "⚠" not in report_text(res, "тест")
    one = evaluate(lps, accounts, DIGITS, BLANK, account_groups(["1300000065", "1300000082", "1300000083"]))
    assert one["not_in_dictionary"] == 1
    assert "нет в словаре: 1 из 3" in report_text(one, "т") and "не того участка" not in report_text(one, "т")
    other = evaluate(lps, accounts, DIGITS, BLANK, account_groups(["1300026377", "1300015718"]))
    assert other["not_in_dictionary"] == 3 and "Похоже, таблица не того участка" in report_text(other, "т")
