"""
Этап 5b: эталон трудных фото (tools/etalon.py) и запас вокруг рамки
серийника (PipelineConfig.serial_crop_pad). Модели — подделки.
"""
import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gmr
from src.gmr.domain import PipelineConfig
from src.gmr.storage import LOG_COLUMNS, photo_fingerprint, save_log
from src.gmr.storage.month import MonthDB
from tests._month import img, make_month
from tests._pipeline import fake_models  # noqa: F401 (фикстура)
from tools import etalon


def row(outcome, source="auto", h="", name="", **kw):
    r = {c: "" for c in LOG_COLUMNS}
    r.update(outcome=outcome, source=source, photo_hash=h, original_filename=name, **kw)
    return r


# ─── Какие фото — трудные ────────────────────────────────────────────────────

def test_hard_answers():
    rows = [
        row("DIGITS_ERROR", h="h1", name="a.jpg", source_folder="Аюб"),
        row("PLUS", "manual", h="h1", name="a.jpg", serial_id="111111", reading="01234"),
        row("SERIAL_NOT_FOUND", h="h2"), row("REPEAT", "manual", h="h2", serial_id="2", reading="5"),
        row("PLUS", h="h3", name="c.jpg", serial_id="333333", reading="777",
            notes="ок | исправлено при проверке (абонент A → B)"),
        row("PLUS", h="h4", serial_id="4", reading="4", verified_by="Оператор"),          # «Верно» — не трудное
        row("ERROR", h="h5"), row("PLUS", "manual", h="h5", serial_id="5", reading="5"),   # сбой программы
        row("SERIAL_NOT_FOUND", h="h6", name="f.jpg"), row("SERIAL_LOW_CONF", h="h6", name="f.jpg"),
        row("MINUS", "manual", h="h6", serial_id="666666", reading="12.0",
            notes="серийник в базе с ошибкой: в базе 666660, на фото 666666"),
        row("NO_METER", h="h7"), row("UNREADABLE", "manual", h="h7"),
        row("DIGITS_ERROR", name="old.jpg", source_folder="Сулиман С"),                   # старый лог: без отпечатка
        row("PLUS", "manual", name="old.jpg", source_folder="Сулиман С", serial_id="888", reading="42"),
        row("PLUS", "pre_existing", name="__pre_existing__", serial_id="9", reading="9"),
        row("NO_SERIAL", h="h8"), row("PLUS", "manual", h="h8", serial_id="", reading="10"),  # без номера
    ]
    got = [(a.photo_hash, a.reason, a.serial, a.reading, a.original_filename, a.source_folder)
           for a in etalon.hard_answers(rows)]
    assert got == [
        ("h1", "DIGITS_ERROR", "111111", "1234", "a.jpg", "Аюб"),
        ("h3", "CORRECTED", "333333", "777", "c.jpg", ""),
        ("h6", "SERIAL_LOW_CONF", "666666", "12", "f.jpg", ""),
        ("", "DIGITS_ERROR", "888", "42", "old.jpg", "Сулиман С"),
    ]


def test_every_third_photo_to_etalon():
    hashes = [hashlib.sha256(str(i).encode()).hexdigest()[:16] for i in range(3000)]
    share = sum(map(etalon.to_etalon, hashes)) / len(hashes)
    assert 0.30 < share < 0.37
    assert [etalon.to_etalon(h) for h in hashes[:50]] == [etalon.to_etalon(h) for h in hashes[:50]]


# ─── Пополнение эталона ──────────────────────────────────────────────────────

def _hard_month(tmp_path, n=12):
    """Месяц с n трудными фото (DIGITS_ERROR → «Принять»)."""
    f = make_month(tmp_path, [("100000", "A1", "1", "")], name="m")
    rows = []
    for i in range(n):
        p = img(f.photos / "Аюб" / f"p{i}.jpg", 10 + i)
        h = photo_fingerprint(str(p))
        rows += [row("DIGITS_ERROR", h=h, name=p.name, source_folder="Аюб"),
                 row("PLUS", "manual", h=h, name=p.name, serial_id=f"10000{i}", reading=f"{i}00")]
    with MonthDB(f.db) as db, db.transaction():
        db.append_log_rows(rows)
    return f, rows


def _answers(et):
    with open(et / "answers.csv", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_add_from_month(tmp_path):
    f, rows = _hard_month(tmp_path)
    hashes = [r["photo_hash"] for r in rows if r["source"] == "manual"]
    want = [h for h in hashes if etalon.to_etalon(h)]
    assert 0 < len(want) < len(hashes)                        # тестовые фото попали в обе стороны
    et = tmp_path / "etalon"
    k = hashes.index(want[0])
    moved = f.photos / "Аюб" / f"p{k}.jpg"
    moved.rename(f.photos / "Аюб" / "renamed.jpg")            # фото переименовали — найдётся по отпечатку,
    img(moved, 200)                                           # а другое фото с его прежним именем не возьмётся
    rep = etalon.add(f.root, etalon=et)
    assert (rep.found, rep.etalon, rep.added, rep.already, rep.training, rep.missing) == (
        12, len(want), len(want), 0, 12 - len(want), 0)
    assert (et / "photos" / f"{want[0]}.jpg").read_bytes() == (f.photos / "Аюб" / "renamed.jpg").read_bytes()
    got = _answers(et)
    assert [a["photo_hash"] for a in got] == want
    a = got[0]
    i = hashes.index(a["photo_hash"])
    assert a["file"] == f"{a['photo_hash']}.jpg" and a["serial"] == f"10000{i}" and a["reading"] == f"{i}00"
    assert a["reason"] == "DIGITS_ERROR" and a["source"] == "m"
    assert (et / "photos" / a["file"]).read_bytes() == next(
        p for p in (f.photos / "Аюб").iterdir() if photo_fingerprint(str(p)) == a["photo_hash"]).read_bytes()
    again = etalon.add(f.root, etalon=et)                     # второй раз — ничего не дублируется
    assert (again.added, again.already) == (0, len(want)) and len(_answers(et)) == len(want)
    assert f"в эталон (каждое третье): {len(want)} — добавлено 0, уже были {len(want)}" in again.text()
    more = []                                                 # месяц пополнился — дописываются новые
    for i in range(12, 30):
        p = img(f.photos / "Аюб" / f"q{i}.jpg", 10 + i)
        h = photo_fingerprint(str(p))
        more += [row("NO_SERIAL", h=h, name=p.name, source_folder="Аюб"),
                 row("PLUS", "manual", h=h, name=p.name, serial_id="5", reading="5")]
    with MonthDB(f.db) as db, db.transaction():
        db.append_log_rows(more)
    third = etalon.add(f.root, etalon=et)
    assert third.added > 0 and len(_answers(et)) == len(want) + third.added
    assert all(a["photo_hash"] != "photo_hash" for a in _answers(et))   # заголовок не повторился


def test_add_missing_photo_counted(tmp_path):
    f, rows = _hard_month(tmp_path)
    for p in (f.photos / "Аюб").iterdir():
        p.unlink()
    rep = etalon.add(f.root, etalon=tmp_path / "e")
    assert rep.missing == rep.etalon > 0 and rep.added == 0 and not (tmp_path / "e" / "answers.csv").exists()
    assert "⚠ фото не найдено на диске" in rep.text()


def test_add_from_old_csv_log(tmp_path):
    photos = tmp_path / "старые фото"
    rows = []
    for i in range(9):
        img(photos / "Сулиман С" / f"o{i}.jpg", 50 + i)
        rows += [row("SERIAL_NOT_FOUND", name=f"o{i}.jpg", source_folder="Сулиман С"),
                 row("PLUS", "manual", name=f"o{i}.jpg", source_folder="Сулиман С", serial_id=f"7{i}", reading="5")]
    rows += [row("DIGITS_ERROR", name="нет.jpg"), row("PLUS", "manual", name="нет.jpg", serial_id="1", reading="1")]
    log = tmp_path / "Реестр_log.csv"
    save_log(str(log), rows)
    with pytest.raises(ValueError, match="--photos"):
        etalon.add(log, etalon=tmp_path / "e")
    rep = etalon.add(log, photos, etalon=tmp_path / "e")
    assert rep.found == 10 and rep.missing == 1 and rep.etalon + rep.training == 9
    for a in _answers(tmp_path / "e"):                         # отпечаток посчитан по найденному файлу
        assert photo_fingerprint(str(tmp_path / "e" / "photos" / a["file"])) == a["photo_hash"]
        assert a["reason"] == "SERIAL_NOT_FOUND" and a["source"] == "Реестр_log.csv"


def test_add_not_a_month(tmp_path, capsys):
    assert gmr.main(["etalon", "add", str(tmp_path)]) == 1
    assert "не папка месяца" in capsys.readouterr().out


# ─── Проверка моделей ────────────────────────────────────────────────────────

def test_serial_and_reading_match():
    assert etalon.serial_matches("123456", "0123456") and etalon.serial_matches("123456", "00123456")
    assert etalon.serial_matches(" 123456", "123456")
    assert not etalon.serial_matches("12345", "123456") and not etalon.serial_matches("", "")
    cfg = PipelineConfig()
    assert etalon.reading_matches(1234, "01234", cfg)
    assert not etalon.reading_matches(1230, "1234", cfg)        # «0» вместо настоящей цифры — неверно
    assert not etalon.reading_matches(2234, "1234", cfg) and not etalon.reading_matches(None, "1234", cfg)


def _etalon_with(tmp_path, answers):
    et = tmp_path / "etalon"
    (et / "photos").mkdir(parents=True)
    with open(et / "answers.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=etalon.ANSWER_COLUMNS)
        w.writeheader()
        for h, serial, reading, reason in answers:
            w.writerow({"photo_hash": h, "file": f"{h}.jpg", "serial": serial, "reading": reading,
                        "reason": reason, "source": "m", "added_at": ""})
            (et / "photos" / f"{h}.jpg").write_bytes(b"x")
    return et


def _rec(serial=None, conf=0.9, number=None, meter=True):
    return SimpleNamespace(meter={} if meter else None, serial={} if serial is not None else None,
                           serial_prediction=SimpleNamespace(text=serial, confidence=conf) if serial is not None else None,
                           digits=SimpleNamespace(number=number) if meter else None)


def test_check(tmp_path, monkeypatch):
    et = _etalon_with(tmp_path, [
        ("a", "111111", "1234", "SERIAL_NOT_FOUND"),   # теперь всё верно
        ("b", "222222", "1234", "SERIAL_NOT_FOUND"),   # номер уверенно, но неверно
        ("c", "333333", "5678", "DIGITS_ERROR"),       # номер верно, цифры не прочитаны
        ("d", "444444", "5678", "DIGITS_ERROR"),       # цифры уверенно, но неверно
        ("e", "555555", "1", "NO_METER"),              # счётчик не найден
        ("f", "666666", "1", "CORRECTED"),             # фото пропало из папки
    ])
    (et / "photos" / "f.jpg").unlink()
    recs = {"a": _rec("111111", number=1234), "b": _rec("222221", number=1234),
            "c": _rec("333333", number=None), "d": _rec("444444", 0.4, number=9678),
            "e": _rec(None, meter=False)}
    import src.gmr.application as app
    monkeypatch.setattr(app, "recognize_photo", lambda models, path, cfg, **kw: recs[Path(path).stem])
    text, out = etalon.check(PipelineConfig(), et, models=object(), title="Модели: как в работе")
    lines = text.splitlines()
    assert lines[0] == "Модели: как в работе"
    by = {ln.split("  ")[0]: ln for ln in lines if ln and not ln.startswith(("Модели", "причина", " "))}
    assert "Серийник не найден" in by and "Ошибка цифр" in by and "Нет счётчика" in by
    total = by["ВСЕГО"].split()
    assert total[1] == "5"                                       # f — без файла
    assert "2 (100%)" in by["Серийник не найден"] and "1 (50%)" in by["Серийник не найден"]
    assert "⚠ нет файла фото в эталоне: 1" in text
    with open(out, encoding="utf-8-sig", newline="") as fh:
        res = {r["photo_hash"]: r for r in csv.DictReader(fh)}
    assert res["a"]["serial_ok"] == res["a"]["reading_ok"] == "True"
    assert res["b"]["serial_wrong_sure"] == "True" and res["b"]["serial_ok"] == "False"
    assert res["c"]["serial_ok"] == "True" and res["c"]["reading_ok"] == "False"
    assert res["c"]["reading_wrong_sure"] == "False"
    assert res["d"]["serial_ok"] == "False" and res["d"]["serial_wrong_sure"] == "False"   # неуверенный номер
    assert res["d"]["reading_wrong_sure"] == "True" and res["d"]["model_reading"] == "9678"
    assert res["e"]["meter_found"] == "False"
    assert out.parent == et / "checks"


def test_check_summary_counts():
    r = [etalon.PhotoCheck("a", "DIGITS_ERROR", True, True, True, False, True, False),
         etalon.PhotoCheck("b", "DIGITS_ERROR", True, True, False, True, False, False),
         etalon.PhotoCheck("c", "DIGITS_ERROR", False, False, False, False, True, False),
         etalon.PhotoCheck("d", "DIGITS_ERROR", False, False, False, False, False, True)]
    line = [ln for ln in etalon.summary(r, "t").splitlines() if ln.startswith("Ошибка цифр")][0]
    assert line.split()[2:] == ["4", "2", "(50%)", "1", "(25%)", "2", "(50%)", "1", "(25%)", "2", "(50%)"]


def test_check_empty_etalon(tmp_path, capsys):
    assert gmr.main(["etalon", "check", "--etalon", str(tmp_path)]) == 1
    assert "эталон пуст" in capsys.readouterr().out


def test_check_command_config(tmp_path, monkeypatch, capsys):
    et = _etalon_with(tmp_path, [("a", "111111", "1234", "DIGITS_ERROR")])
    seen = {}
    from src.gmr.ml import loader
    monkeypatch.setattr(loader, "load_models", lambda cfg: seen.setdefault("cfg", cfg))
    import src.gmr.application as app
    monkeypatch.setattr(app, "recognize_photo", lambda models, path, cfg, **kw: _rec("111111", number=1234))
    assert gmr.main(["etalon", "--etalon", str(et), "check", "--serial-pad", "0.05",
                     "--serial-weights", "new.pt", "--digit-weights", "d.pth"]) == 0
    cfg = seen["cfg"]
    assert cfg.serial_crop_pad == 0.05 and cfg.serial_ocr_model == "new.pt" and cfg.digit_ocr_model == "d.pth"
    assert cfg.meter_detect_model == PipelineConfig().meter_detect_model
    out = capsys.readouterr().out
    assert "запас вокруг рамки серийника = 0.05" in out and "serial_ocr_model = new.pt" in out
    assert gmr.main(["etalon", "check", "--etalon", str(et)]) == 0
    assert "Модели: как в работе (PipelineConfig)" in capsys.readouterr().out


def test_status(tmp_path, capsys):
    et = _etalon_with(tmp_path, [("a", "1", "1", "DIGITS_ERROR"), ("b", "1", "1", "DIGITS_ERROR"),
                                 ("c", "1", "1", "CORRECTED")])
    assert gmr.main(["etalon", "--etalon", str(et)]) == 0
    out = capsys.readouterr().out
    assert "фото: 3" in out and "Ошибка цифр: 2" in out and "Исправлено на «Проверке»: 1" in out


# ─── Запас вокруг рамки серийника ────────────────────────────────────────────

class _V:
    def __init__(self, v):
        self.v = v

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.v)

    def item(self):
        return self.v


class _FakeYolo:
    names = {0: "serial_number", 1: "gas_meter"}

    def __call__(self, img, conf, verbose):
        return [SimpleNamespace(boxes=_Boxes())]


class _Boxes:
    xyxy = [_V([10, 5, 110, 25]), _V([150, 0, 198, 40]), _V([5, 50, 105, 90])]
    conf = [_V(0.9), _V(0.5), _V(0.8)]
    cls = [_V(0), _V(0), _V(1)]

    def __len__(self):
        return 3


def _crops(pad):
    from models.yolo_all_detect.infer_yolo import YOLOInferer
    inf = YOLOInferer(_FakeYolo(), straighten=False, pad=pad)
    dets = inf._process_img(np.zeros((100, 200, 3), np.uint8), max_per_class=2)
    return {(d["class"], d["conf"]): (d["bbox"], d["crop"].shape[:2]) for d in dets}


def test_serial_pad_widens_only_serial():
    plain, padded = _crops(None), _crops({"serial_number": 0.1})
    assert plain[("serial_number", 0.9)] == ((10, 5, 110, 25), (20, 100))
    assert padded[("serial_number", 0.9)] == ((0, 5, 120, 25), (20, 120))     # по 10% ширины с боков
    assert padded[("serial_number", 0.5)] == ((145, 0, 200, 40), (40, 55))    # у края фото — до края
    assert padded[("gas_meter", 0.8)] == plain[("gas_meter", 0.8)] == ((5, 50, 105, 90), (40, 100))


def test_loader_passes_pad_only_when_set(monkeypatch):
    from src.gmr.ml import loader
    calls = []
    monkeypatch.setattr(loader, "YOLOInferer", lambda *a, **k: calls.append(k) or object())
    loader.load_meter_detector(PipelineConfig())
    loader.load_meter_detector(PipelineConfig(serial_crop_pad=0.07))
    assert "pad" not in calls[0] and calls[1]["pad"] == {"serial_number": 0.07}
    assert PipelineConfig().serial_crop_pad == 0.0                              # в работе — как было


def test_inspect_serial_pad(tmp_path):
    from tests.test_models_tools import _fake_loaders
    from tools import inspect_model
    photo = tmp_path / "a.jpg"
    inspect_model.write_image(photo, np.full((100, 200, 3), 90, np.uint8))
    factory = _fake_loaders()
    inspect_model.main(["serial", str(photo), "--out", str(tmp_path / "o"), "--serial-pad", "0.1"], factory)
    assert factory.cfg.serial_crop_pad == 0.1
    with pytest.raises(SystemExit):
        inspect_model.main(["digit", str(photo), "--serial-pad", "0.1"], factory)


def test_check_with_real_pipeline_code(tmp_path, fake_models):  # noqa: F811
    # настоящие load_models и recognize_photo, модели — подделки tests/_pipeline
    import cv2

    from tests import _pipeline as pl
    et = _etalon_with(tmp_path, [("p0", "11111", "1200", "DIGITS_ERROR"),
                                 ("p1", "22222", "4000", "SERIAL_NOT_FOUND"),
                                 ("p2", "99998", "1000", "SERIAL_NOT_FOUND"),
                                 ("p4", "33333", "100", "NO_METER")])
    for name in ("p0", "p1", "p2", "p4"):
        cv2.imwrite(str(et / "photos" / f"{name}.jpg"), pl.image(f"{name}.jpg"))
    text, out = etalon.check(PipelineConfig(), et)
    with open(out, encoding="utf-8-sig", newline="") as fh:
        res = {r["photo_hash"]: r for r in csv.DictReader(fh)}
    assert res["p0"]["serial_ok"] == res["p0"]["reading_ok"] == "True"
    assert res["p1"]["serial_ok"] == res["p1"]["reading_ok"] == "True"
    assert res["p2"]["serial_wrong_sure"] == "True" and res["p2"]["model_serial"] == "99999"
    assert res["p4"]["meter_found"] == "False"
    assert [ln for ln in text.splitlines() if ln.startswith("ВСЕГО")][0].split()[1] == "4"
