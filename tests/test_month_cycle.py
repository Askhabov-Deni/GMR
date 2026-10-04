"""
Месячный цикл (этап 3, решения владельца 2026-10-03): папку фото за месяц
только пополняют, программа её не меняет; каждый прогон читает только новые
фото, каждое фото разбирается один раз.

  1а. Разобранное фото (показание, повтор, решение оператора) при повторном
      прогоне пропускается молча: не копируется, строки в лог не пишется.
  2а. Фото с авто-ошибкой ждёт оператора и не перечитывается; заново —
      по `process --reread` или SERIAL_NOT_FOUND / «Нет в базе», если номер
      появился в таблице. Фото, на котором программа упала (ERROR), читается
      заново (этап 2.1b).
  3.  Когда у счёта появилось показание, его фото в question/ (цифры,
      подозрительно) уходят в repeat/.
  4б. Фото в корне фото\\ быть не может: прогон не начинается.
"""
from src.gmr.domain import PhotoState, RunSelection, RunSelectionPolicy, photo_state


def r(outcome, source="auto", **kw):
    return {"outcome": outcome, "source": source, **kw}


# ─── Состояние фото по логу ──────────────────────────────────────────────────

def test_photo_state():
    assert photo_state([]) == PhotoState.NEW
    assert photo_state([r("PLUS")]) == PhotoState.DONE
    assert photo_state([r("REPEAT")]) == PhotoState.DONE
    assert photo_state([r("DIGITS_ERROR")]) == PhotoState.WAITING
    assert photo_state([r("ERROR")]) == PhotoState.ERROR
    assert photo_state([r("DIGITS_ERROR"), r("PLUS", "manual")]) == PhotoState.DONE
    assert photo_state([r("SERIAL_NOT_FOUND"), r("NOT_IN_DB", "manual")]) == PhotoState.NOT_IN_DB
    # оператор отметил «Нет в базе», номер появился, прочитано заново
    assert photo_state([r("SERIAL_NOT_FOUND"), r("NOT_IN_DB", "manual"), r("PLUS")]) == PhotoState.DONE
    # старые логи: после решения оператора прогоны писали REPEAT «already in log»
    assert photo_state([r("NO_METER"), r("UNREADABLE", "manual"), r("REPEAT")]) == PhotoState.DONE
    # строка-копия в прогоне ничего не говорит о фото
    assert photo_state([r("REPEAT", notes="duplicate file in current run: a.jpg")]) == PhotoState.NEW


# ─── Читать или пропустить ───────────────────────────────────────────────────

POLICY = RunSelectionPolicy()


def _decide(rows, in_table=(), copy=False, reread=False):
    return POLICY.decide(rows, copy, lambda s: s in in_table, reread)


def test_new_and_done():
    assert _decide([]) == RunSelection.READ
    assert _decide([r("PLUS")]) == RunSelection.SKIP_DONE
    assert _decide([r("MINUS")], reread=True) == RunSelection.SKIP_DONE      # --reread не трогает разобранные
    assert _decide([r("UNREADABLE", "manual")], reread=True) == RunSelection.SKIP_DONE
    assert _decide([], copy=True) == RunSelection.SKIP_COPY


def test_waiting_not_read_again_without_reason():
    for outcome in ("NO_METER", "NO_SERIAL", "SERIAL_LOW_CONF", "SERIAL_NOT_FOUND",
                    "DIGITS_ERROR", "SUSPICIOUS", "SERIAL_AMBIGUOUS"):
        assert _decide([r(outcome, serial_id="11111")]) == RunSelection.SKIP_WAITING, outcome
        assert _decide([r(outcome, serial_id="11111")], reread=True) == RunSelection.READ, outcome


def test_error_read_again():
    assert _decide([r("ERROR")]) == RunSelection.READ


def test_serial_appeared_in_table():
    snf = r("SERIAL_NOT_FOUND", serial_id="99999")
    assert _decide([snf]) == RunSelection.SKIP_WAITING
    assert _decide([snf], in_table={"99999"}) == RunSelection.READ
    # только «номер не найден»: у других ошибок номер есть в таблице и так
    assert _decide([r("DIGITS_ERROR", serial_id="99999")], in_table={"99999"}) == RunSelection.SKIP_WAITING


def test_not_in_db_rule():
    rows = [r("SERIAL_NOT_FOUND", serial_id="9999"), r("NOT_IN_DB", "manual", serial_id="99999")]
    assert _decide(rows) == RunSelection.SKIP_DONE
    assert _decide(rows, reread=True) == RunSelection.SKIP_DONE              # решение оператора
    assert _decide(rows, in_table={"99999"}) == RunSelection.READ             # номер оператора
    assert _decide([r("NOT_IN_DB", "manual", serial_id="")], in_table={""}) == RunSelection.SKIP_DONE


# ─── Прогоны по папке месяца (модели — фейки) ────────────────────────────────
# Фото — PNG-байты, пиксель [0,0] — номер записи в PHOTOS (имя файла не важно).
# До этапа 3 эти фейки жили в tests/test_known_issues.py.

from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import gmr  # noqa: E402
import reader  # noqa: E402
from src.gmr.application import month  # noqa: E402
from src.gmr.application.cycle import CLOSED_NOTE  # noqa: E402
from src.gmr.ml import loader  # noqa: E402
from src.gmr.render import read_image  # noqa: E402
from src.gmr.storage import LOG_COLUMNS  # noqa: E402
from src.gmr.storage.month import MonthDB, MonthFolder  # noqa: E402
from tests._month import make_month  # noqa: E402

# номер -> (серийник, 5 цифр или None = цифры не прочитаны)
PHOTOS = {
    1: ("11111", "01200"),     # A-1
    2: ("22222", "04000"),     # A-2 (MINUS)
    3: ("33333", None),        # A-3, цифры не прочитаны
    4: ("33333", "00150"),     # A-3, хорошее фото
    5: ("99999", "00500"),     # номера нет в таблице
    6: ("33333", "90000"),     # A-3, подозрительно (разница > 10000)
    7: ("11111", "01300"),     # A-1, другое фото того же счётчика
    8: ("33333", None),        # A-3, ещё одно неудачное фото
}
TABLE = [("11111", "A-1", "1000", ""), ("22222", "A-2", "5000", ""), ("33333", "A-3", "100", "")]


class _Meter:
    calls = 0

    def process_image(self, img_path, **kwargs):
        _Meter.calls += 1
        t = int(read_image(img_path)[0, 0, 0])
        return [
            {"class": "gas_meter", "conf": .95, "angle": 0., "bbox": (0, 0, 200, 40),
             "crop": np.full((40, 200, 3), t, np.uint8), "path": None},
            {"class": "serial_number", "conf": .95, "angle": 0., "bbox": (0, 40, 100, 60),
             "crop": np.full((20, 100, 3), t, np.uint8), "path": None},
        ]


class _Digits:
    def process_array(self, img, **kwargs):
        digits = PHOTOS[int(img[0, 0, 0])][1]
        if digits is None:
            return []
        return [{"class": "digit", "conf": .95, "angle": 0., "bbox": (i * 40, 0, i * 40 + 20, 30),
                 "crop": np.full((30, 20, 3), int(d), np.uint8), "path": None}
                for i, d in enumerate(digits)]


class _DigitOCR:
    def predict(self, img):
        return {"digit": int(img[0, 0, 0]), "confidence": .95}


class _SerialOCR:
    def predict_with_details(self, img):
        return {"text": PHOTOS[int(img[0, 0, 0])][0], "avg_confidence": .95, "details": []}


@pytest.fixture
def models(monkeypatch):
    _Meter.calls = 0
    loads = []
    real_load = reader.load_models
    monkeypatch.setattr(reader, "load_models", lambda *a, **k: loads.append(1) or real_load(*a, **k))
    monkeypatch.setattr(loader, "YOLOInferer",
                        lambda m, conf_thresh=.8, straighten=True, output_dir=None:
                        _Digits() if straighten is False else _Meter())
    monkeypatch.setattr(loader, "CNNInferer", lambda *a, **k: _DigitOCR())
    monkeypatch.setattr(loader, "CRNNInferer", lambda *a, **k: _SerialOCR())
    return loads


def _photo(path: Path, n: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(".png", np.full((100, 200, 3), n, np.uint8))[1].tobytes())


def _month(root: Path, table=TABLE) -> MonthFolder:
    return make_month(root, table)


def _run(f: MonthFolder, **kw):
    reader.run_pipeline(reader.PipelineConfig(month_dir=str(f.root), **kw))


def _log(f: MonthFolder) -> list[dict]:
    with MonthDB(f.db) as db:
        return db.log_rows()


def _reading(f: MonthFolder, account: str) -> str:
    with MonthDB(f.db) as db:
        r = db.reading(account)
    return r.value if r else ""


def _files(f: MonthFolder) -> list[str]:
    return sorted(p.relative_to(f.results).as_posix() for p in f.results.rglob("*.jpg"))


def _report(f: MonthFolder, controller="Аюб") -> str:
    return (f.results / controller / "report.txt").read_text(encoding="utf-8")


def _day1(tmp_path):
    """a1 — показание A-1, a3 — цифры A-3 не прочитаны, a5 — номера нет в таблице."""
    f = _month(tmp_path)
    for name, n in (("a1.jpg", 1), ("a3.jpg", 3), ("a5.jpg", 5)):
        _photo(f.photos / "Аюб" / name, n)
    _run(f)
    return f


# ─── 1а, 2а: каждый прогон читает только новые фото ──────────────────────────

def test_next_run_reads_only_new_photos(tmp_path, models):
    f = _day1(tmp_path)
    rows, calls, files = len(_log(f)), _Meter.calls, _files(f)
    assert files == ["Аюб/plus/A-1.jpg", "Аюб/question/digits_error/A-3.jpg",
                     "Аюб/question/serial_not_found/a5.jpg"]

    _photo(f.photos / "Аюб" / "a2.jpg", 2)                     # на следующий день докинули фото
    _run(f)
    assert [(r["original_filename"], r["outcome"]) for r in _log(f)[rows:]] == [("a2.jpg", "MINUS")]
    assert _Meter.calls == calls + 1                             # прочитано одно фото
    assert _files(f) == sorted(files + ["Аюб/minus/A-2.jpg"])   # прежние файлы не тронуты
    report = _report(f)
    assert "Новых фото в этом прогоне: 1" in report
    assert "Уже разобраны раньше (пропущены): 1" in report
    assert "Ждут оператора с прошлых прогонов: 2" in report


def test_run_without_new_photos_does_not_load_models(tmp_path, models):
    f = _day1(tmp_path)
    rows = len(_log(f))
    _run(f)
    assert len(models) == 1 and len(_log(f)) == rows             # модели грузились только в 1-й раз
    assert "(новых фото нет)" in _report(f)
    assert f.export_xlsx.is_file()


def test_same_file_again_under_other_name_is_skipped(tmp_path, models):
    f = _day1(tmp_path)
    rows = len(_log(f))
    data = (f.photos / "Аюб" / "a1.jpg").read_bytes()
    (f.photos / "Сулиман С").mkdir()
    (f.photos / "Сулиман С" / "переслано.jpg").write_bytes(data)  # тот же файл, другой контролёр
    _run(f)
    assert len(_log(f)) == rows
    assert "Уже разобраны раньше (пропущены): 1" in _report(f, "Сулиман С")
    assert not (f.results / "Сулиман С" / "repeat").exists()


def test_copy_in_one_run_is_skipped(tmp_path, models):
    f = _month(tmp_path)
    _photo(f.photos / "Аюб" / "a.jpg", 5)
    (f.photos / "Аюб" / "b (1).jpg").write_bytes((f.photos / "Аюб" / "a.jpg").read_bytes())
    _run(f)
    assert [r["original_filename"] for r in _log(f)] == ["a.jpg"]
    assert _files(f) == ["Аюб/question/serial_not_found/a.jpg"]   # оператору — один раз
    assert "Тот же файл ещё раз (копия): 1" in _report(f)


def test_reread_reads_waiting_photos_again(tmp_path, models, monkeypatch):
    f = _day1(tmp_path)
    rows = len(_log(f))
    monkeypatch.setitem(PHOTOS, 3, ("33333", "00150"))            # «модель стала лучше»
    _run(f, reread_errors=True)
    assert [(r["original_filename"], r["outcome"]) for r in _log(f)[rows:]] == [
        ("a3.jpg", "PLUS"), ("a5.jpg", "SERIAL_NOT_FOUND")]       # a1 не перечитывается
    assert _reading(f, "A-3") == "150"
    assert _files(f) == ["Аюб/plus/A-1.jpg", "Аюб/plus/A-3.jpg",   # из question/ — убрано
                         "Аюб/question/serial_not_found/a5.jpg"]


def test_serial_appeared_after_table_update(tmp_path, models):
    f = _day1(tmp_path)
    rows = len(_log(f))
    _run(f)
    assert len(_log(f)) == rows                                   # без новой таблицы — ждёт
    month.load_table(str(f.root), str(_with_99999(tmp_path)))
    _run(f)
    assert [(r["original_filename"], r["outcome"], r["account_id"]) for r in _log(f)[rows:]] == [
        ("a5.jpg", "PLUS", "A-9")]
    assert _reading(f, "A-9") == "500"
    assert "Аюб/question/serial_not_found/a5.jpg" not in _files(f)
    assert "Аюб/plus/A-9.jpg" in _files(f)


def test_not_in_db_read_again_when_serial_appeared(tmp_path, models):
    f = _day1(tmp_path)
    a5 = next(r for r in _log(f) if r["original_filename"] == "a5.jpg")
    nid = f.results / "Аюб" / "not_in_db"                          # оператор: «Нет в базе»
    nid.mkdir()
    (f.results / "Аюб" / "question" / "serial_not_found" / "a5.jpg").rename(nid / "a5.jpg")
    with MonthDB(f.db) as db, db.transaction():
        db.append_log_rows([{c: "" for c in LOG_COLUMNS} | {
            "original_filename": "a5.jpg", "serial_id": "99999", "outcome": "NOT_IN_DB",
            "source": "manual", "processed_by": "Оператор", "photo_hash": a5["photo_hash"],
            "source_folder": "Аюб"}])
    rows = len(_log(f))
    _run(f, reread_errors=True)                                   # решение оператора --reread не трогает
    assert [r["original_filename"] for r in _log(f)[rows:]] == ["a3.jpg"]
    month.load_table(str(f.root), str(_with_99999(tmp_path)))
    _run(f)
    assert _log(f)[-1]["outcome"] == "PLUS" and _reading(f, "A-9") == "500"
    assert not (nid / "a5.jpg").exists()


def _with_99999(tmp_path):
    upd = tmp_path / "новая.csv"
    upd.write_text("Номер счетчика,Лицевой счет,Последние показания,Текущие показания\n"
                   + "".join(f"{s},{a},{last},\n" for s, a, last, _ in TABLE) + "99999,A-9,400,\n",
                   encoding="utf-8")
    return upd


def test_two_photos_with_same_result_name_do_not_overwrite(tmp_path, models):
    f = _month(tmp_path)
    _photo(f.photos / "Аюб" / "a.jpg", 1)
    _photo(f.photos / "Аюб" / "b.jpg", 7)                         # A-1 ещё раз — повтор
    (f.photos / "Аюб" / "c.jpg").write_bytes((f.photos / "Аюб" / "b.jpg").read_bytes())  # копия b
    _photo(f.photos / "Сулиман С" / "d.jpg", 1)                   # копия a в другой папке
    _run(f)
    _photo(f.photos / "Аюб" / "e.jpg", 3)
    _photo(f.photos / "Аюб" / "g.jpg", 8)                         # два неудачных фото A-3
    _run(f)
    assert _files(f) == ["Аюб/plus/A-1.jpg", "Аюб/question/digits_error/A-3.jpg",
                         "Аюб/question/digits_error/A-3_2.jpg", "Аюб/repeat/A-1.jpg"]
    finals = {r["original_filename"]: r["final_filename"] for r in _log(f)}
    assert (finals["e.jpg"], finals["g.jpg"]) == ("A-3.jpg", "A-3_2.jpg")


# ─── 4б: фото в корне фото\ быть не может ────────────────────────────────────

def test_photo_in_root_stops_run(tmp_path, models):
    f = _month(tmp_path)
    _photo(f.photos / "Аюб" / "p1.jpg", 1)
    _photo(f.photos / "случайно.jpg", 4)
    with pytest.raises(reader.PhotosInRoot, match="случайно.jpg"):
        _run(f)
    assert _log(f) == [] and _files(f) == [] and not f.backups.exists()
    assert not (f.results / "run_logs").exists() and models == []   # прогон не начинался


def test_cli_photo_in_root(tmp_path, models, capsys):
    f = _month(tmp_path)
    _photo(f.photos / "случайно.jpg", 4)
    assert gmr.main(["process", str(f.root)]) == 1
    assert "без папки контролёра" in capsys.readouterr().out


# ─── 3: неудачное фото не блокирует хорошее ──────────────────────────────────

def _closed(f, name):
    return [r for r in _log(f) if r["original_filename"] == name and CLOSED_NOTE in r["notes"]]


def test_good_photo_after_failed_in_same_run(tmp_path, models):
    # было известной ошибкой: хорошее фото становилось REPEAT «duplicate in current run»
    f = _month(tmp_path)
    _photo(f.photos / "Аюб" / "y1.jpg", 3)                        # A-3, цифры не прочитаны
    _photo(f.photos / "Аюб" / "y2.jpg", 4)                        # A-3, хорошее фото
    _run(f)
    assert _reading(f, "A-3") == "150"
    assert [(r["original_filename"], r["outcome"]) for r in _log(f)] == [
        ("y1.jpg", "DIGITS_ERROR"), ("y2.jpg", "PLUS"), ("y1.jpg", "REPEAT")]
    (c,) = _closed(f, "y1.jpg")
    assert c["account_id"] == "A-3" and "y2.jpg" in c["notes"] and c["source_folder"] == "Аюб"
    assert _files(f) == ["Аюб/plus/A-3.jpg", "Аюб/repeat/A-3.jpg"]   # оператору ничего
    _run(f)                                                        # и на следующий день
    assert len(_log(f)) == 3 and _reading(f, "A-3") == "150"


def test_good_photo_next_day_closes_waiting(tmp_path, models):
    f = _month(tmp_path)
    _photo(f.photos / "Аюб" / "y1.jpg", 3)
    _run(f)
    assert _files(f) == ["Аюб/question/digits_error/A-3.jpg"]
    _photo(f.photos / "Сулиман С" / "y2.jpg", 4)                  # другой контролёр
    _run(f)
    assert _reading(f, "A-3") == "150" and len(_closed(f, "y1.jpg")) == 1
    assert _files(f) == ["Аюб/repeat/A-3.jpg", "Сулиман С/plus/A-3.jpg"]


def test_suspicious_and_two_failed_photos_closed(tmp_path, models):
    f = _month(tmp_path)
    for name, n in (("y1.jpg", 3), ("y2.jpg", 8), ("y3.jpg", 6)):
        _photo(f.photos / "Аюб" / name, n)
    _run(f)
    assert _files(f) == ["Аюб/question/digits_error/A-3.jpg", "Аюб/question/digits_error/A-3_2.jpg",
                         "Аюб/question/suspicious/A-3.jpg"]
    _photo(f.photos / "Аюб" / "y4.jpg", 4)
    _run(f)
    assert [len(_closed(f, n)) for n in ("y1.jpg", "y2.jpg", "y3.jpg")] == [1, 1, 1]
    assert _files(f) == ["Аюб/plus/A-3.jpg", "Аюб/repeat/A-3.jpg", "Аюб/repeat/A-3_2.jpg",
                         "Аюб/repeat/A-3_3.jpg"]


def test_failed_photo_after_good_is_repeat(tmp_path, models):
    f = _month(tmp_path)
    _photo(f.photos / "Аюб" / "a.jpg", 4)                         # сначала хорошее
    _photo(f.photos / "Аюб" / "b.jpg", 3)
    _run(f)
    assert [(r["original_filename"], r["outcome"]) for r in _log(f)] == [
        ("a.jpg", "PLUS"), ("b.jpg", "REPEAT")]
    assert _files(f) == ["Аюб/plus/A-3.jpg", "Аюб/repeat/A-3.jpg"]


def test_closing_skips_photo_already_handled_by_operator(tmp_path, models):
    f = _month(tmp_path)
    _photo(f.photos / "Аюб" / "y1.jpg", 3)
    _run(f)
    y1 = _log(f)[0]
    with MonthDB(f.db) as db, db.transaction():                   # оператор: «Нечитаемо»
        db.append_log_rows([{c: "" for c in LOG_COLUMNS} | {
            "original_filename": "A-3.jpg", "outcome": "UNREADABLE", "source": "manual",
            "photo_hash": y1["photo_hash"], "source_folder": "Аюб"}])
    _photo(f.photos / "Аюб" / "y2.jpg", 4)
    _run(f)
    assert _closed(f, "y1.jpg") == []


# ─── Готово этапа 3: кусками за несколько «дней» — как всё разом ──────────────

MONTH_PHOTOS = [  # (контролёр, имя, номер в PHOTOS)
    ("Аюб", "IMG-20261001-WA0001.jpg", 3), ("Аюб", "IMG-20261001-WA0002.jpg", 5),
    ("Сулиман С", "IMG-20261002-WA0001.jpg", 1), ("Аюб", "IMG-20261003-WA0001.jpg", 4),
    ("Сулиман С", "IMG-20261003-WA0002.jpg", 7), ("Аюб", "IMG-20261004-WA0001.jpg", 2),
    ("Сулиман С", "IMG-20261005-WA0001.jpg", 6), ("Аюб", "IMG-20261005-WA0002.jpg", 8),
]


def _state(f):
    """Что получилось: показания и что ждёт оператора (без времени и порядка)."""
    with MonthDB(f.db) as db:
        readings = {a: r.value for a, r in db.readings().items()}
    waiting = sorted(p for p in _files(f) if "/question/" in p)
    return readings, waiting


def test_month_in_daily_batches_same_as_all_at_once(tmp_path, models):
    whole = _month(tmp_path / "разом")
    for ctrl, name, n in MONTH_PHOTOS:
        _photo(whole.photos / ctrl / name, n)
    _run(whole)

    daily = _month(tmp_path / "по дням")
    for day in range(0, len(MONTH_PHOTOS), 2):                     # по два фото в день
        for ctrl, name, n in MONTH_PHOTOS[day:day + 2]:
            _photo(daily.photos / ctrl / name, n)
        _run(daily)
        _run(daily)                                                # и лишний запуск в тот же день

    assert _state(daily) == _state(whole)
    assert _state(whole) == ({"A-1": "1200", "A-2": "4000", "A-3": "150"},
                             ["Аюб/question/serial_not_found/IMG-20261001-WA0002.jpg"])
    assert len(_log(daily)) == len(_log(whole))                    # по строке на фото, без повторов


def test_photo_closed_earlier_in_run_counts_as_done(tmp_path, models):
    f = _month(tmp_path)
    _photo(f.photos / "Сулиман С" / "y1.jpg", 3)                  # ждёт оператора
    _run(f)
    _photo(f.photos / "Аюб" / "y2.jpg", 4)                        # Аюб читается раньше Сулимана
    _run(f)
    assert len(_closed(f, "y1.jpg")) == 1
    report = _report(f, "Сулиман С")
    assert "Уже разобраны раньше (пропущены): 1" in report and "Ждут оператора" not in report


def test_copy_rows_of_old_logs_ignored():
    # до этапа 3 копия файла в прогоне получала строку REPEAT с этой пометкой
    copy = r("REPEAT", notes="duplicate file in current run: b.jpg")
    assert _decide([r("DIGITS_ERROR"), copy]) == RunSelection.SKIP_WAITING
    assert _decide([r("SERIAL_NOT_FOUND", serial_id="99999"), copy], in_table={"99999"}) == RunSelection.READ


def test_serial_lookup_like_reader():
    import pandas as pd
    from reader import _serial_lookup
    cfg = reader.PipelineConfig()
    in_table = _serial_lookup(pd.DataFrame({cfg.col_serial: ["0099999", " 123 "]}), cfg)
    assert in_table("99999") and in_table("0099999") and in_table("123") and not in_table("9999")


def test_log_rows_of_photo_match_like_policy(tmp_path):
    f = _month(tmp_path)
    rows = [{"original_filename": "a.jpg", "photo_hash": "h", "outcome": "NO_METER"},
            {"original_filename": "a.jpg", "photo_hash": "", "outcome": "UNREADABLE"},   # без отпечатка
            {"original_filename": "b.jpg", "photo_hash": "", "outcome": "NO_METER"},
            {"original_filename": "a.jpg", "photo_hash": "h2", "outcome": "PLUS"}]       # другое фото
    with MonthDB(f.db) as db:
        with db.transaction():
            db.append_log_rows([{c: "" for c in LOG_COLUMNS} | x for x in rows])
        assert [x["outcome"] for x in db.log_rows_of_photo("h", "a.jpg")] == ["NO_METER", "UNREADABLE"]
        assert [x["outcome"] for x in db.log_rows_of_photo("", "a.jpg")] == ["NO_METER", "UNREADABLE", "PLUS"]


def test_waiting_photo_with_copy_row_is_closed(tmp_path, models):
    f = _month(tmp_path)
    _photo(f.photos / "Аюб" / "y1.jpg", 3)
    _run(f)
    y1 = _log(f)[0]
    with MonthDB(f.db) as db, db.transaction():                   # строка-копия из старого лога
        db.append_log_rows([{c: "" for c in LOG_COLUMNS} | {
            "original_filename": "y1 (1).jpg", "outcome": "REPEAT", "account_id": "A-3", "source": "auto",
            "notes": "duplicate file in current run: y1 (1).jpg", "photo_hash": y1["photo_hash"],
            "source_folder": "Аюб"}])
    _photo(f.photos / "Аюб" / "y2.jpg", 4)
    _run(f)
    assert len(_closed(f, "y1.jpg")) == 1


def test_photo_read_again_for_other_account_not_closed(tmp_path, models, monkeypatch):
    f = _month(tmp_path)
    _photo(f.photos / "Аюб" / "y1.jpg", 3)                        # A-3, цифры не прочитаны
    _run(f)
    monkeypatch.setitem(PHOTOS, 3, ("22222", None))               # перечитали: это A-2, цифры снова нет
    _run(f, reread_errors=True)
    _photo(f.photos / "Аюб" / "y2.jpg", 4)                        # у A-3 появилось показание
    _run(f)
    assert _closed(f, "y1.jpg") == []                             # фото — уже про A-2


# ─── Файлы, которые программа не берёт, — в отчёте (решение 2026-10-03) ──────

def test_files_not_taken_listed_in_reports(tmp_path, models):
    f = _month(tmp_path)
    _photo(f.photos / "Аюб" / "a1.jpg", 1)
    _photo(f.photos / "Аюб" / "a2.JPG", 2)                        # большие буквы — тоже фото
    for name in ("b.webp", "c.HEIC", "видео.mp4", "Thumbs.db", "desktop.ini", "~$черновик.docx"):
        (f.photos / "Аюб" / name).write_bytes(b"x")
    _photo(f.photos / "Аюб" / "пачка2" / "d.jpg", 2)              # папка внутри — не читается
    (f.photos / "Сулиман С").mkdir()
    (f.photos / "Сулиман С" / "e.webp").write_bytes(b"x")         # фото нет совсем
    (f.photos / "список.xlsx").write_bytes(b"x")                  # не фото в корне — не ошибка
    _run(f)
    assert [r["original_filename"] for r in _log(f)] == ["a1.jpg", "a2.JPG"]
    assert ("⚠ Не взяты (не фото .jpg/.jpeg/.png или папка внутри): 4 — "
            "b.webp, c.HEIC, видео.mp4, пачка2\\") in _report(f)
    overall = (f.results / "report.txt").read_text(encoding="utf-8")
    assert "Не взяты в «Аюб» (не фото .jpg/.jpeg/.png или папка внутри): 4" in overall
    assert "Не взяты в «Сулиман С» (не фото .jpg/.jpeg/.png или папка внутри): 1 — e.webp" in overall
    assert "Не взяты в корне фото\\ (не фото .jpg/.jpeg/.png или папка внутри): 1 — список.xlsx" in overall
    journal = sorted((f.results / "run_logs").glob("run_*.txt"))[-1].read_text(encoding="utf-8")
    import re
    assert re.search(r"WARNING\s+⚠ Не взяты в «Сулиман С» .*: 1 — e\.webp", journal)   # сразу, в начале
    assert "Thumbs.db" not in journal


def test_no_not_taken_line_when_all_photos(tmp_path, models):
    f = _month(tmp_path)
    _photo(f.photos / "Аюб" / "a1.jpg", 1)
    _run(f)
    assert "Не взяты" not in _report(f) and "Не взяты" not in (f.results / "report.txt").read_text(encoding="utf-8")


def test_not_taken_list_is_shortened():
    from reader import _not_taken_line
    line = _not_taken_line([f"{i}.webp" for i in range(12)])
    assert line.endswith("8.webp, 9.webp …") and ": 12 — 0.webp" in line


# ─── 6а: итог месяца (python gmr.py month <папка>) ───────────────────────────

def test_month_summary(tmp_path, models):
    f = make_month(tmp_path, TABLE + [("44444", "A-4", "10", "15")])   # у A-4 показание было в таблице
    for name, n in (("a1.jpg", 1), ("a3.jpg", 3), ("a5.jpg", 5), ("a6.jpg", 6)):
        _photo(f.photos / "Аюб" / name, n)
    _run(f)
    with MonthDB(f.db) as db, db.transaction():                   # оператор: A-2 и исправленный номер
        db.put_reading(month.Reading("A-2", "4100", "", "manual", "p.jpg", "t", "Оператор"))
        db.add_change("Оператор", month.SERIAL_FIX_ACTION, "A-2", "Номер счетчика", "22222", "22229")
    text = month.month_summary(str(f.root))
    assert "абонентов в таблице: 4" in text
    assert "с показанием: 3 — программа 1, оператор 1, было в таблице 1" in text
    assert f"без показания: 1 — список: {f.root / 'без_показаний.csv'}" in text
    assert "ждут оператора: 3 фото — Ошибка цифр 1, Подозрительно 1, Серийник не найден 1" in text
    assert "номера счётчиков исправлены оператором: 1 (для компании — db_serial_fix.csv)" in text
    lines = (f.root / "без_показаний.csv").read_text(encoding="utf-8-sig").splitlines()
    assert lines[0].split(";")[:2] == ["Номер счетчика", "Лицевой счет"]
    assert lines[1:] == ["33333;A-3;100;"]


def test_month_summary_list_locked(tmp_path, models, monkeypatch):
    f = _month(tmp_path)
    real_open = open

    def locked(path, *a, **k):
        if str(path).endswith("без_показаний.csv"):
            raise PermissionError(13, "файл занят", str(path))
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", locked)
    text = month.month_summary(str(f.root))
    assert "без показания: 3" in text and "закройте" in text


def test_not_in_db_renamed_file_removed_on_reread(tmp_path, models):
    # «Нет в базе» при занятом имени: файл a5_2.jpg (имя — в final_filename)
    f = _day1(tmp_path)
    a5 = next(r for r in _log(f) if r["original_filename"] == "a5.jpg")
    nid = f.results / "Аюб" / "not_in_db"
    nid.mkdir()
    _photo(nid / "a5.jpg", 2)                                     # чужое фото с тем же именем
    (f.results / "Аюб" / "question" / "serial_not_found" / "a5.jpg").rename(nid / "a5_2.jpg")
    with MonthDB(f.db) as db, db.transaction():
        db.append_log_rows([{c: "" for c in LOG_COLUMNS} | {
            "original_filename": "a5.jpg", "final_filename": "a5_2.jpg", "serial_id": "99999",
            "outcome": "NOT_IN_DB", "source": "manual", "photo_hash": a5["photo_hash"],
            "source_folder": "Аюб"}])
    month.load_table(str(f.root), str(_with_99999(tmp_path)))
    _run(f)
    assert not (nid / "a5_2.jpg").exists() and (nid / "a5.jpg").exists()


# ─── Этап 4: прогресс для окна и «Остановить» ────────────────────────────────

def _journal(f):
    return sorted((f.results / "run_logs").glob("run_*.txt"))[-1].read_text(encoding="utf-8")


def test_progress_lines_count_only_new_photos(tmp_path, models):
    f = _day1(tmp_path)                                           # a1, a3, a5 уже разобраны
    _photo(f.photos / "Аюб" / "b2.jpg", 2)
    _photo(f.photos / "Сулиман С" / "c4.jpg", 4)
    (f.photos / "Сулиман С" / "c4 (1).jpg").write_bytes((f.photos / "Сулиман С" / "c4.jpg").read_bytes())
    _run(f)
    j = _journal(f)
    assert "Новых фото к чтению: 2" in j                          # копия и старые — не в счёт
    assert "Фото 1 из 2: b2.jpg" in j and "Фото 2 из 2: c4 (1).jpg" in j   # «c4 (1)» раньше «c4»
    assert "Фото 3 из" not in j


def test_stop_file_stops_between_photos(tmp_path, models, monkeypatch):
    f = _month(tmp_path)
    for name, n in (("a.jpg", 1), ("b.jpg", 2), ("c.jpg", 4)):
        _photo(f.photos / "Аюб" / name, n)
    real = reader.process_photo

    def press_stop_after_first(*a, **k):                          # оператор нажал «Остановить»
        res = real(*a, **k)
        f.stop_file.write_text("", encoding="utf-8")
        return res

    monkeypatch.setattr(reader, "process_photo", press_stop_after_first)
    _run(f)
    assert [r["original_filename"] for r in _log(f)] == ["a.jpg"]   # a.jpg записан целиком
    assert "Остановлено оператором: прочитано 1 из 3" in _journal(f)
    assert not f.stop_file.exists() and f.export_xlsx.is_file()   # выгрузка сделана
    monkeypatch.setattr(reader, "process_photo", real)
    _run(f)                                                        # следующий запуск дочитывает
    assert [r["original_filename"] for r in _log(f)] == ["a.jpg", "b.jpg", "c.jpg"]


def test_stop_stops_whole_run_not_only_one_controller(tmp_path, models, monkeypatch):
    f = _month(tmp_path)
    _photo(f.photos / "Аюб" / "a.jpg", 1)
    _photo(f.photos / "Аюб" / "b.jpg", 2)
    _photo(f.photos / "Сулиман С" / "c.jpg", 4)
    real = reader.process_photo
    monkeypatch.setattr(reader, "process_photo",
                        lambda *a, **k: (real(*a, **k), f.stop_file.write_text("", encoding="utf-8"))[0])
    _run(f)
    j = _journal(f)
    assert j.count("Остановлено оператором") == 1 and "Папка контролёра: Сулиман С" not in j
    assert [r["original_filename"] for r in _log(f)] == ["a.jpg"]


def test_two_runs_in_one_second_keep_own_journals(tmp_path, models, monkeypatch):
    class SameSecond(reader.datetime):
        @classmethod
        def now(cls, tz=None):
            return reader.datetime(2026, 10, 3, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(reader, "datetime", SameSecond)
    f = _month(tmp_path)
    _photo(f.photos / "Аюб" / "a.jpg", 1)
    _run(f)
    _run(f)
    logs = f.results / "run_logs"
    assert "Новых фото к чтению: 1" in (logs / "run_2026-10-03_120000.txt").read_text(encoding="utf-8")
    assert "Новых фото к чтению: 0" in (logs / "run_2026-10-03_120000_2.txt").read_text(encoding="utf-8")


def test_old_stop_file_ignored(tmp_path, models):
    f = _month(tmp_path)
    _photo(f.photos / "Аюб" / "a.jpg", 1)
    f.stop_file.write_text("", encoding="utf-8")                  # осталась от прошлого раза
    _run(f)
    assert len(_log(f)) == 1 and not f.stop_file.exists()


def test_progress_lines_parse_in_window(tmp_path, models):
    # строки прогона, по которым окно оператора двигает полосу (src/gmr/ui/run_dialog.py)
    from src.gmr.ui.run_dialog import parse_progress
    f = _day1(tmp_path)
    got = [p for p in map(parse_progress, _journal(f).splitlines()) if p]
    assert got == [("total", 3), ("models",), ("photo", 1, 3, "a1.jpg"), ("photo", 2, 3, "a3.jpg"),
                   ("photo", 3, 3, "a5.jpg")]
