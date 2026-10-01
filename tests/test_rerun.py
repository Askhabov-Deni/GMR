"""
Повторный прогон по той же папке месяца (правило Г, решение владельца
2026-09-29; docs/contract_reader_program2.md, раздел 5, правило 1):
разобранное фото (PLUS/MINUS/REPEAT или решение оператора) не возвращается в
question/ и не читается моделями заново, фото с авто-ошибками читаются
заново. Модели — фейки (tests/_pipeline.py).

До этапа 2.3b — tests/test_shadow_run_pipeline.py (по таблице и CSV-логу,
вместе со сверкой shadow-run CSV/SQLite, которая убрана).
"""
import shutil

from src.gmr.storage import LOG_COLUMNS
from src.gmr.storage.month import MonthDB
from tests import _pipeline as pl
from tests._pipeline import fake_models  # noqa: F401 (фикстура)


def _rerun(folder):
    shutil.rmtree(folder.results)
    pl.run(folder)
    return pl.log(folder)[len(pl.PHOTOS):]


def test_first_run_outcomes(tmp_path, fake_models):  # noqa: F811
    # самопроверка фейков: прогон дал именно те исходы, что задуманы
    f = pl.setup_month(tmp_path)
    pl.run(f)
    assert [r["outcome"] for r in pl.log(f)] == pl.OUTCOMES
    assert (pl.reading(f, "A-1"), pl.reading(f, "A-2")) == ("1200", "4000")
    assert pl.output_tree(f) == ["minus/A-2.jpg", "plus/A-1.jpg", "question/no_meter/p4.jpg",
                                 "question/serial_not_found/p2.jpg", "repeat/A-1.jpg"]


def test_rerun_reads_only_failures_again(tmp_path, fake_models):  # noqa: F811
    f = pl.setup_month(tmp_path)
    pl.run(f)
    calls = pl.MeterDetector.calls
    second = _rerun(f)
    # успешные и REPEAT -> REPEAT без моделей, авто-ошибки перечитываются
    assert [r["outcome"] for r in second] == ["REPEAT", "REPEAT", "SERIAL_NOT_FOUND", "REPEAT", "NO_METER"]
    assert pl.MeterDetector.calls == calls + 2
    assert (pl.reading(f, "A-1"), pl.reading(f, "A-2")) == ("1200", "4000")


def test_variant_g_manually_handled_photo_not_returned_to_question(tmp_path, fake_models):  # noqa: F811
    f = pl.setup_month(tmp_path)
    pl.run(f)
    # оператор в program2.py пометил p2 (SERIAL_NOT_FOUND) как «нет в базе»
    with MonthDB(f.db) as db, db.transaction():
        db.append_log_rows([{c: "" for c in LOG_COLUMNS} | {
            "original_filename": "p2.jpg", "outcome": "NOT_IN_DB", "source": "manual",
            "processed_by": "Оператор"}])

    second = _rerun(f)[1:]
    by_name = {r["original_filename"]: r["outcome"] for r in second}
    assert by_name == {"p0.jpg": "REPEAT", "p1.jpg": "REPEAT", "p2.jpg": "REPEAT",
                       "p3.jpg": "REPEAT", "p4.jpg": "NO_METER"}
    tree = pl.output_tree(f)
    assert not any("serial_not_found" in p for p in tree)   # p2 не вернулся оператору
    assert any("no_meter" in p for p in tree)               # p4 не разобран — снова в question


def test_variant_g_improved_model_reads_old_failure(tmp_path, fake_models, monkeypatch):  # noqa: F811
    f = pl.setup_month(tmp_path)
    pl.run(f)                                               # p4: NO_METER
    # «обучили модель получше»: теперь на p4 находится счётчик 33333
    monkeypatch.setitem(pl.PHOTOS, "p4.jpg", ("33333", "00150", True))
    last = _rerun(f)[-1]
    assert (last["original_filename"], last["outcome"], last["reading"]) == ("p4.jpg", "PLUS", "150")
    assert pl.reading(f, "A-3") == "150"
