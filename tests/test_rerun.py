"""
Повторный прогон по той же папке месяца (docs/contract_reader_program2.md,
раздел 5, правило 1): разобранное фото (PLUS/MINUS/REPEAT или решение
оператора) не возвращается в question/ и не читается моделями заново.
С этапа 3 (решения владельца 2026-10-03) прогон читает только новые фото:
разобранные и ждущие оператора пропускаются молча, фото с авто-ошибками
читаются заново только по `process --reread` (подробно —
tests/test_month_cycle.py). Модели — фейки (tests/_pipeline.py).
"""
from src.gmr.storage import LOG_COLUMNS
from src.gmr.storage.month import MonthDB
from tests import _pipeline as pl
from tests._pipeline import fake_models  # noqa: F401 (фикстура)


def test_first_run_outcomes(tmp_path, fake_models):  # noqa: F811
    # самопроверка фейков: прогон дал именно те исходы, что задуманы
    f = pl.setup_month(tmp_path)
    pl.run(f)
    assert [r["outcome"] for r in pl.log(f)] == pl.OUTCOMES
    assert (pl.reading(f, "A-1"), pl.reading(f, "A-2")) == ("1200", "4000")
    assert pl.output_tree(f) == ["minus/A-2.jpg", "plus/A-1.jpg", "question/no_meter/p4.jpg",
                                 "question/serial_not_found/p2.jpg", "repeat/A-1.jpg"]


def test_rerun_reads_nothing_again(tmp_path, fake_models):  # noqa: F811
    f = pl.setup_month(tmp_path)
    pl.run(f)
    rows, calls, tree = len(pl.log(f)), pl.MeterDetector.calls, pl.output_tree(f)
    pl.run(f)
    assert len(pl.log(f)) == rows and pl.MeterDetector.calls == calls
    assert pl.output_tree(f) == tree
    assert (pl.reading(f, "A-1"), pl.reading(f, "A-2")) == ("1200", "4000")


def test_manually_handled_photo_not_returned_to_question(tmp_path, fake_models):  # noqa: F811
    f = pl.setup_month(tmp_path)
    pl.run(f)
    p2 = pl.log_by_name(f)["p2.jpg"]
    # оператор в program2.py разобрал p2 (SERIAL_NOT_FOUND): «Нечитаемо»
    (pl.results_dir(f) / "question" / "serial_not_found" / "p2.jpg").unlink()
    with MonthDB(f.db) as db, db.transaction():
        db.append_log_rows([{c: "" for c in LOG_COLUMNS} | {
            "original_filename": "p2.jpg", "outcome": "UNREADABLE", "source": "manual",
            "processed_by": "Оператор", "photo_hash": p2["photo_hash"], "source_folder": "Аюб"}])
    rows = len(pl.log(f))
    pl.run(f, reread_errors=True)                           # даже с --reread
    assert [r["original_filename"] for r in pl.log(f)[rows:]] == ["p4.jpg"]   # только авто-ошибка
    assert not any("serial_not_found" in p for p in pl.output_tree(f))       # p2 не вернулся оператору
    assert any("no_meter" in p for p in pl.output_tree(f))


def test_improved_model_reads_old_failure_with_reread(tmp_path, fake_models, monkeypatch):  # noqa: F811
    f = pl.setup_month(tmp_path)
    pl.run(f)                                               # p4: NO_METER
    # «обучили модель получше»: теперь на p4 находится счётчик 33333
    monkeypatch.setitem(pl.PHOTOS, "p4.jpg", ("33333", "00150", True))
    pl.run(f)
    assert pl.reading(f, "A-3") == ""                       # обычный прогон p4 не перечитывает
    pl.run(f, reread_errors=True)
    last = pl.log(f)[-1]
    assert (last["original_filename"], last["outcome"], last["reading"]) == ("p4.jpg", "PLUS", "150")
    assert pl.reading(f, "A-3") == "150"
    assert "question/no_meter/p4.jpg" not in pl.output_tree(f)
