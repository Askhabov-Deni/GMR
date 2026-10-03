"""
Действия окна оператора с базой месяца (src/gmr/application/operator.py,
этап 2.3) — без окна.
"""
import csv
import datetime as dt
from types import SimpleNamespace

import pytest

from src.gmr.application import month
from src.gmr.application.operator import CORRECTED_NOTE, OperatorSession
from src.gmr.storage import LOG_COLUMNS
from src.gmr.storage.month import MonthDB, Reading
from tests._month import changes, log_rows, make_month, readings, serial_of

TODAY = dt.date.today().strftime("%d %m %Y")
TABLE = [("1284567", "A1", "1000", ""), ("1234567", "A2", "5000", ""), ("7777777", "A3", "10", "20")]


def _row(**kw):
    return {c: "" for c in LOG_COLUMNS} | {"source": "manual", "processed_by": "Оператор"} | kw


@pytest.fixture
def session(tmp_path):
    s = OperatorSession(str(make_month(tmp_path, TABLE).root))
    yield s
    s.close()


def test_not_a_month(tmp_path):
    with pytest.raises(month.NotAMonth, match="Месяц не создан"):
        OperatorSession(str(tmp_path / "нет"))


def test_table_rows_have_readings(session):
    rows = session.table_rows()
    assert session.table_columns() == ["Номер счетчика", "Лицевой счет", "Последние показания",
                                       "Текущие показания"]
    assert all(list(r) == session.table_columns() for r in rows)
    assert [r["Лицевой счет"] for r in rows] == ["A1", "A2", "A3"]
    assert [r["Текущие показания"] for r in rows] == ["", "", "20"]


def test_accept_writes_reading_change_and_log_together(session):
    session.accept(_row(original_filename="p1.jpg", account_id="A1", reading="1200", outcome="PLUS"),
                   "IMG-20260915-WA0001.jpg")
    r = session.reading("A1")
    assert (r.value, r.date, r.source, r.updated_by, r.photo) == ("1200", "15 09 2026", "manual", "Оператор", "p1.jpg")
    assert [x["outcome"] for x in log_rows(session.folder)] == ["PLUS"]
    ch = changes(session.folder)[-1]
    assert (ch["who"], ch["action"], ch["account"], ch["old"], ch["new"]) == (
        "Оператор", "показание записано", "A1", "", "1200")


def test_accept_failure_writes_nothing(session, monkeypatch):
    def broken(rows):
        raise OSError("диск")
    monkeypatch.setattr(session.db, "append_log_rows", broken)
    with pytest.raises(OSError):
        session.accept(_row(original_filename="p1.jpg", account_id="A1", reading="1200"), "p1.jpg")
    assert session.reading("A1") is None                     # показание не записалось без строки лога


def test_add_row_has_no_reading(session):
    session.add_row(_row(original_filename="p3.jpg", outcome="UNREADABLE"))
    assert [r["outcome"] for r in log_rows(session.folder)] == ["UNREADABLE"]
    assert readings(session.folder) == {"A3": "20"}


def test_fix_serial_and_accept(session):
    old = session.fix_serial_and_accept(
        _row(original_filename="w.jpg", account_id="A1", reading="1050", outcome="PLUS",
             serial_id="1284999"), "1284999", "w.jpg")
    assert old == "1284567"
    assert serial_of(session.folder, "A1") == "1284999"
    assert session.reading("A1").value == "1050" and session.reading("A1").date == TODAY
    actions = [(c["action"], c["field"], c["old"], c["new"]) for c in changes(session.folder)]
    assert ("номер исправлен оператором", "Номер счетчика", "1284567", "1284999") in actions
    assert ("показание записано", "Текущие показания", "", "1050") in actions


def test_operator_serial_fix_survives_table_update(session, tmp_path):
    session.fix_serial_and_accept(_row(original_filename="w.jpg", account_id="A1", reading="1050",
                                       outcome="PLUS"), "1284999", "w.jpg")
    upd = tmp_path / "обновлённая.csv"                    # компания номер ещё не исправила
    with open(upd, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Номер счетчика", "Лицевой счет", "Последние показания", "Текущие показания"])
        w.writerows(TABLE)
    rep = month.load_table(str(session.folder.root), str(upd))
    assert rep.serial_fixes_kept == [("A1", "1284567", "1284999")]
    assert serial_of(session.folder, "A1") == "1284999"


def _auto(session, **kw):
    """Строка reader.py PLUS по A1 (1300) и её показание — как после прогона."""
    with session.db.transaction():
        session.db.append_log_rows([_row(source="auto", processed_by="auto", original_filename="w.jpg",
                                         final_filename="A1.jpg", serial_id="1284567", account_id="A1",
                                         reading="1300", last_reading="1000.0", delta="300",
                                         outcome="PLUS", model_reading_str="01300", **kw)])
        session.db.put_reading(Reading("A1", "1300", "15 09 2026", "auto", "w.jpg", "t", "auto"))
    return session.log_rows()[-1]


def _c(serial, reading, old, new, last, delta, outcome):
    return SimpleNamespace(serial=serial, reading=reading, old_account=old, new_account=new,
                           last_reading=last, delta=delta, outcome=outcome, account_changed=old != new)


def test_mark_verified_only_this_row(session):
    r = _auto(session)
    session.add_row(_row(original_filename="x.jpg", final_filename="A1.jpg", outcome="REPEAT"))
    session.mark_verified(r, "Оператор", "01.10.2026 12:00")
    rows = log_rows(session.folder)
    assert [x["verified_by"] for x in rows] == ["Оператор", ""]


def test_correct_reading_same_account(session):
    r = _auto(session)
    session.apply_correction(r, _c("1284567", 950, "A1", "A1", 1000.0, -50.0, "MINUS"),
                             "Оператор", "01.10.2026 12:00")
    # дата показания остаётся прежней (не «сегодня»)
    assert session.reading("A1").value == "950" and session.reading("A1").date == "15 09 2026"
    saved = log_rows(session.folder)[0]
    assert (saved["reading"], saved["outcome"], saved["verified_by"]) == ("950", "MINUS", "Оператор")
    assert CORRECTED_NOTE in saved["notes"] and saved["model_reading_str"] == "01300"


def test_correct_moves_reading_to_other_account(session):
    r = _auto(session)
    session.apply_correction(r, _c("1234567", 5300, "A1", "A2", 5000.0, 300.0, "PLUS"),
                             "Оператор", "01.10.2026 12:00")
    assert session.reading("A1") is None and session.reading("A2").value == "5300"
    saved = log_rows(session.folder)[0]
    assert (saved["account_id"], saved["serial_id"], saved["final_filename"]) == ("A2", "1234567", "A2.jpg")
    assert "A1 → A2" in saved["notes"]
    assert any(c["action"] == "показание стёрто при проверке" and c["account"] == "A1"
               for c in changes(session.folder))


def test_correct_keeps_old_account_value_if_changed_by_someone_else(session):
    r = _auto(session)
    with session.db.transaction():
        session.db.put_reading(Reading("A1", "1111", "", "manual", "", "t", "кто-то"))
    session.apply_correction(r, _c("1234567", 5300, "A1", "A2", 5000.0, 300.0, "PLUS"), "Оп", "t")
    assert session.reading("A1").value == "1111"


def test_rename_photo_in_log(session):
    r = _auto(session)
    session.rename_photo_in_log(r, "A1_2.jpg")
    assert log_rows(session.folder)[0]["final_filename"] == "A1_2.jpg"


# ─── reader.py пишет в то же время: окно его записи не затирает ──────────────
# (в режиме месяца исправлены известные ошибки «Верно стирает строки reader.py»
#  и «Принять стирает показания reader.py»)

def test_operator_actions_keep_rows_written_by_reader_meanwhile(session):
    r = _auto(session)
    rows_seen_by_window = session.log_rows()
    with MonthDB(session.folder.db) as reader_db, reader_db.transaction():     # reader.py
        reader_db.append_log_rows([_row(source="auto", original_filename="new.jpg", account_id="A2",
                                        reading="5100", outcome="PLUS")])
        reader_db.put_reading(Reading("A2", "5100", "", "auto", "new.jpg", "t", "auto"))
    session.mark_verified(rows_seen_by_window[0], "Оператор", "t")
    session.accept(_row(original_filename="p.jpg", account_id="A3", reading="30", outcome="PLUS"), "p.jpg")
    assert "new.jpg" in [x["original_filename"] for x in log_rows(session.folder)]
    assert readings(session.folder)["A2"] == "5100"
    assert r["verified_by"] == ""                                # другая строка не тронута


def test_export(session):
    session.accept(_row(original_filename="p1.jpg", account_id="A1", reading="1200", outcome="PLUS"), "p1.jpg")
    rep = session.export()
    assert rep.with_reading == 2 and session.folder.export_xlsx.is_file()


def test_backup(session):
    assert (session.backup() / "gmr.sqlite").is_file()


# ─── Этап 3, пункт 3: у счёта появилось показание — его ждущие фото закрываются

def _waiting(session, name, account="A1", outcome="DIGITS_ERROR", h="h-w"):
    """Авто-строка «цифры не прочитаны» и её файл в question/ контролёра Аюб."""
    from tests._month import img
    folder = {"DIGITS_ERROR": "digits_error", "SUSPICIOUS": "suspicious"}[outcome]
    path = img(session.folder.results / "Аюб" / "question" / folder / f"{account}.jpg")
    with session.db.transaction():
        session.db.append_log_rows([_row(source="auto", processed_by="auto", original_filename=name,
                                         final_filename=f"{account}.jpg", serial_id="1284567",
                                         account_id=account, outcome=outcome, photo_hash=h,
                                         source_folder="Аюб")])
    return path


def test_accept_closes_waiting_photos_of_account(session):
    path = _waiting(session, "w.jpg")
    session.accept(_row(original_filename="p.jpg", account_id="A1", reading="1200", outcome="PLUS"), "p.jpg")
    assert not path.exists()
    assert (session.folder.results / "Аюб" / "repeat" / "A1.jpg").is_file()
    closed = log_rows(session.folder)[-1]
    assert (closed["original_filename"], closed["outcome"], closed["source"], closed["photo_hash"]) == (
        "w.jpg", "REPEAT", "auto", "h-w")
    assert "у счёта появилось показание" in closed["notes"] and "p.jpg" in closed["notes"]
    assert closed["processed_by"] == "Оператор"


def test_accept_keeps_photo_operator_is_handling(session):
    path = _waiting(session, "w.jpg")
    # оператор разбирает этот самый файл (отпечаток не нашёлся — строка без него)
    session.accept(_row(original_filename="A1.jpg", account_id="A1", reading="1200", outcome="PLUS"),
                   "A1.jpg", keep=str(path))
    assert path.exists() and log_rows(session.folder)[-1]["source"] == "manual"


def test_other_account_and_other_outcomes_untouched(session):
    a2 = _waiting(session, "w2.jpg", account="A2", h="h2")
    with session.db.transaction():
        session.db.append_log_rows([_row(source="auto", original_filename="n.jpg", account_id="A1",
                                         outcome="SERIAL_AMBIGUOUS", photo_hash="h3")])
    session.accept(_row(original_filename="p.jpg", account_id="A1", reading="1200", outcome="PLUS"), "p.jpg")
    assert a2.exists() and log_rows(session.folder)[-1]["original_filename"] == "p.jpg"


def test_fix_serial_and_correction_close_waiting(session):
    s1 = _waiting(session, "s.jpg", outcome="SUSPICIOUS", h="h-s")
    session.fix_serial_and_accept(_row(original_filename="w.jpg", account_id="A1", reading="1050",
                                       outcome="PLUS"), "1284999", "w.jpg")
    assert not s1.exists()
    a2 = _waiting(session, "w2.jpg", account="A2", h="h2")
    r = _auto(session)                                      # reader: A1 1300 по w.jpg
    session.apply_correction(r, _c("1234567", 5300, "A1", "A2", 5000.0, 300.0, "PLUS"), "Оп", "t")
    assert not a2.exists()                                  # показание перенесено к A2


def test_old_rows_without_hash_matched_by_name(session):
    # строки старых логов без отпечатка: фото узнаётся по имени файла
    path = _waiting(session, "w.jpg", h="")
    with session.db.transaction():                            # оператор разобрал это фото
        session.db.append_log_rows([_row(original_filename="w.jpg", outcome="UNREADABLE")])
    session.accept(_row(original_filename="p.jpg", account_id="A1", reading="1200", outcome="PLUS"), "p.jpg")
    assert path.exists() and log_rows(session.folder)[-1]["original_filename"] == "p.jpg"
