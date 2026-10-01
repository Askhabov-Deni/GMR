"""
Копии базы месяца, журнал прогона, консоль без падений на ⚠️ (2026-10-01,
этап 2.1 — результат прогона не меняется; с этапа 2.3b — только папка месяца).
"""
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from src.gmr.storage.backup import backup_sqlite, remove_old
from src.gmr.storage.month import MonthDB
from tests import _pipeline as pl
from tests._pipeline import fake_models  # noqa: F401 (фикстура)

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.usefixtures("fake_models")   # модели — фейки


# ─── Копии ───────────────────────────────────────────────────────────────────

def _db(path, value):
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE IF NOT EXISTS t (v TEXT)")
        c.execute("DELETE FROM t")
        c.execute("INSERT INTO t VALUES (?)", (value,))
    c.close()


def _value(path):
    with sqlite3.connect(path) as c:
        v = c.execute("SELECT v FROM t").fetchone()[0]
    c.close()
    return v


def test_backup_sqlite_copies_db_and_skips_missing(tmp_path):
    _db(tmp_path / "gmr.sqlite", "старое")
    dest = backup_sqlite(tmp_path / "gmr.sqlite", tmp_path / "b")
    assert sorted(p.name for p in dest.iterdir()) == ["gmr.sqlite"]
    assert _value(dest / "gmr.sqlite") == "старое"
    assert backup_sqlite(tmp_path / "нет.sqlite", tmp_path / "b2") is None
    assert not (tmp_path / "b2").exists()


def test_two_backups_in_one_second_do_not_overwrite(tmp_path, monkeypatch):
    from src.gmr.storage import backup as backup_mod

    class FrozenTime:
        @staticmethod
        def now():
            import datetime as dt
            return dt.datetime(2026, 10, 1, 12, 0, 0)

    monkeypatch.setattr(backup_mod, "datetime", FrozenTime)
    db = tmp_path / "gmr.sqlite"
    _db(db, "старое")
    first = backup_sqlite(db, tmp_path / "b")
    _db(db, "новое")
    second = backup_sqlite(db, tmp_path / "b")
    assert (first.name, second.name) == ("2026-10-01_120000", "2026-10-01_120000_2")
    assert _value(first / "gmr.sqlite") == "старое"


def test_backup_sqlite_keeps_last_30(tmp_path):
    for day in range(1, 36):
        (tmp_path / "b" / f"2026-09-{day:02d}_120000").mkdir(parents=True)
    _db(tmp_path / "gmr.sqlite", "a")
    dest = backup_sqlite(tmp_path / "gmr.sqlite", tmp_path / "b")
    names = sorted(p.name for p in (tmp_path / "b").iterdir())
    assert len(names) == 30 and names[-1] == dest.name and names[0] == "2026-09-07_120000"


def test_remove_old_keeps_newest(tmp_path):
    for day in range(1, 36):
        (tmp_path / f"2026-10-{day:02d}_120000").mkdir()
    remove_old(tmp_path, 30)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert len(names) == 30 and names[0] == "2026-10-06_120000" and names[-1] == "2026-10-35_120000"


def test_run_backs_up_month_db_before_changing_it(tmp_path):
    f = pl.setup_month(tmp_path)
    pl.run(f)
    copies = sorted(f.backups.iterdir())
    assert len(copies) == 1, "копия не сделана"
    with MonthDB(copies[0] / "gmr.sqlite") as before:          # копия — до прогона
        assert before.reading("A-1") is None and before.log_rows() == []
    assert pl.reading(f, "A-1") == "1200"


# ─── Журнал прогона ──────────────────────────────────────────────────────────

def _journals(f):
    return sorted((f.results / "run_logs").glob("run_*.txt"))


def test_run_writes_journal_file(tmp_path):
    f = pl.setup_month(tmp_path)
    old = f.results / "run_logs"
    old.mkdir(parents=True)
    for i in range(35):                                        # старые журналы
        (old / f"run_2026-09-{i:02d}_000000.txt").write_text("", encoding="utf-8")
    pl.run(f)
    journals = _journals(f)
    assert len(journals) == 30                                 # хранятся последние 30
    text = journals[-1].read_text(encoding="utf-8")
    assert "p0.jpg" in text and "PLUS" in text and "Копия базы месяца" in text


def test_interrupted_run_leaves_reason_in_journal(tmp_path, monkeypatch):
    f = pl.setup_month(tmp_path)

    def ctrl_c(self, image_input):
        raise KeyboardInterrupt

    monkeypatch.setattr(pl.SerialOCR, "predict_with_details", ctrl_c)
    with pytest.raises(KeyboardInterrupt):
        pl.run(f)
    text = _journals(f)[-1].read_text(encoding="utf-8")
    assert "Прогон прерван" in text and "KeyboardInterrupt" in text


# ─── Консоль ─────────────────────────────────────────────────────────────────

def _python_cp1251(code):
    env = dict(os.environ, PYTHONIOENCODING="cp1251")   # так пишет русская Windows в файл
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True)


def test_without_safe_console_emoji_crashes_cp1251():
    # проверка самой проверки: без защиты ⚠ в cp1251 роняет программу
    assert _python_cp1251("print('\\u26a0')").returncode != 0


def test_gmr_survives_emoji_in_cp1251_output():
    out = _python_cp1251("import gmr; gmr.main(['--help']); print('\\u26a0 после')")
    assert out.returncode == 0, out.stderr.decode("cp1251", "replace")[-300:]
    assert "? после".encode("cp1251") in out.stdout
