"""
Копии таблицы и лога, журнал прогона, консоль без падений на ⚠️ (2026-10-01,
этап 2.1 — результат прогона не меняется).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

import reader
from src.gmr.storage import backup_dir_for, backup_files, load_table
from src.gmr.storage.backup import remove_old
from tests import test_shadow_run_pipeline as srp
from tests.test_shadow_run_pipeline import _setup_workspace, fake_models  # noqa: F401 (фикстура)

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.usefixtures("fake_models")   # модели — фейки


def _cfg(root, inp, table):
    return reader.PipelineConfig(input_dir=str(inp), output_base_dir=str(root / "out"),
                                 table_path=str(table))


# ─── Копии ───────────────────────────────────────────────────────────────────

def test_backup_files_copies_existing_and_skips_missing(tmp_path):
    (tmp_path / "t.csv").write_text("a\n", encoding="utf-8")
    dest = backup_files([str(tmp_path / "t.csv"), str(tmp_path / "нет_лога.csv")], tmp_path / "b")
    assert sorted(p.name for p in dest.iterdir()) == ["t.csv"]
    assert backup_files([str(tmp_path / "нет.csv")], tmp_path / "b2") is None
    assert not (tmp_path / "b2").exists()


def test_two_backups_in_one_second_do_not_overwrite(tmp_path, monkeypatch):
    from src.gmr.storage import backup as backup_mod

    class FrozenTime:
        @staticmethod
        def now():
            import datetime as dt
            return dt.datetime(2026, 10, 1, 12, 0, 0)

    monkeypatch.setattr(backup_mod, "datetime", FrozenTime)
    t = tmp_path / "t.csv"
    t.write_text("старое\n", encoding="utf-8")
    first = backup_files([str(t)], tmp_path / "b")
    t.write_text("новое\n", encoding="utf-8")
    second = backup_files([str(t)], tmp_path / "b")
    assert (first.name, second.name) == ("2026-10-01_120000", "2026-10-01_120000_2")
    assert (first / "t.csv").read_text(encoding="utf-8") == "старое\n"


def test_backup_files_keeps_last_30(tmp_path):
    for day in range(1, 36):
        (tmp_path / "b" / f"2026-09-{day:02d}_120000").mkdir(parents=True)
    (tmp_path / "t.csv").write_text("a\n", encoding="utf-8")
    dest = backup_files([str(tmp_path / "t.csv")], tmp_path / "b")
    names = sorted(p.name for p in (tmp_path / "b").iterdir())
    assert len(names) == 30 and names[-1] == dest.name and names[0] == "2026-09-07_120000"


def test_remove_old_keeps_newest(tmp_path):
    for day in range(1, 36):
        (tmp_path / f"2026-10-{day:02d}_120000").mkdir()
    remove_old(tmp_path, 30)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert len(names) == 30 and names[0] == "2026-10-06_120000" and names[-1] == "2026-10-35_120000"


def test_backup_dir_next_to_table():
    assert backup_dir_for(os.path.join("D:", "месяц", "Октябрь.xlsx")) == \
        Path("D:", "месяц", "gmr_backups", "Октябрь")


def test_run_backs_up_table_and_log_before_changing_them(tmp_path):
    inp, table = _setup_workspace(tmp_path)
    reader.run_pipeline(_cfg(tmp_path, inp, table))          # 1-й прогон: лога ещё нет
    reader.run_pipeline(_cfg(tmp_path, inp, table))          # 2-й: копия таблицы и лога
    copies = sorted((tmp_path / "gmr_backups" / "table").iterdir())
    assert copies, "копия не сделана"
    first = copies[0]
    assert sorted(p.name for p in first.iterdir()) == ["table.csv"]   # до 1-го прогона
    before = load_table(str(first / "table.csv"))
    assert before.loc[before["Лицевой счет"] == "A-1", "Текущие показания"].isna().all()
    after = load_table(str(table))
    assert after.loc[after["Лицевой счет"] == "A-1", "Текущие показания"].iloc[0] == "1200"
    assert sorted(p.name for p in copies[-1].iterdir()) == ["table.csv", "table_log.csv"]


# ─── Журнал прогона ──────────────────────────────────────────────────────────

def _journals(root):
    return sorted((root / "out" / "run_logs").glob("run_*.txt"))


def test_run_writes_journal_file(tmp_path):
    inp, table = _setup_workspace(tmp_path)
    old = tmp_path / "out" / "run_logs"
    old.mkdir(parents=True)
    for i in range(35):                                        # старые журналы
        (old / f"run_2026-09-{i:02d}_000000.txt").write_text("", encoding="utf-8")
    reader.run_pipeline(_cfg(tmp_path, inp, table))
    journals = _journals(tmp_path)
    assert len(journals) == 30                                 # хранятся последние 30
    text = journals[-1].read_text(encoding="utf-8")
    assert "p0.jpg" in text and "PLUS" in text and "Копия таблицы" in text


def test_interrupted_run_leaves_reason_in_journal(tmp_path, monkeypatch):
    inp, table = _setup_workspace(tmp_path)

    def ctrl_c(self, image_input):
        raise KeyboardInterrupt

    monkeypatch.setattr(srp._SerialOCR, "predict_with_details", ctrl_c)
    with pytest.raises(KeyboardInterrupt):
        reader.run_pipeline(_cfg(tmp_path, inp, table))
    text = _journals(tmp_path)[-1].read_text(encoding="utf-8")
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
