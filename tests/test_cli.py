"""
Фаза 7: gmr.py — одна точка входа. process вызывает тот же run_pipeline с теми
же настройками, что `python reader.py <папка месяца>`; остальные команды —
обёртки над tools/. С этапа 2.3b process работает только с папкой месяца.
"""
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

import gmr
import reader

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def captured(monkeypatch):
    got = {}
    monkeypatch.setattr(reader, "run_pipeline", lambda cfg: got.setdefault("cfg", cfg))
    return got


def test_process_defaults_equal_reader_main(captured):
    assert gmr.main(["process", "Октябрь"]) == 0
    cfg = captured["cfg"]
    # `python reader.py Октябрь` — те же настройки
    assert cfg == reader.PipelineConfig(month_dir="Октябрь")
    # настройки обычного прогона (до 2026-10-01 — reader.default_run_config)
    assert (cfg.move_photos, cfg.ignore_last_digits, cfg.debug_digits, cfg.draw_boxes) == (False, 2, False, False)


def test_process_reread(captured):
    gmr.main(["process", "Октябрь", "--reread"])
    assert captured["cfg"].reread_errors is True and captured["cfg"].move_photos is False


def test_process_move(captured):
    gmr.main(["process", "Октябрь", "--move"])
    cfg = captured["cfg"]
    assert cfg.month_dir == "Октябрь" and cfg.move_photos is True and cfg.reread_errors is False
    assert cfg.digit_conf_thresh == 0.6 and cfg.serial_conf_thresh == 0.6   # пороги не трогаются


@pytest.mark.parametrize("argv", [[], ["--input", "in", "--output", "out", "--table", "t.csv"],
                                  ["Октябрь", "--shadow-sqlite"]])
def test_process_old_mode_removed(captured, argv):
    # старый режим (таблица и CSV-лог) убран на этапе 2.3b: нужна папка месяца
    with pytest.raises(SystemExit):
        gmr.main(["process", *argv])
    assert "cfg" not in captured


def _reader_script(monkeypatch, *args):
    # `python reader.py …` в этом же процессе (отдельный процесс грузил бы модули ~5 с)
    monkeypatch.setattr(sys, "argv", ["reader.py", *args])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(ROOT / "reader.py"), run_name="__main__")
    return e.value.code


def test_reader_script_needs_month(tmp_path, monkeypatch, capsys):
    assert _reader_script(monkeypatch) == 2
    assert "python reader.py <папка месяца>" in capsys.readouterr().out
    assert _reader_script(monkeypatch, str(tmp_path / "нет")) == 1
    assert "Месяц не создан" in capsys.readouterr().out


def test_other_commands_delegate(monkeypatch):
    from tools import analyze_log, check, inspect_model, weights_manifest
    calls = []
    # analyze и inspect возвращают отчёт — код выхода от этого не меняется
    monkeypatch.setattr(analyze_log, "main", lambda argv: calls.append(("analyze", argv)) or "отчёт")
    monkeypatch.setattr(inspect_model, "main", lambda argv: calls.append(("inspect", argv)) or [{}])
    monkeypatch.setattr(weights_manifest, "main", lambda argv: calls.append(("weights", argv)))
    monkeypatch.setattr(check, "main", lambda argv: calls.append(("check", argv)) or 1)
    assert gmr.main(["analyze", "log.csv", "--details"]) == 0
    assert gmr.main(["inspect", "digit", "x"]) == 0
    assert gmr.main(["weights", "--write"]) == 0
    assert gmr.main(["check"]) == 1             # проверка не прошла -> код выхода 1
    assert calls == [("analyze", ["log.csv", "--details"]), ("inspect", ["digit", "x"]),
                     ("weights", ["--write"]), ("check", [])]


def test_help_and_unknown_command(capsys):
    assert gmr.main([]) == 0
    assert "process" in capsys.readouterr().out
    assert gmr.main(["frobnicate"]) == 2


def test_runs_as_script():
    out = subprocess.run([sys.executable, "gmr.py", "process", "--help"], cwd=ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0 and "папка месяца" in out.stdout and "--input" not in out.stdout


# ─── Сквозной прогон: gmr.py process == run_pipeline (фейковые модели) ───────

from tests import _pipeline as pl  # noqa: E402
from tests._pipeline import fake_models  # noqa: E402, F401 (фикстура)


def test_cli_process_end_to_end_same_as_reader(tmp_path, fake_models):  # noqa: F811
    cli = pl.setup_month(tmp_path / "cli")                 # прогон через CLI
    assert gmr.main(["process", str(cli.root)]) == 0
    direct = pl.setup_month(tmp_path / "direct")           # тот же прогон напрямую
    reader.run_pipeline(reader.PipelineConfig(month_dir=str(direct.root)))

    assert pl.log_without_timestamps(cli) == pl.log_without_timestamps(direct)
    assert pl.output_tree(cli) == pl.output_tree(direct)
    assert [r["outcome"] for r in pl.log(cli)] == pl.OUTCOMES
    assert pl.reading(cli, "A-1") == pl.reading(direct, "A-1") == "1200"


# ─── Этап 4: ярлык «Счётчики» на рабочем столе ───────────────────────────────

from tools import shortcut  # noqa: E402


def test_shortcut_script():
    ps = shortcut.powershell_script("Счётчики", Path("C:/x/.venv/Scripts/pythonw.exe"),
                                    Path("C:/x/program2.py"), Path("C:/x"))
    assert "GetFolderPath('Desktop')" in ps and "'Счётчики.lnk'" in ps
    assert f"$s.TargetPath = '{Path('C:/x/.venv/Scripts/pythonw.exe')}'" in ps
    assert f"$s.Arguments = '\"{Path('C:/x/program2.py')}\"'" in ps
    assert "$s.Save()" in ps
    assert shortcut._q("O'Brien") == "'O''Brien'"                 # кавычка в пути не ломает команду


def test_shortcut_only_windows(monkeypatch, capsys):
    monkeypatch.setattr(shortcut, "_is_windows", lambda: False)
    assert gmr.main(["shortcut"]) == 1
    assert "только на Windows" in capsys.readouterr().out


def test_shortcut_creates_via_powershell(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(shortcut, "_is_windows", lambda: True)
    monkeypatch.setattr(shortcut.sys, "executable", str(tmp_path / "python.exe"))
    assert gmr.main(["shortcut"]) == 1                             # pythonw.exe нет — ошибка
    assert "pythonw.exe" in capsys.readouterr().out
    (tmp_path / "pythonw.exe").write_bytes(b"")
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="C:\\Users\\op\\Desktop\\Счётчики.lnk\n", stderr="")

    monkeypatch.setattr(shortcut.subprocess, "run", run)
    assert gmr.main(["shortcut"]) == 0
    (cmd,) = calls
    assert cmd[0] == "powershell" and str(tmp_path / "pythonw.exe") in cmd[-1]
    assert str(ROOT / "program2.py") in cmd[-1]
    assert "Ярлык создан: C:\\Users\\op\\Desktop\\Счётчики.lnk" in capsys.readouterr().out
    monkeypatch.setattr(shortcut.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="доступ запрещён"))
    assert gmr.main(["shortcut"]) == 1
    assert "ярлык не создан" in capsys.readouterr().out
