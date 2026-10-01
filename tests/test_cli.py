"""
Фаза 7: gmr.py — одна точка входа. process вызывает тот же run_pipeline с теми
же настройками, что `python reader.py`; остальные команды — обёртки над tools/.
"""
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
    assert gmr.main(["process"]) == 0
    cfg, ref = captured["cfg"], reader.default_run_config()
    assert cfg == ref
    # то, что задаёт `python reader.py` поверх дефолтов PipelineConfig
    assert (cfg.move_photos, cfg.ignore_last_digits, cfg.debug_digits, cfg.draw_boxes) == (False, 2, False, False)


def test_process_overrides(captured, tmp_path):
    gmr.main(["process", "--input", "in", "--output", "out", "--table", "t.csv", "--move", "--shadow-sqlite"])
    cfg = captured["cfg"]
    assert (cfg.input_dir, cfg.output_base_dir, cfg.table_path) == ("in", "out", "t.csv")
    assert cfg.move_photos is True and cfg.shadow_sqlite_log is True
    assert cfg.digit_conf_thresh == 0.6 and cfg.serial_conf_thresh == 0.6   # пороги не трогаются


def test_other_commands_delegate(monkeypatch):
    from tools import analyze_log, inspect_model, weights_manifest
    calls = []
    monkeypatch.setattr(analyze_log, "main", lambda argv: calls.append(("analyze", argv)))
    monkeypatch.setattr(inspect_model, "main", lambda argv: calls.append(("inspect", argv)))
    monkeypatch.setattr(weights_manifest, "main", lambda argv: calls.append(("weights", argv)))
    gmr.main(["analyze", "log.csv", "--details"])
    gmr.main(["inspect", "digit", "x"])
    gmr.main(["weights", "--write"])
    assert calls == [("analyze", ["log.csv", "--details"]), ("inspect", ["digit", "x"]),
                     ("weights", ["--write"])]


def test_help_and_unknown_command(capsys):
    assert gmr.main([]) == 0
    assert "process" in capsys.readouterr().out
    assert gmr.main(["frobnicate"]) == 2


def test_runs_as_script():
    out = subprocess.run([sys.executable, "gmr.py", "process", "--help"], cwd=ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--input" in out.stdout


# ─── Сквозной прогон: gmr.py process == run_pipeline (фейковые модели) ───────

from tests.test_shadow_run_pipeline import (  # noqa: E402
    _log_without_timestamps, _output_tree, _setup_workspace, fake_models,  # noqa: F401 (фикстура)
)
from src.gmr.storage import log_path_for  # noqa: E402


def test_cli_process_end_to_end_same_as_reader(tmp_path, fake_models):  # noqa: F811
    # прогон через CLI
    inp, table = _setup_workspace(tmp_path / "cli")
    gmr.main(["process", "--input", str(inp), "--output", str(tmp_path / "cli" / "out"),
              "--table", str(table)])
    # тот же прогон напрямую, с настройками `python reader.py`
    inp2, table2 = _setup_workspace(tmp_path / "direct")
    reader.run_pipeline(reader.default_run_config(
        input_dir=str(inp2), output_base_dir=str(tmp_path / "direct" / "out"), table_path=str(table2)))

    assert _log_without_timestamps(log_path_for(str(table))) == \
        _log_without_timestamps(log_path_for(str(table2)))
    assert Path(table).read_bytes() == Path(table2).read_bytes()
    assert _output_tree(tmp_path / "cli") == _output_tree(tmp_path / "direct")
    outcomes = [r["outcome"] for r in _log_without_timestamps(log_path_for(str(table)))]
    assert outcomes == ["PLUS", "MINUS", "SERIAL_NOT_FOUND", "REPEAT", "NO_METER"]
