"""Собранная программа (PyInstaller, docs/BUILD.md): пути и запуск обработки."""
import sys
from pathlib import Path

from src.gmr import frozen
from src.gmr.ui import run_dialog


def test_not_frozen_everything_as_before():
    assert not frozen.FROZEN
    assert frozen.resolve("meter_detect/a.pt") == "meter_detect/a.pt"      # от текущей папки, как всегда
    assert frozen.program_dir() == frozen.bundle_dir() == Path(__file__).resolve().parents[1]


def test_frozen_paths(monkeypatch, tmp_path):
    exe = tmp_path / "GMR Оператор" / "GMR Оператор.exe"
    monkeypatch.setattr(frozen, "FROZEN", True)
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr(sys, "_MEIPASS", str(exe.parent / "_internal"), raising=False)
    assert frozen.program_dir() == exe.parent                            # настройки и журнал окна — рядом с exe
    assert frozen.bundle_dir() == exe.parent / "_internal"
    assert frozen.resolve("meter_detect/a.pt") == str(exe.parent / "_internal" / "meter_detect/a.pt")
    assert frozen.resolve(str(tmp_path / "b.pt")) == str(tmp_path / "b.pt")   # полный путь — как есть
    assert frozen.resolve("") == ""                                      # модели нет


def test_frozen_window_runs_gmr_exe(monkeypatch, tmp_path):
    exe = tmp_path / "GMR Оператор" / "GMR Оператор.exe"
    monkeypatch.setattr(frozen, "FROZEN", True)
    monkeypatch.setattr(sys, "executable", str(exe))
    assert run_dialog.process_command(tmp_path, "D:/GMR/Май", reread=True) == [
        str(exe.with_name("gmr.exe")), "process", "D:/GMR/Май", "--reread"]
