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


def test_frozen_window_runs_itself_for_processing(monkeypatch, tmp_path):
    # одна программа: обработку окно запускает тем же exe с параметрами
    exe = tmp_path / "GMR Оператор" / "GMR Оператор.exe"
    monkeypatch.setattr(frozen, "FROZEN", True)
    monkeypatch.setattr(sys, "executable", str(exe))
    assert run_dialog.process_command(tmp_path, "D:/GMR/Май", reread=True) == [
        str(exe), "process", "D:/GMR/Май", "--reread"]


def test_attach_std_streams_keeps_existing(monkeypatch):
    import io
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    frozen.attach_std_streams()
    assert sys.stdout is out and sys.stderr is err          # с консолью — ничего не трогаем


def test_entry_with_arguments_runs_command(monkeypatch, capsys):
    import gmr_operator
    from src.gmr.version import __version__
    monkeypatch.setattr(sys, "argv", ["GMR Оператор.exe", "--version"])
    assert gmr_operator.main() == 0
    assert capsys.readouterr().out.strip() == f"GMR {__version__}"
