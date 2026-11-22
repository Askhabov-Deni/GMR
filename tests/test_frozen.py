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
    exe = tmp_path / "GMR" / "GMR.exe"
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
    exe = tmp_path / "GMR" / "GMR.exe"
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
    monkeypatch.setattr(sys, "argv", ["GMR.exe", "--version"])
    assert gmr_operator.main() == 0
    assert capsys.readouterr().out.strip() == f"GMR {__version__}"


def test_frozen_output_switches_to_window_encoding(monkeypatch):
    # 1.5.1: сборка писала в трубу окна в cp1251 (PyInstaller не читает PYTHONIOENCODING) —
    # окно не узнавало «Фото N из M», ход обработки стоял на «Запуске»
    import io
    raw = io.BytesIO()
    out = io.TextIOWrapper(raw, encoding="cp1251")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1251"))
    monkeypatch.setattr(frozen, "FROZEN", True)
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    frozen.attach_std_streams()
    assert sys.stdout is out and out.encoding == "utf-8" and out.line_buffering
    print("Фото 3 из 40: a.jpg")
    assert raw.getvalue().decode("utf-8").rstrip() == "Фото 3 из 40: a.jpg"     # сразу, построчно


def test_window_reads_old_cp1251_output():
    from src.gmr.ui.run_dialog import decode_line, parse_progress
    line = "12:00:00  INFO      Фото 3 из 40: a.jpg"
    assert parse_progress(decode_line(line.encode("utf-8")))[:3] == ("photo", 3, 40)
    assert parse_progress(decode_line(line.encode("cp1251")))[:3] == ("photo", 3, 40)
