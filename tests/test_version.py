"""Версия программы (src/gmr/version.py) и сборка для офиса (tools/build_exe.py)."""
import re

import gmr
from src.gmr.version import __version__
from tools import build_exe


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_cli_prints_version(capsys):
    assert gmr.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"GMR {__version__}"


def test_build_refuses_uncommitted_code(monkeypatch):
    status = " M src/gmr/operator/window.py\n?? tools/new.py\n M docs/CHANGELOG.md\n?? account_ocr/runs/x/\n"
    monkeypatch.setattr(build_exe, "git", lambda *a: status if a[0] == "status" else "abc1234")
    assert build_exe.dirty_code() == [" M src/gmr/operator/window.py", "?? tools/new.py"]   # документы и прогоны — не код
    assert build_exe.main(["--out", "нет"]) == 1


def test_version_file_names_version_commit_and_weights(monkeypatch):
    monkeypatch.setattr("tools.weights_manifest.manifest",
                        lambda: [{"role": "цифры", "path": "a/best.pt", "sha256": "0123456789abcdef"}])
    text = build_exe.version_text("1.2.3", "abc1234", [])
    assert text.startswith("GMR Оператор 1.2.3\n") and "Коммит: abc1234\n" in text
    assert "цифры: a/best.pt  sha256 0123456789abcdef" in text
    assert "незакоммиченные" in build_exe.version_text("1.2.3", "abc1234", ["x"])
