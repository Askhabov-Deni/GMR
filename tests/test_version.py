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
    status = (" M src/gmr/operator/window.py\n?? tools/new.py\n M docs/CHANGELOG.md\n?? account_ocr/runs/x/\n"
              " M docs/OPERATOR.md\n")
    monkeypatch.setattr(build_exe, "git", lambda *a: status if a[0] == "status" else "abc1234")
    # код и инструкция (она в сборке) — да; CHANGELOG и прогоны обучения — нет
    assert build_exe.dirty_code() == [" M src/gmr/operator/window.py", "?? tools/new.py", " M docs/OPERATOR.md"]
    assert build_exe.main(["--out", "нет"]) == 1


def test_version_file_names_version_commit_and_weights(monkeypatch):
    monkeypatch.setattr("tools.weights_manifest.manifest",
                        lambda: [{"role": "цифры", "path": "a/best.pt", "sha256": "0123456789abcdef"}])
    text = build_exe.version_text("1.2.3", "abc1234", [])
    assert text.startswith("GMR Оператор 1.2.3\n") and "Коммит: abc1234\n" in text
    assert "цифры: a/best.pt  sha256 0123456789abcdef" in text
    assert "незакоммиченные" in build_exe.version_text("1.2.3", "abc1234", ["x"])


def test_readme_says_how_to_install_and_where_manual_is():
    text = build_exe.readme_text("1.2.3")
    assert text.startswith("GMR Оператор 1.2.3")
    assert "GMR Оператор.exe" in text and "Инструкция оператора.html" in text and "Ярлык на рабочем столе" in text
    assert "gmr.exe" not in text                                  # одна программа


def test_manual_html_is_one_file_with_pictures(tmp_path):
    from tools import manual_html
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "a.png").write_bytes(b"\x89PNG fake")
    md = ("# Заголовок\n\nАбзац с **жирным** и `кодом`.\n\n1. Шаг\n   - пункт\n2. Второй\n\n"
          "![Снимок](img/a.png)\n\n| A | B |\n|---|---|\n| 1 | <2> |\n")
    page = manual_html.to_html(md, tmp_path, version="1.2.3")
    assert "<h1>Заголовок</h1>" in page and "<b>жирным</b>" in page and "<code>кодом</code>" in page
    assert page.count("<ol>") == 1 and page.count("<ul>") == 1 and "data:image/png;base64," in page
    assert "<td>&lt;2&gt;</td>" in page and "GMR Оператор 1.2.3" in page
    assert "img/a.png" not in page                                # картинка внутри файла, не ссылкой


def test_build_does_not_touch_folder_of_running_program(monkeypatch, tmp_path, capsys):
    # открытая программа держит файлы: сборка не должна удалять её папку наполовину
    app = tmp_path / "dist" / build_exe.APP
    (app / "_internal").mkdir(parents=True)
    (app / "settings.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(build_exe, "dirty_code", lambda: [])
    monkeypatch.setattr(build_exe, "git", lambda *a: "abc1234")
    real_rename = type(app).rename
    monkeypatch.setattr(type(app), "rename", lambda self, target: (_ for _ in ()).throw(PermissionError("занято"))
                        if self == app else real_rename(self, target))
    assert build_exe.main(["--out", str(tmp_path)]) == 1
    assert "занята" in capsys.readouterr().out and (app / "settings.json").is_file()
