"""tools/check.py — проверка перед коммитом (`python gmr.py check`)."""
import shutil
import subprocess

import pytest

from tools import check


def test_data_files_found_anywhere_except_training_metrics():
    paths = [
        "reader.py", "docs/BACKLOG.md", "tests/test_check.py",
        "Октябрь.xlsx", "data/таблица.csv", "data/таблица_log.csv",
        "fotos/IMG-20261001-WA0012.jpg", "x/scan.JPEG", "gmr.sqlite",
        "meter_detect/runs/detect/gas_meter_all_classes_s_v1/results.csv",
        "meter_ocr/runs/yolo/digits_detect_v4/results.csv",
    ]
    assert check.data_files(paths) == sorted([
        "Октябрь.xlsx", "data/таблица.csv", "data/таблица_log.csv",
        "fotos/IMG-20261001-WA0012.jpg", "x/scan.JPEG", "gmr.sqlite",
    ])


@pytest.mark.skipif(shutil.which("git") is None, reason="нужен git")
def test_git_files_sees_new_files_but_not_ignored(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / ".gitignore").write_text("*.xlsx\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "Октябрь.xlsx").write_bytes(b"x")       # закрыт .gitignore
    (tmp_path / "таблица.csv").write_text("x\n", encoding="utf-8")   # попадёт в git по `git add .`
    assert sorted(check.git_files(tmp_path)) == [".gitignore", "a.py", "таблица.csv"]


def _main(files, codes):
    calls = []

    def run(cmd):
        calls.append(cmd[2])           # "ruff" / "pytest"
        return codes.get(cmd[2], 0)
    rc = check.main([], run=run, list_files=lambda: files)
    return rc, calls


def test_main_all_green(capsys):
    rc, calls = _main(["reader.py"], {})
    assert rc == 0 and calls == ["ruff", "pytest"]
    assert "ВСЁ В ПОРЯДКЕ" in capsys.readouterr().out


@pytest.mark.parametrize("files, codes", [
    (["таблица.csv"], {}),             # данные в git
    (["reader.py"], {"ruff": 1}),      # ошибка в коде
    (["reader.py"], {"pytest": 1}),    # тесты не прошли
])
def test_main_any_failure_is_exit_1(files, codes, capsys):
    rc, calls = _main(files, codes)
    assert rc == 1 and calls == ["ruff", "pytest"]     # все шаги выполняются, итог один
    out = capsys.readouterr().out
    assert "ЕСТЬ ОШИБКИ" in out
    if files == ["таблица.csv"]:
        assert "таблица.csv" in out


def test_main_without_git_is_exit_1(capsys):
    def no_git():
        raise FileNotFoundError("git")
    rc = check.main([], run=lambda cmd: 0, list_files=no_git)
    out = capsys.readouterr().out
    assert rc == 1 and "не удалось спросить git" in out and "ЕСТЬ ОШИБКИ" in out
