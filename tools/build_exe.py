"""
tools/build_exe.py — сборка программы для офиса одной командой (docs/BUILD.md).

  python gmr.py build [--out D:\\GMR_build] [--dirty]

1. Код в git должен быть закоммичен: версия — это коммит (с --dirty — можно,
   но в ВЕРСИЯ.txt будет пометка).
2. PyInstaller по packaging/gmr.spec → <out>\\dist\\GMR Оператор\\.
3. ВЕРСИЯ.txt рядом с exe: версия, дата сборки, коммит, файлы весов (SHA-256).
4. Проверка: gmr.exe --version печатает эту версию.
5. Архив <out>\\GMR Оператор <версия>.zip — его и переносить в офис.
"""
import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = "GMR Оператор"
# что попадает в сборку: незакоммиченные изменения здесь — сборка не совпадёт с коммитом
CODE = ("src/", "models/", "tools/", "packaging/", "gmr.py", "reader.py", "program2.py", "gmr_operator.py")


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()


def dirty_code() -> list[str]:
    """Изменённые, но не закоммиченные файлы кода (новые неотслеживаемые — тоже)."""
    out = []
    for line in git("status", "--porcelain").splitlines():
        path = line[3:].strip().strip('"')
        if path.startswith(CODE):
            out.append(line)
    return out


def version_text(version: str, commit: str, dirty: list[str]) -> str:
    from tools.weights_manifest import manifest
    lines = [f"{APP} {version}",
             f"Собрано: {datetime.now():%Y-%m-%d %H:%M}",
             f"Коммит: {commit}" + ("  (+ незакоммиченные изменения!)" if dirty else ""),
             "",
             "Модели (файлы весов внутри _internal\\):"]
    lines += [f"  {r['role']}: {r['path']}  sha256 {r['sha256']}" for r in manifest()]
    lines += ["", "gmr.exe --version — та же версия. Что нового — docs/CHANGELOG.md в проекте."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python gmr.py build", description="Сборка программы для офиса (exe).")
    p.add_argument("--out", default=r"D:\GMR_build", help="куда собирать (по умолчанию D:\\GMR_build)")
    p.add_argument("--dirty", action="store_true", help="собрать с незакоммиченным кодом")
    p.add_argument("--no-zip", action="store_true", help="без архива")
    args = p.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from src.gmr.version import __version__ as version

    dirty = dirty_code()
    if dirty and not args.dirty:
        print("ОШИБКА: код не закоммичен — сборка не совпала бы с версией в git:")
        print("\n".join("  " + d for d in dirty))
        print("Закоммитьте или соберите с --dirty (для проверки, не для офиса).")
        return 1
    commit = git("rev-parse", "--short", "HEAD")
    out = Path(args.out)
    dist, work = out / "dist", out / "work"
    app = dist / APP
    print(f"Сборка {APP} {version} (коммит {commit}) → {app}")
    shutil.rmtree(app, ignore_errors=True)
    env = dict(os.environ, NO_ALBUMENTATIONS_UPDATE="1")
    r = subprocess.run([sys.executable, "-m", "PyInstaller", str(ROOT / "packaging" / "gmr.spec"), "--noconfirm",
                        "--distpath", str(dist), "--workpath", str(work)], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    (out / "build.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode != 0 or not (app / "gmr.exe").is_file():
        print(f"ОШИБКА: PyInstaller не собрал программу, подробности — {out / 'build.log'}")
        print("\n".join((r.stdout + r.stderr).splitlines()[-15:]))
        return 1

    (app / "ВЕРСИЯ.txt").write_text(version_text(version, commit, dirty), encoding="utf-8")
    check = subprocess.run([str(app / "gmr.exe"), "--version"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if check.stdout.strip() != f"GMR {version}":
        print(f"ОШИБКА: gmr.exe --version напечатал «{check.stdout.strip()}» вместо «GMR {version}»\n{check.stderr}")
        return 1
    size = sum(f.stat().st_size for f in app.rglob("*") if f.is_file()) / 2**30
    print(f"Готово: {app} ({size:.1f} ГБ), gmr.exe --version: {check.stdout.strip()}")
    if not args.no_zip:
        zip_base = out / f"{APP} {version}"
        print("Архив…")
        zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=dist, base_dir=APP)
        print(f"Архив: {zip_path} ({Path(zip_path).stat().st_size / 2**30:.1f} ГБ)")
    print(f"Дальше: git tag v{version} и git push origin v{version} (метка — какой коммит собран).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
