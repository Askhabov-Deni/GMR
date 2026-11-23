"""
tools/build_exe.py — сборка программы для офиса одной командой (docs/BUILD.md).

  python gmr.py build [--out D:\\GMR_build] [--dirty]

1. Код в git должен быть закоммичен: версия — это коммит (с --dirty — можно,
   но в ВЕРСИЯ.txt будет пометка).
2. PyInstaller по packaging/gmr.spec → <out>\\dist\\GMR\\.
3. Рядом с exe: ВЕРСИЯ.txt (версия, дата, коммит, файлы весов), ПРОЧТИ.txt
   (как поставить) и «Инструкция оператора.html» (docs/OPERATOR.md).
4. Проверка: «GMR.exe --version» печатает эту версию.
5. Архив <out>\\GMR <версия>.zip — его и переносить в офис.
"""
import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = "GMR"
EXE = f"{APP}.exe"                          # одна программа: окно и обработка (packaging/gmr.spec)
MANUAL = "Инструкция оператора.html"        # src/gmr/operator/theme.MANUAL_NAME
# что попадает в сборку: незакоммиченные изменения здесь — сборка не совпадёт с коммитом
CODE = ("src/", "models/", "tools/", "packaging/", "gmr.py", "reader.py", "program2.py", "gmr_operator.py",
        "docs/OPERATOR.md", "docs/img/")


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
    lines += ["", "Что нового — docs/CHANGELOG.md в проекте."]
    return "\n".join(lines) + "\n"


def readme_text(version: str) -> str:
    """ПРОЧТИ.txt — первое, что видно в архиве: как поставить и где инструкция."""
    rows = [(EXE, "программа: окно оператора, обработку фото запускает сама"),
            (MANUAL, "инструкция оператора со снимками экрана"),
            ("ВЕРСИЯ.txt", "версия программы и её модели"),
            ("_internal", "служебная папка (Python, модели) — не трогать")]
    return "\n".join([
        f"{APP} {version} — показания газовых счётчиков по фото контролёров.",
        "",
        "УСТАНОВКА (один раз)",
        f"  1. Распакуйте архив, например в D:\\ — получится папка «{APP}».",
        "     Python и интернет не нужны.",
        f"  2. Откройте «{EXE}» двойным щелчком.",
        "  3. В окне: меню «Справка» → «Ярлык на рабочем столе».",
        "     Дальше открывайте программу этим ярлыком.",
        "",
        "КАК РАБОТАТЬ",
        f"  «{MANUAL}» в этой папке (откроется в браузере)",
        "  или в окне программы: меню «Справка» → «Инструкция оператора».",
        "",
        "НОВАЯ ВЕРСИЯ",
        "  Распакуйте новый архив поверх этой папки. Папки месяцев (фото, база,",
        "  показания) лежат отдельно и не трогаются.",
        "  До 1.3.0 программа называлась «GMR Оператор»: распакуйте «GMR» рядом",
        "  со старой папкой — имя и месяц программа возьмёт оттуда. Новый ярлык —",
        "  «Справка» → «Ярлык на рабочем столе»; старые папку и ярлык можно удалить.",
        "",
        "ЧТО В ПАПКЕ",
        *[f"  {name:<28}{what}" for name, what in rows],
        "  settings.json и program2.log появятся после первого запуска: настройки и журнал окна.",
    ]) + "\n"


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
    from models.yolo_all_detect.openvino_net import export_openvino, is_current
    from src.gmr.domain import PipelineConfig
    from src.gmr.ml.loader import quiet_ultralytics
    cfg = PipelineConfig()
    for weights, imgsz in ((cfg.meter_detect_model, 640), (cfg.digit_detect_model, 480)):
        if not is_current(ROOT / weights):      # детекторы в OpenVINO (1.6.0): нет выгрузки или от других весов
            quiet_ultralytics()
            print(f"Выгрузка в OpenVINO: {export_openvino(ROOT / weights, imgsz)}")
    out = Path(args.out)
    dist, work = out / "dist", out / "work"
    app = dist / APP
    print(f"Сборка {APP} {version} (коммит {commit}) → {app}")
    keep = None                      # settings.json того, кто запускал программу из этой папки
    if app.exists():
        # открытая программа держит свои файлы: удаление выбило бы её папку
        # наполовину (2026-11-22) — сначала переименовать целиком, не вышло — стоп
        old = app.with_name(app.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        try:
            app.rename(old)
        except OSError:
            print(f"ОШИБКА: папка {app} занята — программа оттуда открыта? Закройте её и повторите. "
                  "Ничего не удалено.")
            return 1
        if (old / "settings.json").is_file():
            keep = (old / "settings.json").read_bytes()
        shutil.rmtree(old, ignore_errors=True)
    env = dict(os.environ, NO_ALBUMENTATIONS_UPDATE="1")
    r = subprocess.run([sys.executable, "-m", "PyInstaller", str(ROOT / "packaging" / "gmr.spec"), "--noconfirm",
                        "--distpath", str(dist), "--workpath", str(work)], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    (out / "build.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode != 0 or not (app / EXE).is_file():
        print(f"ОШИБКА: PyInstaller не собрал программу, подробности — {out / 'build.log'}")
        print("\n".join((r.stdout + r.stderr).splitlines()[-15:]))
        return 1

    (app / "ВЕРСИЯ.txt").write_text(version_text(version, commit, dirty), encoding="utf-8")
    (app / "ПРОЧТИ.txt").write_text(readme_text(version), encoding="utf-8-sig")   # с BOM — Блокнот поймёт
    from tools.manual_html import write_manual
    write_manual(app / MANUAL, version)
    # та же программа с параметрами — команда без окна (так окно запускает обработку фото)
    check = subprocess.run([str(app / EXE), "--version"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if check.stdout.strip() != f"GMR {version}":
        print(f"ОШИБКА: «{EXE} --version» напечатал «{check.stdout.strip()}» вместо «GMR {version}»\n"
              f"{check.stderr}")
        return 1
    size = sum(f.stat().st_size for f in app.rglob("*") if f.is_file()) / 2**30
    print(f"Готово: {app} ({size:.1f} ГБ), «{EXE} --version»: {check.stdout.strip()}")
    if not args.no_zip:
        zip_base = out / f"{APP} {version}"
        print("Архив…")
        zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=dist, base_dir=APP)
        print(f"Архив: {zip_path} ({Path(zip_path).stat().st_size / 2**30:.1f} ГБ)")
    if keep is not None:             # после архива: в архив чужие настройки не попадают
        (app / "settings.json").write_bytes(keep)
    print(f"Дальше: git tag v{version} и git push origin v{version} (метка — какой коммит собран).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
