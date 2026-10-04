"""
tools/check.py — проверка перед коммитом: `python gmr.py check`.

Три шага, по порядку:
  1. данные — в git не попадут таблицы, логи, базы и фото (персональные
     данные абонентов). Смотрятся файлы, которые уже в git или попадут туда
     по `git add .` (то есть не закрыты .gitignore);
  2. код — ruff ищет ошибки: неизвестные имена, лишние импорты и т.п.
     (настройки — ruff.toml, стиль не проверяется);
  3. тесты — `python -m pytest tests/ -q`.

Всё зелёное — можно коммитить. Код возврата: 0 — всё в порядке, 1 — нет.
"""
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parent.parent

# Файлы с такими расширениями — данные, а не код.
DATA_EXTENSIONS = {
    ".csv", ".xlsx", ".xls", ".sqlite", ".db",
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".mp4",
}
# Исключения: метрики обучения моделей (results.csv YOLO), файлы для тестов
# с ВЫДУМАННЫМИ данными (tests/data/) и снимки экрана инструкции оператора на
# выдуманных данных (docs/img/) лежат в git намеренно.
ALLOWED_PREFIXES = ("meter_detect/runs/", "meter_ocr/runs/", "serial_id_ocr/runs/", "tests/data/",
                    "docs/img/")


def data_files(paths: Iterable[str]) -> list[str]:
    """Из списка путей (как их печатает git, через «/») — те, что похожи на данные."""
    found = []
    for p in paths:
        if Path(p).suffix.lower() in DATA_EXTENSIONS and not p.startswith(ALLOWED_PREFIXES):
            found.append(p)
    return sorted(found)


def git_files(root: Path = ROOT) -> list[str]:
    """Файлы в git и новые файлы, не закрытые .gitignore (их возьмёт `git add .`)."""
    out = subprocess.run(
        ["git", "-c", "core.quotepath=off", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root, capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd, cwd=ROOT).returncode


def main(argv=None, run: Callable[[list[str]], int] = _run,
         list_files: Callable[[], list[str]] = git_files) -> int:
    ok = True

    print("1/3 Данные в git (таблицы, логи, фото)…")
    try:
        bad = data_files(list_files())
    except (OSError, subprocess.CalledProcessError) as e:
        ok = False
        bad = []
        print(f"    ОШИБКА: не удалось спросить git ({e}). Git установлен? Это папка проекта?")
    if bad:
        ok = False
        print("    ОШИБКА: эти файлы попадут в git, а это похоже на данные:")
        for p in bad:
            print(f"      {p}")
        print("    Уберите их из папки проекта или добавьте в .gitignore.")
    elif ok:
        print("    в порядке")

    print("2/3 Код (ruff)…")
    if run([sys.executable, "-m", "ruff", "check", "--quiet", "."]) != 0:
        ok = False
        print("    ОШИБКА: ruff нашёл ошибки (список выше)")
    else:
        print("    в порядке")

    print("3/3 Тесты…")
    if run([sys.executable, "-m", "pytest", "tests/", "-q"]) != 0:
        ok = False
        print("    ОШИБКА: тесты не прошли (список выше)")
    else:
        print("    в порядке")

    print()
    print("ВСЁ В ПОРЯДКЕ — можно коммитить." if ok else "ЕСТЬ ОШИБКИ — не коммитить.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
