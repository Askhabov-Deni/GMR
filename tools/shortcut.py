"""
tools/shortcut.py — ярлык «GMR Оператор» на рабочем столе (этап 4, решение
владельца 2026-10-03: 4а; до 2026-11-21 ярлык назывался «Счётчики»).

  python gmr.py shortcut

Ярлык запускает окно оператора (gmr_operator.py) через pythonw.exe из .venv —
без чёрного окна PowerShell. Делается один раз; если папку проекта или .venv
перенесли — команду повторить. Только Windows: ярлык создаёт сама Windows
(WScript.Shell через PowerShell), лишних пакетов не нужно.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "GMR Оператор"
WINDOW_EXE = "GMR Оператор.exe"        # окно в собранной программе (packaging/gmr.spec)


def _q(text) -> str:
    """Строка для PowerShell в одинарных кавычках."""
    return "'" + str(text).replace("'", "''") + "'"


def _is_windows() -> bool:
    return os.name == "nt"


def powershell_script(name: str, pythonw: Path, script, workdir: Path) -> str:
    """script None — ярлык прямо на exe (собранная программа, docs/BUILD.md)."""
    args = [] if script is None else [f"$s.Arguments = {_q(chr(34) + str(script) + chr(34))}"]
    return "\n".join([
        "$desk = [Environment]::GetFolderPath('Desktop')",
        f"$path = Join-Path $desk {_q(name + '.lnk')}",
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($path)",
        f"$s.TargetPath = {_q(pythonw)}",
        *args,
        f"$s.WorkingDirectory = {_q(workdir)}",
        "$s.Description = 'Окно оператора: показания газовых счётчиков'",
        "$s.Save()",
        "Write-Output $path",
    ])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python gmr.py shortcut",
                                description="Ярлык «GMR Оператор» на рабочем столе: окно оператора без "
                                            "чёрного окна PowerShell (только Windows).")
    p.add_argument("--name", default=NAME, help=f"имя ярлыка (по умолчанию «{NAME}»)")
    args = p.parse_args(argv)
    if not _is_windows():
        print("Ярлык на рабочем столе делается только на Windows.")
        return 1
    from src.gmr.frozen import FROZEN, program_dir
    if FROZEN:                                   # собранная программа: ярлык прямо на exe окна
        return _create(powershell_script(args.name, program_dir() / WINDOW_EXE, None, program_dir()))
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.is_file():
        print(f"ОШИБКА: не найден {pythonw}. Запустите команду из .venv проекта "
              "(.\\.venv\\Scripts\\Activate.ps1).")
        return 1
    return _create(powershell_script(args.name, pythonw, ROOT / "gmr_operator.py", ROOT))


def _create(script: str) -> int:
    out = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(f"ОШИБКА: ярлык не создан.\n{out.stderr.strip()}")
        return 1
    print(f"Ярлык создан: {out.stdout.strip()}\nОткрывайте программу двойным щелчком по нему.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
