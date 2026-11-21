"""
tools/shortcut.py — ярлык «GMR» на рабочем столе (этап 4, решение
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
NAME = "GMR"


def _q(text) -> str:
    """Строка для PowerShell в одинарных кавычках."""
    return "'" + str(text).replace("'", "''") + "'"


def _is_windows() -> bool:
    return os.name == "nt"


ICON = ROOT / "packaging" / "gmr.ico"          # значок программы (tools/make_icon.py)


def powershell_script(name: str, pythonw: Path, script, workdir: Path) -> str:
    """script None — ярлык прямо на exe (собранная программа, docs/BUILD.md:
    значок — из exe). Без сборки — на pythonw.exe, значок — packaging/gmr.ico."""
    args = [] if script is None else [f"$s.Arguments = {_q(chr(34) + str(script) + chr(34))}"]
    if script is not None and ICON.is_file():
        args.append(f"$s.IconLocation = {_q(str(ICON) + ',0')}")
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
                                description="Ярлык «GMR» на рабочем столе: окно оператора без "
                                            "чёрного окна PowerShell (только Windows).")
    p.add_argument("--name", default=NAME, help=f"имя ярлыка (по умолчанию «{NAME}»)")
    args = p.parse_args(argv)
    if not _is_windows():
        print("Ярлык на рабочем столе делается только на Windows.")
        return 1
    ok, message = create_shortcut(args.name)
    print(message if ok else f"ОШИБКА: {message}")
    return 0 if ok else 1


def create_shortcut(name: str = NAME) -> tuple[bool, str]:
    """Ярлык на рабочем столе (и из меню окна «Справка»): (получилось, что сказать)."""
    from src.gmr.frozen import FROZEN, program_dir
    if FROZEN:                                   # собранная программа: ярлык прямо на её exe
        return _create(powershell_script(name, Path(sys.executable), None, program_dir()))
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.is_file():
        return False, (f"не найден {pythonw}. Запустите команду из .venv проекта "
                       "(.\\.venv\\Scripts\\Activate.ps1).")
    return _create(powershell_script(name, pythonw, ROOT / "gmr_operator.py", ROOT))


def _create(script: str) -> tuple[bool, str]:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)       # без мелькания окна PowerShell
    out = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                         capture_output=True, text=True, creationflags=flags)
    if out.returncode != 0:
        return False, f"ярлык не создан.\n{out.stderr.strip()}"
    return True, f"Ярлык создан: {out.stdout.strip()}\nОткрывайте программу двойным щелчком по нему."


if __name__ == "__main__":
    sys.exit(main())
