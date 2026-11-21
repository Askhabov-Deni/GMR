# -*- mode: python ; coding: utf-8 -*-
# Сборка программы для офиса (docs/BUILD.md):
#   python -m PyInstaller packaging/gmr.spec --distpath <куда> --workpath <временная папка>
# Результат — папка «GMR Оператор» с двумя exe и общей папкой _internal:
#   «GMR Оператор.exe» — окно оператора (без чёрного окна);
#   gmr.exe            — обработка фото и команды (окно запускает его само).
# Веса моделей — те, что в PipelineConfig (src/gmr/domain/config.py), образцы
# символов надписей — models/overlay/glyphs.npz.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
sys.path.insert(0, str(ROOT))
from src.gmr.domain.config import PipelineConfig  # noqa: E402

cfg = PipelineConfig()
weights = [cfg.meter_detect_model, cfg.digit_detect_model, cfg.digit_ocr_model, cfg.serial_ocr_model,
           cfg.account_ocr_model]
datas = [(str(ROOT / w), str(Path(w).parent)) for w in weights if w]
datas += [(str(ROOT / "models" / "overlay" / "glyphs.npz"), "models/overlay")]
datas += collect_data_files("ultralytics")
missing = [d for d, _ in datas[:len(weights)] if not Path(d).is_file()]
if missing:
    raise SystemExit(f"Нет файлов весов: {missing}")

hidden = collect_submodules("src") + ["reader"]
# torchvision 0.29: модуль операций — _C_stable.pyd (хук PyInstaller ищет старый _C)
import torchvision  # noqa: E402
_tv = Path(torchvision.__file__).parent
binaries = [(str(f), "torchvision") for f in _tv.iterdir() if f.suffix.lower() in (".pyd", ".dll")]
excludes = ["matplotlib", "IPython", "jupyter", "notebook", "tensorboard", "pytest"]


def analysis(script):
    return Analysis([str(ROOT / script)], pathex=[str(ROOT)], datas=datas, binaries=binaries, hiddenimports=hidden,
                    excludes=excludes, noarchive=False)


# Версия в свойствах exe (Windows: «Свойства» → «Подробно») — из src/gmr/version.py
from PyInstaller.utils.win32.versioninfo import (  # noqa: E402
    FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo, VarStruct, VSVersionInfo)
from src.gmr.version import __version__  # noqa: E402

_v = tuple(int(x) for x in __version__.split(".")) + (0,)


def version_info(description):
    return VSVersionInfo(
        ffi=FixedFileInfo(filevers=_v, prodvers=_v),
        kids=[StringFileInfo([StringTable("041904B0", [
            StringStruct("ProductName", "GMR Оператор"), StringStruct("FileDescription", description),
            StringStruct("FileVersion", __version__), StringStruct("ProductVersion", __version__)])]),
            VarFileInfo([VarStruct("Translation", [0x0419, 1200])])])


win = analysis("gmr_operator.py")
cli = analysis("gmr.py")

win_exe = EXE(PYZ(win.pure), win.scripts, [], exclude_binaries=True, name="GMR Оператор",
              console=False, upx=False, version=version_info("GMR Оператор — окно оператора"))
cli_exe = EXE(PYZ(cli.pure), cli.scripts, [], exclude_binaries=True, name="gmr",
              console=True, upx=False, version=version_info("GMR — обработка фото и команды"))

coll = COLLECT(win_exe, win.binaries, win.datas, cli_exe, cli.binaries, cli.datas,
               upx=False, name="GMR Оператор")
