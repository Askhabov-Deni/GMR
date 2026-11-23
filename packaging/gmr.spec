# -*- mode: python ; coding: utf-8 -*-
# Сборка программы для офиса (docs/BUILD.md):
#   python -m PyInstaller packaging/gmr.spec --distpath <куда> --workpath <временная папка>
# Результат — папка «GMR»: «GMR.exe» (одна программа: без
# параметров — окно оператора, с параметрами — команды gmr.py; так окно само
# запускает обработку фото) и _internal (Python, библиотеки, веса).
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
datas += [(str(ROOT / "src" / "gmr" / "operator" / "gmr.png"), "src/gmr/operator")]   # значок окна
datas += collect_data_files("ultralytics")
missing = [d for d, _ in datas[:len(weights)] if not Path(d).is_file()]
if missing:
    raise SystemExit(f"Нет файлов весов: {missing}")
# 1.6.0: детекторы в OpenVINO (python gmr.py openvino; build_exe выгружает сам, если нет)
from models.yolo_all_detect.openvino_net import is_current, openvino_dir  # noqa: E402
for w in (cfg.meter_detect_model, cfg.digit_detect_model):
    d = ROOT / openvino_dir(w)
    if not is_current(ROOT / w):
        raise SystemExit(f"Нет выгрузки OpenVINO из этих весов: {d} — python gmr.py openvino")
    datas += [(str(f), str(openvino_dir(w))) for f in d.iterdir() if f.is_file()]

hidden = collect_submodules("src") + ["reader", "gmr"]
# torchvision 0.29: модуль операций — _C_stable.pyd (хук PyInstaller ищет старый _C)
import torchvision  # noqa: E402
_tv = Path(torchvision.__file__).parent
binaries = [(str(f), "torchvision") for f in _tv.iterdir() if f.suffix.lower() in (".pyd", ".dll")]
# OpenVINO: только расчёт на процессоре и чтение готовой модели — без видеокарты,
# нейропроцессора, конвертеров и телеметрии (иначе +180 МБ)
import openvino  # noqa: E402
_OV_LIBS = {"openvino.dll", "openvino_c.dll", "openvino_intel_cpu_plugin.dll", "openvino_ir_frontend.dll",
            "tbb12.dll", "tbbbind_2_5.dll", "tbbmalloc.dll", "tbbmalloc_proxy.dll"}
_ov_libs = Path(openvino.__file__).parent / "libs"
binaries += [(str(f), "openvino/libs") for f in _ov_libs.iterdir() if f.name.lower() in _OV_LIBS]
hidden += ["openvino"]
excludes = ["matplotlib", "IPython", "jupyter", "notebook", "tensorboard", "pytest", "openvino_telemetry",
            "openvino.tools", "onnx"]


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
            StringStruct("ProductName", "GMR"), StringStruct("FileDescription", description),
            StringStruct("FileVersion", __version__), StringStruct("ProductVersion", __version__)])]),
            VarFileInfo([VarStruct("Translation", [0x0419, 1200])])])


# одна программа: окно и команды gmr.py (gmr_operator.py импортирует gmr только с параметрами)
app = analysis("gmr_operator.py")
exe = EXE(PYZ(app.pure), app.scripts, [], exclude_binaries=True, name="GMR",
          console=False, upx=False, icon=str(ROOT / "packaging" / "gmr.ico"),  # tools/make_icon.py
          version=version_info("GMR — показания газовых счётчиков"))
coll = COLLECT(exe, app.binaries, app.datas, upx=False, name="GMR")
