"""
Общий код reader.py и program2.py (src/gmr/): лог, нормализация серийника,
чтение и запись картинок. program2.py не зависит от reader.py
(с Фазы 6; до 2026-10-01 файл назывался test_phase6_boundaries.py).
"""
import ast
import subprocess
import sys
from pathlib import Path

import reader
from src.gmr.domain.serial_match import normalize_serial
from src.gmr.storage import load_log, log_path_for, save_log

ROOT = Path(__file__).resolve().parent.parent


def test_program2_does_not_import_reader():
    tree = ast.parse((ROOT / "program2.py").read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add(node.module or "")
    assert not {m for m in mods if m == "reader" or m.startswith("reader.")}


def test_program2_import_does_not_load_reader():
    code = "import sys, program2; print('reader' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_reader_legacy_names_removed():
    # Фаза 6 оставила старые имена ссылками, Фаза 7 их удалила (правило 3 ТЗ)
    for name in ("_load_table", "_save_table", "_load_log", "_save_log", "_append_log_row",
                 "_log_path", "_normalize_serial", "_draw_annotation", "_LOG_COLUMNS",
                 "_read_meter_digits", "_find_crop_entry", "_find_crop", "_log_filenames"):
        assert not hasattr(reader, name), name


def test_old_mode_removed():
    # таблица и CSV-лог (старый режим) убраны на этапе 2.3b — только папка месяца
    import src.gmr.storage as storage
    for name in ("_maybe_init_log", "_TableRun", "_backup_before_run", "_report_shadow_run",
                 "load_table", "save_table", "append_log_row", "ShadowLogStore"):
        assert not hasattr(reader, name), name
    for name in ("load_table", "save_table", "append_log_row", "ShadowLogStore",
                 "backup_files", "backup_dir_for"):
        assert not hasattr(storage, name), name
    for field in ("table_path", "shadow_sqlite_log"):
        assert field not in reader.PipelineConfig.__dataclass_fields__, field


def test_log_functions(tmp_path):
    # CSV-лог старого режима: при создании месяца он переносится в базу
    assert log_path_for(str(tmp_path / "meters_table.csv")) == str(tmp_path / "meters_table_log.csv")
    lp = str(tmp_path / "l.csv")
    assert load_log(lp) == []
    save_log(lp, [{"original_filename": "a.jpg", "outcome": "PLUS"},
                  {"original_filename": "b.jpg", "outcome": "MINUS"}])
    assert [r["original_filename"] for r in load_log(lp)] == ["a.jpg", "b.jpg"]


def test_normalize_serial():
    assert normalize_serial("  007123 ") == "007123"


# ─── Картинки при путях с кириллицей (2026-10-01) ────────────────────────────

import numpy as np  # noqa: E402

from src.gmr.render import read_image, write_image  # noqa: E402


def test_image_io_roundtrip_cyrillic_path(tmp_path):
    d = tmp_path / "Сулиман С" / "plus"
    d.mkdir(parents=True)
    img = np.zeros((20, 30, 3), np.uint8)
    img[5, 5] = (10, 200, 30)
    assert write_image(d / "Фото 1.png", img) is True
    back = read_image(d / "Фото 1.png")
    assert back is not None and back.shape == (20, 30, 3) and tuple(back[5, 5]) == (10, 200, 30)


def test_image_io_errors_like_cv2(tmp_path):
    assert read_image(tmp_path / "нет.jpg") is None
    (tmp_path / "пусто.jpg").write_bytes(b"")
    assert read_image(tmp_path / "пусто.jpg") is None
    (tmp_path / "мусор.jpg").write_bytes(b"not an image")
    assert read_image(tmp_path / "мусор.jpg") is None
    assert write_image(tmp_path / "x.unknownext", np.zeros((5, 5, 3), np.uint8)) is False
    assert write_image(tmp_path / "нет_папки" / "x.jpg", np.zeros((5, 5, 3), np.uint8)) is False


def test_program2_redraw_annotation_on_cyrillic_path(tmp_path):
    import program2
    p = tmp_path / "Сулиман С" / "plus" / "A1.jpg"
    p.parent.mkdir(parents=True)
    write_image(p, np.full((400, 600, 3), 255, np.uint8))
    program2.redraw_annotation(str(p), "1234567", "01200")
    after = read_image(p)
    assert max(after[2, 2].tolist()) < 20      # тёмная плашка подписи (JPEG чуть искажает цвет)


def test_reader_save_annotated_on_cyrillic_path(tmp_path):
    from src.gmr.domain import Outcome, PhotoResult
    src = tmp_path / "Аюб" / "фото.jpg"
    src.parent.mkdir(parents=True)
    write_image(src, np.full((400, 600, 3), 255, np.uint8))
    r = PhotoResult(photo_path=str(src), outcome=Outcome.PLUS, serial_text="1", reading=1200)
    reader._save_annotated(str(src), str(tmp_path / "out" / "Аюб" / "plus"), "A1.jpg", r, move=False)
    out = read_image(tmp_path / "out" / "Аюб" / "plus" / "A1.jpg")
    assert out is not None and max(out[2, 2].tolist()) < 20
