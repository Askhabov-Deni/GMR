"""
tools/doc_screenshots.py — снимки окна «GMR Оператор» для docs/OPERATOR.md на
ВЫДУМАННЫХ данных: нарисованные счётчики, вымышленные номера и счета
(docs/img/ — в git, tools/check.py их пропускает).

  python -m tools.doc_screenshots [<папка снимков, по умолчанию docs/img>]

Окно на несколько секунд появляется на экране (снимок — с экрана). Месяц
«Октябрь_2026» создаётся во временной папке и потом удаляется.
"""
import ctypes
import shutil
import sys
import tempfile
import time
from pathlib import Path

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)      # снимок — в настоящих пикселях экрана
except Exception:                                       # noqa: BLE001
    ctypes.windll.user32.SetProcessDPIAware()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw, ImageFont, ImageGrab  # noqa: E402

from tools import ui_scenarios as ui  # noqa: E402

FONT = r"C:\Windows\Fonts\consolab.ttf"
HAND = r"C:\Windows\Fonts\segoepr.ttf"
MENU_H = ctypes.windll.user32.GetSystemMetrics(15) + 1      # строка меню над окном
FAKE_RUN = ("import time\nprint('Новых фото к чтению: 24', flush=True)\n"
            "for i in range(1, 25):\n"
            "    print(f'Фото {i} из 24: IMG-20261004-WA{i:04d}.jpg', flush=True); time.sleep(0.35)\n")


def meter(path: Path, digits: str, serial: str, marker: str = "", red: int = 2):
    """Нарисованный счётчик: барабан, номер на табличке, надпись маркером."""
    im = Image.new("RGB", (900, 1200), (152, 160, 150))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((110, 120, 790, 640), 40, fill=(226, 221, 206), outline=(80, 80, 80), width=8)
    d.rectangle((200, 240, 700, 380), fill=(30, 30, 30))
    f = ImageFont.truetype(FONT, 110)
    for i, ch in enumerate(digits):
        x = 215 + i * 97
        d.rectangle((x, 255, x + 82, 365), fill=(200, 40, 40) if i >= len(digits) - red else (248, 248, 244))
        d.text((x + 41, 312), ch, font=f, fill=(20, 20, 20), anchor="mm")
    if serial:
        d.text((450, 470), f"№ {serial}", font=ImageFont.truetype(FONT, 60), fill=(30, 30, 30), anchor="mm")
    d.text((450, 560), "G4   2019", font=ImageFont.truetype(FONT, 52), fill=(60, 60, 60), anchor="mm")
    if marker:
        d.text((450, 900), marker, font=ImageFont.truetype(HAND, 150), fill=(25, 25, 35), anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=90)


def build(root: Path):
    from tests._month import make_month
    table = [(f"0010001{i:02d}", f"13000010{i:02d}", str(1000 + 50 * i), "") for i in range(1, 13)]
    a, b = "Контролёр 1", "Контролёр 2"
    auto = [
        dict(original_filename="IMG-20261001-WA0003.jpg", final_filename="1300001003.jpg", outcome="DIGITS_ERROR",
             serial_id="001000103", account_id="1300001003", last_reading="1150", source_folder=a),
        dict(original_filename="IMG-20261001-WA0006.jpg", final_filename="1300001006.jpg", outcome="SUSPICIOUS",
             serial_id="001000106", account_id="1300001006", reading="31862", last_reading="1300",
             delta="30562.0", source_folder=b),
        dict(original_filename="IMG-20261002-WA0011.jpg", final_filename="IMG-20261002-WA0011.jpg",
             outcome="SERIAL_NOT_FOUND", serial_id="001000911", source_folder=a),
        dict(original_filename="IMG-20261003-WA0005.jpg", final_filename="IMG-20261003-WA0005.jpg",
             outcome="NO_METER", source_folder=b),
        dict(original_filename="IMG-20261001-WA0001.jpg", final_filename="1300001001.jpg", outcome="PLUS",
             serial_id="001000101", account_id="1300001001", reading="01412", last_reading="1050",
             delta="362.0", source_folder=a, processed_at="2026-10-01T09:12:00"),
        dict(original_filename="IMG-20261001-WA0002.jpg", final_filename="1300001002.jpg", outcome="MINUS",
             serial_id="001000102", account_id="1300001002", reading="01088", last_reading="1100",
             delta="-12.0", source_folder=b, processed_at="2026-10-01T09:14:00"),
    ]
    for r in auto:
        r.update(source="auto", processed_by="auto", photo_hash="h-" + r["original_filename"])
    f = make_month(root, table, auto, name="Октябрь_2026")
    res = f.results
    meter(res / a / "question/digits_error/1300001003.jpg", "01862", "001000103", "1003")
    meter(res / b / "question/suspicious/1300001006.jpg", "31862", "001000106", "1006")
    meter(res / a / "question/serial_not_found/IMG-20261002-WA0011.jpg", "01277", "001000911")
    meter(res / b / "question/no_meter/IMG-20261003-WA0005.jpg", "00000", "")
    meter(res / a / "plus/1300001001.jpg", "01412", "001000101", "1001")
    meter(res / b / "minus/1300001002.jpg", "01088", "001000102", "1002")
    return f


def grab(w, path, menu=True, over=None):
    w.lift()
    w.attributes("-topmost", True)
    if over is not None:                 # окно поверх главного (прогон, настройки)
        over.lift()
        over.attributes("-topmost", True)
    ui.pump(w, 0.8)
    x, y = w.winfo_rootx(), w.winfo_rooty() - (MENU_H if menu else 0)
    ImageGrab.grab(bbox=(x, y, x + w.winfo_width(), w.winfo_rooty() + w.winfo_height()),
                   all_screens=True).save(path)
    w.attributes("-topmost", False)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else ROOT / "docs" / "img"
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="gmr_shots_"))
    try:
        f = build(tmp)
        ui.OPERATOR = "Оператор"
        w = ui.setup(f.root)
        w.attributes("-alpha", 1.0)
        w.geometry("1100x720+40+40")
        ui.pump(w, 0.8)
        grab(w, out / "operator_main.png")

        x, y = w.winfo_rootx(), w.winfo_rooty() - MENU_H

        def shot_menu():               # меню на Windows держит свой цикл: снимок и Esc — по таймеру
            ImageGrab.grab(bbox=(x, y, x + 1100, y + 330), all_screens=True).save(out / "operator_menu.png")
            ctypes.windll.user32.keybd_event(0x1B, 0, 0, 0)
            ctypes.windll.user32.keybd_event(0x1B, 0, 2, 0)
        w.after(900, shot_menu)
        w._month_menu.post(x + 4, w.winfo_rooty())
        ui.pump(w, 0.5)

        from src.gmr.ui import run_dialog
        script = tmp / "fake_run.py"
        script.write_text(FAKE_RUN, encoding="utf-8")
        dlg = run_dialog.ProcessDialog(w, [sys.executable, str(script)], tmp / "stop", lambda c, lines: None)
        ui.pump(w, 5.0)
        grab(w, out / "operator_progress.png", over=dlg)
        ui.pump(w, 5.0)
        if dlg.winfo_exists():
            dlg.destroy()
        ui.pump(w, 0.5)

        import program2
        reason, _, path = program2.list_question_photos(w.results_dir)[0]
        w.open_edit_screen(path, reason)
        ui.pump(w, 1.0)
        scr = w._current_screen
        scr._reading_widget.set_digits("01862", None)
        scr._update_delta()
        scr._update_accept_state()
        grab(w, out / "operator_edit.png")
        w._show_lists()

        w._notebook.select(w._verify_tab)
        w._verify_tab.refresh()
        w._verify_tab._open()
        ui.pump(w, 1.0)
        grab(w, out / "operator_verify.png")
        w._show_lists()

        def shot_settings():
            d = next(t for t in w.winfo_children() if isinstance(t, program2.PresetDialog))
            d._tabs.select(1)
            ui.pump(w, 0.6)
            d.lift()
            d.attributes("-topmost", True)
            ui.pump(w, 0.6)
            ImageGrab.grab(bbox=(d.winfo_rootx(), d.winfo_rooty(), d.winfo_rootx() + d.winfo_width(),
                                 d.winfo_rooty() + d.winfo_height()), all_screens=True).save(
                out / "operator_settings.png")
            d.destroy()
        w.after(400, shot_settings)
        from src.gmr.domain.preset import RecognitionPreset
        w.ask_preset(w, RecognitionPreset.new_month(), "Настройки распознавания")

        w.show_summary()
        ui.pump(w, 0.8)
        top = [t for t in w.winfo_children() if t.winfo_class() == "Toplevel"][-1]
        grab(top, out / "operator_summary.png", menu=False)
        top.destroy()
        w.destroy()
        time.sleep(0.2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"Снимки: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
