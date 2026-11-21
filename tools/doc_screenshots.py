"""
tools/doc_screenshots.py — снимки окна «GMR» для docs/OPERATOR.md на
ВЫДУМАННЫХ данных: нарисованные счётчики, вымышленные номера и счета
(docs/img/ — в git, tools/check.py их пропускает).

  python -m tools.doc_screenshots [<папка снимков, по умолчанию docs/img>]

С 1.3.0 окно стоит за краем экрана и снимается через Windows (PrintWindow):
на рабочем столе ничего не появляется и фокус не уходит — можно снимать,
пока на компьютере работают. Так не снять меню «Месяц» (всплывающее окно)
и диалоги — настройки распознавания и полосу обработки (Windows не отдаёт
их содержимое за краем экрана): operator_menu.png, operator_settings.png и
operator_progress.png остаются прежними. Меню заново — с --menu (окно на
несколько секунд появится на экране). Месяц «Октябрь_2026»
создаётся во временной папке и потом удаляется.
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
import ctypes.wintypes as wt  # noqa: E402

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


OFF = "+-6000+-6000"                    # за краем экрана


class _BIH(ctypes.Structure):
    _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG), ("biPlanes", wt.WORD),
                ("biBitCount", wt.WORD), ("biCompression", wt.DWORD), ("biSizeImage", wt.DWORD),
                ("biXPelsPerMeter", wt.LONG), ("biYPelsPerMeter", wt.LONG), ("biClrUsed", wt.DWORD),
                ("biClrImportant", wt.DWORD)]


def window_image(w) -> tuple[Image.Image, tuple[int, int]]:
    """Окно целиком (с заголовком) через PrintWindow — и где его левый верхний угол."""
    u, gdi = ctypes.windll.user32, ctypes.windll.gdi32
    hwnd = u.GetParent(w.winfo_id())
    r = wt.RECT()
    u.GetWindowRect(hwnd, ctypes.byref(r))
    ww, hh = r.right - r.left, r.bottom - r.top
    hdc = u.GetWindowDC(hwnd)
    mem = gdi.CreateCompatibleDC(hdc)
    bmp = gdi.CreateCompatibleBitmap(hdc, ww, hh)
    gdi.SelectObject(mem, bmp)
    u.PrintWindow(hwnd, mem, 2)                       # 2 — PW_RENDERFULLCONTENT
    buf = ctypes.create_string_buffer(ww * hh * 4)
    bih = _BIH(ctypes.sizeof(_BIH), ww, -hh, 1, 32, 0, 0, 0, 0, 0, 0)
    gdi.GetDIBits(mem, bmp, 0, hh, buf, ctypes.byref(bih), 0)
    gdi.DeleteObject(bmp)
    gdi.DeleteDC(mem)
    u.ReleaseDC(hwnd, hdc)
    return Image.frombuffer("RGBA", (ww, hh), buf, "raw", "BGRA", 0, 1).convert("RGB"), (r.left, r.top)


def settle(w, sec=1.5):
    """Окно за краем экрана рисуется не сразу — дать ему дорисоваться."""
    w.update_idletasks()
    ui.pump(w, sec)


def grab(w, path, menu=True, over=None):
    """Окно без заголовка (со строкой меню — menu=True); over — диалог поверх него."""
    settle(w)
    img, (left, top) = window_image(w)
    if over is not None:
        settle(over, 0.5)
        dimg, (dl, dt) = window_image(over)
        img.paste(dimg, (dl - left, dt - top))
    x0, y0 = w.winfo_rootx() - left, w.winfo_rooty() - top - (MENU_H if menu else 0)
    img.crop((x0, y0, x0 + w.winfo_width(), w.winfo_rooty() - top + w.winfo_height())).save(path)


def shot_menu_on_screen(w, out):
    """Меню «Месяц» — только на экране: окно на несколько секунд показывается."""
    w.geometry("1100x720+40+40")
    ui.pump(w, 0.8)
    x, y = w.winfo_rootx(), w.winfo_rooty() - MENU_H

    def shot():                   # меню на Windows держит свой цикл: снимок и Esc — по таймеру
        ImageGrab.grab(bbox=(x, y, x + 1100, y + 330), all_screens=True).save(out / "operator_menu.png")
        ctypes.windll.user32.keybd_event(0x1B, 0, 0, 0)
        ctypes.windll.user32.keybd_event(0x1B, 0, 2, 0)
    w.after(900, shot)
    w._month_menu.post(x + 4, w.winfo_rooty())
    ui.pump(w, 0.5)
    w.geometry("1100x720" + OFF)


def shot_welcome(w, out):
    """Экран входа — в отдельном окне того же размера (в главном он уже прошёл)."""
    import tkinter as tk

    import program2
    from src.gmr.operator.theme import BG
    from src.gmr.operator.welcome import WelcomeScreen
    top = tk.Toplevel(w, bg=BG)
    top.geometry("1100x620" + OFF)
    scr = WelcomeScreen(top)
    scr._take_focus = lambda: None           # не выводить окно вперёд: фокус — у того, кто работает
    top.after(100, lambda: (grab(top, out / "operator_welcome.png", menu=False), scr._continue()))
    scr.ask(program2.AppSettings("Оператор", "D:/GMR/Октябрь_2026"))
    top.destroy()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    with_menu = "--menu" in argv
    argv = [a for a in argv if a != "--menu"]
    out = Path(argv[0]) if argv else ROOT / "docs" / "img"
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="gmr_shots_"))
    try:
        f = build(tmp)
        ui.OPERATOR = "Оператор"
        w = ui.setup(f.root)
        w.attributes("-alpha", 1.0)
        w.geometry("1100x720" + OFF)
        # два новых фото в папке контролёра — значок «● 2 новых фото» у кнопки обработки
        meter(f.photos / "Контролёр 1" / "IMG-20261005-WA0001.jpg", "01530", "001000104")
        meter(f.photos / "Контролёр 1" / "IMG-20261005-WA0002.jpg", "01211", "001000105")
        w._watcher.check_now()
        settle(w, 2.0)
        grab(w, out / "operator_main.png")
        shot_welcome(w, out)
        if with_menu:
            shot_menu_on_screen(w, out)

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
        w._verify_tab._sort.click("delta")             # отсортировано по «Разнице»
        grab(w, out / "operator_verify_list.png")
        w._verify_tab._open()
        ui.pump(w, 1.0)
        grab(w, out / "operator_verify.png")
        w._show_lists()

        w.show_summary()
        grab(w, out / "operator_summary.png")
        w._current_screen._back()

        w.destroy()
        time.sleep(0.2)
    finally:
        import gc
        for _ in range(5):              # файлы месяца держат объекты окна, пока их не соберёт gc
            gc.collect()
            shutil.rmtree(tmp, ignore_errors=True)
            if not tmp.exists():
                break
            time.sleep(0.5)
        if tmp.exists():
            print(f"Не удалось удалить временную папку {tmp} — удалите вручную")
    print(f"Снимки: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
