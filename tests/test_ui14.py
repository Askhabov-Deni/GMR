"""1.4.0 — внешний вид и удобство окна: масштаб экрана, фото, сообщения, клавиши."""
import pytest
from PIL import Image

from src.gmr.operator import theme
from src.gmr.operator.process_screen import minutes_text
from src.gmr.ui.run_dialog import report_numbers
from tests._window import has_display, new_window

needs_display = pytest.mark.skipif(not has_display(), reason="нет экрана (запускать под xvfb-run)")


class _Root:
    def __init__(self, dpi):
        self.dpi = dpi

    def winfo_fpixels(self, what):
        return self.dpi


def test_scale_follows_screen_dpi(monkeypatch):
    # масштаб Windows 125% — 120 точек на дюйм: размеры в пикселях ×1.25, иначе текст мыльный
    monkeypatch.setattr(theme, "SCALE", 1.0)
    assert theme.init_scale(_Root(120.1)) == 1.25 and theme.px(32) == 40 and theme.px(0) == 0
    assert theme.init_scale(_Root(96)) == 1.0 and theme.px(32) == 32
    assert theme.init_scale(_Root(72)) == 1.0                 # мельче 100% — не уменьшать


def test_ctrl_keys_in_both_layouts():
    assert theme.ctrl_sequences("e") == ["<Control-e>", "<Control-E>", "<Control-Cyrillic_u>", "<Control-Cyrillic_U>"]
    assert "<Control-Cyrillic_sha>" in theme.ctrl_sequences("i")


def test_report_numbers_from_real_report_format():
    text = ("=======\nОБЩИЙ ОТЧЁТ ПО ВСЕМ ПАПКАМ\n=======\nНовых фото в этом прогоне: 1082\n"
            "Тот же файл ещё раз (копия): 29\n\n  PLUS   429  39.6%\n-----\n"
            "  Успешно (PLUS+MINUS):       709    65.5%\n  На проверку (question/):    304    28.1%\n"
            "  Дубли (REPEAT):              69     6.4%\n")
    assert report_numbers(text) == {"new": 1082, "copies": 29, "written": 709, "queue": 304, "repeat": 69}
    assert report_numbers("ничего") == {}


def test_minutes_text():
    assert minutes_text(20) == "меньше минуты" and minutes_text(150) == "≈ 2 мин"
    assert minutes_text(3 * 3600 + 5 * 60) == "≈ 3 ч 05 мин"


@needs_display
def test_photo_zoom_keeps_point_under_mouse(tmp_path):
    import tkinter as tk

    from src.gmr.operator.widgets import PhotoView
    path = tmp_path / "f.jpg"
    Image.new("RGB", (800, 600), (120, 130, 140)).save(path)
    root = new_window(tk.Tk)
    root.geometry("400x300+-3000+-3000")
    try:
        v = PhotoView(root, str(path), width=400, height=300)
        v.pack(fill=tk.BOTH, expand=True)
        root.update()
        w, h = v.winfo_width(), v.winfo_height()
        s0 = v._scale(w, h)
        at = (w * 0.25, h * 0.25)
        before = (v._center[0] + (at[0] - w / 2) / s0, v._center[1] + (at[1] - h / 2) / s0)
        v.zoom_by(2.0, at=at)                                    # колёсико над точкой
        s1 = v._scale(w, h)
        after = (v._center[0] + (at[0] - w / 2) / s1, v._center[1] + (at[1] - h / 2) / s1)
        assert v.zoom == 2.0 and abs(before[0] - after[0]) < 1 and abs(before[1] - after[1]) < 1
        v.FULL_LIMIT = 1                                         # крупно — только видимая часть
        v.zoom_by(3.0)
        root.update()
        assert v._tk_img.width() < w * 1.1 and v._tk_img.height() < h * 1.1     # не всё фото ×6
        v.zoom_by(100)
        assert v.zoom == v.MAX_ZOOM
        v.reset()
        assert v.zoom == 1.0 and v._center == [400, 300]
        PhotoView(root, str(tmp_path / "нет.jpg")).render()      # нет файла — без ошибки
    finally:
        root.destroy()


@needs_display
def test_toast_shows_and_hides():
    import tkinter as tk
    root = new_window(tk.Tk)
    root.withdraw()
    try:
        t = theme.Toast(root)
        t.show("Готово", ms=50)
        assert t.frame is not None and t.text == "Готово"
        t.show("Ещё раз")                                       # второе — вместо первого
        assert len([c for c in root.winfo_children() if isinstance(c, tk.Frame)]) == 1
        t.hide()
        assert t.frame is None
    finally:
        root.destroy()


@needs_display
def test_keys_dialog_lists_all_sections(tmp_path):
    import tkinter as tk

    from src.gmr.operator.dialogs import KeysDialog
    root = new_window(tk.Tk)
    root.withdraw()
    try:
        d = KeysDialog(root, [("Везде", [("F1", "инструкция")]), ("Фото", [("+ / −", "масштаб")])])
        texts = []

        def walk(w):
            for c in w.winfo_children():
                if c.winfo_class() == "Label":
                    texts.append(c.cget("text"))
                walk(c)
        walk(d)
        assert {"ВЕЗДЕ", "ФОТО", "F1", "+ / −", "масштаб"} <= set(texts)
        d.destroy()
    finally:
        root.destroy()
