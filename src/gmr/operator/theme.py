"""
Внешний вид окна «GMR»: цвета, шрифты, стили ttk, кнопки действий,
горячие клавиши в обеих раскладках, масштаб экрана, сообщения внутри окна.

Спокойная светлая тема для работы целый день: белые карточки на светло-сером
фоне, тёмная шапка, одна главная кнопка на экране (зелёная «Принять» / синяя
«Обработать»), остальные — белые.

Масштаб экрана (1.4.0). На ноутбуке с масштабом Windows 125–150% программа,
которая не говорит Windows, что умеет в масштаб, рисуется мелко и
растягивается картинкой — текст и значки мыльные (так было до 1.4.0:
1920×1080 при 125% — окно рисовалось как на 1536×864). Теперь программа
объявляет себя «знающей масштаб» (enable_dpi_awareness — до первого окна):
шрифты Windows рисует чётко в настоящих пикселях, а размеры в пикселях
(отступы, высота строк, ширина панелей, рисунки итога) умножаются на масштаб
— px().
"""
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

APP_NAME = "GMR"                             # до 1.3.0 — «GMR Оператор»
ICON_PNG = Path(__file__).with_name("gmr.png")   # значок окна (tools/make_icon.py; в exe — packaging/gmr.ico)
MANUAL_NAME = "Инструкция оператора.html"     # рядом с exe (python gmr.py build, tools/manual_html.py)
FONT = "Segoe UI"

BG = "#eef1f5"            # фон окна
SURFACE = "#ffffff"       # карточки, таблицы
BORDER = "#d5dbe3"
TEXT = "#1f2933"
MUTED = "#66788a"
HEADER = "#1f2d3d"        # шапка
HEADER_TEXT = "#ffffff"
HEADER_MUTED = "#a9b6c4"
ACCENT = "#1f6feb"
ACCENT_DARK = "#1858bd"
SUCCESS = "#2e7d32"
SUCCESS_DARK = "#256628"
DANGER = "#c62828"
WARNING = "#b45309"
SELECT = "#dbe8fd"
ZEBRA = "#f6f8fb"
HOVER = "#e8edf3"
PHOTO_BG = "#2b2f36"      # под фото

# фон, фон под мышью, текст
BUTTON_KINDS = {
    "primary":   (SUCCESS, SUCCESS_DARK, "#ffffff"),
    "accent":    (ACCENT, ACCENT_DARK, "#ffffff"),
    "secondary": (SURFACE, HOVER, TEXT),
    "soft":      ("#edf1f6", "#dfe5ed", TEXT),        # второстепенная — на белой карточке
    "danger":    (SURFACE, "#fdecec", DANGER),
}

# Буква горячей клавиши → та же клавиша в русской раскладке (D — «в», …):
# без этого D/N/B/E при русской раскладке не срабатывали
_CYRILLIC = {"d": "ve", "n": "te", "b": "i", "e": "u", "i": "sha", "o": "shcha"}


def key_sequences(letter: str) -> list[str]:
    """«d» → <d>, <D>, <Cyrillic_ve>, <Cyrillic_VE>."""
    seqs = [f"<{letter.lower()}>", f"<{letter.upper()}>"]
    cyr = _CYRILLIC.get(letter.lower())
    if cyr:
        seqs += [f"<Cyrillic_{cyr}>", f"<Cyrillic_{cyr.upper()}>"]
    return seqs


def ctrl_sequences(letter: str) -> list[str]:
    """«e» → <Control-e>, <Control-E>, <Control-Cyrillic_u>, … (Ctrl+буква в обеих раскладках)."""
    return [s.replace("<", "<Control-", 1) for s in key_sequences(letter)]


# ─── Масштаб экрана ──────────────────────────────────────────────────────────

SCALE = 1.0               # во сколько раз пиксели экрана мельче обычных (125% — 1.25)


def enable_dpi_awareness() -> None:
    """Сказать Windows, что программа сама учитывает масштаб экрана — до
    первого окна. Без этого при масштабе 125% окно мыльное (рисуется мельче и
    растягивается). «Системный» режим: на втором мониторе с другим масштабом
    Windows подгонит окно сама."""
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError, ImportError):     # не Windows
        pass


def init_scale(root: tk.Misc) -> float:
    """Масштаб — по тому, сколько пикселей в дюйме у экрана (96 — 100%)."""
    global SCALE
    try:
        SCALE = max(1.0, round(root.winfo_fpixels("1i") / 96, 2))
    except tk.TclError:
        SCALE = 1.0
    return SCALE


def px(n: float) -> int:
    """Размер в пикселях при масштабе 100% → в пикселях этого экрана."""
    return int(round(n * SCALE))


# ─── Стили ───────────────────────────────────────────────────────────────────

def apply_theme(root: tk.Misc) -> None:
    init_scale(root)
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(bg=BG)
    style.configure(".", background=BG, foreground=TEXT, font=(FONT, 10))
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=SURFACE)
    style.configure("Header.TFrame", background=HEADER)

    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Muted.TLabel", foreground=MUTED)
    style.configure("Title.TLabel", font=(FONT, 12, "bold"))
    style.configure("Card.TLabel", background=SURFACE)
    style.configure("CardMuted.TLabel", background=SURFACE, foreground=MUTED, font=(FONT, 9))
    style.configure("CardTitle.TLabel", background=SURFACE, foreground=MUTED, font=(FONT, 9, "bold"))
    style.configure("CardBig.TLabel", background=SURFACE, font=(FONT, 13))
    style.configure("HeaderTitle.TLabel", background=HEADER, foreground=HEADER_TEXT, font=(FONT, 13, "bold"))
    style.configure("HeaderInfo.TLabel", background=HEADER, foreground=HEADER_MUTED, font=(FONT, 10))

    style.configure("TButton", padding=(px(12), px(6)), background=SURFACE, bordercolor=BORDER,
                    lightcolor=SURFACE, darkcolor=SURFACE, focuscolor=SELECT)
    style.map("TButton", background=[("active", HOVER)])
    style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff", bordercolor=ACCENT,
                    lightcolor=ACCENT, darkcolor=ACCENT, font=(FONT, 10, "bold"))
    style.map("Accent.TButton", background=[("active", ACCENT_DARK)])
    style.configure("Header.TButton", background=HEADER, foreground=HEADER_TEXT, bordercolor="#3a4a5c",
                    lightcolor=HEADER, darkcolor=HEADER, padding=(px(12), px(7)))
    style.map("Header.TButton", background=[("active", "#2d3f53"), ("disabled", HEADER)],
              foreground=[("disabled", "#5d6d7e")])

    style.configure("TEntry", padding=px(6), fieldbackground=SURFACE, bordercolor=BORDER,
                    lightcolor=BORDER, darkcolor=BORDER)
    style.map("TEntry", bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)])
    # флажки и переключатели: в «clam» они 9 пикселей — на экране 125% почти не видно
    for kind in ("TRadiobutton", "TCheckbutton"):
        style.configure(kind, background=BG, padding=(0, px(2)), indicatorsize=px(15),
                        indicatormargin=(0, 0, px(6), 0), indicatorbackground=SURFACE, indicatorforeground=ACCENT)
        style.map(kind, indicatorbackground=[("disabled", BG), ("pressed", SELECT)])
    style.configure("Card.TRadiobutton", background=SURFACE)
    style.configure("Card.TCheckbutton", background=SURFACE)

    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", padding=(px(18), px(8)), background="#dde3ea", bordercolor=BORDER,
                    lightcolor="#dde3ea", font=(FONT, 10))
    style.map("TNotebook.Tab", background=[("selected", SURFACE)], lightcolor=[("selected", SURFACE)])

    style.configure("Treeview", rowheight=px(32), background=SURFACE, fieldbackground=SURFACE,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, font=(FONT, 10))
    style.configure("Treeview.Heading", font=(FONT, 10, "bold"), background="#e4e9ef", foreground=TEXT,
                    bordercolor=BORDER, lightcolor="#e4e9ef", darkcolor="#e4e9ef", padding=(px(8), px(7)))
    style.map("Treeview.Heading", background=[("active", "#d7dee7")])       # заголовок — кнопка сортировки
    style.map("Treeview", background=[("selected", SELECT)], foreground=[("selected", TEXT)])
    style.configure("TLabelframe", background=SURFACE, bordercolor=BORDER)
    style.configure("TLabelframe.Label", background=SURFACE, foreground=MUTED, font=(FONT, 9, "bold"))
    for orient in ("Vertical", "Horizontal"):
        style.configure(f"{orient}.TScrollbar", background="#d3dae2", troughcolor=BG, bordercolor=BG,
                        arrowcolor=MUTED, lightcolor="#d3dae2", darkcolor="#d3dae2", arrowsize=px(13))
        style.map(f"{orient}.TScrollbar", background=[("active", "#b9c3ce")])
    style.configure("TProgressbar", background=ACCENT, troughcolor="#dde3ea", bordercolor="#dde3ea",
                    lightcolor=ACCENT, darkcolor=ACCENT, thickness=px(10))


# ─── Значок ──────────────────────────────────────────────────────────────────

def _icon_image(root: tk.Misc, size: int):
    """gmr.png нужного размера — сглаженно (PIL), а не через пиксель."""
    from PIL import Image, ImageTk
    with Image.open(ICON_PNG) as im:
        return ImageTk.PhotoImage(im.convert("RGBA").resize((size, size), Image.LANCZOS), master=root)


def set_window_icon(root: tk.Misc) -> None:
    """Свой значок в заголовке окна и на панели задач (вместо пёрышка Tk)."""
    try:
        images = [_icon_image(root, s) for s in (px(64), px(32), px(16))]
    except (OSError, tk.TclError):   # файла нет — остаётся значок Tk
        return
    root.iconphoto(True, *images)
    root._gmr_icon = images          # картинки держит окно — иначе их уберёт сборщик мусора


def logo_image(root: tk.Misc, size: int) -> "tk.PhotoImage | None":
    """Значок крупно — для экрана входа (size — при масштабе 100%)."""
    try:
        return _icon_image(root, px(size))
    except (OSError, tk.TclError):
        return None


# ─── Кнопки и карточки ───────────────────────────────────────────────────────

def action_button(parent, text: str, command, kind: str = "secondary", **kw) -> tk.Button:
    """Плоская кнопка действия с подсветкой под мышью."""
    bg, hover, fg = BUTTON_KINDS[kind]
    strong = kind in ("primary", "accent")
    opts = dict(bg=bg, fg=fg, activebackground=hover, activeforeground=fg,
                disabledforeground="#cfe3d0" if strong else "#b8c4cf", relief="flat", bd=0,
                highlightthickness=0 if strong or kind == "soft" else 1, highlightbackground=BORDER,
                font=(FONT, 11 if strong else 10, "bold" if strong else "normal"),
                padx=px(16 if strong else 12), pady=px(9), cursor="hand2")
    opts.update(kw)
    b = tk.Button(parent, text=text, command=command, **opts)
    b._bg, b._off_bg = opts["bg"], {"primary": "#a9c8ab", "accent": "#a7c2ef"}.get(kind, opts["bg"])
    if str(b["state"]) == "disabled":
        b.config(bg=b._off_bg)
    b.bind("<Enter>", lambda e: b.config(bg=hover) if str(b["state"]) != "disabled" else None)
    b.bind("<Leave>", lambda e: b.config(bg=b._bg if str(b["state"]) != "disabled" else b._off_bg))
    return b


def set_enabled(button: tk.Button, enabled: bool) -> None:
    """Включить или выключить кнопку действия (выключенная главная — бледная)."""
    button.config(state=tk.NORMAL if enabled else tk.DISABLED,
                  bg=getattr(button, "_bg", button["bg"]) if enabled else getattr(button, "_off_bg", button["bg"]))


def card(parent, **pack) -> ttk.Frame:
    """Белая карточка с отступами внутри."""
    f = ttk.Frame(parent, style="Card.TFrame", padding=px(16))
    if pack:
        f.pack(**pack)
    return f


def chip(parent, text: str, bg: str, fg: str) -> tk.Label:
    """Цветная метка: «Ошибка цифр», «Проверка», «На проверке: 12»."""
    return tk.Label(parent, text=text, bg=bg, fg=fg, font=(FONT, 10, "bold"), padx=px(10), pady=px(3))


# ─── Сообщение внутри окна ───────────────────────────────────────────────────

TOAST_KINDS = {"info": ("ℹ", "#8ab4f8"), "success": ("✓", "#81c995"), "warning": ("⚠", "#fdd663")}


class Toast:
    """Короткое сообщение внизу окна, само исчезает через несколько секунд —
    вместо окошка «Готово» с кнопкой ОК (1.4.0: окно не отвлекает лишним
    окном). Щелчок — закрыть сразу."""

    def __init__(self, window: tk.Misc):
        self.window = window
        self.frame: Optional[tk.Frame] = None
        self.text = ""
        self._job = None

    def show(self, text: str, kind: str = "success", ms: int = 4000) -> None:
        self.hide()
        icon, color = TOAST_KINDS.get(kind, TOAST_KINDS["info"])
        self.text = text
        f = tk.Frame(self.window, bg="#1f2933", padx=px(18), pady=px(10), cursor="hand2")
        tk.Label(f, text=icon, bg="#1f2933", fg=color, font=(FONT, 12, "bold")).pack(side=tk.LEFT, padx=(0, px(10)))
        tk.Label(f, text=text, bg="#1f2933", fg="#ffffff", font=(FONT, 10), justify="left",
                 wraplength=px(560)).pack(side=tk.LEFT)
        for w in (f, *f.winfo_children()):
            w.bind("<Button-1>", lambda e: self.hide())
        f.place(relx=0.5, rely=1.0, y=-px(28), anchor="s")
        f.lift()
        self.frame = f
        self._job = self.window.after(ms, self.hide)

    def hide(self) -> None:
        if self._job is not None:
            try:
                self.window.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None
        if self.frame is not None:
            try:
                self.frame.destroy()
            except tk.TclError:
                pass
            self.frame = None
