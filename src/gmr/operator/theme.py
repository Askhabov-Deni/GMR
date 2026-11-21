"""
Внешний вид окна «GMR»: цвета, шрифты, стили ttk, кнопки действий,
горячие клавиши в обеих раскладках.

Спокойная светлая тема для работы целый день: белые карточки на светло-сером
фоне, тёмная шапка, одна главная кнопка на экране (зелёная «Принять» / синяя
«Обработать»), остальные — белые.
"""
import tkinter as tk
from pathlib import Path
from tkinter import ttk

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
PHOTO_BG = "#2b2f36"      # под фото

# фон, фон под мышью, текст
BUTTON_KINDS = {
    "primary":   (SUCCESS, SUCCESS_DARK, "#ffffff"),
    "accent":    (ACCENT, ACCENT_DARK, "#ffffff"),
    "secondary": (SURFACE, "#e8edf3", TEXT),
    "danger":    (SURFACE, "#fdecec", DANGER),
}

# Буква горячей клавиши → та же клавиша в русской раскладке (D — «в», …):
# без этого D/N/B/E при русской раскладке не срабатывали
_CYRILLIC = {"d": "ve", "n": "te", "b": "i", "e": "u"}


def key_sequences(letter: str) -> list[str]:
    """«d» → <d>, <D>, <Cyrillic_ve>, <Cyrillic_VE>."""
    seqs = [f"<{letter.lower()}>", f"<{letter.upper()}>"]
    cyr = _CYRILLIC.get(letter.lower())
    if cyr:
        seqs += [f"<Cyrillic_{cyr}>", f"<Cyrillic_{cyr.upper()}>"]
    return seqs


def apply_theme(root: tk.Misc) -> None:
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

    style.configure("TButton", padding=(12, 6), background=SURFACE, bordercolor=BORDER,
                    lightcolor=SURFACE, darkcolor=SURFACE, focuscolor=SELECT)
    style.map("TButton", background=[("active", "#e8edf3")])
    style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff", bordercolor=ACCENT,
                    lightcolor=ACCENT, darkcolor=ACCENT, font=(FONT, 10, "bold"))
    style.map("Accent.TButton", background=[("active", ACCENT_DARK)])
    style.configure("Header.TButton", background=HEADER, foreground=HEADER_TEXT, bordercolor="#3a4a5c",
                    lightcolor=HEADER, darkcolor=HEADER)
    style.map("Header.TButton", background=[("active", "#2d3f53")])

    style.configure("TEntry", padding=6, fieldbackground=SURFACE, bordercolor=BORDER,
                    lightcolor=BORDER, darkcolor=BORDER)
    style.map("TEntry", bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)])

    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", padding=(18, 8), background="#dde3ea", bordercolor=BORDER,
                    lightcolor="#dde3ea")
    style.map("TNotebook.Tab", background=[("selected", SURFACE)], lightcolor=[("selected", SURFACE)])

    style.configure("Treeview", rowheight=30, background=SURFACE, fieldbackground=SURFACE,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
    style.configure("Treeview.Heading", font=(FONT, 10, "bold"), background="#e4e9ef",
                    bordercolor=BORDER, lightcolor="#e4e9ef", darkcolor="#e4e9ef", padding=(8, 6))
    style.map("Treeview", background=[("selected", SELECT)], foreground=[("selected", TEXT)])
    style.configure("TLabelframe", background=SURFACE, bordercolor=BORDER)
    style.configure("TLabelframe.Label", background=SURFACE, foreground=MUTED, font=(FONT, 9, "bold"))
    style.configure("Vertical.TScrollbar", background="#dde3ea", troughcolor=BG, bordercolor=BG,
                    arrowcolor=MUTED, lightcolor="#dde3ea", darkcolor="#dde3ea")


def set_window_icon(root: tk.Misc) -> None:
    """Свой значок в заголовке окна и на панели задач (вместо пёрышка Tk)."""
    try:
        img = tk.PhotoImage(master=root, file=str(ICON_PNG))
    except tk.TclError:          # файла нет — остаётся значок Tk
        return
    root.iconphoto(True, img)
    root._gmr_icon = img         # картинку держит окно — иначе её уберёт сборщик мусора


def logo_image(root: tk.Misc, size: int) -> "tk.PhotoImage | None":
    """Значок крупно — для экрана входа (gmr.png 128×128; меньше — в целое число раз)."""
    try:
        img = tk.PhotoImage(master=root, file=str(ICON_PNG))
    except tk.TclError:
        return None
    k = max(1, round(img.width() / size))
    return img.subsample(k, k) if k > 1 else img


def action_button(parent, text: str, command, kind: str = "secondary", **kw) -> tk.Button:
    """Плоская кнопка действия с подсветкой под мышью."""
    bg, hover, fg = BUTTON_KINDS[kind]
    strong = kind in ("primary", "accent")
    opts = dict(bg=bg, fg=fg, activebackground=hover, activeforeground=fg,
                disabledforeground="#cfe3d0" if strong else "#b8c4cf", relief="flat", bd=0,
                highlightthickness=0 if strong else 1, highlightbackground=BORDER,
                font=(FONT, 11 if strong else 10, "bold" if strong else "normal"),
                padx=16 if strong else 12, pady=9, cursor="hand2")
    opts.update(kw)
    b = tk.Button(parent, text=text, command=command, **opts)
    b._bg, b._off_bg = opts["bg"], "#a9c8ab" if kind == "primary" else opts["bg"]
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
    f = ttk.Frame(parent, style="Card.TFrame", padding=16)
    if pack:
        f.pack(**pack)
    return f
