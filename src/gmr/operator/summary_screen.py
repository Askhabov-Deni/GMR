"""
Экран «Итог месяца» в окне — цифрами и полосами (2026-11-22, просьба
владельца: «сколько сделала программа, сколько человек, сколько проверено,
сколько ошибок — графиками, а не текстом»).

Текст итога никуда не делся: те же цифры — в итог_месяца.txt в папке месяца
(кнопка «Текстом» его открывает) и в `python gmr.py month <папка>`.

Цвета: у каждого «кто записал» свой цвет (программа — синий, оператор —
оранжевый, было в таблице — бирюзовый), «без показания» — серый остаток.
Подписи с числами — всегда рядом с полосой: цвет не единственная подсказка.
Наведите мышь на часть полосы — всплывёт её число и доля.
"""
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from src.gmr.application.month import SUMMARY_FILE, MonthStats
from src.gmr.application.quality import MIN_CHECKED

from . import theme
from .files import open_in_explorer
from .theme import action_button, BG, BORDER, DANGER, FONT, MUTED, px, SURFACE, TEXT

PROGRAM, OPERATOR, TABLE = "#2a78d6", "#eb6834", "#1baf7a"      # кто записал показание
REST = "#d3dae2"                                                 # без показания / не сделано
TRACK = "#e7ecf2"                                                # дорожка полос
SOURCE_COLORS = {"программа": PROGRAM, "оператор": OPERATOR, "было в таблице": TABLE}
BAR_H, GAP = 28, 2


def fmt(n) -> str:
    """1284 → «1 284»."""
    return f"{int(n):,}".replace(",", " ")


def pct(part, whole) -> str:
    return (f"{part / whole * 100:.0f}%" if whole else "—").replace(".", ",")


def pct1(x: float) -> str:
    return f"{x * 100:.1f}%".replace(".", ",")


def _round_rect(c: tk.Canvas, x0, y0, x1, y1, r=4, **kw):
    r = max(0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1,
           x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return c.create_polygon(pts, smooth=True, **kw)


class Chart(tk.Canvas):
    """Холст карточки: рисуется заново при смене ширины; подсказка под мышью.
    Рисунок — в размерах при масштабе 100% (ширина w — тоже), потом весь
    холст умножается на масштаб экрана (theme.px): шрифты Windows уже
    увеличила сама, вместе — как на обычном экране, только чётче."""

    def __init__(self, parent, height: int, draw: Callable[["Chart", float], None]):
        super().__init__(parent, height=px(height), bg=SURFACE, highlightthickness=0, bd=0)
        self._draw = draw
        self._tip = None
        self.bind("<Configure>", lambda e: self.redraw())

    def redraw(self):
        self.delete("all")
        w = self.winfo_width()
        if w > 20:
            k = theme.SCALE
            self._draw(self, w / k)
            if k != 1:
                self.scale("all", 0, 0, k, k)

    def tip(self, item, text: str):
        """Подсказка при наведении на часть рисунка."""
        self.tag_bind(item, "<Enter>", lambda e: self._show_tip(e, text))
        self.tag_bind(item, "<Motion>", lambda e: self._show_tip(e, text))
        self.tag_bind(item, "<Leave>", lambda e: self._hide_tip())

    def _show_tip(self, e, text):
        self._hide_tip()
        x = min(e.x + px(12), self.winfo_width() - px(8))       # подсказка — в пикселях экрана
        t = self.create_text(x, e.y - px(14), text=text, anchor="sw" if x == e.x + px(12) else "se",
                             font=(FONT, 9), fill="#ffffff")
        x0, y0, x1, y1 = self.bbox(t)
        box = _round_rect(self, x0 - px(6), y0 - px(3), x1 + px(6), y1 + px(3), r=px(4), fill="#1f2933",
                          outline="")
        self.tag_raise(t, box)
        self._tip = (box, t)

    def _hide_tip(self):
        if self._tip:
            for i in self._tip:
                self.delete(i)
            self._tip = None


def segmented_bar(c: Chart, w: int, y: int, parts: list[tuple[str, int, str]], total: Optional[int] = None):
    """Одна полоса из частей (название, число, цвет): ширина — доля; часть с
    нулём не рисуется; между частями — просвет."""
    total = total if total is not None else sum(n for _, n, _ in parts)
    if total <= 0:
        _round_rect(c, 0, y, w, y + BAR_H, fill=TRACK, outline="")
        return
    shown = [(name, n, color) for name, n, color in parts if n > 0]
    x = 0.0
    for i, (name, n, color) in enumerate(shown):
        x1 = w if i == len(shown) - 1 else x + max(3.0, n / total * w)
        item = _round_rect(c, x, y, max(x + 1, x1 - (GAP if i < len(shown) - 1 else 0)), y + BAR_H,
                           fill=color, outline="")
        c.tip(item, f"{name}: {fmt(n)} ({pct(n, total)})")
        x = x1


def text_end(c: Chart, item, x: float) -> float:
    """Где кончается текст, начатый в x (рисунок — до умножения на масштаб, а
    ширину шрифта Windows уже увеличила: делим её обратно)."""
    x0, _, x1, _ = c.bbox(item)
    return x + (x1 - x0) / theme.SCALE


def legend(c: Chart, y: int, parts: list[tuple[str, int, str]], total: int):
    """Подписи под полосой: квадратик цвета, название, число и доля."""
    x = 0
    for name, n, color in parts:
        _round_rect(c, x, y + 3, x + 12, y + 15, r=3, fill=color, outline="")
        t = c.create_text(x + 18, y, anchor="nw", text=name, font=(FONT, 10), fill=TEXT)
        x1 = text_end(c, t, x + 18)
        v = c.create_text(x1 + 6, y, anchor="nw", text=f"{fmt(n)}  ·  {pct(n, total)}", font=(FONT, 10, "bold"),
                          fill=TEXT)
        x = text_end(c, v, x1 + 6) + 28


def meter(c: Chart, w: int, y: int, part: int, whole: int, color: str, h: int = 12):
    """Шкала «сделано из»: заливка на светлой дорожке того же тона."""
    _round_rect(c, 0, y, w, y + h, r=h / 2, fill=TRACK, outline="")
    if whole and part:
        _round_rect(c, 0, y, max(h, part / whole * w), y + h, r=h / 2, fill=color, outline="")


def hbars(c: Chart, w: int, y: int, rows: list[tuple[str, int]], color: str, label_w: int = 250,
          step: int = 30) -> int:
    """Столбики по строкам (название слева, число справа). Возвращает высоту."""
    if not rows:
        return 0
    top = max(n for _, n in rows) or 1
    bar_w = max(40, w - label_w - 60)
    for i, (name, n) in enumerate(rows):
        yy = y + i * step
        c.create_text(0, yy + 9, anchor="w", text=name, font=(FONT, 10), fill=TEXT)
        item = _round_rect(c, label_w, yy + 2, label_w + max(4, n / top * bar_w), yy + 16, r=4, fill=color,
                           outline="")
        c.tip(item, f"{name}: {fmt(n)}")
        c.create_text(label_w + max(4, n / top * bar_w) + 8, yy + 9, anchor="w", text=fmt(n),
                      font=(FONT, 10, "bold"), fill=TEXT)
    return len(rows) * step


class SummaryScreen(ttk.Frame):
    """«Итог месяца»: плитки с главными числами и четыре карточки с полосами."""

    def __init__(self, app, stats: MonthStats, on_back: Callable[[], None]):
        super().__init__(app)
        self.app, self.st, self._on_back = app, stats, on_back
        self._build()
        self._bind_keys()

    # ── клавиши: Esc — назад (общие клавиши окна, как у экранов разбора) ──
    def _key_bindings(self):
        return [("<Escape>", lambda e: self._back()),
                ("<MouseWheel>", self._wheel)]           # прокрутка, если итог не влез в окно

    def _wheel(self, event):
        c = self._scroll_canvas
        if c.winfo_exists() and c.yview() != (0.0, 1.0):
            c.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _bind_keys(self):
        for seq, cb in self._key_bindings():
            self.app.bind_all(seq, cb)
        self.app._keys_owner = self

    def _unbind_keys(self):
        if getattr(self.app, "_keys_owner", None) is not self:
            return
        self.app._keys_owner = None
        for seq, _ in self._key_bindings():
            try:
                self.app.unbind_all(seq)
            except tk.TclError:
                pass

    def _back(self):
        self._unbind_keys()
        self._on_back()

    # ── экран ──
    def _build(self):
        st, q = self.st, self.st.quality
        head = ttk.Frame(self)
        head.pack(fill=tk.X, padx=px(20), pady=(px(16), px(8)))
        ttk.Label(head, text="Итог месяца", font=(FONT, 16, "bold")).pack(side=tk.LEFT)
        ttk.Label(head, text=f"   {st.root.name}", style="Muted.TLabel", font=(FONT, 12)).pack(side=tk.LEFT)
        action_button(head, "←  К спискам   Esc", self._back).pack(side=tk.RIGHT)
        action_button(head, "Без показаний (CSV)", lambda: self._open(st.without_list)).pack(side=tk.RIGHT, padx=px(8))
        action_button(head, "Текстом (TXT)", lambda: self._open(st.root / SUMMARY_FILE)).pack(side=tk.RIGHT)

        # всё ниже заголовка — с прокруткой: в маленьком окне (1100×680) итог выше окна
        c = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=c.yview)
        c.configure(yscrollcommand=vsb.set, yscrollincrement=20)
        body = ttk.Frame(c)
        win = c.create_window(0, 0, window=body, anchor="nw")

        def fit(_=None):
            c.configure(scrollregion=(0, 0, body.winfo_reqwidth(), body.winfo_reqheight()))
            if body.winfo_reqheight() > c.winfo_height() > 1:
                if not vsb.winfo_manager():
                    vsb.pack(side=tk.RIGHT, fill=tk.Y, before=c)
            elif vsb.winfo_manager():
                vsb.pack_forget()
                c.yview_moveto(0)
        body.bind("<Configure>", fit)
        c.bind("<Configure>", lambda e: (c.itemconfigure(win, width=e.width), fit()))
        c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._scroll_canvas = c

        tiles = ttk.Frame(body)
        tiles.pack(fill=tk.X, padx=px(14))
        rate = q.error_rate()
        waiting = sum(st.waiting.values())
        for i, (label, value, sub, color) in enumerate([
            ("Абонентов в таблице", fmt(st.abonents), f"ещё {fmt(st.not_in_table)} не в таблице"
             if st.not_in_table else "", TEXT),
            ("С показанием", fmt(st.with_reading), f"{pct(st.with_reading, st.abonents)} абонентов", TEXT),
            ("Без показания", fmt(st.without), "список — «Без показаний (CSV)»" if st.without else "все с показанием",
             TEXT),
            ("Ждут оператора", fmt(waiting), "фото в «Очереди»" if waiting else "очередь пуста", TEXT),
            ("Ошибки программы", pct1(rate[0]) if rate else "—",
             f"проверено {fmt(q.checked)}, исправлено {fmt(q.corrected)}" if rate
             else f"проверьте {MIN_CHECKED}–50 фото на «Проверке»", DANGER if rate and q.corrected else TEXT),
        ]):
            t = tk.Frame(tiles, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER, padx=px(16), pady=px(12))
            t.grid(row=0, column=i, sticky="nsew", padx=px(6), pady=px(6))
            tiles.columnconfigure(i, weight=1, uniform="tile")
            tk.Label(t, text=label, font=(FONT, 9), fg=MUTED, bg=SURFACE).pack(anchor="w")
            tk.Label(t, text=value, font=(FONT, 22, "bold"), fg=color, bg=SURFACE).pack(anchor="w")
            tk.Label(t, text=sub, font=(FONT, 9), fg=MUTED, bg=SURFACE, wraplength=px(165),
                     justify="left").pack(anchor="w")        # узкое окно — в две строки, не обрезать

        grid = ttk.Frame(body)
        grid.pack(fill=tk.BOTH, expand=True, padx=px(14), pady=(0, px(14)))
        for col in (0, 1):
            grid.columnconfigure(col, weight=1, uniform="card")
        # высоты — чтобы всё влезло на экран ноутбука 1366×768
        self._card(grid, 0, 0, "Кто записал показания", "из абонентов таблицы", 96, self._draw_readings)
        self._card(grid, 0, 1, "Фото месяца", "прочитала программа сама — или отдала оператору", 96,
                   self._draw_photos)
        self._card(grid, 1, 0, "Проверка программы", "показания, которые программа записала сама", 150,
                   self._draw_quality)
        self._card(grid, 1, 1, "Очередь оператора", "что оператор решил по фото, которые программа не прочла",
                   150, self._draw_queue)

    def _card(self, grid, row, col, title, sub, height, draw):
        f = tk.Frame(grid, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER, padx=px(18), pady=px(12))
        f.grid(row=row, column=col, sticky="nsew", padx=px(6), pady=px(6))
        tk.Label(f, text=title, font=(FONT, 12, "bold"), fg=TEXT, bg=SURFACE).pack(anchor="w")
        tk.Label(f, text=sub, font=(FONT, 9), fg=MUTED, bg=SURFACE).pack(anchor="w", pady=(0, px(8)))
        Chart(f, height, draw).pack(fill=tk.X)

    # ── рисунки ──
    def _draw_readings(self, c: Chart, w: int):
        st = self.st
        parts = [(name, n, SOURCE_COLORS.get(name, PROGRAM)) for name, n in st.by_source.items()]
        parts.append(("без показания", st.without, REST))
        segmented_bar(c, w, 4, parts, st.abonents)
        shown = [pt for pt in parts if pt[1] or pt[0] != "было в таблице"]    # «было в таблице 0» — шум
        legend(c, 44, shown[:2], st.abonents)
        legend(c, 70, shown[2:], st.abonents)

    def _draw_photos(self, c: Chart, w: int):
        q = self.st.quality
        if not q.photos:
            c.create_text(0, 12, anchor="w", text="Фото ещё не читались — «Обработать новые фото»",
                          font=(FONT, 10), fill=MUTED)
            return
        parts = [("программа сама", q.written, PROGRAM), ("оператору", q.to_operator, OPERATOR)]
        segmented_bar(c, w, 4, parts)
        legend(c, 44, parts, q.photos)
        c.create_text(0, 72, anchor="nw", font=(FONT, 9), fill=MUTED,
                      text=f"всего прочитано фото: {fmt(q.photos)} (каждое — один раз)")

    def _draw_quality(self, c: Chart, w: int):
        q = self.st.quality
        if not q.written:
            c.create_text(0, 12, anchor="w", text="Программа ещё ничего не записала сама", font=(FONT, 10),
                          fill=MUTED)
            return
        rate = q.error_rate()
        if rate is None:
            c.create_text(0, 0, anchor="nw", font=(FONT, 10), fill=MUTED, width=w,
                          text="Процент ошибок появится, когда оператор проверит фото на вкладке «Проверка» "
                               f"(хотя бы {MIN_CHECKED}–50: «Верно» или «Исправить»).")
        else:
            p, lo, hi = rate
            big = c.create_text(0, 0, anchor="nw", font=(FONT, 20, "bold"), fill=DANGER if q.corrected else TEXT,
                                text=pct1(p))
            x = text_end(c, big, 0) + 14
            c.create_text(x, 4, anchor="nw", font=(FONT, 10), fill=TEXT,
                          text=f"ошибок: исправлено {fmt(q.corrected)} из {fmt(q.checked)} проверенных")
            c.create_text(x, 23, anchor="nw", font=(FONT, 9), fill=MUTED,
                          text=f"вероятно от {pct1(lo)} до {pct1(hi)} — чем больше проверено, тем точнее")
            # шкала 0…top: розовая полоса — где вероятно, точка — сколько вышло
            top = max(0.10, hi * 1.25)
            y = 52
            _round_rect(c, 0, y, w, y + 10, r=5, fill=TRACK, outline="")
            band = _round_rect(c, lo / top * w, y, max(lo / top * w + 6, hi / top * w), y + 10, r=5,
                               fill="#f4b4b4", outline="")
            c.tip(band, f"вероятно от {pct1(lo)} до {pct1(hi)}")
            xp = p / top * w
            dot = c.create_oval(xp - 7, y - 2, xp + 7, y + 12, fill=DANGER, outline=SURFACE, width=2)
            c.tip(dot, f"вышло {pct1(p)}")
            c.create_text(0, y + 14, anchor="nw", text="0%", font=(FONT, 8), fill=MUTED)
            c.create_text(w, y + 14, anchor="ne", text=pct1(top).replace(",0%", "%"), font=(FONT, 8), fill=MUTED)
        y = 92
        c.create_text(0, y, anchor="nw", font=(FONT, 10), fill=TEXT,
                      text=f"Проверено на «Проверке»: {fmt(q.checked)} из {fmt(q.written)} ({pct(q.checked, q.written)})")
        meter(c, w, y + 22, q.checked, q.written, PROGRAM, h=10)
        if rate is not None and q.checked >= MIN_CHECKED:
            note = (f"≈ {fmt(round(rate[0] * q.written))} неверных среди {fmt(q.written)} записанных программой "
                    f"(от {fmt(round(rate[1] * q.written))} до {fmt(round(rate[2] * q.written))})")
        else:
            note = f"Для оценки по всему месяцу проверьте хотя бы {MIN_CHECKED} фото"
        c.create_text(0, y + 38, anchor="nw", text=note, font=(FONT, 9), fill=MUTED)

    def _draw_queue(self, c: Chart, w: int):
        q, waiting = self.st.quality, sum(self.st.waiting.values())
        rows = q.decided.most_common()
        if len(rows) > 4:                                # больше 4 строк не влезает — остальное «другое»
            rows = rows[:3] + [("другое", sum(n for _, n in rows[3:]))]
        done = sum(q.decided.values())
        if not rows and not waiting:
            c.create_text(0, 12, anchor="w", text="Оператору ничего не отдавалось", font=(FONT, 10), fill=MUTED)
            return
        head = (f"Разобрано {fmt(done)} из {fmt(done + waiting)}" + (f", ждут ещё {fmt(waiting)}" if waiting
                                                                     else " — очередь пуста ✓"))
        c.create_text(0, 0, anchor="nw", text=head, font=(FONT, 10), fill=TEXT)
        hbars(c, w, 24, rows, OPERATOR, label_w=min(260, int(w * 0.42)), step=26)
        if waiting:
            reasons = ", ".join(f"{r} {fmt(n)}" for r, n in sorted(self.st.waiting.items(), key=lambda kv: -kv[1]))
            c.create_text(0, 24 + 26 * len(rows) + 4, anchor="nw", width=w, font=(FONT, 9), fill=MUTED,
                          text=f"ждут: {reasons}")

    def _open(self, path):
        if path.is_file():
            open_in_explorer(path)
