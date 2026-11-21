"""
Списки «Очередь» и «Проверка»: сортировка щелчком по заголовку столбца (как в
проводнике Windows) и цветные стрелки расхода (просьба владельца 2026-11-22).

Первый щелчок по числу или дате — от больших к меньшим, по тексту — по
алфавиту; второй щелчок — наоборот. Стрелка у заголовка показывает, по чему и
куда отсортировано. Порядок списка — и порядок «следующего фото» на экране
разбора: после «Верно» открывается следующее в том же порядке.
"""
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from .theme import DANGER, SUCCESS

UP, DOWN = " ▲", " ▼"


def number(text) -> Optional[float]:
    """«+1 234», «12,5», «00123» → число; пусто или не число — None."""
    s = str(text or "").replace("−", "-").replace(" ", "").replace(" ", "").replace(",", ".").lstrip("+")
    try:
        return float(s)
    except ValueError:
        return None


def sort_key(kind: str) -> Callable:
    """Ключ сортировки: пустые — всегда в конце (см. sort_items)."""
    if kind == "number":
        return number
    return lambda v: str(v or "").casefold() or None


def sort_items(items: list, value: Callable, kind: str, descending: bool) -> list:
    """Сортировка списка; строки без значения — в конце при любом направлении."""
    key = sort_key(kind)
    have = [(key(value(it)), i, it) for i, it in enumerate(items)]
    full = [h for h in have if h[0] is not None]
    empty = [h[2] for h in have if h[0] is None]
    full.sort(key=lambda h: (h[0], h[1]), reverse=descending)
    return [h[2] for h in full] + empty


class SortableTable:
    """Сортировка Treeview по заголовкам. columns: [(ключ, заголовок, вид)],
    вид — "number" | "date" | "text". on_change() — список надо перерисовать."""

    def __init__(self, tree: ttk.Treeview, columns: list[tuple[str, str, str]], on_change: Callable[[], None]):
        self.tree = tree
        self.titles = {key: title for key, title, _ in columns}
        self.kinds = {key: kind for key, _, kind in columns}
        self.on_change = on_change
        self.column: Optional[str] = None      # None — порядок программы
        self.descending = False
        for key, title, _ in columns:
            tree.heading(key, text=title, command=lambda k=key: self.click(k))

    def click(self, key: str) -> None:
        if key == self.column:
            self.descending = not self.descending
        else:
            self.column = key
            self.descending = self.kinds[key] in ("number", "date")
        self._titles()
        self.on_change()

    def _titles(self) -> None:
        for key, title in self.titles.items():
            mark = (DOWN if self.descending else UP) if key == self.column else ""
            self.tree.heading(key, text=title + mark)

    def order(self, items: list, value: Callable[[object, str], object]) -> list:
        """items в выбранном порядке; value(элемент, ключ столбца) → значение ячейки."""
        if self.column is None:
            return list(items)
        col = self.column
        return sort_items(items, lambda it: value(it, col), self.kinds[col], self.descending)


def arrow_images(master) -> dict[str, tk.PhotoImage]:
    """Стрелки расхода для списка: зелёная вверх — больше прошлого, красная вниз — меньше."""
    return {"PLUS": _triangle(master, SUCCESS, up=True), "MINUS": _triangle(master, DANGER, up=False)}


def _triangle(master, color: str, up: bool, size: int = 13) -> tk.PhotoImage:
    img = tk.PhotoImage(master=master, width=size, height=size)
    top, bottom = 2, size - 2
    for y in range(top, bottom):
        t = (y - top) / max(1, bottom - top - 1)          # 0 — острие, 1 — основание
        if not up:
            t = 1 - t
        half = t * (size / 2 - 1)
        x0, x1 = round(size / 2 - half - 0.5), round(size / 2 + half - 0.5)
        img.put(color, to=(max(0, x0), y, min(size, x1 + 1), y + 1))
    return img
