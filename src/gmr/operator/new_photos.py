"""
Сколько новых фото ждут обработки — значок у кнопки «Обработать новые фото»
(2026-11-22, просьба владельца: «добавились фото — чтобы было видно, сколько»).

Новое фото — файл .jpg/.png в папке контролёра (<месяц>/фото/<контролёр>/),
которого нет в логе месяца: ни по имени в этой папке, ни по отпечатку
содержимого (то же фото под другим именем программа читать не станет).

Не тормозит окно:
  - папки просматриваются раз в 10 секунд и в отдельном потоке — окно в это
    время работает как обычно;
  - просмотр папки — только список файлов (без чтения фото);
  - отпечаток (SHA-256 файла) считается только у файлов с незнакомым именем
    и один раз: дальше — из памяти, пока файл не изменился;
  - пока идёт обработка фото, проверка не запускается.
Точное число «к чтению» считает сама обработка; здесь — то же правило без
моделей, поэтому числа совпадают, кроме редкого: фото, которое уже лежит в
очереди с ошибкой, программа без «Прочитать заново» не перечитывает.
"""
import os
import threading
from pathlib import Path
from typing import Callable, Optional

from src.gmr.storage.fingerprint import photo_fingerprint

PHOTO_EXTS = (".jpg", ".jpeg", ".png")
INTERVAL_MS = 10_000


def known_photos(log_rows: list[dict]) -> tuple[set, set, set]:
    """Что программа уже видела: (папка, имя), имена без папки (старые строки
    лога), отпечатки."""
    by_folder, names, hashes = set(), set(), set()
    for r in log_rows:
        name = r.get("original_filename") or ""
        if name:
            folder = r.get("source_folder") or ""
            by_folder.add((folder, name))
            if not folder:
                names.add(name)
        if r.get("photo_hash"):
            hashes.add(r["photo_hash"])
    return by_folder, names, hashes


def count_new_photos(photos_dir: Path, known: tuple[set, set, set], cache: dict) -> int:
    """Сколько фото в папках контролёров ещё не читались. cache — отпечатки
    {(путь, размер, время): отпечаток}, переживает вызовы."""
    by_folder, names, hashes = known
    n = 0
    try:
        folders = [d for d in os.scandir(photos_dir) if d.is_dir()]
    except OSError:                       # папки нет или недоступна — новых нет
        return 0
    for d in folders:
        try:
            entries = list(os.scandir(d.path))
        except OSError:
            continue
        for e in entries:
            if not (e.is_file() and e.name.lower().endswith(PHOTO_EXTS)):
                continue
            if (d.name, e.name) in by_folder or e.name in names:
                continue
            try:
                st = e.stat()
            except OSError:
                continue
            key = (e.path, st.st_size, st.st_mtime_ns)
            if key not in cache:
                try:
                    cache[key] = photo_fingerprint(e.path)
                except OSError:           # файл ещё копируется — посчитаем в следующий раз
                    continue
            if cache[key] not in hashes:
                n += 1
    return n


def new_photos_text(n: int) -> str:
    """1 → «1 новое фото», 3 → «3 новых фото», 21 → «21 новое фото»."""
    word = "новое" if n % 10 == 1 and n % 100 != 11 else "новых"
    return f"{n} {word} фото"


class NewPhotosWatcher:
    """Раз в INTERVAL_MS пересчитывает новые фото в фоне и отдаёт число в
    on_count(n) — в потоке окна. source() → (папка фото, строки лога) или None
    (месяц не открыт); busy() → True — идёт обработка, не проверять."""

    def __init__(self, window, source: Callable[[], Optional[tuple[Path, list]]],
                 on_count: Callable[[int], None], busy: Callable[[], bool] = lambda: False,
                 interval_ms: int = INTERVAL_MS):
        self.window, self.source, self.on_count, self.busy = window, source, on_count, busy
        self.interval_ms = interval_ms
        self._cache: dict = {}
        self._scan: Optional[tuple[threading.Thread, dict, int]] = None   # поток, его ответ, поколение
        self._generation = 0         # check_now — ответ прошлого просмотра уже устарел
        self._job = None
        self._stopped = False

    def start(self, delay_ms: int = 500) -> None:
        self._stopped = False
        self._schedule(delay_ms)

    def stop(self) -> None:
        self._stopped = True
        if self._job is not None:
            try:
                self.window.after_cancel(self._job)
            except Exception:
                pass
            self._job = None

    def check_now(self) -> None:
        """Пересчитать сразу (после обработки, смены месяца): ответ просмотра,
        начатого раньше, не показывается."""
        self._generation += 1
        self.stop()
        self.start(50)

    def _schedule(self, ms: int) -> None:
        if not self._stopped:
            self._job = self.window.after(ms, self._tick)

    def _tick(self) -> None:
        self._job = None
        if self._scan is not None:                   # прошлый просмотр
            thread, box, generation = self._scan
            if thread.is_alive():
                self._schedule(200)
                return
            self._scan = None
            if generation == self._generation:
                if box.get("n") is not None:
                    self.on_count(box["n"])
                self._schedule(self.interval_ms)
                return
        src = None if self.busy() else self.source()
        if src is None:
            self._schedule(self.interval_ms)
            return
        photos_dir, log_rows = src
        known = known_photos(log_rows)               # в потоке окна: лог меняет только окно
        box: dict = {}

        def scan():
            try:
                box["n"] = count_new_photos(photos_dir, known, self._cache)
            except Exception:                        # счётчик — подсказка: сбой не мешает работе
                box["n"] = None
        thread = threading.Thread(target=scan, name="new-photos", daemon=True)
        self._scan = (thread, box, self._generation)
        thread.start()
        self._schedule(100)


class NewPhotosBadge:
    """Значок «● 3 новых фото» в шапке рядом с «Обработать новые фото»: нет
    новых — не виден; число выросло — на секунду ярче. Щелчок — обработать."""

    COLOR, BRIGHT, TEXT = "#eb6834", "#ff8f5c", "#ffffff"

    def __init__(self, parent, bg: str, on_click: Callable[[], None]):
        import tkinter as tk
        self._tk = tk
        self.canvas = tk.Canvas(parent, height=30, width=10, bg=bg, highlightthickness=0, bd=0, cursor="hand2")
        self.canvas.bind("<Button-1>", lambda e: on_click())
        self.count = 0
        self._pack: Optional[dict] = None

    def pack(self, **kw):
        self._pack = kw              # показывается, только когда есть новые фото
        return self

    def set_count(self, n: int) -> None:
        grew = n > self.count
        self.count = n
        if n <= 0:
            self.canvas.pack_forget()
            return
        self._draw(self.BRIGHT if grew else self.COLOR)
        if not self.canvas.winfo_manager() and self._pack is not None:
            self.canvas.pack(**self._pack)
        if grew:
            self.canvas.after(1200, lambda: self.count == n and self._draw(self.COLOR))

    def _draw(self, color: str) -> None:
        c = self.canvas
        c.delete("all")
        t = c.create_text(0, 15, anchor="w", text="●  " + new_photos_text(self.count), fill=self.TEXT,
                          font=("Segoe UI", 10, "bold"))
        x0, y0, x1, y1 = c.bbox(t)
        w = x1 - x0 + 28
        c.configure(width=w)
        c.coords(t, 14, 15)
        h = 26
        r = h / 2
        pill = c.create_polygon([r, 2, w - r, 2, w, 2, w, 2 + r, w, 2 + h, w - r, 2 + h, r, 2 + h, 0, 2 + h, 0, 2 + r,
                                 0, 2], smooth=True, fill=color, outline="")
        c.tag_lower(pill, t)
