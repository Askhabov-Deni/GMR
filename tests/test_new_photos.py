"""Счётчик новых фото у кнопки «Обработать новые фото» (src/gmr/operator/new_photos.py)."""
import time

import pytest

from src.gmr.operator.new_photos import (
    NewPhotosBadge,
    NewPhotosWatcher,
    count_new_photos,
    known_photos,
    new_photos_text,
)
from src.gmr.storage.fingerprint import photo_fingerprint

from tests._window import has_display, new_window

needs_display = pytest.mark.skipif(not has_display(), reason="нет экрана (запускать под xvfb-run)")


def _photo(path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_counts_only_photos_the_program_has_not_seen(tmp_path):
    photos = tmp_path / "фото"
    read = _photo(photos / "Аюб" / "a.jpg", b"a")
    _photo(photos / "Аюб" / "b.jpg", b"b")                      # новое
    _photo(photos / "Аюб" / "копия a.jpg", b"a")                # то же фото под другим именем — не новое
    _photo(photos / "Аюб" / "заметка.txt", b"x")                # не фото
    _photo(photos / "Сулиман" / "a.jpg", b"c")                  # то же имя, но у другого контролёра — новое
    _photo(photos / "Сулиман" / "old.PNG", b"d")                # в старой строке лога без папки — не новое
    log = [{"original_filename": "a.jpg", "source_folder": "Аюб", "photo_hash": photo_fingerprint(str(read))},
           {"original_filename": "old.PNG", "source_folder": "", "photo_hash": ""}]
    cache = {}
    assert count_new_photos(photos, known_photos(log), cache) == 2
    hashed = len(cache)
    assert count_new_photos(photos, known_photos(log), cache) == 2
    assert len(cache) == hashed                                 # отпечатки — один раз


def test_no_photos_folder_means_no_new(tmp_path):
    assert count_new_photos(tmp_path / "нет", known_photos([]), {}) == 0


@pytest.mark.parametrize("n, text", [(1, "1 новое фото"), (3, "3 новых фото"), (11, "11 новых фото"),
                                     (21, "21 новое фото"), (25, "25 новых фото")])
def test_text(n, text):
    assert new_photos_text(n) == text


@needs_display
def test_watcher_counts_in_background_and_badge_shows(tmp_path):
    import tkinter as tk
    photos = tmp_path / "фото"
    _photo(photos / "Аюб" / "a.jpg", b"a")
    root = new_window(tk.Tk)
    root.withdraw()
    try:
        bar = tk.Frame(root)
        bar.pack()
        badge = NewPhotosBadge(bar, "#000000", lambda: None).pack(side=tk.RIGHT)
        busy = [True]
        seen = []
        w = NewPhotosWatcher(root, lambda: (photos, []), lambda n: (seen.append(n), badge.set_count(n)),
                             busy=lambda: busy[0], interval_ms=50)
        w.start(10)

        def pump(until, sec=5.0):
            end = time.time() + sec
            while time.time() < end and not until():
                root.update()
                time.sleep(0.01)
        pump(lambda: False, 0.3)
        assert seen == []                                       # идёт обработка — не проверяет
        busy[0] = False
        pump(lambda: seen)
        assert seen[-1] == 1 and badge.canvas.winfo_manager() == "pack"
        _photo(photos / "Аюб" / "b.jpg", b"b")
        pump(lambda: seen[-1] == 2)
        assert seen[-1] == 2 and badge.count == 2
        badge.set_count(0)
        assert not badge.canvas.winfo_manager()                 # нет новых — значка нет
        # «проверить сейчас» (после обработки): ответ прошлого просмотра не показывается
        w.interval_ms = 60_000
        pump(lambda: w._scan is None, 2.0)
        _photo(photos / "Аюб" / "c.jpg", b"c")
        n_before = len(seen)
        w.check_now()
        pump(lambda: len(seen) > n_before)
        assert seen[n_before:] == [3]
        w.stop()
    finally:
        root.destroy()
