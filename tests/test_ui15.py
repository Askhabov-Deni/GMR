"""1.5.0 — просьбы владельца 2026-11-22: двойной щелчок, перетаскивание, «Добавить фото»,

скрытое служебное в папке месяца, ход обработки."""

import sys

from types import SimpleNamespace



import pytest



from tests._window import has_display, new_window



needs_display = pytest.mark.skipif(not has_display(), reason="нет экрана (запускать под xvfb-run)")

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="только Windows")





@needs_display

def test_double_click_opens_only_on_row():

    # быстрый двойной щелчок по заголовку (сортировка) или пустому месту открывал первое фото

    import tkinter as tk

    from tkinter import ttk



    from src.gmr.operator.window import open_on_row

    root = new_window(tk.Tk)

    root.geometry("400x300+-3000+-3000")

    try:

        tree = ttk.Treeview(root, columns=("a",), show="headings")

        tree.heading("a", text="A")

        tree.pack(fill=tk.BOTH, expand=True)

        rows = [tree.insert("", "end", values=(i,)) for i in range(3)]

        tree.selection_set(rows[0])

        root.update()

        opened = []



        def click(x, y):

            open_on_row(tree, SimpleNamespace(x=x, y=y), lambda: opened.append(tree.selection()))

        click(20, 5)                                         # заголовок

        click(20, tree.winfo_height() - 5)                   # пусто под строками

        assert opened == []

        y = tree.bbox(rows[1])[1] + 2

        click(20, y)

        assert opened == [(rows[1],)]

    finally:

        root.destroy()





# ─── «Добавить фото»: копирование и объединение папок ────────────────────────



def _jpg(path, color=(1, 2, 3)):

    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (8, 8), color).save(path)

    return path





def test_find_groups_one_controller_folder_of_folders_and_loose(tmp_path):

    from src.gmr.application.add_photos import find_groups

    _jpg(tmp_path / "Аюб 2" / "a.jpg")

    _jpg(tmp_path / "Аюб 2" / "день 2" / "b.JPG")                  # вложенная папка — туда же

    (tmp_path / "Аюб 2" / "заметки.txt").write_text("x")

    g = find_groups([tmp_path / "Аюб 2"])

    assert [(x.name, [p.name for p in x.files]) for x in g] == [("Аюб 2", ["a.jpg", "b.JPG"])]

    _jpg(tmp_path / "июнь" / "Фатима Я" / "c.jpg")

    _jpg(tmp_path / "июнь" / "Аюб" / "d.png")

    (tmp_path / "июнь" / "пусто").mkdir()

    assert [(x.name, len(x.files)) for x in find_groups([tmp_path / "июнь"])] == [("Аюб", 1), ("Фатима Я", 1)]

    loose = find_groups([_jpg(tmp_path / "x.jpg"), tmp_path / "заметки.txt", tmp_path / "нет.jpg"])

    assert [(x.name, len(x.files)) for x in loose] == [("", 1)]

    assert find_groups([tmp_path / "июнь" / "пусто"]) == []





def test_add_photos_merges_into_existing_controller(tmp_path):

    from src.gmr.application.add_photos import PhotoGroup, add_photos, check_name, match_controller

    photos = tmp_path / "фото"

    _jpg(photos / "Аюб" / "a.jpg", (1, 1, 1))

    _jpg(photos / "Аюб" / "b.jpg", (2, 2, 2))

    src = tmp_path / "Аюб новые"

    a, b, c = _jpg(src / "a.jpg", (1, 1, 1)), _jpg(src / "b.jpg", (9, 9, 9)), _jpg(src / "c.jpg", (3, 3, 3))

    name = match_controller(" аюб ", ["Аюб", "Фатима Я"])

    assert name == "Аюб"

    rep = add_photos(photos, [PhotoGroup(name, [a, b, c])])

    assert (rep.copied, rep.same, rep.renamed, rep.new_folders) == ({"Аюб": 2}, 1, 1, [])

    assert sorted(p.name for p in (photos / "Аюб").iterdir()) == ["a.jpg", "b (2).jpg", "b.jpg", "c.jpg"]

    assert src.is_dir() and len(list(src.iterdir())) == 3                 # оригиналы на месте

    assert "Добавлено фото: 2" in rep.text() and "уже были в месяце: 1" in rep.text()

    again = add_photos(photos, [PhotoGroup("Аюб", [a, b, c])])            # второй раз — ничего нового

    assert again.total == 0 and again.same == 3 and again.text().startswith("Новых фото нет")

    new = add_photos(photos, [PhotoGroup("Магомед", [c])])

    assert new.new_folders == ["Магомед"] and (photos / "Магомед" / "c.jpg").is_file()

    assert check_name("") and check_name("a/b") and check_name("Аюб") is None

    with pytest.raises(ValueError):

        add_photos(photos, [PhotoGroup("", [c])])

    assert not list(photos.rglob("*.копируется"))





@needs_display

def test_window_add_photos(tmp_path, monkeypatch):

    import program2

    from src.gmr.operator import window as op_window

    from tests._month import make_month

    from tests.test_program2_processing import _window

    f = make_month(tmp_path, [("1234567", "A1", "1000", "")])

    (f.photos / "Аюб").mkdir(parents=True)

    src = tmp_path / "с телефона" / "аюб"

    _jpg(src / "a.jpg")

    offered = []



    def pick(window, existing, paths=None):

        from src.gmr.application.add_photos import find_groups, match_controller

        offered.append((existing, paths))

        groups = find_groups(paths)

        groups[0].name = match_controller(groups[0].name, existing)

        return groups

    monkeypatch.setattr(op_window, "pick_photos", pick)

    w = _window(monkeypatch, program2.AppSettings(operator_name="Оператор", month_dir=str(f.root)))

    try:

        w._on_drop([str(src)])                                       # папку бросили на окно

        assert offered == [(["Аюб"], [str(src)])]

        assert (f.photos / "Аюб" / "a.jpg").is_file()

        assert "Добавлено фото: 1" in w.toast.text

        w._on_drop([str(tmp_path / "Июнь.xlsx")])                    # таблица на списках — подсказка

        assert "Новый месяц" in w.toast.text and len(offered) == 1

        w._processing = True

        assert w.add_photos([str(src)]) is None and len(offered) == 1

    finally:

        w._processing = False

        w.destroy()





# ─── Поле «перетащите сюда» и перетаскивание из проводника ──────────────────



@needs_display

def test_table_drop_panel(tmp_path):

    import tkinter as tk



    from src.gmr.operator.drop_panel import TableDropPanel

    root = new_window(tk.Tk)

    root.geometry("900x700+-3000+-3000")

    try:

        table = tmp_path / "Июнь.xlsx"

        table.write_bytes(b"x")

        p = TableDropPanel(root, "Новый месяц", "таблица")

        seen = []



        def drop():

            p.dropped([str(tmp_path / "фото.jpg")])                 # не таблица — красная строка

            seen.append((p._error.cget("text"), p._done.get()))

            p.dropped([str(table)])

        root.after(50, drop)

        assert p.wait() == str(table) and root._drop_target is None

        assert "Нужна одна таблица" in seen[0][0] and not seen[0][1]

        p2 = TableDropPanel(root, "Новый месяц", "таблица")

        root.after(80, p2.cancel)                                   # «Отмена» / Esc

        assert p2.wait() is None

    finally:

        root.destroy()





@needs_display

def test_photo_drop_panel_asks_whose_photos(tmp_path):

    import tkinter as tk



    from src.gmr.operator.drop_panel import PhotoDropPanel

    _jpg(tmp_path / "IMG" / "a.jpg")

    root = new_window(tk.Tk)

    root.geometry("900x700+-3000+-3000")

    try:

        p = PhotoDropPanel(root, ["Аюб", "Фатима Я"])

        notes = []



        def steps():

            p.dropped([str(tmp_path / "IMG")])

            notes.append(p._note.cget("text"))

            p._name_var.set("фатима я")

            notes.append(p._note.cget("text"))

            p.confirm()

        root.after(50, steps)

        groups = p.wait()

        assert [(g.name, len(g.files)) for g in groups] == [("Фатима Я", 1)]

        assert "новая папка" in notes[0] and "уже есть" in notes[1]

    finally:

        root.destroy()





@windows_only
def test_drop_from_explorer_while_window_runs():
    # 1.5.0 в офисе: перетаскивание закрывало программу — вызов Tk из оконной
    # процедуры. Отдельный процесс (tests/_drop_probe.py): падение не роняет тесты.
    import os
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, str(root / "tests" / "_drop_probe.py"), str(root)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})     # вывод — не в cp1251
    assert r.returncode == 0, r.stderr[-800:]
    assert "GOT [['C:/Рабочий стол/реестр.xlsx', 'папка'], ['второе']]" in r.stdout, r.stdout


# ─── Служебное в папке месяца — скрытое ─────────────────────────────────────



@windows_only

def test_month_service_files_hidden_and_still_writable(tmp_path):

    from src.gmr.application import month as month_app

    from src.gmr.application.operator import OperatorSession

    from src.gmr.storage.hidden import is_hidden

    from tests._month import make_month

    f = make_month(tmp_path, [("1234567", "A1", "1000", "")])

    month_app.export_month(str(f.root))

    month_app.month_summary(str(f.root))

    for name in ("gmr.sqlite", "таблицы", "лог.csv", "итог_месяца.txt", "без_показаний.csv"):

        assert is_hidden(f.root / name), name

    assert not is_hidden(f.export_xlsx)

    month_app.export_month(str(f.root))                             # перезапись скрытых — без «Отказано в доступе»

    month_app.month_summary(str(f.root))

    assert is_hidden(f.export_log) and is_hidden(f.root / "итог_месяца.txt")

    from src.gmr.storage.hidden import unhide

    f.results.mkdir(exist_ok=True)

    unhide(f.results)                                               # старый месяц: служебное было видно

    f.hide_service()

    assert is_hidden(f.root / "результат")

    OperatorSession(str(f.root)).close()                            # база скрыта — открывается





# ─── Ход обработки ──────────────────────────────────────────────────────────



def test_clock_text():

    from src.gmr.operator.process_screen import clock_text

    assert (clock_text(42), clock_text(725), clock_text(3753)) == ("0:42", "12:05", "1:02:33")

