"""
tools/ui_scenarios.py — окно «GMR» с клавиатуры, как работает оператор
(перед выпуском версии, docs/BUILD.md).

  python -m tools.ui_scenarios <КОПИЯ папки месяца> [--photos <папка с новыми фото>] [--exe <GMR.exe>]

Месяц меняется (отметки, решения, обработка) — только копия! Окно настоящее
(Tk), но невидимое и за краем экрана: Enter, Esc, Del, стрелки, цифры идут в
виджет с фокусом, как с клавиатуры, мышь и клавиатура компьютера не трогаются.
Буквы-горячие клавиши (D, N, B, E) Tk в Windows не имитирует в чужой раскладке —
они вызываются через ту же привязку экрана, что срабатывает при нажатии.

Сценарии: «Проверка» (Enter / → / E / Esc, сколько осталось), исправление
показания, «Очередь» (Enter, Del, D, N, B, цифры с клавиатуры, Enter = «Принять»),
список под экраном, настройки (обычные и расширенные), «Справка», меню «Месяц»
(новый месяц, другой месяц, обновить таблицу, перечитать), «Обработать новые
фото» (в Python и через собранный exe), закрытие. В конце — «ИТОГ: N из M».
Нужны в месяце: фото в «Очереди» (ошибка цифр / подозрительно) и на «Проверке».
"""
import argparse
import ctypes
import shutil
import sqlite3
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import program2  # noqa: E402
from src.gmr.operator import edit_screen as op_edit  # noqa: E402
from src.gmr.operator import window as op_window  # noqa: E402
from src.gmr.operator.edit_screen import EditScreen  # noqa: E402
from src.gmr.operator.verify_screen import VerifyScreen  # noqa: E402

ASKED, TOLD, TEXTS, RESULTS = [], [], [], []
ANSWER = {"yes": True, "text": None}
# раскладка сейчас: буквы нажимаются той клавишей, что даёт эта раскладка
RUSSIAN = sys.platform == "win32" and (ctypes.windll.user32.GetKeyboardLayout(0) & 0xFFFF) == 0x0419
OPERATOR = "Сценарий"


class Run:
    def __init__(self, month: Path, photos: list[Path], exe: Path | None):
        self.month, self.photos, self.exe = month, photos, exe

    def take_photos(self, n: int) -> list[Path]:
        got, self.photos = self.photos[:n], self.photos[n:]
        return got


# ─── Окно и клавиши ──────────────────────────────────────────────────────────

def setup(month: Path):
    op_window.load_settings = lambda: program2.AppSettings(operator_name=OPERATOR, month_dir=str(month))
    op_window.save_settings = lambda s: None
    op_window.ask_login = lambda *a, **k: "continue"
    mb = program2.messagebox
    for n in ("showinfo", "showwarning", "showerror"):
        setattr(mb, n, lambda title, msg="", *a, _n=n, **k: TOLD.append((_n, title, msg)))
    mb.askyesno = lambda title, msg="", *a, **k: ASKED.append((title, msg)) or ANSWER["yes"]
    mb.askretrycancel = lambda *a, **k: False
    op_edit.ask_text = lambda parent, title, prompt, suggested="": TEXTS.append((title, prompt)) or ANSWER["text"]
    w = program2.MainWindow()
    w.state("normal")
    w.attributes("-alpha", 0.0)
    w.geometry("1280x800+-5000+-5000")
    pump(w, 0.5)
    return w


def pump(w, sec=0.15):
    end = time.time() + sec
    while time.time() < end:
        w.update()
        time.sleep(0.01)


def focus_widget(w):
    name = str(w.tk.call("focus", "-lastfor", w))
    try:
        return w.nametowidget(name) if int(w.tk.call("winfo", "exists", name)) else w
    except Exception:            # noqa: BLE001
        return w


def press(w, keysym, wait=0.25):
    """Нажать и отпустить клавишу — в виджет с фокусом, как с клавиатуры."""
    if len(keysym) == 1 and keysym.isalpha():
        return letter(w, keysym, wait)
    try:
        focus_widget(w).event_generate("<KeyPress>", when="tail", keysym=keysym)
        pump(w, 0.03)
    finally:
        focus_widget(w).event_generate("<KeyRelease>", when="tail", keysym=keysym)   # фокус мог смениться
    pump(w, wait)


def letter(w, key, wait=0.25):
    from src.gmr.operator.theme import key_sequences
    seqs = key_sequences(key)
    want = [q for q in seqs if ("Cyrillic" in q) == RUSSIAN] or seqs
    scr = getattr(w, "_current_screen", None)
    owner = getattr(w, "_keys_owner", None)
    if scr is None or owner is not scr:
        check(f"клавиша {key}: горячие клавиши у открытого экрана", False,
              f"экран {type(scr).__name__}, клавиши у {type(owner).__name__}")
        return
    ev = type("E", (), {"keysym": want[0], "time": 0, "widget": focus_widget(w), "char": ""})()
    dict(scr._key_bindings())[want[0]](ev)
    pump(w, wait)


def screens(w):
    return [c for c in w.winfo_children() if isinstance(c, (EditScreen, VerifyScreen)) and c.winfo_manager()]


def close_toplevels(w):
    for t in [t for t in w.winfo_children() if t.winfo_class() == "Toplevel"]:
        t.destroy()


def check(name, cond, detail=""):
    RESULTS.append((bool(cond), name, detail))
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""), flush=True)


def one_screen(w, kind, what):
    s = screens(w)
    check(f"{what}: один экран {kind.__name__}", len(s) == 1 and isinstance(s[0], kind),
          f"экранов {len(s)}: {[type(x).__name__ for x in s]}")
    return s[0] if s else None


def db(run, sql, *args):
    con = sqlite3.connect(f"file:{run.month / 'gmr.sqlite'}?mode=ro", uri=True)
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


def verified(run):
    return db(run, "select count(*) from processing_log where verified_by=?", OPERATOR)[0][0]


def queue_len(w):
    return len(program2.list_question_photos(w.results_dir))


def to_tab(w, tab):
    w._notebook.select(tab)
    tab.refresh()
    pump(w, 0.3)


# ─── Сценарии ────────────────────────────────────────────────────────────────

def s_verify(w, run):
    print("Проверка: Enter / → / E / Esc, сколько осталось")
    to_tab(w, w._verify_tab)
    n0 = len(w._verify_tab._items)
    first = w._verify_tab._items[0]["path"]
    press(w, "Return")
    scr = one_screen(w, VerifyScreen, "Enter в списке")
    check("открыто выбранное фото", scr and scr.item["path"] == first)
    check("«На проверке» = все фото", scr and scr.total == n0, f"{scr and scr.total} ≠ {n0}")
    press(w, "Return")                                         # «Верно»
    scr = one_screen(w, VerifyScreen, "Enter = «Верно»")
    check("отмечено ровно одно фото", verified(run) == 1, f"отмечено {verified(run)}")
    check("открыто следующее фото", scr and scr.item["path"] != first)
    check("«На проверке» — на одно меньше", scr and scr.total == n0 - 1, f"{scr and scr.total}")
    check("вкладка — тоже", w._notebook.tab(w._verify_tab, "text") == f"Проверка  {n0 - 1}",
          w._notebook.tab(w._verify_tab, "text"))
    for _ in range(3):
        press(w, "Return")
    scr = one_screen(w, VerifyScreen, "4 × Enter")
    check("4 × Enter — 4 отметки, «На проверке» −4", verified(run) == 4 and scr.total == n0 - 4,
          f"отмечено {verified(run)}, на проверке {scr.total}")
    before = scr
    press(w, "Right")                                          # «Пропустить»
    scr = one_screen(w, VerifyScreen, "→ = «Пропустить»")
    check("→ не отмечает и не уменьшает", verified(run) == 4 and scr.item["path"] != before.item["path"]
          and scr.total == n0 - 4)
    press(w, "e")
    check("E — режим исправления", scr._edit_mode)
    press(w, "Escape")
    check("Esc при исправлении — отменить исправление (1.4.0)", screens(w) and not scr._edit_mode)
    press(w, "Escape")
    check("Esc — к списку", not screens(w) and not w.screen_open())
    check("фокус — в списке", focus_widget(w) is w._verify_tab._tree)
    check("в списке на 4 фото меньше", len(w._verify_tab._items) == n0 - 4)
    press(w, "Return")
    one_screen(w, VerifyScreen, "Enter после возврата")
    press(w, "Escape")


def s_verify_edit(w, run):
    print("Проверка: исправить показание")
    to_tab(w, w._verify_tab)
    press(w, "Return")
    scr = screens(w)[0]
    row = scr.item["log_row"]
    press(w, "e")
    new = f"{(int(row['reading']) + 1):05d}"
    scr._reading_widget.set_digits(new, None)
    press(w, "Return")                                         # «Сохранить исправление»
    one_screen(w, VerifyScreen, "сохранить исправление")
    val = db(run, "select value from readings where account=?", row["account_id"])
    check("исправленное показание в базе", val and val[0][0] == str(int(new)), f"{val} ≠ {int(new)}")
    press(w, "Escape")


def s_keys_14(w, run):
    print("1.4.0: масштаб фото + − 0, ← на «Проверке», Esc отменяет исправление, F5")
    to_tab(w, w._verify_tab)
    press(w, "Return")
    scr = screens(w)[0]
    press(w, "plus")
    check("+ — фото крупнее", scr._canvas.zoom > 1, f"{scr._canvas.zoom}")
    press(w, "minus")
    press(w, "plus")
    press(w, "0")
    check("0 — фото целиком", scr._canvas.zoom == 1, f"{scr._canvas.zoom}")
    i0 = scr.index
    press(w, "Right")
    scr = screens(w)[0]
    press(w, "Left")
    scr = screens(w)[0]
    check("→ и ← — следующее и предыдущее фото", scr.index == i0, f"{scr.index} ≠ {i0}")
    press(w, "e")
    press(w, "Escape")
    scr2 = screens(w)
    check("Esc при исправлении — отменить, а не к списку", scr2 and scr2[0] is scr and not scr._edit_mode)
    press(w, "Escape")
    check("Esc — к списку", not w.screen_open())
    press(w, "F5")
    check("F5 — списки обновлены (сообщение в окне)", w.toast.text == "Списки обновлены", w.toast.text)


def s_queue(w, run):
    print("Очередь: Enter, Del, D, N, Esc — горячие клавиши на каждом фото")
    to_tab(w, w._proc_tab)
    q0 = queue_len(w)
    press(w, "Return")
    scr = one_screen(w, EditScreen, "Enter в списке очереди")
    p1 = scr.photo_path
    ANSWER["yes"] = True
    press(w, "Delete")
    scr = one_screen(w, EditScreen, "Del = «Нечитаемо»")
    check("Del: фото ушло из очереди", queue_len(w) == q0 - 1 and not Path(p1).exists())
    p2 = scr.photo_path
    press(w, "d")
    scr = one_screen(w, EditScreen, "D на втором фото")
    check("D на втором фото сработало", queue_len(w) == q0 - 2 and not Path(p2).exists(), f"в очереди {queue_len(w)}")
    p3 = scr.photo_path
    ANSWER["text"] = "0123456"
    press(w, "n")
    check("N на третьем фото сработало", not Path(p3).exists(), f"вопросы: {TEXTS[-1:]}")
    press(w, "Escape")
    check("Esc — к списку", not screens(w))
    check("фокус — в списке очереди", focus_widget(w) is w._proc_tab._tree)
    rows = db(run, "select outcome from processing_log where processed_by=? and source='manual'", OPERATOR)
    check("в лог — 3 решения оператора", sorted(r[0] for r in rows) == ["NOT_IN_DB", "REPEAT", "UNREADABLE"], f"{rows}")


def s_queue_accept(w, run):
    print("Очередь: цифры с клавиатуры и Enter = «Принять»")
    to_tab(w, w._proc_tab)
    tree, items = w._proc_tab._tree, w._proc_tab._items
    pick = next((i for i, (reason, _, _) in enumerate(items) if reason in ("digits_error", "suspicious")), None)
    if pick is None:
        check("в очереди есть «ошибка цифр» / «подозрительно»", False)
        return
    tree.selection_set(tree.get_children()[pick])
    tree.focus(tree.get_children()[pick])
    tree.focus_set()
    press(w, "Return")
    scr = one_screen(w, EditScreen, "открыть фото с ошибкой цифр")
    account = scr._account_var.get()
    scr._reading_widget._entries[0].focus_set()
    for d in "54321":
        press(w, d, wait=0.05)
    check("цифры набраны с клавиатуры", scr._reading_widget.get_string() == "54321", scr._reading_widget.get_string())
    press(w, "Return")
    val = db(run, "select value from readings where account=?", account)
    check("Enter в поле цифр — «Принять», показание в базе", val and val[0][0] == "54321", f"{account}: {val}")
    check("после «Принять» — не больше одного экрана", len(screens(w)) <= 1)
    if screens(w):
        press(w, "Escape")


def s_queue_b(w, run):
    print("Очередь: B = «Номер в таблице с ошибкой»")
    to_tab(w, w._proc_tab)
    press(w, "Return")
    scr = one_screen(w, EditScreen, "открыть фото")
    account = db(run, "select account from abonents where account not in (select account from readings) limit 1")[0][0]
    scr._account_var.set(account)
    scr._flush_lookups()
    scr._reading_widget.set_digits("01234", None)
    pump(w)
    ANSWER["text"] = "0123999"
    photo = scr.photo_path
    letter(w, "b")
    check("B: фото ушло из очереди", not Path(photo).exists())
    check("B: номер в базе исправлен", db(run, "select serial from abonents where account=?", account)[0][0] == "0123999")
    check("B: показание записано", db(run, "select value from readings where account=?", account) == [("1234",)])
    check("B: номер — в списке для компании", (run.month / "db_serial_fix.csv").is_file())
    if screens(w):
        press(w, "Escape")


def s_list_keys_inert(w, run):
    print("Список под открытым экраном не открывает второй")
    to_tab(w, w._verify_tab)
    press(w, "Return")
    n = verified(run)
    w._verify_tab._open()                                      # как если бы Enter дошёл до списка
    one_screen(w, VerifyScreen, "открыть из списка при открытом экране")
    check("ничего не отмечено", verified(run) == n)
    press(w, "Escape")


def s_settings(w, run):
    print("Настройки распознавания: обычные и расширенные")
    from src.gmr.application import month as month_app

    def change(dlg):
        tabs = [dlg._tabs.tab(i, "text") for i in dlg._tabs.tabs()]
        check("две вкладки: «Обычные», «Расширенные»", tabs == ["Обычные", "Расширенные"], f"{tabs}")
        dlg._vars["serial_conf"].set("0,55")
        dlg._vars["turn_photo"].set(True)
        dlg._save()
    w.after(150, lambda: change(next(x for x in w.winfo_children() if isinstance(x, program2.PresetDialog))))
    w.edit_preset()
    p = month_app.month_preset(str(run.month))
    check("сохранено в месяц: уверенность номера 0,55, поворот фото", p and p.serial_conf == 0.55 and p.turn_photo)
    check("окно читает с новой уверенностью", w.config.serial_conf_thresh == 0.55)

    def standard(dlg):
        dlg._fill(type(p).new_month())
        dlg._save()
    w.after(150, lambda: standard(next(x for x in w.winfo_children() if isinstance(x, program2.PresetDialog))))
    w.edit_preset()
    check("«Вернуть стандартные»", month_app.month_preset(str(run.month)) == type(p).new_month())


def s_help(w, run):
    print("Справка")
    from src.gmr.version import __version__
    check("«О программе» — версия", __version__ in w.about_text())
    path = w.manual_path()
    check("инструкция оператора есть", path is not None and path.is_file() and path.stat().st_size > 10_000)
    labels = [w._help_menu.entrycget(i, "label") for i in range(w._help_menu.index("end") + 1)
              if w._help_menu.type(i) == "command"]
    check("меню «Справка»", labels == ["Инструкция оператора", "Горячие клавиши", "Ярлык на рабочем столе", "О программе"], f"{labels}")


def s_menu(w, run):
    print("Итог месяца, выгрузка")
    w.show_summary()
    pump(w)
    scr = getattr(w, "_current_screen", None)
    check("«Итог месяца» открылся экраном", type(scr).__name__ == "SummaryScreen", f"{type(scr).__name__}")
    check("итог текстом — итог_месяца.txt", (run.month / "итог_месяца.txt").is_file())
    press(w, "Escape")
    check("Esc — к спискам", not w.screen_open())
    TOLD.clear()
    for fmt in ("xlsx", "xls", "csv", "ods"):
        rep = w.export_as(fmt)
        check(f"«Выгрузить» в .{fmt}", Path(rep.path).is_file() and rep.path.endswith(f"показания.{fmt}"), rep.path)


def _process(w, run, n, how):
    photos = run.take_photos(n)
    if len(photos) < n:
        check(f"{how}: есть {n} новых фото (--photos)", False)
        return
    dst = run.month / "фото" / photos[0].parent.name
    dst.mkdir(parents=True, exist_ok=True)
    for p in photos:
        shutil.copy2(p, dst / p.name)
    n0 = db(run, "select count(*) from processing_log")[0][0]
    t = time.time()
    w.process_new()
    pump(w, 0.5)
    n1 = db(run, "select count(*) from processing_log")[0][0]
    titles = [t_.title() for t_ in w.winfo_children() if t_.winfo_class() == "Toplevel"]
    scr = getattr(w, "_current_screen", None)
    check(f"{how}: {n} новых фото прочитаны", n1 - n0 >= n, f"+{n1 - n0}, окна {titles}")
    check(f"{how}: итог — экраном в окне, без окон поверх", type(scr).__name__ == "ProcessScreen"
          and bool(scr.report) and not titles, f"{type(scr).__name__}, окна {titles}")
    press(w, "Escape")
    check(f"{how}: Esc — к спискам", not w.screen_open())
    print(f"     {time.time() - t:.0f} с", flush=True)


def s_process(w, run):
    print("«Обработать новые фото»")
    _process(w, run, 4, "Python")


def s_process_exe(w, run):
    if run.exe is None:
        return
    print("«Обработать новые фото» — собранная программа сама себя с параметрами")
    orig = op_window.process_command
    op_window.process_command = lambda project, month, reread=False: [str(run.exe), "process", month] + (
        ["--reread"] if reread else [])
    try:
        _process(w, run, 2, "exe")
    finally:
        op_window.process_command = orig


def s_months(w, run):
    print("Меню «Месяц»: новый месяц, другой месяц, обновить таблицу, перечитать")
    table = next((run.month / "таблицы").glob("*.xls*"), None) or next((run.month / "таблицы").glob("*.csv"))
    new = run.month.parent / (run.month.name + "_новый")
    shutil.rmtree(new, ignore_errors=True)
    fd = op_window.filedialog
    fd.askopenfilename = lambda **k: str(table)
    fd.asksaveasfilename = lambda **k: str(new)
    offered = []
    real_ask = program2.MainWindow.ask_preset
    program2.MainWindow.ask_preset = staticmethod(
        lambda parent, preset, title, ok_text="Сохранить": offered.append(preset) or preset)
    try:
        w.new_month()
        pump(w, 0.3)
        close_toplevels(w)
        check("новый месяц создан и открыт", w.session.folder.root == new and (new / "gmr.sqlite").is_file())
        check("в новом месяце очередь пуста", queue_len(w) == 0)
        fd.askdirectory = lambda **k: str(run.month)
        w.choose_month()
        check("«Открыть другой месяц» — обратно", w.session.folder.root == run.month)
        w.update_table()
        pump(w, 0.3)
        close_toplevels(w)
        check("«Обновить таблицу» без ошибок", not [x for x in TOLD if x[0] == "showerror"], f"{TOLD[-2:]}")
        q0 = queue_len(w)
        w.reread_errors()
        pump(w, 0.3)
        close_toplevels(w)
        check("«Прочитать заново фото с ошибками»", queue_len(w) <= q0 + 1, f"было {q0}, стало {queue_len(w)}")
    finally:
        program2.MainWindow.ask_preset = real_ask
        shutil.rmtree(new, ignore_errors=True)


SCENARIOS = (s_verify, s_verify_edit, s_keys_14, s_queue, s_queue_accept, s_queue_b, s_list_keys_inert, s_settings,
             s_help, s_menu, s_process, s_process_exe, s_months)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.ui_scenarios", description=__doc__.split("\n\n")[0])
    p.add_argument("month", help="КОПИЯ папки месяца (меняется)")
    p.add_argument("--photos", help="папка с новыми фото для «Обработать новые фото» (берутся 6)")
    p.add_argument("--exe", help="собранная программа «GMR.exe» — обработка через неё")
    p.add_argument("--only", help="только эти сценарии, через запятую (s_verify,…)")
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    photos = sorted(f for f in Path(a.photos).iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")) if a.photos else []
    run = Run(Path(a.month), photos[-6:], Path(a.exe) if a.exe else None)
    w = setup(run.month)
    for s in SCENARIOS:
        if a.only and s.__name__ not in a.only.split(","):
            continue
        w._show_lists()
        w._return_released = True
        try:
            s(w, run)
        except Exception as e:      # noqa: BLE001
            check(f"{s.__name__}: без исключений", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    try:
        w._on_close()
        check("окно закрылось", True)
    except Exception as e:          # noqa: BLE001
        check("закрытие без исключений", False, f"{e}")
    bad = [r for r in RESULTS if not r[0]]
    print(f"\nИТОГ: {len(RESULTS) - len(bad)} из {len(RESULTS)} проверок прошли")
    for _, name, detail in bad:
        print("  FAIL", name, detail[:600])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
