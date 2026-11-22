"""Главное окно: вход, месяц, меню, вкладки «Обработка» и «Проверка»."""
import sqlite3
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import pandas as pd

from src.gmr.application import add_photos as photos_app
from src.gmr.application import month as month_app
from src.gmr.application.month import ExportLocked, NotAMonth
from src.gmr.application.operator import OperatorSession
from src.gmr.domain import PipelineConfig, find_auto_row_for_output_file
from src.gmr.domain.area_code import accounts_by_controller, controller_code
from src.gmr.domain.preset import RecognitionPreset
from src.gmr.storage import LOG_COLUMNS as _LOG_COLUMNS
from src.gmr.storage import log_path_for
from src.gmr.frozen import program_dir
from src.gmr.version import __version__
from src.gmr.ui.run_dialog import process_command, show_text

from .common import (
    _BASE,
    _LOG_FILE,
    REASON_LABELS,
    _log_handlers,
    log,
)
from .dialogs import ExportDialog, KeysDialog, PresetDialog
from .drop_panel import TABLE_EXTS, PhotoDropPanel, TableDropPanel
from .dropfiles import enable_drop
from .theme import (
    action_button,
    APP_NAME,
    apply_theme,
    ctrl_sequences,
    HEADER,
    MANUAL_NAME,
    MUTED,
    px,
    set_enabled,
    set_window_icon,
    SURFACE,
    Toast,
    ZEBRA,
)
from .edit_screen import EditScreen
from .files import move_photo, now_iso, open_in_explorer, place_decision
from .hints import SerialIndex, build_serial_index
from .new_photos import NewPhotosBadge, NewPhotosWatcher
from .photos import (
    controller_dir,
    controller_name,
    list_question_photos,
    list_verify_photos,
)
from .settings import AppSettings, default_month_name, load_settings, save_settings
from .tables import SortableTable, arrow_images, number
from .process_screen import ProcessScreen
from .summary_screen import SummaryScreen
from .verify_screen import VerifyScreen
from .welcome import WelcomeScreen


def ask_login(welcome: WelcomeScreen, settings: AppSettings) -> str:
    """«Продолжить» / «Сменить пользователя» / закрыть окно (тесты подменяют)."""
    return welcome.ask(settings)


def ask_setup(welcome: WelcomeScreen, settings: AppSettings, new_month, error=None,
              can_cancel: bool = False) -> Optional[AppSettings]:
    """Имя и папка месяца на экране входа (тесты подменяют)."""
    return welcome.ask_setup(settings, new_month, error, can_cancel)


def pick_table(window, title: str, subtitle: str) -> Optional[str]:
    """Таблица компании: перетащить в окно или выбрать (тесты подменяют)."""
    return TableDropPanel(window, title, subtitle).wait()


def pick_photos(window, existing: list, paths: Optional[list] = None):
    """«Добавить фото»: группы фото по контролёрам или None (тесты подменяют)."""
    return PhotoDropPanel(window, existing, paths).wait()


def open_on_row(tree, event, open_fn):
    """Двойной щелчок открывает фото, только если пришёлся на строку: по
    заголовку (сортировка), пустому месту или полосе прокрутки — ничего
    (раньше открывалось выделенное или первое фото списка)."""
    row = tree.identify_row(event.y)
    if not row or tree.identify_region(event.x, event.y) not in ("cell", "tree"):
        return "break"
    tree.selection_set(row)
    tree.focus(row)
    open_fn()
    return "break"


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} {__version__}")
        self.geometry(f"{px(1280)}x{px(800)}")
        self.minsize(px(1100), px(680))  # меньше — кнопки шапки и разбора фото не помещаются
        try:
            self.state("zoomed")         # на весь экран: на ноутбуке 1366×768 окно 1280×800 не влезает
        except tk.TclError:              # не Windows
            pass
        apply_theme(self)
        set_window_icon(self)
        # файлы и папки из проводника — в любое место окна (1.5.0): таблица
        # нового месяца, папка с фото (_on_drop)
        self._drop_target = None
        self.after_idle(lambda: enable_drop(self, self._on_drop))

        self.settings  = load_settings()
        self.session:   Optional[OperatorSession] = None
        self.df: Optional[pd.DataFrame] = None
        self.log_rows:  list[dict] = []
        self.config      = PipelineConfig()

        # Удержание Enter: автоповтор Windows шлёт нажатия без отпускания —
        # без защиты «Верно» отмечало подряд фото, которых оператор не видел
        # (замер 2026-11-20: 3 нажатия подряд — 3 фото). Действие — только
        # если Enter отпускали после прошлого действия или нажатие через
        # секунду после прошлого (отпускание могло уйти в окно-вопрос).
        self._return_released = True
        self._last_return_time = 0
        self.bind_all("<KeyRelease-Return>", self._on_return_release, add="+")

        self._do_login()

    def _on_return_release(self, event=None):
        self._return_released = True

    def take_return(self, event=None) -> bool:
        """Enter с клавиатуры: True — выполнить действие, False — это автоповтор."""
        t = int(getattr(event, "time", 0) or 0)
        ok = (self._return_released or t - self._last_return_time > 1000
              or t < self._last_return_time)
        self._last_return_time = t
        if ok:
            self._return_released = False
        return ok

    def report_callback_exception(self, exc, val, tb):
        """Ошибка в обработчике кнопки: в журнал и оператору (с ярлыка консоли
        нет — без этого ошибка была бы не видна)."""
        log.error("Ошибка в окне", exc_info=(exc, val, tb))
        where = (f"в журнале {_LOG_FILE.name} (папка программы)" if _log_handlers
                 else "в окне PowerShell, из которого запущена программа")
        messagebox.showerror("Ошибка программы", f"{exc.__name__}: {val}\n\nПодробности — {where}.",
                             parent=self)

    def _do_login(self):
        """Вход — экраном в этом же окне (welcome.py): и вход, и первая
        настройка, и «Сменить пользователя или месяц» — без окон поверх."""
        welcome = WelcomeScreen(self)
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        complete = bool(self.settings.operator_name and self.settings.month_dir)
        setup, error = not complete, None
        if complete:
            action = ask_login(welcome, self.settings)
            if action == "quit":
                self.destroy()
                return
            setup = action == "change"
        while True:
            if setup:
                new = ask_setup(welcome, self.settings, self.ask_new_month, error, can_cancel=complete)
                if new is not None:
                    self.settings = new
                    save_settings(self.settings)
                elif welcome.quit or not complete:          # закрыли окно / первый запуск без настроек
                    self.destroy()
                    return
                # «Отмена» — работа с прежними настройками
            welcome.status("Открываю месяц…")
            if self._load_data():
                break
            welcome.status(None)
            setup, complete = True, False     # папка — не месяц: выбрать другую (отмены нет — месяца нет)
            error = (f"В папке «{self.settings.month_dir}» месяц не создан. Выберите папку месяца или "
                     "создайте новый месяц из таблицы компании.")
        welcome.destroy()
        self._build_ui()

    def _load_data(self) -> bool:
        """Открывает базу папки месяца (и делает её копию). False — папка не месяц."""
        if self.session is not None:
            self.session.close()
            self.session = None
        try:
            self.session = OperatorSession(self.settings.month_dir, self.config)
        except NotAMonth as e:
            # этап 4: месяц можно создать прямо отсюда, без PowerShell
            if not (self.settings.month_dir and messagebox.askyesno(
                    "Папка месяца",
                    f"{e}\n\nСоздать в этой папке новый месяц из таблицы компании?", parent=self)):
                return False
            if not self.create_month(self.settings.month_dir):
                return False
            self.session = OperatorSession(self.settings.month_dir, self.config)
        try:
            dest = self.session.backup()
            if dest is not None:
                log.info(f"Копия базы месяца: {dest}")
        except (OSError, sqlite3.Error) as e:
            log.warning(f"Не удалось сделать копию базы месяца: {e}")
        self._apply_preset()
        self.reload()
        return True

    def reload(self):
        """Абоненты, показания и лог — заново из базы (их мог изменить reader.py)."""
        self.df = pd.DataFrame(self.session.table_rows(), columns=self.session.table_columns())
        self.log_rows = self.session.log_rows()
        self._serial_index = None
        log.info(f"Месяц: {self.session.folder.root} — абонентов {len(self.df)}, строк лога {len(self.log_rows)}")

    @property
    def results_dir(self) -> str:
        return str(self.session.folder.results)

    # ─── Месяц: создать, обновить таблицу, итог (этап 4) ───────────────────────
    def _ask_table(self, parent=None, update: bool = False) -> Optional[str]:
        if update:
            return pick_table(self, "Обновить таблицу компании",
                              "Новая таблица компании для этого месяца. Показания, которые уже записаны, "
                              "останутся. Оригинал таблицы не меняется.")
        return pick_table(self, "Новый месяц — таблица компании",
                          "Таблица компании на месяц: номера счётчиков, лицевые счета, прошлые показания. "
                          "Программа сделает себе копию — оригинал не меняется.")

    def ask_new_month(self, parent=None, who: Optional[str] = None) -> Optional[str]:
        """Новый месяц без PowerShell (2026-10-07): таблица компании, потом —
        как «Сохранить как»: где и под каким именем создать папку месяца (по
        умолчанию «Октябрь_2026» рядом с текущим месяцем). Возвращает папку
        созданного месяца или None."""
        parent = parent or self
        table = self._ask_table(parent)
        if not table:
            return None
        here = Path(self.settings.month_dir).parent if self.settings.month_dir else Path.home()
        folder = filedialog.asksaveasfilename(
            parent=parent, title="Где создать папку нового месяца", initialdir=str(here),
            initialfile=default_month_name(), confirmoverwrite=False)
        if not folder:
            return None
        return folder if self.create_month(folder, table, parent=parent, who=who) else None

    def create_month(self, folder: str, table: Optional[str] = None, parent=None,
                     who: Optional[str] = None) -> bool:
        """Новый месяц в папке folder из таблицы компании (то же, что
        `gmr.py month <папка> --table <таблица>`). True — месяц создан."""
        parent = parent or self
        if month_app.is_month(month_app.MonthFolder(Path(folder))):
            messagebox.showerror("Новый месяц", f"В папке {folder} месяц уже есть.\n"
                                 "Новая таблица компании — меню «Месяц» → «Обновить таблицу…».", parent=parent)
            return False
        table = table or self._ask_table(parent)
        if not table:
            return False
        old_log = Path(log_path_for(table))
        import_log = old_log.is_file() and messagebox.askyesno(
            "Старый лог",
            f"Рядом с таблицей лежит лог старой версии программы:\n{old_log.name}\n\n"
            "Перенести его в базу месяца? Тогда фото из этого лога будут считаться уже разобранными.\n"
            "Обычно — «Нет»: месяц начинается с чистого листа.", default="no", parent=parent)
        try:
            n = month_app.table_readings_count(table, self.config)
        except (ValueError, OSError) as e:
            messagebox.showerror("Новый месяц", str(e), parent=parent)
            return False
        clear = n > 0 and messagebox.askyesno(
            "Показания в таблице",
            f"В таблице уже есть показания у {n} абонентов.\n\nОчистить их?\n\n"
            "«Да» — месяц начинается без показаний, программа прочитает все фото.\n"
            "«Нет» — эти показания считаются уже записанными.\n\n"
            "Оригинал таблицы не меняется.", parent=parent)
        preset = self.ask_preset(parent, self._previous_preset(), "Настройки распознавания нового месяца",
                                 ok_text="Дальше")
        if preset is None:
            return False
        try:
            rep = month_app.load_table(folder, table, self.config, who=who or self.settings.operator_name,
                                       import_old_log=import_log, clear_readings=clear, preset=preset)
            exported = month_app.export_month(folder, self.config).text()
        except (ValueError, OSError, ExportLocked) as e:
            messagebox.showerror("Новый месяц", str(e), parent=parent)
            return False
        show_text(parent, "Месяц создан", rep.text() + "\n\n" + exported)
        return True

    # ─── Настройки распознавания месяца (2026-10-11, решение 2а) ──────────────
    @staticmethod
    def ask_preset(parent, preset: RecognitionPreset, title: str, ok_text: str = "Сохранить"):
        return PresetDialog(parent, preset, title, ok_text=ok_text,
                            marker_model=bool(PipelineConfig().account_ocr_model)).result

    def _previous_preset(self) -> RecognitionPreset:
        """Для нового месяца — как в открытом месяце (с решениями владельца для
        новых месяцев), иначе — всё подключено."""
        if self.session is not None:
            prev = month_app.month_preset(str(self.session.folder.root))
            if prev is not None:
                return prev.next_month()
        return RecognitionPreset.new_month()

    def _apply_preset(self):
        """Настройки месяца — в self.config (порог разницы, столбцы — как у прогона)."""
        preset = month_app.month_preset(str(self.session.folder.root))
        self.config = preset.apply(PipelineConfig()) if preset is not None else PipelineConfig()

    def edit_preset(self):
        """Меню «Месяц» → «Настройки распознавания…»."""
        root = str(self.session.folder.root)
        current = month_app.month_preset(root) or RecognitionPreset()
        new = self.ask_preset(self, current, "Настройки распознавания")
        if new is None or new == current:
            return
        month_app.set_month_preset(root, new, who=self.settings.operator_name)
        self._apply_preset()

    def _switch_month(self, folder: str):
        """Окно — на другой папке месяца (настройки сохраняются). Если месяц
        не открылся, окно остаётся в прежнем месяце."""
        self._close_screen()
        old = self.settings
        self.settings = AppSettings(old.operator_name, folder)
        opened = False
        try:
            opened = self._load_data()
        finally:
            if not opened:
                self.settings = old
                self._load_data()
        save_settings(self.settings)
        self._show_month()
        self.refresh_tabs()
        self._watcher.check_now()              # новые фото — уже в другом месяце

    def new_month(self):
        """Меню «Месяц» → «Новый месяц…»: таблица компании и папка нового месяца."""
        folder = self.ask_new_month()
        if folder:
            self._switch_month(folder)

    def update_table(self):
        """Меню «Месяц» → «Обновить таблицу…»: компания прислала новую таблицу."""
        table = self._ask_table(update=True)
        if not table or not messagebox.askyesno(
                "Обновить таблицу", f"Загрузить таблицу\n{table}\nв месяц {self.session.folder.root}?\n\n"
                "Сверка — по лицевому счёту; показания из базы сохраняются.", parent=self):
            return
        self._close_screen()
        try:
            rep = month_app.load_table(str(self.session.folder.root), table, self.config,
                                       who=self.settings.operator_name)
            exported = month_app.export_month(str(self.session.folder.root), self.config).text()
        except (ValueError, OSError, ExportLocked) as e:
            messagebox.showerror("Обновить таблицу", str(e), parent=self)
            return
        self.refresh_tabs()
        show_text(self, "Таблица обновлена", rep.text() + "\n\n" + exported)

    def choose_month(self):
        """Меню «Месяц» → «Открыть другой месяц…»."""
        folder = filedialog.askdirectory(parent=self, title="Папка месяца")
        if not folder:
            return
        if not month_app.is_month(month_app.MonthFolder(Path(folder))):
            if not messagebox.askyesno("Папка месяца", f"В папке {folder} месяц не создан.\n\n"
                                       "Создать новый месяц из таблицы компании?", parent=self):
                return
            if not self.create_month(folder):
                return
        self._switch_month(folder)

    # ─── «Справка» (2026-11-21): инструкция, ярлык, версия — без командной строки ──
    def manual_path(self) -> Optional[Path]:
        """Инструкция оператора: в собранной программе — рядом с exe (`python gmr.py
        build`), без сборки — из docs/OPERATOR.md во временную папку."""
        path = program_dir() / MANUAL_NAME
        if path.is_file():
            return path
        try:
            import tempfile

            from tools.manual_html import write_manual
            return write_manual(Path(tempfile.gettempdir()) / MANUAL_NAME, __version__)
        except (ImportError, OSError) as e:
            log.warning(f"Инструкция не собрана: {e}")
            return None

    def open_manual(self):
        path = self.manual_path()
        if path is None:
            messagebox.showwarning("Инструкция", f"Файла «{MANUAL_NAME}» нет в папке программы.", parent=self)
            return
        open_in_explorer(path)

    def make_shortcut(self):
        from tools.shortcut import create_shortcut
        ok, message = create_shortcut()
        if ok:
            self.toast.show("Ярлык «GMR» — на рабочем столе. Дальше открывайте программу им.")
        else:
            messagebox.showerror("Ярлык на рабочем столе", message, parent=self)

    def about_text(self) -> str:
        cfg = PipelineConfig()
        models = [("цифры", cfg.digit_ocr_model), ("номер счётчика", cfg.serial_ocr_model),
                  ("надпись маркером", cfg.account_ocr_model or "не подключена")]
        lines = [f"{APP_NAME} {__version__}", "", f"Папка программы: {program_dir()}", "", "Модели:"]
        lines += [f"  {name}: {Path(path).parent.name if '/' in path else path}" for name, path in models]
        return "\n".join(lines)

    def show_about(self):
        messagebox.showinfo("О программе", self.about_text(), parent=self)

    def show_summary(self):
        """«Итог месяца»: экран с полосами (summary_screen.py); тот же итог
        текстом — итог_месяца.txt в папке месяца и `gmr.py month <папка>`."""
        stats = month_app.month_stats(str(self.session.folder.root))
        if stats is None:
            show_text(self, "Итог месяца", month_app.month_summary(str(self.session.folder.root)))
            return
        month_app.save_summary(stats)
        self._show_screen(lambda: SummaryScreen(self, stats, on_back=self._back_from_screen))

    def open_month_folder(self):
        open_in_explorer(self.session.folder.root)

    def reread_errors(self):
        """Меню «Месяц» → «Прочитать заново фото с ошибками…» (после замены модели)."""
        if messagebox.askyesno(
                "Прочитать заново", "Прочитать заново все фото с ошибками, которые ждут оператора?\n\n"
                "Нужно, например, после замены модели. Разобранные фото не трогаются.", parent=self):
            self.process_new(reread=True)

    def _build_ui(self):
        menubar = tk.Menu(self)
        m = tk.Menu(menubar, tearoff=False)
        m.add_command(label="Добавить фото…", accelerator="Ctrl+O", command=self.add_photos)
        m.add_command(label="Обработать новые фото", command=self.process_new)
        m.add_command(label="Итог месяца", accelerator="Ctrl+I", command=self.show_summary)
        m.add_command(label="Выгрузить показания…", accelerator="Ctrl+E", command=self.export)
        m.add_command(label="Открыть папку месяца", command=self.open_month_folder)
        m.add_separator()
        m.add_command(label="Новый месяц…", command=self.new_month)
        m.add_command(label="Обновить таблицу компании…", command=self.update_table)
        m.add_command(label="Открыть другой месяц…", command=self.choose_month)
        m.add_separator()
        m.add_command(label="Прочитать заново фото с ошибками…", command=self.reread_errors)
        m.add_command(label="Настройки распознавания…", command=self.edit_preset)
        menubar.add_cascade(label="Месяц", menu=m)
        h = tk.Menu(menubar, tearoff=False)
        h.add_command(label="Инструкция оператора", accelerator="F1", command=self.open_manual)
        h.add_command(label="Горячие клавиши", command=self.show_keys)
        h.add_command(label="Ярлык на рабочем столе", command=self.make_shortcut)
        h.add_separator()
        h.add_command(label="О программе", command=self.show_about)
        menubar.add_cascade(label="Справка", menu=h)
        self.configure(menu=menubar)        # self.config — это PipelineConfig, поэтому configure
        self._menubar = menubar
        self._month_menu = m
        self._help_menu = h

        # Шапка: программа, месяц, оператор; справа — главные действия месяца
        top = ttk.Frame(self, style="Header.TFrame", padding=(px(16), px(10)))
        top.pack(fill=tk.X)
        ttk.Label(top, text=APP_NAME, style="HeaderTitle.TLabel").pack(side=tk.LEFT)
        self._month_var = tk.StringVar()
        # имя месяца — последним: в узком окне сжимается оно, а не кнопки и значок новых фото
        self._month_label = ttk.Label(top, textvariable=self._month_var, style="HeaderInfo.TLabel")
        export_btn = ttk.Button(top, text="⇩  Выгрузить показания…", style="Header.TButton", command=self.export)
        export_btn.pack(side=tk.RIGHT)
        summary_btn = ttk.Button(top, text="Итог месяца", style="Header.TButton", command=self.show_summary)
        summary_btn.pack(side=tk.RIGHT, padx=(0, px(8)))
        self._process_btn = action_button(top, "▶  Обработать новые фото", self.process_new, kind="accent",
                                          highlightbackground=HEADER)
        self._process_btn.pack(side=tk.RIGHT, padx=(0, px(12)))
        add_btn = ttk.Button(top, text="＋  Добавить фото…", style="Header.TButton", command=self.add_photos)
        add_btn.pack(side=tk.RIGHT, padx=(0, px(8)))
        self._header_buttons = (export_btn, summary_btn, add_btn)
        # сколько новых фото ждут обработки (new_photos.py): проверка в фоне раз в 10 с;
        # значок появляется левее кнопки (before= — раньше имени месяца в порядке шапки)
        self._month_label.pack(side=tk.LEFT, padx=(px(18), 0))
        self._new_badge = NewPhotosBadge(top, HEADER, self.process_new).pack(side=tk.RIGHT, padx=(0, px(10)),
                                                                           before=self._month_label)
        self._processing = False
        self._watcher = NewPhotosWatcher(
            self, lambda: (self.session.folder.photos, self.log_rows) if self.session is not None else None,
            self._new_badge.set_count, busy=lambda: self._processing)
        self._watcher.start()
        self._show_month()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=px(12), pady=px(12))

        self._proc_tab = ProcessingTab(self._notebook, self)
        self._verify_tab = VerifyTab(self._notebook, self)

        self._notebook.add(self._proc_tab,   text="Очередь")
        self._notebook.add(self._verify_tab, text="Проверка")
        self._update_tab_titles()
        # вкладка сменилась — клавиши её списку (иначе Enter уходил в скрытый список другой вкладки)
        self._notebook.bind("<<NotebookTabChanged>>", lambda e: self.after_idle(self._focus_list))
        self._notebook.enable_traversal()                  # Ctrl+Tab / Ctrl+Shift+Tab — между вкладками
        self.toast = Toast(self)
        self._bind_window_keys()

    # ─── Клавиши окна (1.4.0): работают в списках и в «Итоге месяца» ─────────
    WINDOW_KEYS = (("F1", "инструкция оператора"), ("F5", "обновить списки"),
                   ("Ctrl+O", "добавить фото (или перетащить папку в окно)"),
                   ("Ctrl+I", "итог месяца"), ("Ctrl+E", "выгрузить показания"),
                   ("Ctrl+Tab", "другая вкладка: «Очередь» / «Проверка»"))

    def _bind_window_keys(self):
        def guard(fn, lists_only=False):
            def handler(e):
                scr = getattr(self, "_current_screen", None)
                if self._processing or (self.screen_open() and (lists_only or not isinstance(scr, SummaryScreen))):
                    return None                    # идёт обработка или открыто фото — не уводить с экрана
                fn()
                return "break"
            return handler
        self.bind_all("<F1>", lambda e: self.open_manual())
        self.bind_all("<F5>", guard(self.refresh_all, lists_only=True))
        for seq in ctrl_sequences("i"):
            self.bind_all(seq, guard(self.show_summary))
        for seq in ctrl_sequences("e"):
            self.bind_all(seq, guard(self.export))
        for seq in ctrl_sequences("o"):
            self.bind_all(seq, guard(self.add_photos, lists_only=True))

    # ─── Перетаскивание и «Добавить фото» (1.5.0) ─────────────────────────────
    def _on_drop(self, paths: list):
        """Файлы из проводника упали в окно: открыто поле «перетащите сюда» —
        в него; иначе папка или фото на списках — «Добавить фото» с ними."""
        target = self._drop_target
        if target is not None and target.winfo_exists():
            target.dropped(paths)
            return
        toast = getattr(self, "toast", None)
        if toast is None:                        # экран входа — перетаскивать некуда
            return
        if self._processing or self.screen_open():
            toast.show("Сначала вернитесь к спискам — тогда можно перетащить фото", "info")
            return
        if all(Path(x).suffix.lower() in TABLE_EXTS for x in paths):
            toast.show("Таблицу — через меню «Месяц»: «Новый месяц…» или «Обновить таблицу…»", "info", ms=5000)
            return
        self.add_photos(paths)

    def add_photos(self, paths: Optional[list] = None) -> Optional[photos_app.AddReport]:
        """Кнопка «Добавить фото…» / Ctrl+O / папка, перетащенная в окно."""
        if self._processing or self.session is None:
            return None
        photos_dir = self.session.folder.photos
        groups = pick_photos(self, photos_app.controllers(photos_dir), paths)
        if not groups:
            return None
        try:
            rep = photos_app.add_photos(photos_dir, groups)
        except (OSError, ValueError) as e:
            messagebox.showerror("Добавить фото", f"Фото добавлены не все: {e}", parent=self)
            return None
        log.info(f"Добавить фото: {rep.text()}")
        self._watcher.check_now()
        if rep.total:
            self.toast.show(rep.text() + ". Чтобы прочитать — «Обработать новые фото».", "success", ms=7000)
        else:
            self.toast.show("Новых фото нет — все эти фото уже есть в месяце", "info", ms=5000)
        return rep

    def refresh_all(self):
        """F5 и «Обновить»: списки заново и новые фото — сразу."""
        self.refresh_tabs()
        self._watcher.check_now()
        self.toast.show("Списки обновлены", "info", ms=1500)

    def show_keys(self):
        """Справка → «Горячие клавиши»."""
        KeysDialog(self, [
            ("Везде", self.WINDOW_KEYS),
            ("Списки", (("Enter или 2× щелчок", "открыть фото"), ("Del", "«Нечитаемо» (в «Очереди»)"),
                        ("щелчок по заголовку", "сортировка, второй — наоборот"))),
            ("Разбор фото", (("Enter", "Принять"), ("D", "Дубль"), ("Del", "Нечитаемо"), ("N", "Нет в базе"),
                             ("B", "Номер в таблице с ошибкой"), ("Esc", "к списку (из поля ввода — выйти из поля)"))),
            ("Проверка фото", (("Enter", "Верно (при исправлении — Сохранить)"), ("E", "Исправить"),
                               ("→ / ←", "следующее / предыдущее фото"), ("Esc", "к списку (при исправлении — отменить)"))),
            ("Фото", (("колёсико", "масштаб к месту под мышью"), ("+ / −", "крупнее / мельче"),
                      ("0 или 2× щелчок", "фото целиком"), ("перетащить мышью", "сдвинуть"))),
        ])

    def _set_busy(self, busy: bool):
        """Идёт обработка: шапка и меню выключены (разбирать фото нельзя — решение 2а)."""
        state = tk.DISABLED if busy else tk.NORMAL
        for b in self._header_buttons:
            b.state(["disabled"] if busy else ["!disabled"])
        set_enabled(self._process_btn, not busy)
        for i in range(self._menubar.index("end") + 1):
            try:
                self._menubar.entryconfigure(i, state=state)
            except tk.TclError:
                pass

    def _show_month(self):
        """В шапке — имя папки месяца и оператор (полный путь — «Месяц» →
        «Открыть папку месяца»): длинный путь вытеснял кнопки за край окна."""
        self._month_var.set(f"{self.session.folder.root.name}   ·   {self.settings.operator_name}")
        self.title(f"{APP_NAME} {__version__} — {self.session.folder.root.name}")

    def _update_tab_titles(self):
        if not hasattr(self, "_notebook"):
            return
        q, v = len(self._proc_tab._items), len(self._verify_tab._items)
        self._notebook.tab(self._proc_tab, text=f"Очередь  {q}" if q else "Очередь")
        self._notebook.tab(self._verify_tab, text=f"Проверка  {v}" if v else "Проверка")

    def refresh_tabs(self):
        self.reload()
        self._proc_tab.refresh()
        self._verify_tab.refresh()
        self._update_tab_titles()

    def _close_screen(self):
        """Вернуться к спискам (экран разбора закрывается)."""
        scr = getattr(self, "_current_screen", None)
        if scr is not None and scr.winfo_exists():
            scr._back()

    def process_new(self, reread: bool = False):
        """«Обработать новые» (этап 4): прогон новых фото месяца. С 1.4.0 ход
        и итог — экраном в окне (process_screen.py), не окнами поверх; пока
        идёт, разбирать фото нельзя (решение 2а). Возвращается, когда
        обработка закончилась (итог остаётся на экране)."""
        if self.session is None or self._processing:
            return
        self._close_screen()
        self._processing = True                # счётчик новых фото — не во время обработки
        self._close_after = False
        self._set_busy(True)
        try:
            cmd = process_command(_BASE, str(self.session.folder.root), reread)
            title = "Обработка фото с ошибками" if reread else "Обработка новых фото"
            scr = self._show_screen(lambda: ProcessScreen(self, cmd, self.session.folder.stop_file, title, _BASE,
                                                          self._after_process))
            self._process_screen = scr
            self.protocol("WM_DELETE_WINDOW", self._close_while_processing)
            self.wait_variable(scr.done)
        finally:
            self._processing = False
            self._set_busy(False)
            self.protocol("WM_DELETE_WINDOW", self._on_close)
            if hasattr(self, "_watcher"):
                self._watcher.check_now()
        if self._close_after:
            self._on_close()

    def _close_while_processing(self):
        """Крестик окна во время обработки: остановить, закрыть — после текущего фото."""
        self._close_after = True
        self._process_screen.stop()
        self.toast.show("Останавливаем обработку — окно закроется после текущего фото", "info", ms=8000)

    def _after_process(self, code: int, lines: list[str]):
        self.session.folder.hide_service()       # обработка могла создать «результат» заново
        self.refresh_tabs()
        path = self.session.folder.results / "report.txt"
        report = path.read_text(encoding="utf-8") if code == 0 and path.is_file() else ""
        self._process_screen.show_result(code, report, lines, len(self._proc_tab._items),
                                         open_queue=self._open_queue_tab, back=self._show_lists)

    def _open_queue_tab(self):
        self._show_lists()
        self._notebook.select(self._proc_tab)
        self.after_idle(self._focus_list)

    def export(self) -> bool:
        """Кнопка «Выгрузить»: окно выбора формата (xlsx, xls, csv, ods), там же
        «Открыть файл» / «Открыть папку». True — выгружено."""
        dlg = ExportDialog(self, month_app.EXPORT_FORMATS, self.settings.export_format,
                           lambda fmt: (lambda rep: (rep.path, rep.text()))(self.export_as(fmt)),
                           self._open_export)
        return dlg.chosen is not None

    def export_as(self, fmt: str):
        """Выгрузка в формате fmt (показания.xlsx и лог.csv — всегда); формат
        запоминается — в следующий раз он выбран сразу. ExportLocked — файл открыт."""
        rep = self.session.export(formats=(fmt,))
        if fmt != self.settings.export_format:
            self.settings = replace(self.settings, export_format=fmt)
            save_settings(self.settings)
        return rep

    @staticmethod
    def _open_export(path, folder: bool = False):
        if folder:
            import subprocess
            try:                       # проводник с выделенным файлом
                subprocess.Popen(["explorer", "/select,", str(path)])
                return
            except OSError:
                path = Path(path).parent
        open_in_explorer(path)

    def _on_close(self):
        """Закрытие окна: выгрузка, потом выход. Файл открыт в Excel —
        «Повторить» после того, как его закроют, или выйти без выгрузки
        (показания в базе, выгрузить можно потом: gmr.py export)."""
        while self.session is not None:
            try:
                self.session.export()
                break
            except ExportLocked as e:
                if not messagebox.askretrycancel(
                    "Выгрузка",
                    f"{e}\n\n«Повторить» — после того как закроете файл.\n"
                    "«Отмена» — выйти без выгрузки (показания сохранены в базе).",
                ):
                    break
        self.destroy()

    def destroy(self):
        if getattr(self, "_watcher", None) is not None:
            self._watcher.stop()
        if getattr(self, "session", None) is not None:
            self.session.close()
            self.session = None
        super().destroy()

    def _with_photo_identity(self, row: dict, photo_path: str) -> Optional[dict]:
        """
        Отпечаток исходного фото — из автоматической строки, которая создала
        этот файл (файл в question/ пересохранён с аннотацией, по нему отпечаток
        не посчитать). Нужен, чтобы reader.py узнал разобранное фото по
        содержимому, а не только по имени. Возвращает эту строку (или None).
        """
        src = find_auto_row_for_output_file(
            self.log_rows, row.get("original_filename", ""),
            controller_name(photo_path, self.results_dir),
        )
        if src is not None and row.get("source") == "manual" and not row.get("photo_hash"):
            row["photo_hash"] = src.get("photo_hash", "")
            row["source_folder"] = src.get("source_folder", "")
        return src

    def append_log(self, row: dict, photo_path: str):
        """Решение оператора без показания: «Дубль», «Нечитаемо», «Нет в базе»."""
        self._with_photo_identity(row, photo_path)
        self.session.add_row(row)
        self.log_rows.append(row)

    def existing_reading(self, account: str):
        """Показание, уже записанное лицевому счёту из таблицы (None — нет;
        счёта нет в таблице — показание «Принять» и не пишет)."""
        if not account or self.session is None or self.df is None:
            return None
        if not (self.df[self.config.col_account_id].astype(str).str.strip() == account).any():
            return None
        return self.session.reading(account)

    def _sync_df_reading(self, account: str):
        """Текущее показание абонента в таблице окна — как в базе."""
        r = self.session.reading(account)
        acc = self.df[self.config.col_account_id].astype(str).str.strip()
        self.df.loc[acc == account, self.config.col_new_reading] = r.value if r else ""

    def accept_reading(self, row: dict, photo_path: str):
        """«Принять»: показание и строка лога одной записью в базу. Если лицевого
        счёта нет в таблице — только строка лога (как и раньше)."""
        src = self._with_photo_identity(row, photo_path)
        account = row.get("account_id", "")
        in_table = account and (self.df[self.config.col_account_id].astype(str).str.strip() == account).any()
        if in_table:
            date_name = (src or {}).get("original_filename") or row.get("original_filename", "")
            self.session.accept(row, date_name, keep=photo_path)
            self._sync_df_reading(account)
        else:
            self.session.add_row(row)
        self.log_rows.append(row)

    def fix_serial_and_accept(self, row: dict, photo_path: str, photo_serial: str) -> str:
        """«Серийник в базе с ошибкой»: номер в базе — как на фото, показание сразу."""
        src = self._with_photo_identity(row, photo_path)
        date_name = (src or {}).get("original_filename") or row.get("original_filename", "")
        old = self.session.fix_serial_and_accept(row, photo_serial, date_name, keep=photo_path)
        acc = self.df[self.config.col_account_id].astype(str).str.strip()
        self.df.loc[acc == row["account_id"], self.config.col_serial] = photo_serial
        self._sync_df_reading(row["account_id"])
        self._serial_index = None
        self.log_rows.append(row)
        return old

    def controller_code(self, controller: str) -> Optional[str]:
        """Код участка контролёра — по счетам таблицы из его строк лога (area_code.py)."""
        if self.df is None or not controller:
            return None
        table = set(self.df[self.config.col_account_id].astype(str).str.strip())
        return controller_code(accounts_by_controller(self.log_rows, table).get(controller, ()))

    def serial_index(self) -> Optional[SerialIndex]:
        """Индекс серийников таблицы для подсказки (строится один раз: номера,
        лицевые счета и последние показания program2 не меняет)."""
        if getattr(self, "_serial_index", None) is None and self.df is not None:
            self._serial_index = build_serial_index(self.df, self.config)
        return getattr(self, "_serial_index", None)

    # ─── Экран фото: всегда один ─────────────────────────────────────────────
    # 2026-11-21: Enter в списке открывает фото, а фокус клавиатуры оставался в
    # списке под экраном — Enter на экране срабатывал и в списке: открывался
    # второй экран под первым («экран поделился на два»). Теперь прежний экран
    # закрывается до того, как строится новый (его горячие клавиши общие —
    # bind_all), фокус уходит на экран, а список при открытом экране не
    # открывает фото.
    def screen_open(self) -> bool:
        scr = getattr(self, "_current_screen", None)
        return scr is not None and bool(scr.winfo_exists())

    def _drop_screen(self):
        scr = getattr(self, "_current_screen", None)
        if scr is None:
            return
        del self._current_screen
        try:
            scr._unbind_keys()
            scr.pack_forget()
            scr.destroy()
        except tk.TclError:            # уже закрыт
            pass

    def _show_screen(self, factory):
        self._drop_screen()
        self._notebook.pack_forget()
        screen = factory()
        screen.pack(fill=tk.BOTH, expand=True)
        self._current_screen = screen
        screen.focus_set()             # клавиши — экрану, а не списку под ним
        return screen

    def _show_lists(self):
        self._drop_screen()
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=px(12), pady=px(12))
        self.refresh_tabs()
        self._focus_list()

    def _focus_list(self):
        """Клавиши — списку открытой вкладки: Enter открывает выбранное фото."""
        if self.screen_open() or not self._notebook.select():
            return
        tree = getattr(self.nametowidget(self._notebook.select()), "_tree", None)
        if tree is not None:
            tree.focus_set()

    def open_edit_screen(self, photo_path: str, reason: str):
        return self._show_screen(lambda: EditScreen(self, photo_path, reason, on_back=self._back_from_screen))

    def open_verify_screen(self, item: dict, index: int = 0, total: Optional[int] = None):
        if total is None:
            total = len(list_verify_photos(self.results_dir, self.log_rows))
        self._set_verify_title(total)
        return self._show_screen(lambda: VerifyScreen(self, item, on_back=self._back_from_screen, index=index,
                                                      total=total))

    def _set_verify_title(self, total: int):
        """Вкладка «Проверка N» — сразу, а не только по возвращении к списку."""
        if hasattr(self, "_notebook"):
            self._notebook.tab(self._verify_tab, text=f"Проверка  {total}" if total else "Проверка")

    def open_verify_at(self, index: int):
        """
        Следующее фото на вкладке «Проверка» — как во вкладке «Очередь».
        Список строится заново: проверенное фото из него выпадает, поэтому
        после «Верно»/«Сохранить» следующее стоит на том же номере.
        """
        items = self._verify_tab.ordered(list_verify_photos(self.results_dir, self.log_rows))
        if index < len(items):
            self.open_verify_screen(items[index], index, total=len(items))
        else:
            self._show_lists()
            self.toast.show("Все фото на проверке просмотрены")

    def _back_from_screen(self):
        self._show_lists()

    def open_next_or_back(self, current_path: str):
        photos = self._proc_tab.ordered(list_question_photos(self.results_dir))
        if photos:
            reason, _, next_path = photos[0]
            self.open_edit_screen(next_path, reason)
        else:
            self._show_lists()
            self.toast.show("Все фото в очереди разобраны")

# ─── Вкладка «Очередь» (фото, которые программа не прочитала сама) ──────────
def _table_with_buttons(tab: ttk.Frame, columns: list, buttons: list) -> ttk.Treeview:
    """Таблица со скроллом и колонка кнопок справа (первая — главная). Столбец
    "#0" (первым в списке) — со значком: в нём стрелка расхода."""
    body = ttk.Frame(tab)
    body.pack(fill=tk.BOTH, expand=True, padx=px(16), pady=(0, px(16)))
    with_icon = columns[0][0] == "#0"
    tree = ttk.Treeview(body, columns=[c for c, *_ in columns if c != "#0"],
                        show="tree headings" if with_icon else "headings", selectmode="browse")
    for col, title, width, stretch in columns:
        tree.heading(col, text=title, anchor="w")
        tree.column(col, width=px(width), minwidth=px(60), stretch=stretch, anchor="w")
    tree.tag_configure("odd", background=ZEBRA)
    vsb = ttk.Scrollbar(body, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vsb.pack(side=tk.LEFT, fill=tk.Y)
    side = ttk.Frame(body)
    side.pack(side=tk.LEFT, fill=tk.Y, padx=(px(16), 0))
    for i, (text, command) in enumerate(buttons):
        action_button(side, text, command, kind="accent" if i == 0 else "secondary",
                      width=16).pack(fill=tk.X, pady=(0, px(8)))
    return tree


def _tab_header(tab: ttk.Frame, hint: str) -> tk.StringVar:
    head = ttk.Frame(tab)
    head.pack(fill=tk.X, padx=px(16), pady=(px(14), px(10)))
    var = tk.StringVar()
    ttk.Label(head, textvariable=var, style="Title.TLabel").pack(side=tk.LEFT)
    ttk.Label(head, text=hint, style="Muted.TLabel").pack(side=tk.RIGHT)
    return var


class ProcessingTab(ttk.Frame):
    def __init__(self, parent, app: MainWindow):
        super().__init__(parent)
        self.app = app
        self._build()
        self.refresh()

    def _build(self):
        self._counter_var = _tab_header(self, "Enter или 2× щелчок — открыть фото   ·   Del — нечитаемо   ·   "
                                              "щелчок по заголовку — сортировка")
        self._items: list[tuple[str, str, str]] = []
        self._tree = _table_with_buttons(
            self,
            [("reason", "Причина", 260, False), ("controller", "Контролёр", 200, False),
             ("filename", "Фото", 360, True)],
            [("Открыть", self._open), ("Нечитаемо", self._unreadable), ("⟳  Обновить", self.refresh)])
        self._sort = SortableTable(self._tree, [("reason", "Причина", "text"), ("controller", "Контролёр", "text"),
                                                ("filename", "Фото", "text")], self.refresh)
        self._empty = ttk.Label(self._tree, text="Очередь пуста — все фото разобраны",
                                background=SURFACE, foreground=MUTED, font=("Segoe UI", 12))
        self._tree.bind("<Double-Button-1>", lambda e: open_on_row(self._tree, e, self._open))
        self._tree.bind("<Return>", lambda e: self._open())
        self._tree.bind("<Delete>", lambda e: self._unreadable())    # как на экране разбора

    def refresh(self):
        self._tree.delete(*self._tree.get_children())
        self._items = self.ordered(list_question_photos(self.app.results_dir))
        for i, item in enumerate(self._items):
            self._tree.insert("", tk.END, values=tuple(self._cell(item, c) for c in ("reason", "controller", "filename")),
                              tags=("odd",) if i % 2 else ())

        remaining = len(self._items)
        processed = sum(
            1 for r in self.app.log_rows
            if r.get("source") == "manual"
            and r.get("outcome") not in ("", None)
        )
        self._counter_var.set(f"Ждут разбора: {remaining}" + (f"   ·   разобрано оператором: {processed}"
                                                              if processed else ""))
        if remaining:
            self._empty.place_forget()
            first = self._tree.get_children()[0]
            self._tree.selection_set(first)
            self._tree.focus(first)
        else:
            self._empty.place(relx=0.5, rely=0.4, anchor="center")

    def _cell(self, item: tuple[str, str, str], column: str) -> str:
        reason, fname, path = item
        if column == "reason":
            return REASON_LABELS.get(reason, reason)
        if column == "controller":
            return controller_name(path, self.app.results_dir)
        return fname

    def ordered(self, items: list) -> list:
        """Фото очереди в порядке списка (щелчок по заголовку меняет порядок)."""
        return self._sort.order(items, self._cell)

    def _selected_path(self) -> Optional[tuple[str, str]]:
        sel = self._tree.selection()
        if not sel:
            self.app.toast.show("Сначала выберите фото в списке", "info", ms=2500)
            return None
        idx = self._tree.index(sel[0])
        if idx < len(self._items):
            reason, _, path = self._items[idx]
            return reason, path
        return None

    def _open(self):
        if self.app.screen_open():          # Enter дошёл до списка под открытым экраном
            return
        sel = self._selected_path()
        if sel:
            reason, path = sel
            self.app.open_edit_screen(path, reason)

    def _unreadable(self):
        if self.app.screen_open():
            return
        sel = self._selected_path()
        if not sel:
            return
        reason, path = sel
        if not messagebox.askyesno("Нечитаемо", f"Фото не прочитать?\n{Path(path).name}"):
            return
        row = {k: "" for k in _LOG_COLUMNS}
        row.update({
            "original_filename": Path(path).name,
            "outcome":            "UNREADABLE",
            "source":             "manual",
            "processed_by":      self.app.settings.operator_name,
            "processed_at":      now_iso(),
            "notes":              "нечитаемо (из списка)",
        })
        dst_dir = str(controller_dir(path) / "unreadable")
        name = place_decision(row, path, dst_dir)
        self.app.append_log(row, path)
        try:
            move_photo(path, dst_dir, name)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return
        self.refresh()

# ─── Вкладка «Проверка» (показания, которые программа записала сама) ────────
OUTCOME_WORDS = {"PLUS": "больше прошлого", "MINUS": "меньше прошлого"}   # стрелка — значком


def delta_text(row: dict) -> str:
    """Разница с прошлым показанием со знаком: «+37», «-12» (не по модулю)."""
    d = number(row.get("delta"))
    if d is None:
        reading, last = number(row.get("reading")), number(row.get("last_reading"))
        if reading is None or last is None:
            return ""
        d = reading - last
    return f"{d:+.0f}" if d else "0"


class VerifyTab(ttk.Frame):
    def __init__(self, parent, app: MainWindow):
        super().__init__(parent)
        self.app = app
        self._build()
        self.refresh()

    def _build(self):
        self._counter_var = _tab_header(self, "Показания, которые программа записала сама   ·   Enter — открыть   ·   "
                                              "щелчок по заголовку — сортировка")
        self._items: list[dict] = []
        self._tree = _table_with_buttons(
            self,
            [("#0", "Расход", 175, False), ("controller", "Контролёр", 170, False),
             ("account_id", "Лицевой счёт", 130, False), ("reading", "Показание", 105, False),
             ("delta", "Разница", 95, False), ("processed_at", "Прочитано", 140, True)],
            [("Открыть", self._open), ("⟳  Обновить", self.refresh)])
        self._arrows = arrow_images(self)
        self._sort = SortableTable(
            self._tree, [("#0", "Расход", "text"), ("controller", "Контролёр", "text"),
                         ("account_id", "Лицевой счёт", "text"), ("reading", "Показание", "number"),
                         ("delta", "Разница", "number"), ("processed_at", "Прочитано", "date")], self.refresh)
        self._empty = ttk.Label(self._tree, text="Проверять нечего",
                                background=SURFACE, foreground=MUTED, font=("Segoe UI", 12))
        self._tree.bind("<Double-Button-1>", lambda e: open_on_row(self._tree, e, self._open))
        self._tree.bind("<Return>", lambda e: self._open())

    def refresh(self):
        self._tree.delete(*self._tree.get_children())
        self._items = self.ordered(list_verify_photos(self.app.results_dir, self.app.log_rows))
        for i, item in enumerate(self._items):
            outcome = item["log_row"].get("outcome", "")
            self._tree.insert("", tk.END, text="  " + self._cell(item, "#0"), image=self._arrows.get(outcome, ""),
                              values=tuple(self._cell(item, c) for c in
                                           ("controller", "account_id", "reading", "delta", "processed_at")),
                              tags=("odd",) if i % 2 else ())

        total = len(self._items)
        self._counter_var.set(f"На проверку: {total}")
        if total:
            self._empty.place_forget()
            first = self._tree.get_children()[0]     # как в «Очереди»: Enter сразу открывает фото
            self._tree.selection_set(first)
            self._tree.focus(first)
        else:
            self._empty.place(relx=0.5, rely=0.4, anchor="center")

    def _cell(self, item: dict, column: str) -> str:
        r = item["log_row"]
        if column == "#0":
            return OUTCOME_WORDS.get(r.get("outcome", ""), r.get("outcome", ""))
        if column == "controller":
            return controller_name(item["path"], self.app.results_dir)
        if column == "delta":
            return delta_text(r)
        if column == "processed_at":
            return r.get("processed_at", "")[:16].replace("T", " ")
        return r.get(column, "")

    def ordered(self, items: list) -> list:
        """Фото «Проверки» в порядке списка. Без щелчка по заголовку — вперемешку
        (честная выборка для процента ошибок, photos.list_verify_photos)."""
        return self._sort.order(items, self._cell)

    def _open(self):
        if self.app.screen_open():          # Enter дошёл до списка под открытым экраном
            return
        sel = self._tree.selection()
        if not sel:
            self.app.toast.show("Сначала выберите фото в списке", "info", ms=2500)
            return
        idx = self._tree.index(sel[0])
        if idx < len(self._items):
            self.app.open_verify_screen(self._items[idx], idx, total=len(self._items))
