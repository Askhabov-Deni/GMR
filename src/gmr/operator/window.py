"""Главное окно: вход, месяц, меню, вкладки «Обработка» и «Проверка»."""
import sqlite3
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import pandas as pd

from src.gmr.application import month as month_app
from src.gmr.application.month import ExportLocked, NotAMonth
from src.gmr.application.operator import OperatorSession
from src.gmr.domain import PipelineConfig, find_auto_row_for_output_file
from src.gmr.domain.preset import RecognitionPreset
from src.gmr.storage import LOG_COLUMNS as _LOG_COLUMNS
from src.gmr.storage import log_path_for
from src.gmr.ui.run_dialog import ProcessDialog, process_command, show_text

from .common import (
    _BASE,
    _LOG_FILE,
    CLR_BG,
    CLR_GREEN,
    REASON_LABELS,
    _log_handlers,
    log,
)
from .dialogs import LoginDialog, PresetDialog, SettingsDialog
from .edit_screen import EditScreen
from .files import move_photo, now_iso, open_in_explorer, place_decision
from .hints import SerialIndex, build_serial_index
from .photos import (
    controller_dir,
    controller_name,
    list_question_photos,
    list_verify_photos,
)
from .settings import AppSettings, default_month_name, load_settings, save_settings
from .verify_screen import VerifyScreen


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Обработка газовых счётчиков")
        self.geometry("960x640")
        self.minsize(800, 500)
        self.configure(bg=CLR_BG)

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
        if not self.settings.operator_name or not self.settings.month_dir:
            if not self._open_settings():
                self.destroy()
                return
        else:
            dlg = LoginDialog(self, self.settings)
            if dlg.action == "change":
                self._open_settings()          # «Отмена» — работа с прежними настройками

        while not self._load_data():          # папка месяца не создана — выбрать другую
            if not self._open_settings():
                self.destroy()
                return
        self._build_ui()

    def _open_settings(self) -> bool:
        dlg = SettingsDialog(self, self.settings, new_month=self.ask_new_month)
        if dlg.result:
            self.settings = dlg.result
            save_settings(self.settings)
            return True
        return False

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
                    f"{e}\n\nСоздать в этой папке новый месяц из таблицы компании?")):
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
    def _ask_table(self, parent=None) -> Optional[str]:
        return filedialog.askopenfilename(
            parent=parent or self, title="Таблица компании",
            filetypes=[("Таблицы", "*.xls *.xlsx *.csv"), ("Все файлы", "*.*")]) or None

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
        """Для нового месяца — как в открытом месяце, иначе — как было."""
        if self.session is not None:
            return month_app.month_preset(str(self.session.folder.root)) or RecognitionPreset()
        return RecognitionPreset()

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
        self._month_var.set(f"Месяц: {self.session.folder.root}")
        self.refresh_tabs()

    def new_month(self):
        """Меню «Месяц» → «Новый месяц…»: таблица компании и папка нового месяца."""
        folder = self.ask_new_month()
        if folder:
            self._switch_month(folder)

    def update_table(self):
        """Меню «Месяц» → «Обновить таблицу…»: компания прислала новую таблицу."""
        table = self._ask_table()
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

    def show_summary(self):
        """«Итог месяца»: то же, что `gmr.py month <папка>`."""
        show_text(self, "Итог месяца", month_app.month_summary(str(self.session.folder.root)))

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
        m.add_command(label="Итог месяца", command=self.show_summary)
        m.add_command(label="Открыть папку месяца", command=self.open_month_folder)
        m.add_separator()
        m.add_command(label="Новый месяц…", command=self.new_month)
        m.add_command(label="Обновить таблицу компании…", command=self.update_table)
        m.add_command(label="Открыть другой месяц…", command=self.choose_month)
        m.add_separator()
        m.add_command(label="Прочитать заново фото с ошибками…", command=self.reread_errors)
        m.add_command(label="Настройки распознавания…", command=self.edit_preset)
        menubar.add_cascade(label="Месяц", menu=m)
        self.configure(menu=menubar)        # self.config — это PipelineConfig, поэтому configure
        self._month_menu = m

        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=(8, 0))
        self._month_var = tk.StringVar(value=f"Месяц: {self.session.folder.root}")
        ttk.Label(top, textvariable=self._month_var, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        ttk.Button(top, text="⇩  Выгрузить показания", command=self.export).pack(side=tk.RIGHT)
        ttk.Button(top, text="Итог месяца", command=self.show_summary).pack(side=tk.RIGHT, padx=(0, 8))
        tk.Button(top, text="▶  Обработать новые", command=self.process_new,
                  bg=CLR_GREEN, fg="white", relief="flat", padx=10).pack(side=tk.RIGHT, padx=8)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._proc_tab = ProcessingTab(self._notebook, self)
        self._verify_tab = VerifyTab(self._notebook, self)

        self._notebook.add(self._proc_tab,   text="  Обработка  ")
        self._notebook.add(self._verify_tab, text="  Проверка  ")

    def refresh_tabs(self):
        self.reload()
        self._proc_tab.refresh()
        self._verify_tab.refresh()

    def _close_screen(self):
        """Вернуться к спискам (экран разбора закрывается)."""
        scr = getattr(self, "_current_screen", None)
        if scr is not None and scr.winfo_exists():
            scr._back()

    def process_new(self, reread: bool = False):
        """«Обработать новые» (этап 4): прогон новых фото месяца с полосой
        прогресса; пока он идёт, разбирать фото нельзя (решение 2а)."""
        if self.session is None:
            return
        self._close_screen()
        dlg = ProcessDialog(self, process_command(_BASE, str(self.session.folder.root), reread),
                            self.session.folder.stop_file, self._after_process, cwd=_BASE,
                            title="Обработка фото с ошибками" if reread else "Обработка новых фото")
        self._process_dialog = dlg
        self.wait_window(dlg)

    def _after_process(self, code: int, lines: list[str]):
        self.refresh_tabs()
        report = self.session.folder.results / "report.txt"
        if code == 0 and report.is_file():
            show_text(self, "Итог обработки", report.read_text(encoding="utf-8"))
        else:
            tail = "\n".join(lines[-25:]) or "(нет вывода)"
            show_text(self, "Обработка не удалась", f"Код завершения: {code}\n\n{tail}")

    def export(self) -> bool:
        """Кнопка «Выгрузить показания»: показания.xlsx и лог.csv."""
        try:
            rep = self.session.export()
        except ExportLocked as e:
            messagebox.showwarning("Выгрузка", str(e))
            return False
        messagebox.showinfo("Выгрузка", rep.text())
        return True

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

    def serial_index(self) -> Optional[SerialIndex]:
        """Индекс серийников таблицы для подсказки (строится один раз: номера,
        лицевые счета и последние показания program2 не меняет)."""
        if getattr(self, "_serial_index", None) is None and self.df is not None:
            self._serial_index = build_serial_index(self.df, self.config)
        return getattr(self, "_serial_index", None)

    def open_edit_screen(self, photo_path: str, reason: str):
        self._notebook.pack_forget()
        screen = EditScreen(self, photo_path, reason, on_back=self._back_from_screen)
        screen.pack(fill=tk.BOTH, expand=True)
        self._current_screen = screen

    def open_verify_screen(self, item: dict, index: int = 0):
        self._notebook.pack_forget()
        screen = VerifyScreen(self, item, on_back=self._back_from_screen, index=index)
        screen.pack(fill=tk.BOTH, expand=True)
        self._current_screen = screen

    def open_verify_at(self, index: int):
        """
        Следующее фото на вкладке «Проверка» — как во вкладке «Обработка».
        Список строится заново: проверенное фото из него выпадает, поэтому
        после «Верно»/«Сохранить» следующее стоит на том же номере.
        """
        items = list_verify_photos(self.results_dir, self.log_rows)
        if hasattr(self, "_current_screen"):
            self._current_screen.pack_forget()
            self._current_screen.destroy()
            del self._current_screen
        if index < len(items):
            self.open_verify_screen(items[index], index)
        else:
            self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            self.refresh_tabs()
            messagebox.showinfo("Готово", "Все фото на проверке просмотрены.")

    def _back_from_screen(self):
        if hasattr(self, "_current_screen"):
            self._current_screen.pack_forget()
            self._current_screen.destroy()
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.refresh_tabs()

    def open_next_or_back(self, current_path: str):
        photos = list_question_photos(self.results_dir)

        if hasattr(self, "_current_screen"):
            self._current_screen.pack_forget()
            self._current_screen.destroy()

        if photos:
            reason, fname, next_path = photos[0]
            screen = EditScreen(self, next_path, reason, on_back=self._back_from_screen)
            screen.pack(fill=tk.BOTH, expand=True)
            self._current_screen = screen
        else:
            self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            self.refresh_tabs()
            messagebox.showinfo("Готово", "Все фото в очереди обработаны!")

# ─── Таб «Обработка» ──────────────────────────────────────────────────────────
class ProcessingTab(ttk.Frame):
    def __init__(self, parent, app: MainWindow):
        super().__init__(parent)
        self.app = app
        self._build()
        self.refresh()

    def _build(self):
        self._counter_var = tk.StringVar(value="Загрузка...")
        ttk.Label(self, textvariable=self._counter_var, font=("Segoe UI", 10), wraplength=900).pack(
            anchor="w", padx=12, pady=(10, 4)
        )

        cols = ("reason", "controller", "filename")
        self._tree = ttk.Treeview(self, columns=cols, show="headings", selectmode="browse")
        self._tree.heading("reason",     text="Причина")
        self._tree.heading("controller", text="Контролёр")
        self._tree.heading("filename",   text="Имя файла")
        self._tree.column("reason",     width=240, stretch=False)
        self._tree.column("controller", width=160, stretch=False)
        self._tree.column("filename",   width=320)
        self._items: list[tuple[str, str, str]] = []

        vsb = ttk.Scrollbar(self, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)

        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0), pady=(0, 12))
        vsb.pack(side=tk.LEFT, fill=tk.Y, pady=(0, 12))

        btn_frame = ttk.Frame(self)
        btn_frame.pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=12)

        ttk.Button(btn_frame, text="✎  Открыть",    command=self._open,       width=18).pack(pady=6)
        ttk.Button(btn_frame, text="✗  Нечитаемо",  command=self._unreadable, width=18).pack(pady=6)
        ttk.Button(btn_frame, text="⟳  Обновить",   command=self.refresh,     width=18).pack(pady=6)

        self._tree.bind("<Double-Button-1>", lambda e: self._open())

    def refresh(self):
        self._tree.delete(*self._tree.get_children())
        self._items = list_question_photos(self.app.results_dir)
        for reason, fname, path in self._items:
            label = REASON_LABELS.get(reason, reason)
            ctrl = controller_name(path, self.app.results_dir)
            self._tree.insert("", tk.END, values=(label, ctrl, fname))

        remaining = len(self._items)
        processed = sum(
            1 for r in self.app.log_rows
            if r.get("source") == "manual"
            and r.get("outcome") not in ("", None)
        )
        total = remaining + processed
        self._counter_var.set(f"Всего: {total} | Обработано: {processed} | Осталось: {remaining}")

    def _selected_path(self) -> Optional[tuple[str, str]]:
        sel = self._tree.selection()
        if not sel:
            messagebox.showinfo("Выберите фото", "Сначала выберите фото из списка.")
            return None
        idx = self._tree.index(sel[0])
        if idx < len(self._items):
            reason, _, path = self._items[idx]
            return reason, path
        return None

    def _open(self):
        sel = self._selected_path()
        if sel:
            reason, path = sel
            self.app.open_edit_screen(path, reason)

    def _unreadable(self):
        sel = self._selected_path()
        if not sel:
            return
        reason, path = sel
        if not messagebox.askyesno("Нечитаемо", f"Пометить как нечитаемо?\n{Path(path).name}"):
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

# ─── Таб «Проверка» ───────────────────────────────────────────────────────────
class VerifyTab(ttk.Frame):
    def __init__(self, parent, app: MainWindow):
        super().__init__(parent)
        self.app = app
        self._build()
        self.refresh()

    def _build(self):
        self._counter_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._counter_var, font=("Segoe UI", 10), wraplength=900).pack(
            anchor="w", padx=12, pady=(10, 4)
        )

        cols = ("outcome", "controller", "account_id", "reading", "processed_at")
        self._tree = ttk.Treeview(self, columns=cols, show="headings", selectmode="browse")
        self._tree.heading("controller",   text="Контролёр")
        self._tree.column("controller",   width=140, stretch=False)
        self._tree.heading("outcome",      text="Результат")
        self._tree.heading("account_id",   text="Account ID")
        self._tree.heading("reading",      text="Показания")
        self._tree.heading("processed_at", text="Обработано")
        self._tree.column("outcome",      width=80,  stretch=False)
        self._tree.column("account_id",   width=140, stretch=False)
        self._tree.column("reading",      width=100, stretch=False)
        self._tree.column("processed_at", width=200)

        vsb = ttk.Scrollbar(self, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0), pady=(0, 12))
        vsb.pack(side=tk.LEFT, fill=tk.Y, pady=(0, 12))

        btn_frame = ttk.Frame(self)
        btn_frame.pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=12)
        ttk.Button(btn_frame, text="✎  Открыть",  command=self._open,   width=18).pack(pady=6)
        ttk.Button(btn_frame, text="⟳  Обновить", command=self.refresh, width=18).pack(pady=6)

        self._tree.bind("<Double-Button-1>", lambda e: self._open())
        self._items: list[dict] = []

    def refresh(self):
        self._tree.delete(*self._tree.get_children())
        self._items = list_verify_photos(self.app.results_dir, self.app.log_rows)
        for item in self._items:
            r = item["log_row"]
            self._tree.insert("", tk.END, values=(
                r.get("outcome", ""),
                controller_name(item["path"], self.app.results_dir),
                r.get("account_id", ""),
                r.get("reading", ""),
                r.get("processed_at", "")[:16],
            ))

        total = len(self._items)
        self._counter_var.set(f"Всего для проверки: {total}")

    def _open(self):
        sel = self._tree.selection()
        if not sel:
            messagebox.showinfo("Выберите фото", "Сначала выберите фото из списка.")
            return
        idx = self._tree.index(sel[0])
        if idx < len(self._items):
            self.app.open_verify_screen(self._items[idx], idx)
