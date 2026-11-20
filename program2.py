"""
program2.py — окно оператора: ручной разбор фото из question/, с которыми не
справилась автоматическая обработка, и выборочная проверка её результатов.
Запуск: `python program2.py` (или ярлык «Счётчики»: `python gmr.py shortcut`).

Код окна — src/gmr/operator/; здесь — точка входа и прежние имена
(program2.MainWindow, program2.list_verify_photos, …). Что окно читает и
пишет — docs/contract_reader_program2.md.
"""
import sys
from pathlib import Path

# папка проекта — в sys.path: запуск ярлыком из другой папки
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Имена, которыми пользуются тесты и инструменты (program2.<имя>)
from tkinter import filedialog, messagebox, simpledialog  # noqa: E402, F401
import tkinter as tk  # noqa: E402, F401

from src.gmr.application.month import ExportLocked, NotAMonth  # noqa: E402, F401
from src.gmr.console import safe_console  # noqa: E402
from src.gmr.domain import PipelineConfig  # noqa: E402, F401
from src.gmr.render import read_image, write_image  # noqa: E402, F401
from src.gmr.ui.run_dialog import ProcessDialog, process_command, show_text  # noqa: E402, F401
from src.gmr.operator.common import (  # noqa: E402, F401
    _BASE,
    CLR_BG,
    CLR_BROWN,
    CLR_GRAY,
    CLR_GREEN,
    CLR_ORANGE,
    CLR_RED,
    CLR_WHITE,
    log,
    _LOG_FILE,
    _log_handlers,
    _log_handlers_for,
    PHOTO_EXTS,
    QUESTION_SUBFOLDERS,
    REASON_LABELS,
)
from src.gmr.operator.settings import (  # noqa: E402, F401
    AppSettings,
    default_month_name,
    load_settings,
    RU_MONTHS,
    save_settings,
    SETTINGS_FILE,
)
from src.gmr.operator.dialogs import (  # noqa: E402, F401
    LoginDialog,
    PresetDialog,
    SettingsDialog,
)
from src.gmr.operator.model_bundle import (  # noqa: E402, F401
    _abs_model,
    ModelBundle,
)
from src.gmr.operator.photos import (  # noqa: E402, F401
    controller_dir,
    controller_dirs,
    controller_name,
    find_verify_row,
    list_question_photos,
    list_verify_photos,
    _pick_verify_row,
    RESULT_SUBFOLDERS,
    _to_float,
    VERIFY_FOLDER_OUTCOME,
    _VerifyIndex,
)
from src.gmr.operator.decisions import (  # noqa: E402, F401
    accounts_with_serial,
    not_in_db_serial,
    plan_verify_correction,
    VerifyCorrection,
)
from src.gmr.operator.hints import (  # noqa: E402, F401
    build_serial_index,
    CHOICE_NOTE,
    HINT_NOTE,
    serial_choices,
    serial_hints,
    SerialHint,
    SerialIndex,
)
from src.gmr.operator.files import (  # noqa: E402, F401
    append_db_serial_fix,
    DB_SERIAL_FIX_COLUMNS,
    DB_SERIAL_FIX_LIST,
    describe_reading,
    free_photo_path,
    move_photo,
    now_iso,
    open_in_explorer,
    place_decision,
    redraw_annotation,
    table_serial_for_account,
)
from src.gmr.operator.markup import (  # noqa: E402, F401
    _free_name,
    save_cnn_markup,
    save_crnn_markup,
    save_meter_markup,
)
from src.gmr.operator.widgets import (  # noqa: E402, F401
    bind_tree,
    ReadingWidget,
    unbind_tree,
)
from src.gmr.operator.window import (  # noqa: E402, F401
    MainWindow,
    ProcessingTab,
    VerifyTab,
)
from src.gmr.operator.edit_screen import (  # noqa: E402, F401
    EditScreen,
)
from src.gmr.operator.verify_screen import (  # noqa: E402, F401
    VerifyScreen,
)


if __name__ == "__main__":
    safe_console()
    app = MainWindow()
    app.mainloop()
