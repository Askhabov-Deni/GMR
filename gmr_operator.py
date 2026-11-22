"""
gmr_operator.py — «GMR»: одна программа для оператора. Разбор фото,
которые программа не прочитала сама, выборочная проверка её показаний и
обработка новых фото — всё из этого окна.

Запуск: ярлык «GMR» на рабочем столе (`python gmr.py shortcut`)
или `python gmr_operator.py`. В собранной программе — «GMR.exe».

С параметрами — команды gmr.py без окна: `GMR.exe process <месяц>`.
Так окно само запускает обработку фото (отдельным процессом: модели грузятся
только на время обработки, а если обработка упадёт, окно останется открытым).
Код окна — src/gmr/operator/; program2.py — прежнее имя того же окна.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # запуск ярлыком из другой папки


def main() -> int:
    from src.gmr.frozen import attach_std_streams
    if sys.argv[1:]:
        # команда без окна (обработка, которую запускает окно): вывод — тому, кто запустил
        attach_std_streams()
        import gmr
        return gmr.main(sys.argv[1:]) or 0
    from src.gmr.console import safe_console
    from src.gmr.operator.theme import enable_dpi_awareness
    from src.gmr.operator.window import MainWindow
    safe_console()
    enable_dpi_awareness()      # чёткий текст при масштабе Windows 125–150% (до первого окна)
    try:        # своя кнопка на панели задач со значком GMR, а не значок Python
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("GMR")
    except (AttributeError, OSError):    # не Windows
        pass
    MainWindow().mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
