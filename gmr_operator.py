"""
gmr_operator.py — «GMR Оператор»: окно оператора. Разбор фото, которые
программа не прочитала сама, и выборочная проверка её показаний.

Запуск: ярлык «GMR Оператор» на рабочем столе (`python gmr.py shortcut`)
или `python gmr_operator.py`. Код окна — src/gmr/operator/; program2.py —
прежнее имя того же окна (старые ярлыки и тесты).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # запуск ярлыком из другой папки

from src.gmr.console import safe_console  # noqa: E402
from src.gmr.operator.window import MainWindow  # noqa: E402


def main() -> None:
    safe_console()
    MainWindow().mainloop()


if __name__ == "__main__":
    main()
