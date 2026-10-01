"""
src/gmr/console.py — консоль, которая не роняет программу (2026-10-01).

На русской Windows, если вывод идёт в файл (`python gmr.py process > run.txt`)
или программа запущена без окна консоли, Python пишет в кодировке cp1251. В ней
нет символов ⚠️ ✅ 📁 🔄 из сообщений программы и моделей, и print падает с
UnicodeEncodeError — прогон останавливается (models/cnn/infer_cnn.py печатает
«🔄» уже при загрузке весов). После safe_console() такие символы выводятся как
«?», а программа работает дальше.
"""
import sys


def safe_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):     # None без консоли, у pytest — своя замена
            stream.reconfigure(errors="replace")
