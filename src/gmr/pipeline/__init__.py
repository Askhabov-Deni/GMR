"""
src/gmr/pipeline — автоматическая обработка фото месяца.

  photo.py   — одно фото: модели → серийник → строка таблицы → показание → исход
  run.py     — прогон по папке месяца: что читать, запись в базу, раскладка, журнал
  output.py  — строка лога, фото с подписью, отладочные кропы цифр
  report.py  — отчёты report.txt

Запуск — `python gmr.py process <месяц>` (или `python reader.py <месяц>`).
Как результаты использует окно оператора — docs/contract_reader_program2.md.
"""
from .photo import NoteContext, RunState, account_groups_of, note_context_of, process_photo, serial_groups_of
from .run import PhotosInRoot, run_pipeline

__all__ = [
    "NoteContext",
    "PhotosInRoot",
    "RunState",
    "account_groups_of",
    "note_context_of",
    "process_photo",
    "run_pipeline",
    "serial_groups_of",
]
