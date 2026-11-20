"""
reader.py — автоматическая обработка фото месяца.

Запуск: `python gmr.py process <папка месяца>` или `python reader.py <папка
месяца>` (одинаковые настройки — значения по умолчанию PipelineConfig).
Фото — из <месяц>/фото, результат — в <месяц>/результат, абоненты, показания
и лог — в базе месяца.

Код обработки — src/gmr/pipeline/; здесь — точка входа и прежние имена
(`reader.process_photo`, `reader.run_pipeline`, `reader.PipelineConfig`…).
"""
import sys

from src.gmr.application.month import NotAMonth
from src.gmr.console import safe_console
from src.gmr.domain import OUTCOME_FOLDER, Outcome, PhotoResult, PipelineConfig
from src.gmr.pipeline import (
    NoteContext,
    PhotosInRoot,
    RunState,
    account_groups_of,
    note_context_of,
    process_photo,
    run_pipeline,
    serial_groups_of,
)

__all__ = [
    "NotAMonth",
    "NoteContext",
    "OUTCOME_FOLDER",
    "Outcome",
    "PhotoResult",
    "PhotosInRoot",
    "PipelineConfig",
    "RunState",
    "account_groups_of",
    "note_context_of",
    "process_photo",
    "run_pipeline",
    "serial_groups_of",
]


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("Запуск: python reader.py <папка месяца>  (то же, что python gmr.py process <папка месяца>)")
        return 2
    try:
        run_pipeline(PipelineConfig(month_dir=argv[0]))
    except (NotAMonth, PhotosInRoot) as e:
        print(f"ОШИБКА: {e}")
        return 1
    return 0


if __name__ == "__main__":
    safe_console()
    sys.exit(main(sys.argv[1:]))
