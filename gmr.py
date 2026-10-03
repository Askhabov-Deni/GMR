"""
gmr.py — одна точка входа для всех команд проекта (Фаза 7). Запускать из
папки проекта:

  python gmr.py month <папка> --table <файл>  создать месяц из таблицы компании / загрузить обновлённую
  python gmr.py month <папка>                 сводка месяца
  python gmr.py export <папка>                выгрузить показания.xlsx и лог.csv ещё раз
  python gmr.py process <папка месяца>        прогон reader.py: новые фото из фото\\ → результат\\, база, выгрузка
  python gmr.py process <папка месяца> --reread   то же + заново фото с ошибками (после замены модели)
  python gmr.py analyze <лог.csv> [--table <таблица>] [--details]   качество по логу
  python gmr.py inspect <модель> <фото|папка> [...]                  посмотреть модель
  python gmr.py weights [--write]                                     файлы весов
  python gmr.py check                                                 проверка перед коммитом
  python gmr.py <команда> --help                                      подробности

Ничего нового команды не делают: process вызывает тот же run_pipeline с теми же
настройками, что `python reader.py <папка месяца>` (значения по умолчанию
PipelineConfig); остальные — тонкие обёртки над tools/. Окно оператора —
`python program2.py`.
"""
import argparse
import sys
from typing import Optional


def _process(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="python gmr.py process",
        description="Автоматическая обработка фото месяца (reader.py): новые фото из "
                    "<месяц>\\фото\\<контролёр>, результат в <месяц>\\результат, показания и лог — "
                    "в базе месяца, в конце — выгрузка показания.xlsx. Уже разобранные фото "
                    "пропускаются, фото с ошибками ждут оператора (заново — с --reread).",
    )
    p.add_argument("month", help="папка месяца (создаётся командой `gmr.py month`)")
    p.add_argument("--move", action="store_true", help="перемещать фото, а не копировать")
    p.add_argument("--reread", action="store_true",
                   help="прочитать заново фото с ошибками, которые ждут оператора (например после "
                        "замены модели); обычный прогон читает только новые фото")
    args = p.parse_args(argv)

    import reader
    try:
        reader.run_pipeline(reader.PipelineConfig(month_dir=args.month, move_photos=args.move,
                                                  reread_errors=args.reread))
    except (reader.NotAMonth, reader.PhotosInRoot) as e:
        print(f"ОШИБКА: {e}")
        return 1
    return 0


def _month(argv):
    p = argparse.ArgumentParser(
        prog="python gmr.py month",
        description="Папка месяца: создать её из таблицы компании (.xls/.xlsx/.csv) или загрузить "
                    "обновлённую таблицу; после загрузки — выгрузка показания.xlsx. "
                    "Без --table — сводка месяца.",
    )
    p.add_argument("folder", help="папка месяца, например D:\\GMR\\Октябрь_2026")
    p.add_argument("--table", help="таблица компании: при первом запуске создаёт месяц, потом — обновляет")
    args = p.parse_args(argv)
    from src.gmr.application import month
    try:
        if not args.table:
            print(month.month_summary(args.folder))
            return 0
        print(month.load_table(args.folder, args.table).text())
        print(month.export_month(args.folder).text())
    except (ValueError, OSError, month.ExportLocked) as e:
        print(f"ОШИБКА: {e}")
        return 1
    return 0


def _export(argv):
    p = argparse.ArgumentParser(prog="python gmr.py export",
                                description="Выгрузить показания.xlsx и лог.csv из базы месяца.")
    p.add_argument("folder", help="папка месяца")
    args = p.parse_args(argv)
    from src.gmr.application import month
    try:
        print(month.export_month(args.folder).text())
    except (ValueError, OSError, month.ExportLocked) as e:
        print(f"ОШИБКА: {e}")
        return 1
    return 0


def _analyze(argv):
    from tools import analyze_log
    analyze_log.main(argv)


def _inspect(argv):
    from tools import inspect_model
    inspect_model.main(argv)


def _weights(argv):
    from tools import weights_manifest
    weights_manifest.main(argv)


def _check(argv):
    from tools import check
    return check.main(argv)


COMMANDS = {
    "month":   (_month,   "папка месяца: создать / загрузить обновлённую таблицу / сводка"),
    "export":  (_export,  "выгрузить показания.xlsx и лог.csv"),
    "process": (_process, "автоматическая обработка фото (reader.py)"),
    "analyze": (_analyze, "качество чтения по логу (tools/analyze_log.py)"),
    "inspect": (_inspect, "посмотреть, что делает модель (tools/inspect_model.py)"),
    "weights": (_weights, "какие файлы весов в работе (tools/weights_manifest.py)"),
    "check":   (_check,   "проверка перед коммитом: данные, код, тесты (tools/check.py)"),
}


def main(argv: Optional[list[str]] = None) -> int:
    from src.gmr.console import safe_console
    safe_console()      # ⚠️ и т.п. при выводе в файл на русской Windows — «?», а не падение
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print(__doc__.strip())
        print("\nКоманды:")
        for name, (_, help_) in COMMANDS.items():
            print(f"  {name:<8} {help_}")
        return 0 if (not argv or argv[0] in ("-h", "--help")) else 2
    return COMMANDS[argv[0]][0](argv[1:]) or 0


if __name__ == "__main__":
    sys.exit(main())
