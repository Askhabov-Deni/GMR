"""Отчёты прогона: report.txt в каждой папке результата и общий."""
from pathlib import Path
from typing import Optional

from src.gmr.domain import Outcome, RunSelection


# ─── Формирование отчёта ──────────────────────────────────────────────────────

# Группы исходов для сводных метрик отчёта.
_SUCCESS_OUTCOMES = (Outcome.PLUS, Outcome.MINUS)
_QUESTION_OUTCOMES = (
    Outcome.NO_METER,
    Outcome.NO_SERIAL,
    Outcome.SERIAL_LOW_CONF,
    Outcome.SERIAL_NOT_FOUND,
    Outcome.DIGITS_ERROR,
    Outcome.SUSPICIOUS,
    Outcome.ERROR,
    Outcome.SERIAL_AMBIGUOUS,
)


_SKIP_LABELS = {
    RunSelection.SKIP_DONE:    "Уже разобраны раньше (пропущены)",
    RunSelection.SKIP_WAITING: "Ждут оператора с прошлых прогонов",
    RunSelection.SKIP_COPY:    "Тот же файл ещё раз (копия)",
}


def skip_lines(skipped: Optional[dict]) -> list[str]:
    return [f"{label}: {skipped[k]}" for k, label in _SKIP_LABELS.items() if skipped and skipped.get(k)]


_NOT_TAKEN_SHOWN = 10


def where(folder: str, folder_report: bool = False) -> str:
    if folder_report:
        return ""
    return f" в «{folder}»" if folder else " в корне фото\\"


def not_taken_line(names: list[str], where: str = "") -> str:
    shown = ", ".join(names[:_NOT_TAKEN_SHOWN]) + (" …" if len(names) > _NOT_TAKEN_SHOWN else "")
    return f"⚠ Не взяты{where} (не фото .jpg/.jpeg/.png или папка внутри): {len(names)} — {shown}"


def format_report(title: str, stats: dict["Outcome", int], skipped: Optional[dict] = None,
                   not_taken: Optional[dict] = None, folder_report: bool = False) -> str:
    """
    Формирует текстовый отчёт по статистике этого прогона: количество и
    процент по каждому исходу + сводные метрики (успешно / на проверку /
    дубли); skipped — сколько фото пропущено (RunSelection → число);
    not_taken — что программа не взяла: {папка: [имена]} ("" — корень фото\\);
    в отчёте папки контролёра (folder_report) — без имени папки.
    """
    total = sum(stats.values())
    lines = []
    lines.append("=" * 55)
    lines.append(title)
    lines.append("=" * 55)
    lines.append(f"Новых фото в этом прогоне: {total}")
    lines += skip_lines(skipped)
    for folder, names in (not_taken or {}).items():
        if names:
            lines.append(not_taken_line(names, where(folder, folder_report)))
    lines.append("")

    if total == 0:
        lines.append("(новых фото нет)")
        lines.append("=" * 55)
        return "\n".join(lines)

    for outcome, count in stats.items():
        if count:
            pct = count / total * 100
            lines.append(f"  {outcome.name:<22} {count:>6}   {pct:5.1f}%")

    success = sum(stats.get(o, 0) for o in _SUCCESS_OUTCOMES)
    question = sum(stats.get(o, 0) for o in _QUESTION_OUTCOMES)
    repeat = stats.get(Outcome.REPEAT, 0)

    lines.append("")
    lines.append("-" * 55)
    lines.append(
        f"  Успешно (PLUS+MINUS):    {success:>6}   {success / total * 100:5.1f}%"
    )
    lines.append(
        f"  На проверку (question/): {question:>6}   {question / total * 100:5.1f}%"
    )
    lines.append(
        f"  Дубли (REPEAT):          {repeat:>6}   {repeat / total * 100:5.1f}%"
    )
    lines.append("=" * 55)

    return "\n".join(lines)


def save_report(output_base: str, report_text: str) -> str:
    """Сохраняет отчёт в output_base/report.txt, возвращает путь к файлу."""
    Path(output_base).mkdir(parents=True, exist_ok=True)
    report_path = str(Path(output_base) / "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    return report_path


def format_folder_ranking(per_folder_stats: dict[str, dict["Outcome", int]]) -> str:
    """
    Формирует рейтинг подпапок (контролёров) по доле успешных обработок
    (PLUS+MINUS от общего числа фото). Помогает увидеть, кто фотографирует
    качественнее, а кто хуже.
    """
    rows = []
    for name, stats in per_folder_stats.items():
        total = sum(stats.values())
        if total == 0:
            continue
        success = sum(stats.get(o, 0) for o in _SUCCESS_OUTCOMES)
        question = sum(stats.get(o, 0) for o in _QUESTION_OUTCOMES)
        rows.append((name, total, success, question, success / total * 100))

    if not rows:
        return ""

    rows.sort(key=lambda r: r[4], reverse=True)

    lines = []
    lines.append("=" * 55)
    lines.append("РЕЙТИНГ ПАПОК ПО % УСПЕШНЫХ ОБРАБОТОК")
    lines.append("=" * 55)
    lines.append(f"{'Папка':<30} {'Всего':>6} {'Успех':>6} {'На пров.':>9} {'%':>7}")
    lines.append("-" * 55)
    for name, total, success, question, pct in rows:
        lines.append(f"{name:<30} {total:>6} {success:>6} {question:>9} {pct:>6.1f}%")
    lines.append("=" * 55)

    return "\n".join(lines)
