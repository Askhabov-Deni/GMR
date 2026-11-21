"""Очереди окна: фото на разбор (question/) и на проверку (plus/, minus/)."""
from pathlib import Path
from typing import Optional

from .common import PHOTO_EXTS, QUESTION_SUBFOLDERS

RESULT_SUBFOLDERS = ("question", "plus", "minus", "repeat")


def controller_dirs(results_dir: str) -> list[Path]:
    """
    Папки контролёров в «результате» месяца (<месяц>/результат/<контролёр>), а
    если фото лежали без подпапок — сама папка результата. С этапа 2.3 окно
    показывает всех контролёров сразу («0 из 0» больше не бывает).
    """
    base = Path(results_dir) if results_dir else None
    if base is None or not base.is_dir():
        return []
    dirs = [base] if any((base / f).is_dir() for f in RESULT_SUBFOLDERS) else []
    dirs += sorted(d for d in base.iterdir()
                   if d.is_dir() and any((d / f).is_dir() for f in RESULT_SUBFOLDERS))
    return dirs


def controller_dir(photo_path: str) -> Path:
    """Папка контролёра, в которой лежит фото (…/<контролёр>/question/<причина>/фото
    или …/<контролёр>/plus/фото)."""
    p = Path(photo_path)
    for parent in p.parents:
        if parent.name in RESULT_SUBFOLDERS:
            return parent.parent
    return p.parent


def controller_name(photo_path: str, results_dir: str) -> str:
    """Имя подпапки контролёра (source_folder в логе); "" — фото без подпапок."""
    d = controller_dir(photo_path)
    return "" if Path(results_dir) == d else d.name


def list_question_photos(results_dir: str) -> list[tuple[str, str, str]]:
    """
    Возвращает список (reason, filename, full_path) из question/ всех
    контролёров. Сортировка: по приоритету QUESTION_SUBFOLDERS, затем по
    контролёру и имени.
    """
    result = []
    dirs = controller_dirs(results_dir)
    for subfolder in QUESTION_SUBFOLDERS:
        for d in dirs:
            sub = d / "question" / subfolder
            if not sub.exists():
                continue
            for p in sorted(sub.iterdir()):
                if p.is_file() and p.suffix.lower() in PHOTO_EXTS:
                    result.append((subfolder, p.name, str(p)))
    return result

# Папка в output/ → исход строки лога, которую проверяют в этой папке
VERIFY_FOLDER_OUTCOME = {"plus": "PLUS", "minus": "MINUS"}


def find_verify_row(log_rows: list[dict], file_name: str, folder: str,
                    controller: Optional[str] = None) -> Optional[dict]:
    """
    Авто-строка лога для фото из plus/ или minus/: исход — как у папки
    (PLUS/MINUS), имя файла совпадает с final_filename или original_filename
    (с расширением или без). Строки REPEAT не подходят: у второго фото того же
    счётчика final_filename тот же, но показаний в строке нет (ошибка до
    2026-09-30 — вкладка «Проверка» брала последнюю строку с этим именем).
    Среди подходящих — последняя; если задан controller — предпочитаются
    строки этого контролёра (source_folder).
    """
    wanted = VERIFY_FOLDER_OUTCOME.get(folder)
    stem = Path(file_name).stem
    found = []
    for r in log_rows:
        if r.get("source") != "auto" or r.get("outcome") != wanted:
            continue
        names = {n for n in (r.get("final_filename", ""), r.get("original_filename", "")) if n}
        names |= {Path(n).stem for n in names}
        if file_name in names or stem in names:
            found.append(r)
    return _pick_verify_row(found, controller)


def _pick_verify_row(found: list[dict], controller: Optional[str]) -> Optional[dict]:
    """Из подходящих строк (в порядке лога) — последняя; если задан controller —
    последняя строка этого контролёра, а без таких — просто последняя."""
    if controller is not None:
        same = [r for r in found if (r.get("source_folder") or "") == controller]
        if same:
            return same[-1]
    return found[-1] if found else None


class _VerifyIndex:
    """Тот же поиск, что find_verify_row, но лог просматривается один раз:
    строки PLUS/MINUS разложены по (исход, имя файла / имя без расширения).
    Раньше список «Проверки» сравнивал каждое фото со всем логом — на месяце
    в 1000 фото это 18 с на каждое «Верно» (замер 2026-11-20)."""

    def __init__(self, log_rows: list[dict]):
        wanted = set(VERIFY_FOLDER_OUTCOME.values())
        self._by_name: dict[tuple, list[tuple[int, dict]]] = {}
        for pos, r in enumerate(log_rows):
            outcome = r.get("outcome")
            if r.get("source") != "auto" or outcome not in wanted:
                continue
            names = {n for n in (r.get("final_filename", ""), r.get("original_filename", "")) if n}
            names |= {Path(n).stem for n in names}
            for n in names:
                self._by_name.setdefault((outcome, n), []).append((pos, r))

    def find(self, file_name: str, folder: str, controller: Optional[str] = None) -> Optional[dict]:
        wanted = VERIFY_FOLDER_OUTCOME.get(folder)
        hits = dict(self._by_name.get((wanted, file_name), ()))
        hits.update(self._by_name.get((wanted, Path(file_name).stem), ()))
        return _pick_verify_row([hits[p] for p in sorted(hits)], controller)


def list_verify_photos(results_dir: str, log_rows: list[dict]) -> list[dict]:
    """Фото из plus/ и minus/ всех контролёров, у которых авто-строка PLUS/MINUS
    ещё не проверена."""
    index = _VerifyIndex(log_rows)
    result = []
    for ctrl in controller_dirs(results_dir):
        name = "" if ctrl == Path(results_dir) else ctrl.name
        for folder in ("plus", "minus"):
            d = ctrl / folder
            if not d.exists():
                continue
            for p in sorted(d.iterdir()):
                if not (p.is_file() and p.suffix.lower() in PHOTO_EXTS):
                    continue
                row = index.find(p.name, folder, controller=name)
                if row is not None and not row.get("verified_by"):
                    result.append({"path": str(p), "log_row": row, "folder": folder})
    # Порядок — случайный, но постоянный (по отпечатку фото): сколько бы фото
    # оператор ни проверил, это честная выборка из всех контролёров и дней —
    # по ней считается процент ошибок (src/gmr/application/quality.py)
    result.sort(key=lambda it: _shuffle_key(it["log_row"], it["path"]))
    return result


def _shuffle_key(row: dict, path: str) -> str:
    import hashlib
    return hashlib.sha1((row.get("photo_hash") or Path(path).name).encode("utf-8")).hexdigest()


def _to_float(v) -> Optional[float]:
    s = str(v).strip() if v is not None else ""
    if s in ("", "nan", "None"):
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None
