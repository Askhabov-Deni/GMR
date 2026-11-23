"""
Прогон по папке месяца: какие фото читать, чтение, запись показаний и лога
в базу (одной транзакцией на фото), раскладка по папкам, отчёты, журнал
прогона, выгрузка. Запуск — `python gmr.py process <месяц>`.
"""
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.gmr.application.cycle import close_waiting, free_name, move_files, result_file
from src.gmr.application.month import (
    ExportLocked,
    NotAMonth,
    export_month,
    is_month,
    month_preset,
)
from src.gmr.domain import (
    OUTCOME_FOLDER,
    Outcome,
    PhotoResult,
    PipelineConfig,
    RunSelection,
    RunSelectionPolicy,
)
from src.gmr.domain.area_code import accounts_by_controller, controller_code, only_code
from src.gmr.domain.photo_date import reading_date
from src.gmr.domain.serial_match import lookup_variants, normalize_serial
from src.gmr.ml.loader import default_device, load_models
from src.gmr.storage import photo_fingerprint
from src.gmr.storage.backup import KEEP, backup_sqlite, remove_old
from src.gmr.version import __version__
from src.gmr.storage.month import MonthDB, MonthFolder, Reading, Recognition, now_text

from .output import make_log_row, save_annotated
from .photo import (
    RunState,
    account_groups_of,
    note_context_of,
    read_photo,
    serial_groups_of,
)
from .report import (
    format_folder_ranking,
    format_report,
    not_taken_line,
    save_report,
    skip_lines,
    where,
)
from .seen import safe_what_models_saw

_run_selection = RunSelectionPolicy()


# ─── Вспомогательные функции для определения структуры входной папки ────────

_PHOTO_EXTS = {".jpg", ".jpeg", ".png"}


def list_photos(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _PHOTO_EXTS
    )


class PhotosInRoot(ValueError):
    """В корне <месяц>/фото лежат фото — фото кладут только в папки
    контролёров (решение владельца 2026-10-03, этап 3, пункт 4б)."""


def check_no_photos_in_root(photos_dir: Path) -> None:
    loose = list_photos(photos_dir) if photos_dir.is_dir() else []
    if loose:
        names = ", ".join(p.name for p in loose[:5]) + (" …" if len(loose) > 5 else "")
        raise PhotosInRoot(
            f"В папке {photos_dir} лежат фото без папки контролёра ({len(loose)} шт.: {names}). "
            f"Фото кладут только в папки контролёров: фото\\<контролёр>\\. "
            f"Разложите их и запустите снова — прогон не начинался.")


# служебные файлы Windows/macOS и временные файлы Office — не «файлы контролёров»
_SYSTEM_FILES = {"thumbs.db", "desktop.ini", ".ds_store"}


def not_taken_files(folder: Path, with_dirs: bool = True) -> list[str]:
    """Что в папке программа не возьмёт (этап 3, решение владельца
    2026-10-03 — перечислять в отчёте): файлы не .jpg/.jpeg/.png и папки
    внутри папки контролёра (фото в них не читаются; имя — с «\\» на конце).
    Служебные файлы не считаются."""
    if not folder.is_dir():
        return []
    out = []
    for p in sorted(folder.iterdir(), key=lambda x: x.name):
        if p.name.lower() in _SYSTEM_FILES or p.name.startswith("~$"):
            continue
        if p.is_dir():
            if with_dirs:
                out.append(p.name + "\\")
        elif p.suffix.lower() not in _PHOTO_EXTS:
            out.append(p.name)
    return out


def controller_folders(input_dir: str) -> list[tuple[str, Path]]:
    """Папки контролёров с фото: [(имя, путь), …]. Фото в корне — ошибка
    (check_no_photos_in_root, до начала прогона); папки без фото не берутся."""
    base = Path(input_dir)
    if not base.is_dir():
        return []
    return [(sub.name, sub) for sub in sorted(p for p in base.iterdir() if p.is_dir())
            if list_photos(sub)]


# ─── Пайплайн для целой папки (или набора папок) ─────────────────────────────

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"


def run_pipeline(config: PipelineConfig) -> None:
    """
    Прогон по папке месяца config.month_dir (этап 2.2b; старый режим «таблица
    и CSV-лог» убран на этапе 2.3b): фото из <месяц>/фото, результат в
    <месяц>/результат, абоненты, показания и лог — в базе gmr.sqlite;
    показание и строка лога пишутся одной транзакцией; в конце — выгрузка
    показания.xlsx (_MonthRun). Перед прогоном — копия базы месяца. Всё, что
    прогон пишет в консоль, пишется и в журнал
    `<месяц>/результат/run_logs/run_<дата_время>.txt` (последние 30), вместе
    с причиной, если прогон прервался (2026-10-01).
    """
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt="%H:%M:%S")
    log = logging.getLogger("reader")
    log.setLevel(logging.INFO)

    if not config.month_dir:
        raise NotAMonth("Не указана папка месяца: python gmr.py process <папка месяца>")
    month = MonthFolder(Path(config.month_dir))
    if not is_month(month):
        raise NotAMonth(f"Месяц не создан: {month.root}. Сначала: "
                        f"python gmr.py month \"{month.root}\" --table <таблица компании>")
    check_no_photos_in_root(month.photos)
    config.input_dir = str(month.photos)
    config.output_base_dir = str(month.results)
    month.stop_file.unlink(missing_ok=True)          # старая просьба остановиться — не в счёт

    run_logs = Path(config.output_base_dir) / "run_logs"
    run_logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    journal, n = run_logs / f"run_{stamp}.txt", 2
    while journal.exists():                          # два прогона в одну секунду — свой журнал у каждого
        journal, n = run_logs / f"run_{stamp}_{n}.txt", n + 1
    handler = logging.FileHandler(journal, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(handler)
    try:
        log.info(f"GMR {__version__}. Журнал прогона: {journal}")
        apply_month_preset(config, month, log)
        _backup_month(month, log)
        store = _MonthRun(config, log, month)
        try:
            _run_pipeline(config, log, store)
        finally:
            store.close()
    except BaseException:
        log.exception("Прогон прерван")
        raise
    finally:
        month.stop_file.unlink(missing_ok=True)
        log.removeHandler(handler)
        handler.close()
        remove_old(run_logs, KEEP)


_PRESET_FIELDS = ("missing_digit_mode", "ignore_last_digits", "forgiven_digit_mode",
                  "serial_crop_pad", "account_ocr_model", "drum_rule", "first_digit_from_last",
                  "serial_by_table", "turn_if_nothing", "overlay_notes",
                  "meter_conf_thresh", "digit_detect_conf_thresh", "digit_conf_thresh", "serial_conf_thresh",
                  "account_conf_thresh")


def apply_month_preset(config: PipelineConfig, month: MonthFolder, log: logging.Logger) -> None:
    """Настройки распознавания месяца (src/gmr/domain/preset.py) — в config.
    Месяц без них — всё как в PipelineConfig."""
    preset = month_preset(month.root)
    if preset is None:
        log.info("Настройки распознавания: по умолчанию (как до настроек месяца)")
        return
    applied = preset.apply(config)
    for name in _PRESET_FIELDS:
        setattr(config, name, getattr(applied, name))
    log.info(f"Настройки распознавания месяца: {preset.describe()}")


def _backup_month(month: MonthFolder, log: logging.Logger) -> None:
    try:
        dest = backup_sqlite(month.db, month.backups)
    except (OSError, sqlite3.Error) as e:
        log.warning(f"Не удалось сделать копию базы месяца: {e}")
        return
    if dest is not None:
        log.info(f"Копия базы месяца: {dest}")


class _MonthRun:
    """
    Прогон по папке месяца (этап 2.2b). Абоненты и показания — из базы;
    показание (PLUS/MINUS), запись в журнал изменений и строка лога — одной
    транзакцией: сбой посреди прогона не теряет показаний, при перезапуске
    фото узнаётся по логу. Таблицу во время прогона никто не пишет, поэтому
    открытый Excel ему не мешает; в конце — выгрузка показания.xlsx (если
    файл открыт — показания остаются в базе, выгрузить позже: gmr.py export).
    """

    def __init__(self, config: PipelineConfig, log: logging.Logger, month: MonthFolder):
        self.config, self.log, self.month = config, log, month
        self.db = MonthDB(month.db)
        abonents = self.db.abonents(only_in_table=True)
        readings = self.db.readings()
        cfg = config
        self.df = pd.DataFrame(
            [{cfg.col_serial: a.serial, cfg.col_account_id: a.account,
              cfg.col_last_reading: a.last_reading,
              cfg.col_new_reading: readings[a.account].value if a.account in readings else ""}
             for a in abonents.values()],
            columns=[cfg.col_serial, cfg.col_account_id, cfg.col_last_reading, cfg.col_new_reading],
        )
        # Показания, которые были в таблице при загрузке, — как строки
        # pre_existing старого лога: по такому счёту новое фото → REPEAT
        # (DuplicatePolicy). В базу эти строки не пишутся.
        self.log_rows = self.db.log_rows() + [
            {"original_filename": "__pre_existing__", "source": "pre_existing",
             "outcome": "UNKNOWN", "account_id": account}
            for account, r in readings.items() if r.source == "table"
        ]
        log.info(f"Месяц: {month.root}; абонентов: {len(abonents)}, с показанием: "
                 f"{sum(1 for a in abonents if a in readings)}, строк лога: {len(self.log_rows)}")

    def record(self, result: PhotoResult, log_row: dict, photo_name: str,
               seen: Optional[Recognition] = None) -> list[dict]:
        """Показание и строка лога — одной транзакцией (с ними — что прочитали
        модели, для окна оператора). Если у счёта появилось показание, его фото,
        ждущие оператора, закрываются (этап 3, пункт 3) — возвращаются их строки
        REPEAT."""
        cfg = self.config
        closed, moves = [], []
        with self.db.transaction():
            if result.outcome in (Outcome.PLUS, Outcome.MINUS) and result.account_id:
                account, value = result.account_id, str(result.reading)
                old = self.db.reading(account)
                self.db.put_reading(Reading(account, value, reading_date(photo_name), "auto",
                                            photo_name, now_text(), "auto"))
                self.db.add_change("auto", "показание записано", account, cfg.col_new_reading,
                                   old.value if old else "", value, note=photo_name)
                self.df.loc[self.df[cfg.col_account_id] == account, cfg.col_new_reading] = value
                self.log.info(f"  ✅ Записано в базу: account={account}, reading={value}")
            self.db.append_log_rows([log_row])
            if seen is not None and log_row.get("photo_hash"):
                self.db.put_recognition(log_row["photo_hash"], seen)
            if result.outcome in (Outcome.PLUS, Outcome.MINUS) and result.account_id:
                closed, moves = close_waiting(self.db, self.month.results, result.account_id,
                                              photo_name, "auto")
        for row in closed:
            self.log.info(f"  ↪ {row['original_filename']}: ждало оператора, у л/с "
                          f"{row['account_id']} теперь есть показание → repeat/{row['final_filename']}")
        for err in move_files(moves):
            self.log.warning(f"  ⚠️  Не удалось перенести фото в repeat/: {err}")
        return closed

    def finish(self) -> None:
        try:
            self.log.info(export_month(str(self.month.root), self.config).text())
        except ExportLocked as e:
            self.log.warning(f"⚠️  {e} Показания сохранены в базе.")

    def close(self) -> None:
        self.db.close()


class _PhotoIndex:
    """Строки лога по фото — как ProcessedPhotoPolicy.row_matches: строка с
    отпечатком совпадает по отпечатку, без отпечатка — по имени файла."""

    def __init__(self, rows: list[dict]):
        self._n = 0
        self._by_hash: dict[str, list] = {}
        self._no_hash_by_name: dict[str, list] = {}
        self._by_name: dict[str, list] = {}
        for r in rows:
            self.add(r)

    def add(self, row: dict) -> None:
        self._n += 1
        item = (self._n, row)
        name = row.get("original_filename") or ""
        self._by_name.setdefault(name, []).append(item)
        if row.get("photo_hash"):
            self._by_hash.setdefault(row["photo_hash"], []).append(item)
        else:
            self._no_hash_by_name.setdefault(name, []).append(item)

    def rows(self, name: str, photo_hash: str) -> list[dict]:
        if photo_hash:
            found = self._by_hash.get(photo_hash, []) + self._no_hash_by_name.get(name, [])
        else:
            found = self._by_name.get(name, [])
        return [r for _, r in sorted(found, key=lambda x: x[0])]


def serial_lookup(df: pd.DataFrame, config: PipelineConfig):
    """Есть ли номер в таблице — так же, как ищет process_photo (шаг 3):
    номер, '0'+номер, '00'+номер."""
    serials = {normalize_serial(str(s)) for s in df[config.col_serial]}

    def in_table(serial: str) -> bool:
        s = normalize_serial(serial)
        return any(c in serials for c in lookup_variants(s))
    return in_table


def _run_pipeline(config: PipelineConfig, log: logging.Logger, store) -> None:
    df = store.df
    state = RunState(log_rows=store.log_rows)
    index = _PhotoIndex(store.log_rows)
    serial_in_table = serial_lookup(df, config)
    models = None                                       # загружаются, только если есть что читать
    account_groups_cache = None                         # словарь счетов для модели надписи (один на прогон)
    serial_groups_cache = None                          # словарь номеров для серийника по таблице (этап 6b)
    notes_cache = None                                  # надписи поверх фото: счета таблицы, образцы
    # код участка контролёра (area_code.py): счета таблицы, найденные по его фото
    table_accounts = set(df[config.col_account_id].astype(str).str.strip())
    area_accounts = accounts_by_controller(store.log_rows, table_accounts)
    groups_by_code: dict = {}                           # словарь надписи для каждого кода

    def load():
        log.info("Загружаем модели...")
        device = default_device()
        log.info(f"Устройство: {device}")
        loaded = load_models(config, device=device)
        log.info("Модели загружены ✓")
        for name, det in (("счётчика", loaded.meter_detector), ("цифр", loaded.digit_detector)):
            backend = getattr(getattr(det, "inferer", None), "backend", None)
            if backend:
                log.info(f"  детектор {name}: {backend}")
        if config.ignore_last_digits > 0:
            log.info(
                f"⚙️  ignore_last_digits={config.ignore_last_digits}: "
                f"последние {config.ignore_last_digits} цифр(ы) — ошибки конфиданса прощаются, "
                f"подставляется '{config.forgiven_digit_placeholder}'"
            )
        if config.debug_digits:
            log.info("⚙️  debug_digits=True: кропы цифр и meta.json будут сохраняться в question/digits_error/<account_id>/")
        if config.ignore_last_digits > 0 and config.debug_digits:
            log.warning(
                "⚠️  ignore_last_digits и debug_digits включены одновременно: "
                "ошибки на прощённых позициях НЕ попадут в digits_error и не будут продебажены."
            )
        if config.draw_boxes:
            log.info("⚙️  draw_boxes=True: на фото будут нарисованы боксы детекций")
        if loaded.account_recognizer is not None:
            log.info(f"⚙️  надпись маркером: {config.account_ocr_model} (порог {config.account_conf_thresh})")
        return loaded

    # ── Папки контролёров ────────────────────────────────────────────────────
    base = Path(config.input_dir)
    input_groups = controller_folders(config.input_dir)
    # что программа не возьмёт — в журнал и в отчёты (этап 3)
    not_taken = {"": not_taken_files(base, with_dirs=False)}
    if base.is_dir():
        not_taken |= {d.name: not_taken_files(d) for d in sorted(base.iterdir(), key=lambda x: x.name)
                      if d.is_dir()}
    for folder, names_ in not_taken.items():
        if names_:
            log.warning(not_taken_line(names_, where(folder)))
    if not input_groups:
        log.warning(f"В {config.input_dir} нет папок контролёров с фото.")
    else:
        names = ", ".join(name for name, _ in input_groups)
        log.info(f"Папок контролёров с фото: {len(input_groups)} ({names})")
    if config.reread_errors:
        log.info("⚙️  --reread: фото с ошибками, которые ждут оператора, читаются заново")

    # ── Сколько новых фото прочитать — для полосы прогресса окна (этап 4).
    # Отпечатки считаются один раз и запоминаются.
    hashed: dict[Path, tuple[str, Optional[Exception]]] = {}
    expected, seen = 0, set()
    for _, folder_path in input_groups:
        for p in list_photos(folder_path):
            try:
                h, err = photo_fingerprint(str(p)), None
            except Exception as e:      # файл не читается — при чтении станет ERROR
                h, err = "", e
            hashed[p] = (h, err)
            if _run_selection.decide(index.rows(p.name, h), bool(h) and h in seen, serial_in_table,
                                     reread=config.reread_errors) == RunSelection.READ:
                expected += 1
                seen.add(h)
    log.info(f"Новых фото к чтению: {expected}")
    stop_file = MonthFolder(Path(config.month_dir)).stop_file if config.month_dir else None
    done, stopped = 0, False

    total_stats: dict[Outcome, int] = {o: 0 for o in Outcome}
    total_skipped: dict[str, int] = {}
    per_folder_stats: dict[str, dict[Outcome, int]] = {}

    for subfolder_name, folder_path in input_groups:
        if stopped:
            break
        log.info(f"\n{'=' * 55}")
        log.info(f"Папка контролёра: {subfolder_name}")
        log.info("=" * 55)
        output_base = str(Path(config.output_base_dir) / subfolder_name)

        photos = list_photos(folder_path)
        log.info(f"Фото в папке: {len(photos)}\n")

        stats: dict[Outcome, int] = {o: 0 for o in Outcome}
        skipped: dict[str, int] = {}
        code_logged = None

        for photo_path_obj in photos:
            photo_path = photo_path_obj  # совместимость с process_photo (ожидает str)

            # ── Читать или пропустить (этап 3: каждое фото разбирается один раз)
            photo_hash, hash_error = hashed[photo_path]
            rows = index.rows(photo_path.name, photo_hash)
            choice = _run_selection.decide(
                rows, bool(photo_hash) and photo_hash in state.processed_hashes,
                serial_in_table, reread=config.reread_errors)
            if choice != RunSelection.READ:
                skipped[choice] = skipped.get(choice, 0) + 1
                total_skipped[choice] = total_skipped.get(choice, 0) + 1
                continue

            if stop_file is not None and stop_file.exists():   # «Остановить» в окне (этап 4)
                stopped = True
                log.warning(f"⏹ Остановлено оператором: прочитано {done} из {expected}. "
                            "Остальные фото прочитает следующий запуск.")
                break
            if models is None:                       # до строки «Фото 1 из N»: окно покажет «Загружаем модели…»
                models = load()
                if models.account_recognizer is not None:
                    account_groups_cache = account_groups_of(df, config)
                if config.serial_by_table:
                    serial_groups_cache = serial_groups_of(df, config)
                if config.overlay_notes:
                    notes_cache = note_context_of(df, config)
            done += 1
            log.info(f"Фото {done} из {expected}: {photo_path.name}" + ("  (читается заново)" if rows else ""))
            if rows:
                # прежний файл результата этого фото (question/…, not_in_db/) убрать:
                # сейчас фото разложится заново
                old = result_file(Path(config.output_base_dir), rows[-1])
                if old is not None and old.is_file():
                    old.unlink()

            code = controller_code(area_accounts.get(subfolder_name, ()))
            if code != code_logged:
                log.info(f"  Код участка контролёра: {code}" if code else "  Код участка контролёра пока не известен")
                code_logged = code
            groups = account_groups_cache
            if groups is not None and code is not None:
                if code not in groups_by_code:
                    groups_by_code[code] = only_code(account_groups_cache, code)
                groups = groups_by_code[code]
            try:
                if hash_error is not None:
                    raise hash_error
                result = read_photo(
                    str(photo_path), df, config,
                    models.meter_detector, models.digit_detector,
                    models.digit_recognizer, models.serial_recognizer,
                    photo_hash=photo_hash, reread=bool(rows),
                    account_recognizer=models.account_recognizer,
                    account_groups=groups,
                    serial_groups=serial_groups_cache,
                    notes=notes_cache,
                    state=state,
                    area_code=code,
                )
            except Exception as e:
                # Ошибка на одном фото не останавливает прогон: фото → question/error,
                # при следующем прогоне читается заново. Ctrl+C (KeyboardInterrupt)
                # сюда не попадает и останавливает прогон, как раньше.
                log.exception(f"  Ошибка программы на фото {photo_path.name}")
                result = PhotoResult(
                    photo_path=str(photo_path), outcome=Outcome.ERROR,
                    error_detail=f"ошибка программы: {type(e).__name__}: {e}",
                )
            stats[result.outcome] += 1
            total_stats[result.outcome] += 1
            # что прочитали модели — для окна оператора (этап Е), в базу вместе со строкой лога
            seen = safe_what_models_saw(result, models, config, log) if photo_hash else None

            detail = result.error_detail or ""
            log.info(
                f"  serial={result.serial_text!r}({result.serial_conf or 0:.2f})  "
                f"reading={result.reading_str!r}  delta={result.delta}  "
                f"→ {result.outcome.name}  {detail}"
            )

            # Сохраняем аннотированное фото в нужную папку. Имя занято другим
            # фото — <имя>_2, <имя>_3… (этап 3: файлы не затирают друг друга)
            dst_dir  = str(Path(output_base) / OUTCOME_FOLDER[result.outcome])
            new_name = free_name(Path(dst_dir), result.new_photo_name or photo_path.name)
            if new_name != (result.new_photo_name or photo_path.name):
                result.new_photo_name = new_name
            try:
                save_annotated(
                    str(photo_path), dst_dir, new_name, result,
                    move=config.move_photos,
                    draw_boxes=config.draw_boxes,
                )
            except OSError:
                if result.outcome != Outcome.ERROR:
                    raise
                # файл не читается совсем — остаётся во входной папке, в логе ERROR
                log.exception(f"  Не удалось скопировать {photo_path.name} в {dst_dir}")
            log.info(f"  📁 → {output_base}/{OUTCOME_FOLDER[result.outcome]}/{new_name}")

            # Показание (PLUS/MINUS) и строка лога — после каждого фото
            log_row = make_log_row(
                photo_path_obj.name, result, config,
                photo_hash=photo_hash, source_folder=subfolder_name,
            )
            closed = store.record(result, log_row, photo_path_obj.name, seen=seen)
            for row in [log_row, *closed]:
                state.log_rows.append(row)
                index.add(row)
            for folder, accounts in accounts_by_controller([log_row, *closed], table_accounts).items():
                area_accounts.setdefault(folder, set()).update(accounts)
            # Счёт «обработан в прогоне», только если получил показание: неудачное
            # фото не блокирует следующее фото того же счётчика (этап 3, пункт 3)
            if result.account_id and result.outcome in (Outcome.PLUS, Outcome.MINUS):
                state.processed_accounts.add(result.account_id)
            state.processed_hashes.add(photo_hash)

        if skipped:
            log.info("Пропущено: " + "; ".join(skip_lines(skipped)))
        per_folder_stats[subfolder_name] = stats

        report_text = format_report(f"ОТЧЁТ: {subfolder_name}", stats, skipped,
                                     {subfolder_name: not_taken.get(subfolder_name, [])}, folder_report=True)
        report_path = save_report(output_base, report_text)

        log.info("\n" + report_text)
        log.info(f"💾 Отчёт сохранён: {report_path}")

    overall_report_text = format_report("ОБЩИЙ ОТЧЁТ ПО ВСЕМ ПАПКАМ", total_stats, total_skipped,
                                         {k: v for k, v in not_taken.items() if v} or None)
    if len(per_folder_stats) > 1:
        ranking_text = format_folder_ranking(per_folder_stats)
        if ranking_text:
            overall_report_text = overall_report_text + "\n\n" + ranking_text

    overall_report_path = save_report(config.output_base_dir, overall_report_text)
    log.info("\n" + overall_report_text)
    log.info(f"💾 Общий отчёт сохранён: {overall_report_path}")

    store.finish()
