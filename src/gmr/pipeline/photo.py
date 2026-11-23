"""
Обработка одного фото: модели → серийник → строка таблицы → показание →
исход. Ничего не пишет ни в базу, ни на диск (кроме отладочных кропов при
config.debug_digits) — это делает прогон (run.py).

  process_photo(photo_path, df, config, …) -> PhotoResult
    ├── шаг 0: фото уже разобрано / прочитано?  → REPEAT без запуска моделей
    ├── детектор счётчика (YOLO)  → кропы meter, serial (+ account — надпись маркером)
    ├── надпись маркером (CRNN, если подключена) → лицевой счёт
    ├── серийник (CRNN) → строка таблицы (лицевой счёт, прошлое показание)
    ├── детектор и распознавание цифр → показание
    └── правила (src/gmr/domain/policies.py) → исход и папка
"""
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import pandas as pd

from src.gmr.application import (
    RecognitionModels,
    describe_substitutions,
    detect_meter,
    find_detection,
    read_meter_digits_for_config,
)
from src.gmr.domain import (
    AccountPrediction,
    AccountRecognizer,
    DeltaThresholdPolicy,
    DigitDetector,
    DigitRecognizer,
    DuplicatePolicy,
    MeterDetector,
    Outcome,
    PhotoResult,
    PipelineConfig,
    ProcessedPhotoPolicy,
    SerialRecognizer,
)
from src.gmr.domain.account_match import AccountMarkerPolicy, account_variants
from src.gmr.domain.account_match import account_groups as build_account_groups
from src.gmr.domain.area_code import other_code_note
from src.gmr.domain.datasets import ACCOUNT, METER, SERIAL
from src.gmr.domain.note_match import AccountIndex, interpret, shown
from src.gmr.domain.note_match import rescue as note_rescue
from src.gmr.domain.serial_match import (
    build_serial_groups,
    lookup_variants,
    normalize_serial,
)
from src.gmr.render import read_image

from .output import save_digit_debug

_delta_policy           = DeltaThresholdPolicy()
_duplicate_policy       = DuplicatePolicy()
_processed_photo_policy = ProcessedPhotoPolicy()
_account_policy         = AccountMarkerPolicy()


@dataclass
class RunState:
    """Что прогон уже знает к очередному фото. Один на прогон; process_photo
    его только читает, прогон пополняет после каждого фото.

    log_rows           — строки лога месяца и прочитанные в этом прогоне, плюс
                         строки pre_existing (показание было в таблице при
                         загрузке месяца, в базу не пишутся);
    processed_accounts — счета, получившие показание в этом прогоне;
    processed_hashes   — отпечатки фото, прочитанных в этом прогоне.
    """
    log_rows: list = field(default_factory=list)
    processed_accounts: set = field(default_factory=set)
    processed_hashes: set = field(default_factory=set)


# ─── Чтение цифр счётчика ────────────────────────────────────────────────────

def _digit_bboxes_to_orig(
    digit_bboxes: list[Optional[tuple]],
    meter_bbox: tuple[int, int, int, int],
    meter_crop_shape: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """
    Пересчитывает bbox-ы цифр из координат кропа в координаты оригинального фото.

    digit_bboxes      — [(x1,y1,x2,y2) в пикселях кропа | None для заглушек]
    meter_bbox        — (x1,y1,x2,y2) кропа счётчика в оригинале (от meter_detector)
    meter_crop_shape  — (h, w) кропа (после возможного выпрямления)

    Возвращает список только реальных bbox-ов (заглушки пропускаются).
    """
    orig_x1, orig_y1, orig_x2, orig_y2 = meter_bbox
    crop_h, crop_w = meter_crop_shape[:2]

    # Размер bbox счётчика в оригинале
    bbox_w = orig_x2 - orig_x1
    bbox_h = orig_y2 - orig_y1

    # Масштаб: кроп мог быть выпрямлен и изменить размер,
    # поэтому пересчитываем через соотношение сторон
    scale_x = bbox_w / crop_w
    scale_y = bbox_h / crop_h

    result = []
    for bbox in digit_bboxes:
        if bbox is None:
            continue
        dx1, dy1, dx2, dy2 = bbox
        x1 = orig_x1 + int(dx1 * scale_x)
        y1 = orig_y1 + int(dy1 * scale_y)
        x2 = orig_x1 + int(dx2 * scale_x)
        y2 = orig_y1 + int(dy2 * scale_y)
        result.append((x1, y1, x2, y2))
    return result


# ─── Серийник по таблице (этап 6b) ───────────────────────────────────────────

def serial_groups_of(df: pd.DataFrame, config: PipelineConfig) -> dict:
    """Словарь для серийника по таблице (этап 6b): {номер: варианты записи}."""
    return build_serial_groups(df[config.col_serial])


def _rows_with_serial(df: pd.DataFrame, config: PipelineConfig, serial: str) -> pd.DataFrame:
    norm = normalize_serial(serial)
    return df[df[config.col_serial].apply(lambda x: normalize_serial(str(x)) == norm)]


# ─── Надпись контролёра поверх фото (настройка месяца overlay_notes) ─────────

@dataclass
class NoteContext:
    """Для надписей поверх фото — один на прогон: образцы символов, как
    записывается каждый счёт таблицы, прошлые показания."""
    bank: object
    index: AccountIndex
    lasts: dict


def note_context_of(df: pd.DataFrame, config: PipelineConfig) -> NoteContext:
    from src.gmr.overlay import GlyphBank
    accounts = df[config.col_account_id].astype(str).str.strip()
    lasts = {}
    for account, value in zip(accounts, df[config.col_last_reading]):
        try:
            lasts[account] = int(float(str(value).replace(",", ".")))
        except ValueError:
            continue
    return NoteContext(GlyphBank.load(), AccountIndex(accounts), lasts)


def _read_note(notes: Optional[NoteContext], photo_path: str) -> Optional[tuple[list[str], int]]:
    """(строки надписи, сколько строк похожи на надпись, но не прочитаны);
    None — не смотрим или надписи нет."""
    if notes is None:
        return None
    from src.gmr import overlay
    img = read_image(photo_path)
    if img is None:
        return None
    dropped: list = []
    lines = [ln.text for ln in overlay.read(img, notes.bank, dropped)]
    unread = sum(d.text_like for d in dropped)
    return (lines, unread) if lines or unread else None


# ─── Надпись маркером (лицевой счёт) ─────────────────────────────────────────

def account_groups_of(df: pd.DataFrame, config: PipelineConfig) -> dict:
    """Словарь модели надписи: {счёт: варианты записи} по всей таблице."""
    return build_account_groups(df[config.col_account_id])


def _account_rows(df: pd.DataFrame, config: PipelineConfig, account: str) -> pd.DataFrame:
    return df[df[config.col_account_id].astype(str).str.strip() == str(account)]


def _read_marker(recognizer: Optional[AccountRecognizer], crops: list, groups: Optional[dict],
                 config: PipelineConfig) -> Optional[AccountPrediction]:
    """Что написано маркером и какой это счёт; None — нет модели или надписи."""
    if recognizer is None:
        return None
    det = find_detection(crops, ACCOUNT)
    return None if det is None else recognizer.recognize(det["crop"], groups)


# ─── Надпись — только если без неё фото ушло бы оператору (1.6.0) ───────────

class _Memo:
    """Модель на одно фото: тот же вход (тот же объект кропа, путь) — тот же
    ответ без второго запуска. Ключи держат свои объекты, поэтому id не
    переиспользуется, пока память жива."""

    def __init__(self, inner):
        self._inner, self._seen = inner, {}

    def _call(self, name: str, *args):
        key = (name,) + tuple(a if isinstance(a, str) else id(a) for a in args)
        hit = self._seen.get(key)
        if hit is None:
            hit = self._seen[key] = (args, getattr(self._inner, name)(*args))
        return hit[1]

    def __getattr__(self, name):            # detect_turned и прочее — только если есть у модели
        if not hasattr(self._inner, name):
            raise AttributeError(name)
        return lambda *args: self._call(name, *args)

    def detect(self, x):
        return self._call("detect", x)

    def recognize(self, crop, *rest):
        if rest and rest[0] is not None:     # надпись маркером со словарём: словарь бывает новым dict
            for (args, res) in (v for k, v in self._seen.items() if k[0] == "recognize" and len(k) == 3):
                if args[0] is crop and (args[1] is rest[0] or args[1] == rest[0]):
                    return res
        return self._call("recognize", crop, *rest)


def needs_note(result: PhotoResult) -> bool:
    """Без надписи фото ушло бы оператору (question/, кроме сбоя программы)."""
    from src.gmr.domain import OUTCOME_FOLDER
    return OUTCOME_FOLDER[result.outcome].startswith("question/") and result.outcome != Outcome.ERROR


def read_photo(photo_path: str, df: pd.DataFrame, config: PipelineConfig,
               meter_detector: MeterDetector, digit_detector: DigitDetector,
               digit_recognizer: DigitRecognizer, serial_recognizer: SerialRecognizer,
               notes: Optional["NoteContext"] = None, **kw) -> PhotoResult:
    """process_photo с надписью поверх фото по настройке
    config.overlay_notes_if_unsure (1.6.0, решение владельца 2026-11-23): фото
    читается без надписи; ушло бы оператору — читается ещё раз с надписью,
    и ответ — как до 1.6.0. Модели второй раз не запускаются (_Memo)."""
    if notes is None or not config.overlay_notes_if_unsure:
        return process_photo(photo_path, df, config, meter_detector, digit_detector,
                             digit_recognizer, serial_recognizer, notes=notes, **kw)
    memo = [_Memo(m) for m in (meter_detector, digit_detector, digit_recognizer, serial_recognizer)]
    if kw.get("account_recognizer") is not None:
        kw["account_recognizer"] = _Memo(kw["account_recognizer"])
    result = process_photo(photo_path, df, config, *memo, notes=None, **kw)
    if not needs_note(result):
        return result
    return process_photo(photo_path, df, config, *memo, notes=notes, **kw)


# ─── Основная функция обработки одного фото ──────────────────────────────────

def process_photo(
    photo_path: str,
    df: pd.DataFrame,
    config: PipelineConfig,
    meter_detector: MeterDetector,
    digit_detector: DigitDetector,
    digit_recognizer: DigitRecognizer,
    serial_recognizer: SerialRecognizer,
    photo_hash: Optional[str] = None,
    reread: bool = False,
    account_recognizer: Optional[AccountRecognizer] = None,
    account_groups: Optional[dict] = None,
    serial_groups: Optional[dict] = None,
    notes: Optional["NoteContext"] = None,
    state: Optional[RunState] = None,
    area_code: Optional[str] = None,
) -> PhotoResult:
    """
    Обрабатывает одну фотографию. Возвращает PhotoResult.
    НЕ изменяет df и не трогает файлы — это делает caller (run_pipeline).

    photo_hash — отпечаток содержимого фото (photo_fingerprint). Если задан,
    шаг 0 узнаёт фото в логе по нему, иначе по имени файла.
    reread — прогон уже решил прочитать фото заново (RunSelectionPolicy:
    --reread, номер появился в таблице), шаг 0 пропускается.

    Модели — в виде контрактов src/gmr/domain/ml.py (Фаза 3); реальные
    модели создаёт src.gmr.ml.loader.load_models.

    account_recognizer — модель надписи маркером (None — её нет, всё как до
    неё); account_groups — словарь счетов таблицы для неё
    (account_groups_of(df, config): прогон строит его один раз).
    notes — надписи контролёров поверх фото (note_context_of, настройка
    месяца overlay_notes; None — не смотрим, всё как без них).
    area_code — код участка контролёра (src/gmr/domain/area_code.py); прогон
    тогда даёт account_groups только из счетов этого кода. Счёт по серийнику
    с другим кодом — пометка в account_notes, исход тот же.
    """
    state = state if state is not None else RunState()
    models = RecognitionModels(
        meter_detector=meter_detector,
        digit_detector=digit_detector,
        digit_recognizer=digit_recognizer,
        serial_recognizer=serial_recognizer,
        account_recognizer=account_recognizer,
    )
    result = PhotoResult(photo_path=photo_path, outcome=Outcome.NO_METER)
    ext = Path(photo_path).suffix

    # ── Шаг 0: фото уже обрабатывалось? ──────────────────────────────────────
    # До запуска моделей. Разобранное оператором или успешно прочитанное фото
    # сразу REPEAT; фото только с авто-ошибками в логе обрабатывается заново
    # (ProcessedPhotoPolicy, решение владельца 2026-09-29, вариант Г).
    _seen = _processed_photo_policy.decide(
        Path(photo_path).name, state.log_rows, photo_hash,
        hashes_in_run=frozenset(state.processed_hashes),
    )
    if _seen.skip and not reread:
        row = _seen.row or {}
        result.outcome = Outcome.REPEAT
        result.error_detail = _seen.reason
        result.account_id = row.get("account_id") or None
        result.serial_text = row.get("serial_id") or None
        result.new_photo_name = row.get("final_filename") or Path(photo_path).name
        return result

    # ── Шаг 1: детекция трёх классов ─────────────────────────────────────────
    # (ничего не найдено и включено turn_if_nothing — фото поворачивается, этап 7b)
    crops, result.photo_turn = detect_meter(meter_detector, photo_path, config)
    # ── Шаг 1a: надпись контролёра поверх фото (настройка месяца) ───────────
    # Её показание главнее барабана (решение владельца 2026-10-28): счётчик
    # не найден — фото всё равно читается, счёт и показание — по надписи.
    note = _read_note(notes, photo_path)

    meter_entry  = find_detection(crops, METER)
    serial_entry = find_detection(crops, SERIAL)

    # Сохраняем все найденные детекции для отрисовки боксов
    if crops:
        result.meter_crops_raw = crops

    no_meter: Optional[PhotoResult] = None      # счётчика нет — исход, если надпись не поможет
    if meter_entry is None:
        result.outcome = Outcome.NO_METER
        result.error_detail = "class meter not detected" if crops else "YOLO returned no crops"
        if note is None:
            return result
        no_meter = replace(result)

    def _as_no_meter(hint: Optional[str]) -> PhotoResult:
        """Счётчика нет, надпись не дала счёт и показание — как без неё, с подсказкой."""
        no_meter.account_notes = hint
        return no_meter

    meter_crop = None if meter_entry is None else meter_entry["crop"]
    meter_bbox = None if meter_entry is None else meter_entry["bbox"]   # (x1,y1,x2,y2) в оригинале

    def _read_digits_for_annotation() -> None:
        """Показание для аннотации и оператору, когда исход — не запись."""
        if meter_crop is None:
            return
        _r, _rs, _, _dres, _bboxes = read_meter_digits_for_config(models, meter_crop, config).as_tuple()
        result.reading_str = _rs
        result.reading     = _r
        result.digit_notes = describe_substitutions(_dres)
        result.digit_results, result.digit_bboxes = _dres or [], _bboxes   # [] — читали, цифр нет
        if _bboxes is not None:
            result.digit_bboxes_in_orig = _digit_bboxes_to_orig(_bboxes, meter_bbox, meter_crop.shape)

    # ── Шаг 1b: надпись маркером (лицевой счёт) — только если есть модель ───
    if account_recognizer is not None and account_groups is None:
        account_groups = account_groups_of(df, config)
    marker = _read_marker(account_recognizer, crops, account_groups, config)

    # ── Шаги 2–3: серийник → строка таблицы ─────────────────────────────────
    # failure — почему серийник не дал ровно один счёт (исход и пояснение)
    failure: Optional[tuple[Outcome, str]] = None
    ambiguous: list[str] = []
    matches = pd.DataFrame()

    if serial_entry is None:
        failure = (Outcome.NO_SERIAL, "class serial not detected")
    else:
        serial_res  = serial_recognizer.recognize(serial_entry["crop"])
        serial_text = serial_res.text
        serial_conf = serial_res.confidence

        result.serial_text = serial_text
        result.serial_conf = serial_conf
        result.serial_read = serial_text

        if serial_conf < config.serial_conf_thresh:
            failure = (Outcome.SERIAL_LOW_CONF,
                       f"serial conf={serial_conf:.3f} < {config.serial_conf_thresh}")
        else:
            def _find_serial_in_df(serial: str) -> pd.DataFrame:
                norm = normalize_serial(serial)
                return df[df[config.col_serial].apply(
                    lambda x: normalize_serial(str(x)) == norm
                )]

            serial_candidates = lookup_variants(serial_text)

            serial_used = serial_text
            for candidate in serial_candidates:
                matches = _find_serial_in_df(candidate)
                if not matches.empty:
                    serial_used = candidate
                    break

            if not matches.empty and serial_used != serial_text:
                logging.getLogger("reader").info(
                    f"  ℹ️  serial fallback: '{serial_text}' → '{serial_used}' (добавлены ведущие нули)"
                )

            if matches.empty:
                failure = (Outcome.SERIAL_NOT_FOUND, (
                    f"serial '{serial_text}' not in table "
                    f"(tried: {', '.join(repr(c) for c in serial_candidates)})"
                ))
            else:
                result.serial_text = serial_used
                # Номер в таблице у нескольких лицевых счетов — у одной записи в базе номер
                # с ошибкой; чей счётчик, решает оператор (решение владельца 2026-10-01).
                accounts = list(dict.fromkeys(str(a).strip() for a in matches[config.col_account_id]))
                if len(accounts) > 1:
                    failure = (Outcome.SERIAL_AMBIGUOUS,
                               f"номер {serial_used} у нескольких абонентов: {', '.join(accounts)}")
                    ambiguous = accounts

    # ── Шаг 3a: серийник по таблице (этап 6b, пресет месяца; выключено) ─────
    if (failure is not None and failure[0] in (Outcome.SERIAL_NOT_FOUND, Outcome.SERIAL_LOW_CONF)
            and config.serial_by_table and hasattr(serial_recognizer, "match_table")):
        if serial_groups is None:
            serial_groups = serial_groups_of(df, config)
        m = serial_recognizer.match_table(serial_entry["crop"], serial_groups)
        if m.serial is not None and m.confidence >= config.serial_table_conf_thresh:
            result.serial_notes = (f"серийник по таблице: прочитано {result.serial_text}, "
                                   f"взят {m.serial} (доля {m.confidence:.2f})")
            result.serial_text = m.serial
            matches = _rows_with_serial(df, config, m.serial)
            accounts = list(dict.fromkeys(str(a).strip() for a in matches[config.col_account_id]))
            failure = None
            if len(accounts) > 1:
                failure = (Outcome.SERIAL_AMBIGUOUS,
                           f"номер {m.serial} у нескольких абонентов: {', '.join(accounts)}")
                ambiguous = accounts
        elif m.serial is not None:
            result.serial_notes = (f"серийник по таблице: не уверен — ближе всех {m.serial} "
                                   f"(доля {m.confidence:.2f})")

    if failure is not None:
        # Серийник счёт не дал — может быть, его даёт надпись маркером
        # (src/gmr/domain/account_match.py, правило 2). Без модели надписи — как раньше.
        rows_m = (_account_rows(df, config, marker.account)
                  if _account_policy.confident(marker, config.account_conf_thresh) else df.iloc[0:0])
        decision = _account_policy.rescue(
            failure[0], marker, config.account_conf_thresh, result.serial_text,
            [str(v) for v in rows_m[config.col_serial]], ambiguous,
        )
        result.account_notes = decision.note
        if (decision.account is None or rows_m.empty) and note is not None:
            # …или надпись поверх фото (note_match.rescue): счёт по ней
            # подтверждается показанием барабана — читаем его сначала
            _read_digits_for_annotation()
            lines, unread = note
            answer = interpret(lines, notes.index, notes.lasts, None,
                               None if result.reading is None else int(result.reading), unread)
            rows_n = (_account_rows(df, config, answer.account) if answer.account is not None
                      else df.iloc[0:0])
            decision = note_rescue(failure[0], lines, answer, result.serial_text,
                                   [str(v) for v in rows_n[config.col_serial]], ambiguous)
            result.account_notes = "; ".join(n for n in (result.account_notes, decision.note) if n)
            rows_m = rows_n
        if decision.account is None or rows_m.empty:
            if no_meter is not None:
                return _as_no_meter(result.account_notes)
            result.outcome, result.error_detail = failure
            if failure[0] != Outcome.NO_SERIAL and result.reading is None:
                # Читаем цифры для информативной аннотации
                _read_digits_for_annotation()
            return result
        matches = rows_m
        result.serial_text = normalize_serial(str(rows_m.iloc[0][config.col_serial]))

    row_idx    = matches.index[0]
    account_id = str(df.at[row_idx, config.col_account_id])
    last_val   = df.at[row_idx, config.col_last_reading]

    result.account_id = account_id
    other_code = other_code_note(account_id, area_code)
    if other_code is not None:
        result.account_notes = "; ".join(n for n in (result.account_notes, other_code) if n)

    try:
        last_reading = (
            float(str(last_val).replace(",", "."))
            if pd.notna(last_val) and str(last_val).strip() != ""
            else None
        )
    except ValueError:
        last_reading = None
    result.last_reading = last_reading

    # Надпись поверх фото про этот счёт: подтверждает, спорит, даёт показание
    known = None if note is None else interpret(note[0], notes.index, notes.lasts, account_id, None, note[1])

    # ── Шаг 3b: серийник нашёл счёт сам — сверить с надписью маркером ───────
    # Надпись уверенно указывает на другой счёт → показание не пишем, решает
    # оператор (account_match.py, правило 1). Счёт фото не присваивается: чей
    # это счётчик, неизвестно, и показание счёта по серийнику с другого фото
    # не должно молча закрыть это фото (close_waiting ищет по account_id).
    if failure is None:
        if marker is not None and account_groups is not None and account_id.strip() not in account_groups:
            # словарь — только счета участка, а серийник нашёл счёт другого кода:
            # надпись сверяется и с ним, иначе спор был бы всегда
            marker = _read_marker(account_recognizer, crops,
                                  {**account_groups, account_id.strip(): account_variants(account_id.strip())},
                                  config)
        conflict = _account_policy.conflict(account_id, marker, config.account_conf_thresh)
        if conflict is None and known is not None and known.conflict:
            # надпись поверх фото явно про другой счёт («лс…») — тоже спор
            conflict = f"надпись на фото {shown(note[0])}: {known.conflict}"
        if conflict is not None and no_meter is not None:
            return _as_no_meter(conflict)
        if conflict is not None:
            result.outcome = Outcome.SUSPICIOUS
            result.error_detail = conflict
            result.account_id = None
            result.last_reading = None
            _read_digits_for_annotation()
            return result
    if no_meter is not None and known.reading is None:
        return _as_no_meter("; ".join(n for n in (
            result.account_notes, f"надпись на фото {shown(note[0])}: {known.why}; показания по надписи нет") if n))

    # ── Шаг 4: проверяем дубль ───────────────────────────────────────────────
    # Дубль определяем по логу, а не по col_new_reading.
    #
    # Кейс «показание было в таблице до программы»:
    #   _MonthRun добавляет к логу строки с original_filename="__pre_existing__"
    #   и source="pre_existing" для показаний, внесённых до программы (в базе
    #   source=table). Такой счёт уже закрыт — это тоже дубль. Ищем их по
    #   совпадению account_id в логе с source=pre_existing.
    # Проверка "это фото уже в логе" — на шаге 0 (ProcessedPhotoPolicy).
    _log_rows     = state.log_rows
    _processed_accounts = state.processed_accounts
    _new_val      = df.at[row_idx, config.col_new_reading]
    _table_filled = pd.notna(_new_val) and str(_new_val).strip() not in ("", "nan")

    _dup = _duplicate_policy.decide(
        account_id=account_id,
        table_new_reading_filled=_table_filled,
        processed_accounts=_processed_accounts,
        log_rows=_log_rows,
    )
    if _dup.outcome is not None:
        # SUSPICIOUS здесь означает рассинхронизацию таблицы и лога —
        # показания НЕ перезаписываем (см. DuplicatePolicy).
        result.outcome = _dup.outcome
        result.error_detail = _dup.reason
        result.new_photo_name = f"{account_id}{ext}"
        return result

    # ── Шаг 5: читаем показания счётчика ─────────────────────────────────────
    # (прошлое показание — для старшей цифры по прошлому показанию, этап 6b)
    reading, reading_str, err, digit_results = None, None, None, None
    if meter_crop is not None:
        reading, reading_str, err, digit_results, digit_bboxes = (
            read_meter_digits_for_config(models, meter_crop, config, last_reading=last_reading).as_tuple()
        )

        result.reading_str = reading_str
        result.digit_notes = describe_substitutions(digit_results)
        result.digit_results, result.digit_bboxes = digit_results or [], digit_bboxes

        # Пересчитываем digit bbox-ы в координаты оригинала (для отрисовки)
        if digit_bboxes is not None:
            result.digit_bboxes_in_orig = _digit_bboxes_to_orig(
                digit_bboxes, meter_bbox, meter_crop.shape
            )

    # Показание по надписи поверх фото главнее барабана (решение владельца
    # 2026-10-28); с барабана — что прочитано, для оператора (model_reading_str)
    if known is not None and known.reading is not None:
        if known.reading != reading:
            barrel = (f"барабан: {reading}" if reading is not None else "счётчик не найден" if meter_crop is None
                      else f"барабан не прочитан: '{reading_str}'")
            result.account_notes = "; ".join(n for n in (
                result.account_notes, f"показание по надписи на фото {shown(note[0])}: {known.reading} ({barrel})")
                if n)
        reading = known.reading

    if reading is None:
        result.outcome = Outcome.DIGITS_ERROR
        result.error_detail = err
        result.new_photo_name = f"{account_id}{ext}"

        # ── Debug: сохраняем кропы цифр и meta.json ──────────────────────────
        if config.debug_digits and digit_results is not None:
            debug_dir = str(
                Path(config.output_base_dir)
                / "question/digits_error"
                / account_id
            )
            _raw_crops = digit_detector.detect(meter_crop)

            if _raw_crops:
                _sorted = sorted(_raw_crops, key=lambda c: c["bbox"][0])
                _with_placeholders: list[Optional[dict]] = []
                src_idx = 0
                for res in digit_results:
                    if res.get("note") == "inserted placeholder":
                        _with_placeholders.append(None)
                    else:
                        _with_placeholders.append(
                            _sorted[src_idx] if src_idx < len(_sorted) else None
                        )
                        src_idx += 1
                save_digit_debug(debug_dir, _with_placeholders, digit_results)

        return result

    result.reading     = reading
    result.reading_str = reading_str

    # ── Шаг 6: проверяем подозрительное отклонение ───────────────────────────
    result.outcome, result.delta = _delta_policy.decide(reading, last_reading, config.delta_threshold)

    if result.outcome == Outcome.SUSPICIOUS:
        result.error_detail = f"delta={result.delta:+.0f} > ±{config.delta_threshold:.0f}"
        result.new_photo_name = f"{account_id}{ext}"
        return result

    result.new_photo_name = f"{account_id}{ext}"
    return result
