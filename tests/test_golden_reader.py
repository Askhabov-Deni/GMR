"""
Golden tests для reader.py — docs/MIGRATION_TZ.md, раздел 2.

Пишутся ДО любого рефакторинга структуры (Фаза 0), на текущем reader.py,
с моками YOLOInferer/CRNNInferer/CNNInferer (см. tests/conftest.py).

Каждый тест пронумерован так же, как строка в таблице раздела 2 ТЗ и
содержит ссылку на диапазон строк reader.py, на которую он опирается —
если рефакторинг сдвинет логику, тест обязан продолжать проходить с тем же
наблюдаемым поведением (это и есть цель golden test: контракт поведения,
не контракт реализации).

Нумерация кейсов 1-16 соответствует таблице в MIGRATION_TZ.md §2.
"""
from tests._fixtures import (
    FakeDigitDetector, FakeDigitOCR, FakeMeterDetector, FakeSerialOCR, process_photo,
    make_crop, make_df, make_digit_crops_with_centers, make_meter_crops,
)
import reader
from reader import Outcome


SERIAL = "12345"
ACCOUNT = "A-0001"


def _digits_setup(config, digits, forced_conf=0.95):
    """5 digit-кропов, каждый распознаётся как соответствующий символ digits[i]."""
    assert len(digits) == config.expected_digits
    xs = [i * 40 for i in range(len(digits))]
    crops, arrays = make_digit_crops_with_centers(xs, width=20)
    predictions = {id(arr): (digits[i], forced_conf) for i, arr in enumerate(arrays)}
    return FakeDigitDetector(crops), FakeDigitOCR(predictions)


def _run(config, df, meter_crops, digit_detector, digit_ocr, serial_text, serial_conf,
          photo_name="photo1.jpg"):
    meter_detector = FakeMeterDetector(meter_crops)
    serial_ocr = FakeSerialOCR(serial_text, serial_conf)
    return process_photo(
        photo_name, df, config,
        meter_detector, digit_detector, digit_ocr, serial_ocr,
    )


# ─── 1-2: PLUS / MINUS, delta корректный ─────────────────────────────────────

def test_case1_plus_delta_200(base_config):
    # reader.py:982 — delta = reading - last_reading; delta>=0 → PLUS
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    dd, docr = _digits_setup(base_config, ["0", "1", "2", "0", "0"])  # -> 1200
    result = _run(base_config, df, make_meter_crops(), dd, docr, SERIAL, 0.95)
    assert result.outcome == Outcome.PLUS
    assert result.delta == 200
    assert result.reading == 1200


def test_case2_minus_delta_negative_200(base_config):
    # reader.py:982 — delta<0 → MINUS
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1200"}])
    dd, docr = _digits_setup(base_config, ["0", "1", "0", "0", "0"])  # -> 1000
    result = _run(base_config, df, make_meter_crops(), dd, docr, SERIAL, 0.95)
    assert result.outcome == Outcome.MINUS
    assert result.delta == -200
    assert result.reading == 1000


# ─── 3: SUSPICIOUS по delta_threshold ────────────────────────────────────────

def test_case3_suspicious_delta_over_threshold(base_config):
    # reader.py:976-980 — abs(delta) > delta_threshold(10000) → SUSPICIOUS
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    dd, docr = _digits_setup(base_config, ["2", "0", "0", "0", "0"])  # -> 20000, delta=19000
    result = _run(base_config, df, make_meter_crops(), dd, docr, SERIAL, 0.95)
    assert result.outcome == Outcome.SUSPICIOUS
    assert result.delta == 19000
    assert "delta" in (result.error_detail or "")


# ─── 4: last=None (пустая ячейка) → PLUS, delta=None ─────────────────────────

def test_case4_no_last_reading_is_plus_with_none_delta(base_config):
    # reader.py:983-985
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": ""}])
    dd, docr = _digits_setup(base_config, ["0", "1", "2", "0", "0"])
    result = _run(base_config, df, make_meter_crops(), dd, docr, SERIAL, 0.95)
    assert result.outcome == Outcome.PLUS
    assert result.delta is None
    assert result.last_reading is None


# ─── 5: фото уже в _log_filenames_cache → REPEAT ─────────────────────────────

def test_case5_repeat_photo_already_in_log(base_config):
    # Шаг 0 process_photo (ProcessedPhotoPolicy). До 2026-09-29 решение
    # принималось по набору имён _log_filenames_cache; с варианта Г — по
    # строке лога: успешно прочитанное фото (auto PLUS) сразу REPEAT.
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    base_config._log_rows_cache = [
        {"original_filename": "photo1.jpg", "outcome": "PLUS", "source": "auto",
         "account_id": ACCOUNT, "final_filename": f"{ACCOUNT}.jpg"},
    ]
    dd, docr = _digits_setup(base_config, ["0", "1", "2", "0", "0"])
    result = _run(base_config, df, make_meter_crops(), dd, docr, SERIAL, 0.95,
                  photo_name="photo1.jpg")
    assert result.outcome == Outcome.REPEAT
    assert "already in log" in result.error_detail


# ─── 6: тот же account_id второй раз в текущем прогоне → REPEAT ─────────────

def test_case6_repeat_duplicate_in_current_run(base_config):
    # reader.py:878-883
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    base_config._processed_accounts_cache = {ACCOUNT}
    dd, docr = _digits_setup(base_config, ["0", "1", "2", "0", "0"])
    result = _run(base_config, df, make_meter_crops(), dd, docr, SERIAL, 0.95)
    assert result.outcome == Outcome.REPEAT
    assert "duplicate in current run" in result.error_detail


# ─── 7: col_new_reading заполнен И есть pre_existing запись → REPEAT ─────────

def test_case7_repeat_pre_existing_entry(base_config):
    # reader.py:897-907
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000",
                    "new_reading": "1200"}])
    base_config._log_rows_cache = [{"source": "pre_existing", "account_id": ACCOUNT}]
    dd, docr = _digits_setup(base_config, ["0", "1", "2", "0", "0"])
    result = _run(base_config, df, make_meter_crops(), dd, docr, SERIAL, 0.95)
    assert result.outcome == Outcome.REPEAT
    assert "pre-existing entry" in result.error_detail


# ─── 8: col_new_reading заполнен, НЕТ pre_existing записи → SUSPICIOUS ───────

def test_case8_suspicious_table_log_desync(base_config):
    # reader.py:909-917
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000",
                    "new_reading": "1200"}])
    base_config._log_rows_cache = []  # нет pre_existing записи
    dd, docr = _digits_setup(base_config, ["0", "1", "2", "0", "0"])
    result = _run(base_config, df, make_meter_crops(), dd, docr, SERIAL, 0.95)
    assert result.outcome == Outcome.SUSPICIOUS
    assert "рассинхронизация" in result.error_detail
    # показания не должны перезаписываться при этой аномалии
    assert result.reading is None


# ─── 9: серийник не найден ни в одном из 3 вариантов → SERIAL_NOT_FOUND ─────

def test_case9_serial_not_found(base_config):
    # reader.py:815-836
    df = make_df([{"serial": "99999", "account_id": ACCOUNT, "last_reading": "1000"}])
    dd, docr = _digits_setup(base_config, ["0", "1", "2", "0", "0"])
    result = _run(base_config, df, make_meter_crops(), dd, docr, SERIAL, 0.95)
    assert result.outcome == Outcome.SERIAL_NOT_FOUND
    assert "not in table" in result.error_detail


# ─── 10: serial_conf < serial_conf_thresh → SERIAL_LOW_CONF ─────────────────

def test_case10_serial_low_confidence(base_config):
    # reader.py:768 (serial_conf_thresh по умолчанию 0.6)
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    dd, docr = _digits_setup(base_config, ["0", "1", "2", "0", "0"])
    result = _run(base_config, df, make_meter_crops(), dd, docr, SERIAL, 0.5)
    assert result.outcome == Outcome.SERIAL_LOW_CONF
    assert "0.5" in result.error_detail


# ─── 11: детектор цифр вернул 3 цифры → DIGITS_ERROR ─────────────────────────

def test_case11_digits_error_wrong_count(base_config):
    # reader.py:606
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    crops, arrays = make_digit_crops_with_centers([0, 40, 80])
    predictions = {id(a): ("1", 0.95) for a in arrays}
    dd = FakeDigitDetector(crops)
    docr = FakeDigitOCR(predictions)
    result = _run(base_config, df, make_meter_crops(), dd, docr, SERIAL, 0.95)
    assert result.outcome == Outcome.DIGITS_ERROR
    assert "expected 5 digits, got 3" in result.error_detail


# ─── 12: 4 цифры, gap на позиции >=2 → восстановление успешно ───────────────

def test_case12_recovery_success_gap_at_safe_position(base_config):
    # reader.py:572-603; missing_idx рассчитывается геометрически из центров.
    # centers: 10,60,210,260 -> gaps 50,150,50 avg=83.33, thresh=133.3
    # gap[1]=150 > thresh -> missing_idx = 1+1 = 2 (не критическая позиция)
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    xs = [0, 50, 200, 250]  # x1 значения; centers = x1 + width/2 = x1+10
    crops, arrays = make_digit_crops_with_centers(xs, width=20)
    predictions = {id(a): (d, 0.95) for a, d in zip(arrays, ["1", "2", "0", "0"])}
    dd = FakeDigitDetector(crops)
    docr = FakeDigitOCR(predictions)
    result = _run(base_config, df, make_meter_crops(), dd, docr, SERIAL, 0.95)
    assert result.outcome in (Outcome.PLUS, Outcome.MINUS, Outcome.SUSPICIOUS)
    assert result.reading is not None
    # позиция 2 (индекс missing_idx) должна получить missing_digit_placeholder="5"
    assert result.reading_str[2] == base_config.missing_digit_placeholder
    assert result.reading_str == "12500"


# ─── 13: 4 цифры, gap на позиции 0 или 1 (критическая) → DIGITS_ERROR ───────

def test_case13_recovery_refused_critical_position(base_config):
    # reader.py:597-601
    # centers: 10,210,260,310 -> gaps 200,50,50, avg=100, thresh=160
    # gap[0]=200>160 -> missing_idx=0+1=1 (< 2, критическая позиция) -> ошибка
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    xs = [0, 200, 250, 300]
    crops, arrays = make_digit_crops_with_centers(xs, width=20)
    predictions = {id(a): (d, 0.95) for a, d in zip(arrays, ["1", "2", "0", "0"])}
    dd = FakeDigitDetector(crops)
    docr = FakeDigitOCR(predictions)
    result = _run(base_config, df, make_meter_crops(), dd, docr, SERIAL, 0.95)
    assert result.outcome == Outcome.DIGITS_ERROR
    assert "critical position" in result.error_detail
    assert result.reading is None


# ─── 14: последняя цифра с conf<thresh, но в forgiven_positions → placeholder ─

def test_case14_forgiven_last_digit_low_confidence(base_config):
    # reader.py:635-643; ignore_last_digits=2 (default) -> forgiven positions {3,4}
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    xs = [i * 40 for i in range(5)]
    crops, arrays = make_digit_crops_with_centers(xs, width=20)
    # позиции 0..2 уверенные, позиция 4 (последняя) - низкий conf
    confs = [0.95, 0.95, 0.95, 0.95, 0.3]
    digits = ["0", "1", "2", "0", "9"]
    predictions = {id(a): (d, c) for a, d, c in zip(arrays, digits, confs)}
    dd = FakeDigitDetector(crops)
    docr = FakeDigitOCR(predictions)
    result = _run(base_config, df, make_meter_crops(), dd, docr, SERIAL, 0.95)
    assert result.outcome != Outcome.DIGITS_ERROR
    assert result.reading is not None
    # позиция 4 заменена на forgiven_digit_placeholder="0", а не на "9"
    assert result.reading_str[4] == base_config.forgiven_digit_placeholder
    assert result.reading_str == "01200"


# ─── 15: серийник с ведущими нулями сравнивается без int() ──────────────────

def test_case15_serial_leading_zeros_preserved(base_config):
    # reader.py:185-193 — _normalize_serial делает только .strip(), без int()
    df = make_df([{"serial": " 007123 ", "account_id": ACCOUNT, "last_reading": "1000"}])
    dd, docr = _digits_setup(base_config, ["0", "1", "2", "0", "0"])
    result = _run(base_config, df, make_meter_crops(), dd, docr, "007123", 0.95)
    # серийник нашёлся (не SERIAL_NOT_FOUND) именно за счёт strip-нормализации,
    # ведущие нули сохранены с обеих сторон сравнения
    assert result.outcome != Outcome.SERIAL_NOT_FOUND
    assert result.account_id == ACCOUNT


# ─── 16: детектор цифр вернул 0 кропов ──────────────────────────────────────

def test_case16_digit_detector_found_nothing():
    # reader.py:563-564 — тест на уровне _read_meter_digits напрямую
    class EmptyDigitDetector:
        def process_array(self, img, save_crops=False, max_per_class=5, straighten=None):
            return []

    number, reading_str, err, digit_results, digit_bboxes = reader._read_meter_digits(
        EmptyDigitDetector(), FakeDigitOCR({}),
        meter_crop=make_meter_crops()[0]["crop"],
        conf_thresh=0.6,
        expected_digits=5,
    )
    assert number is None
    assert reading_str is None
    assert err == "digit detector found nothing"
    assert digit_results is None
    assert digit_bboxes is None


# ─── 17-19: NO_METER / NO_SERIAL — до этого 0% покрытия ──────────────────────
# Найдено по расхождению с production-отчётом (docs/audit/baseline_1111.txt):
# NO_METER+NO_SERIAL = 127/1111 фото (11.5%) реального трафика, а Outcome-кейсы
# для них отсутствовали в golden tests (были покрыты только 7 из 9 значений
# Outcome). Три разных условия внутри process_photo (reader.py:736-756) дают
# два одинаковых Outcome с разным error_detail — тестируем все три отдельно,
# чтобы будущий рефакторинг (Фаза 2a) не мог тихо их схлопнуть в одно.

def test_case17_no_meter_empty_crops(base_config):
    # reader.py:736-740 — meter_detector вообще ничего не нашёл (crops == [])
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    result = _run(base_config, df, [], FakeDigitDetector([]), FakeDigitOCR({}), SERIAL, 0.95)
    assert result.outcome == Outcome.NO_METER
    assert result.error_detail == "YOLO returned no crops"


def test_case18_no_meter_class_missing(base_config):
    # reader.py:742-751 — crops не пустой, но класса gas_meter среди них нет
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    crops = [{"class": "serial_number", "conf": 0.9, "angle": 0.0,
              "bbox": (0, 0, 100, 30), "crop": make_crop(30, 100), "path": None}]
    result = _run(base_config, df, crops, FakeDigitDetector([]), FakeDigitOCR({}), SERIAL, 0.95)
    assert result.outcome == Outcome.NO_METER
    assert result.error_detail == "class gas_meter not detected"


def test_case19_no_serial_class_missing(base_config):
    # reader.py:753-756 — gas_meter нашёлся, а serial_number — нет
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    crops = [{"class": "gas_meter", "conf": 0.9, "angle": 0.0,
              "bbox": (0, 0, 300, 100), "crop": make_crop(100, 300), "path": None}]
    result = _run(base_config, df, crops, FakeDigitDetector([]), FakeDigitOCR({}), SERIAL, 0.95)
    assert result.outcome == Outcome.NO_SERIAL
    assert result.error_detail == "class serial_number not detected"


# ─── 20-25: вариант Г — какие фото из лога перечитываются ─────────────────────
# Решение владельца 2026-09-29 (docs/MIGRATION_STATUS.md). Шаг 0 process_photo:
# разобранное оператором или успешно прочитанное фото сразу REPEAT, модели не
# запускаются; фото только с авто-ошибками в логе обрабатывается заново.

class _MustNotRun:
    """Детектор, который валит тест, если его вызвали."""
    def process_image(self, *a, **k):
        raise AssertionError("модель не должна запускаться для уже разобранного фото")
    process_array = process_image


def _log_row(fname, outcome, source="auto", account=ACCOUNT, final=""):
    return {"original_filename": fname, "outcome": outcome, "source": source,
            "account_id": account, "serial_id": SERIAL, "final_filename": final}


def test_case20_done_photo_skips_models(base_config):
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    base_config._log_rows_cache = [_log_row("photo1.jpg", "MINUS", final=f"{ACCOUNT}.jpg")]
    result = process_photo(
        "photo1.jpg", df, base_config,
        _MustNotRun(), _MustNotRun(), FakeDigitOCR({}), FakeSerialOCR(SERIAL, 0.95),
    )
    assert result.outcome == Outcome.REPEAT
    assert result.account_id == ACCOUNT
    assert result.new_photo_name == f"{ACCOUNT}.jpg"


def test_case21_manual_row_skips_even_after_auto_error(base_config):
    # оператор пометил фото "Нечитаемо" в program2.py после авто NO_METER
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    base_config._log_rows_cache = [
        _log_row("photo1.jpg", "NO_METER", account=""),
        _log_row("photo1.jpg", "UNREADABLE", source="manual", account=""),
    ]
    result = process_photo(
        "photo1.jpg", df, base_config,
        _MustNotRun(), _MustNotRun(), FakeDigitOCR({}), FakeSerialOCR(SERIAL, 0.95),
    )
    assert result.outcome == Outcome.REPEAT
    assert "handled manually" in result.error_detail
    assert result.new_photo_name == "photo1.jpg"


def test_case22_auto_digits_error_is_reprocessed(base_config):
    # в прошлый раз цифры не прочитались, теперь модель читает -> PLUS
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    base_config._log_rows_cache = [_log_row("photo1.jpg", "DIGITS_ERROR")]
    dd, docr = _digits_setup(base_config, ["0", "1", "2", "0", "0"])
    result = _run(base_config, df, make_meter_crops(), dd, docr, SERIAL, 0.95)
    assert result.outcome == Outcome.PLUS
    assert result.reading == 1200


def test_case23_auto_early_error_is_reprocessed(base_config):
    # в прошлый раз NO_METER -> фото снова проходит модели (здесь снова NO_METER)
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    base_config._log_rows_cache = [_log_row("photo1.jpg", "NO_METER", account="")]
    result = _run(base_config, df, [], FakeDigitDetector([]), FakeDigitOCR({}), SERIAL, 0.95)
    assert result.outcome == Outcome.NO_METER
    assert result.error_detail == "YOLO returned no crops"


def test_case24_table_filled_explained_by_log_is_repeat_not_suspicious(base_config):
    # счётчик уже прочитан по ДРУГОМУ фото (PLUS в логе), таблица заполнена.
    # Раньше это давало SUSPICIOUS "рассинхронизация"; с 2026-09-29 — REPEAT.
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000",
                    "new_reading": "1200"}])
    base_config._log_rows_cache = [
        _log_row("other.jpg", "PLUS"),
        _log_row("photo1.jpg", "DIGITS_ERROR"),
    ]
    dd, docr = _digits_setup(base_config, ["0", "1", "2", "0", "0"])
    result = _run(base_config, df, make_meter_crops(), dd, docr, SERIAL, 0.95)
    assert result.outcome == Outcome.REPEAT
    assert "already has reading in log" in result.error_detail


def test_case25_auto_repeat_counts_as_done(base_config):
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    base_config._log_rows_cache = [_log_row("photo1.jpg", "REPEAT")]
    result = process_photo(
        "photo1.jpg", df, base_config,
        _MustNotRun(), _MustNotRun(), FakeDigitOCR({}), FakeSerialOCR(SERIAL, 0.95),
    )
    assert result.outcome == Outcome.REPEAT


# ─── 26-28: фото узнаётся по отпечатку, старые строки — по имени ──────────────
# Решение владельца 2026-09-29 ("хеш + подпапка", docs/MIGRATION_STATUS.md).

def _hashed_row(fname, outcome, photo_hash):
    return _log_row(fname, outcome) | {"photo_hash": photo_hash}


def test_case26_same_name_other_hash_is_processed(base_config):
    # в логе IMG.jpg из другой подпапки (другое фото, другой отпечаток)
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    base_config._log_rows_cache = [_hashed_row("photo1.jpg", "PLUS", "aaaa")]
    dd, docr = _digits_setup(base_config, ["0", "1", "2", "0", "0"])
    result = process_photo(
        "photo1.jpg", df, base_config, FakeMeterDetector(make_meter_crops()), dd, docr,
        FakeSerialOCR(SERIAL, 0.95), photo_hash="bbbb",
    )
    assert result.outcome == Outcome.PLUS


def test_case27_same_hash_other_name_skips_models(base_config):
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    base_config._log_rows_cache = [_hashed_row("original.jpg", "PLUS", "aaaa")]
    result = process_photo(
        "renamed.jpg", df, base_config,
        _MustNotRun(), _MustNotRun(), FakeDigitOCR({}), FakeSerialOCR(SERIAL, 0.95),
        photo_hash="aaaa",
    )
    assert result.outcome == Outcome.REPEAT


def test_case28_legacy_row_without_hash_matches_by_name(base_config):
    # строки, записанные до 2026-09-29, отпечатка не имеют — узнаём по имени
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    base_config._log_rows_cache = [_log_row("photo1.jpg", "PLUS")]
    result = process_photo(
        "photo1.jpg", df, base_config,
        _MustNotRun(), _MustNotRun(), FakeDigitOCR({}), FakeSerialOCR(SERIAL, 0.95),
        photo_hash="cccc",
    )
    assert result.outcome == Outcome.REPEAT


# ─── 29-30: побайтная копия файла в текущем прогоне ───────────────────────────
# Решение владельца 2026-09-29: копия уже обработанного в ЭТОМ прогоне файла —
# сразу REPEAT при любом исходе первой копии. Между прогонами — правило Г.

def test_case29_copy_in_current_run_is_repeat_even_after_error(base_config):
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    base_config._log_rows_cache = [_hashed_row("first.jpg", "NO_METER", "aaaa") | {"account_id": ""}]
    base_config._processed_hashes_cache = {"aaaa"}
    result = process_photo(
        "copy.jpg", df, base_config,
        _MustNotRun(), _MustNotRun(), FakeDigitOCR({}), FakeSerialOCR(SERIAL, 0.95),
        photo_hash="aaaa",
    )
    assert result.outcome == Outcome.REPEAT
    assert "duplicate file in current run" in result.error_detail
    assert result.new_photo_name == "copy.jpg"


def test_case30_same_error_from_previous_run_is_reprocessed(base_config):
    # та же строка лога, но отпечаток НЕ из этого прогона -> правило Г, перечитываем
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    base_config._log_rows_cache = [_hashed_row("first.jpg", "NO_METER", "aaaa") | {"account_id": ""}]
    base_config._processed_hashes_cache = set()
    result = process_photo(
        "copy.jpg", df, base_config, FakeMeterDetector([]), FakeDigitDetector([]),
        FakeDigitOCR({}), FakeSerialOCR(SERIAL, 0.95), photo_hash="aaaa",
    )
    assert result.outcome == Outcome.NO_METER


def test_case31_copy_repeat_rows_do_not_block_retry(base_config):
    # в логе: оригинал NO_METER + его копия REPEAT ("duplicate file in current run")
    # из прошлого прогона. Строка копии не считается "разобрано" -> перечитываем.
    df = make_df([{"serial": SERIAL, "account_id": ACCOUNT, "last_reading": "1000"}])
    base_config._log_rows_cache = [
        _hashed_row("first.jpg", "NO_METER", "aaaa") | {"account_id": ""},
        _hashed_row("copy.jpg", "REPEAT", "aaaa") | {"account_id": "",
            "notes": "duplicate file in current run: copy.jpg"},
    ]
    base_config._processed_hashes_cache = set()
    result = process_photo(
        "first.jpg", df, base_config, FakeMeterDetector([]), FakeDigitDetector([]),
        FakeDigitOCR({}), FakeSerialOCR(SERIAL, 0.95), photo_hash="aaaa",
    )
    assert result.outcome == Outcome.NO_METER
