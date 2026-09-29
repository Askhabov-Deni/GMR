"""
src/gmr/domain/policies.py — явные policy-объекты для бизнес-правил,
вынесенных из reader.py (Фаза 2a, docs/MIGRATION_TZ.md).

Правило переноса: значения и порядок проверок не меняются. Каждый класс
здесь — прямая пересадка соответствующего блока reader.py в изолированную,
не зависящую от torch/cv2/ultralytics форму (только примитивы: int, float,
str, list, set, dict). Соответствие golden tests (tests/test_golden_reader.py)
и правилам см. в докстринге каждого класса.

Эти классы принимают уже готовые данные (bbox-центры, флаги, confidence),
а не сырые кропы/модели — сам вызов моделей (YOLO/CNN/CRNN) остаётся в
reader.py, как и предписано Фазой 2a ("reader.py продолжает работать,
просто дёргая новые классы вместо инлайн-кода").
"""
from dataclasses import dataclass
from typing import Optional

from .models import Outcome


# ─── Восстановление пропущенной цифры (кейсы 12-13) ──────────────────────────

@dataclass
class RecoveryDecision:
    """missing_idx — куда вставить заглушку; reason — None при успехе,
    иначе 'gap not found' или 'critical position' (см. _read_meter_digits)."""
    missing_idx: Optional[int]
    reason: Optional[str]


class MissingDigitRecoveryPolicy:
    """
    Что делать, если детектор нашёл ровно (expected_digits - 1) цифр вместо
    expected_digits: попытаться геометрически восстановить пропущенную позицию
    по разрыву между центрами соседних bbox-ов (или по отступу от края, если
    пропала крайняя цифра), и отказать, если позиция критическая (0 или 1) —
    первые цифры счётчика слишком значимы, чтобы гадать заглушкой.

    Перенесено из reader.py:_read_meter_digits (ветка `n == expected_digits - 1`)
    без изменения порогов (1.6× среднего шага) и без изменения условия
    критической позиции (< 2). Golden tests: case12 (успешное восстановление,
    позиция 2) и case13 (отказ, критическая позиция).
    """

    GAP_RATIO = 1.6          # во сколько раз разрыв должен превышать средний шаг
    CRITICAL_POSITION = 2    # позиции < этого числа — восстановление запрещено

    def decide(
        self,
        centers: list[int],
        crop_width: int,
        expected_digits: int,
    ) -> RecoveryDecision:
        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        avg_step = sum(gaps) / len(gaps)

        missing_idx = None
        for i, gap in enumerate(gaps):
            if gap > avg_step * self.GAP_RATIO:
                missing_idx = i + 1
                break

        if missing_idx is None:
            left_margin = centers[0]
            right_margin = crop_width - centers[-1]
            if right_margin > left_margin * self.GAP_RATIO:
                missing_idx = expected_digits - 1
            elif left_margin > right_margin * self.GAP_RATIO:
                missing_idx = 0
            else:
                return RecoveryDecision(None, "gap not found")

        if missing_idx < self.CRITICAL_POSITION:
            return RecoveryDecision(missing_idx, "critical position")

        return RecoveryDecision(missing_idx, None)


# ─── Прощение низкого конфиданса на хвостовых позициях (кейс 14) ─────────────

class DigitForgivenessPolicy:
    """
    Какие позиции цифр счётчика "прощаются" при низком конфидансе OCR —
    последние `ignore_last_digits` позиций (десятые/сотые доли на счётчике
    часто не несут веса). При прощении используется forgiven_digit_placeholder
    вместо '?' и она НЕ считается ошибкой (has_error не выставляется).

    Перенесено из reader.py:_read_meter_digits без изменения условия
    (`conf < conf_thresh and position in forgiven_positions`).
    Golden test: case14.
    """

    def forgiven_positions(self, expected_digits: int, ignore_last_digits: int) -> set[int]:
        return set(range(expected_digits - ignore_last_digits, expected_digits))

    def is_forgiven(
        self,
        position: int,
        confidence: float,
        conf_thresh: float,
        forgiven_positions: set[int],
    ) -> bool:
        return confidence < conf_thresh and position in forgiven_positions


# ─── PLUS / MINUS / SUSPICIOUS по дельте показаний (кейсы 1-3) ───────────────

class DeltaThresholdPolicy:
    """
    Итоговый Outcome по разнице между новыми и последними показаниями:
      - last_reading отсутствует        → PLUS, delta=None
      - |delta| > delta_threshold        → SUSPICIOUS
      - delta >= 0                       → PLUS
      - delta < 0                        → MINUS

    Перенесено из reader.py:process_photo, Шаг 6, без изменения порогов.
    Golden tests: case1 (PLUS), case2 (MINUS), case3 (SUSPICIOUS),
    case4 (last_reading=None → PLUS, delta=None).
    """

    def decide(
        self,
        reading: int,
        last_reading: Optional[float],
        delta_threshold: float,
    ) -> tuple[Outcome, Optional[float]]:
        if last_reading is None:
            return Outcome.PLUS, None

        delta = reading - last_reading
        if abs(delta) > delta_threshold:
            return Outcome.SUSPICIOUS, delta

        return (Outcome.PLUS if delta >= 0 else Outcome.MINUS), delta


# ─── Дубли: лог → processed-in-run → pre-existing → рассинхронизация ─────────

@dataclass
class DuplicateDecision:
    """outcome=None означает "не дубль и не аномалия, можно читать показания".
    Иначе — Outcome.REPEAT или Outcome.SUSPICIOUS с объяснением в reason."""
    outcome: Optional[Outcome]
    reason: Optional[str]


class DuplicatePolicy:
    """
    Проверки по лицевому счёту (вызывается на шаге 4 process_photo, когда
    серийник уже найден в таблице):

      1. этот account_id уже обработан в текущем прогоне    → REPEAT
      2. в таблице уже стоит показание:
         2a. и это pre_existing запись (инициализация из таблицы) → REPEAT
         2b. в логе есть PLUS/MINUS по этому счёту (auto или manual,
             например по другому фото того же счётчика)          → REPEAT
         2c. ни одна строка лога его не объясняет                 → SUSPICIOUS
             (рассинхронизация таблицы и лога — показания НЕ трогаем)
      3. иначе — не дубль, идём читать показания

    Проверка "это фото уже в логе" отсюда вынесена в ProcessedPhotoPolicy
    (шаг 0, до запуска моделей) — решение владельца 2026-09-29, вариант Г,
    см. docs/MIGRATION_STATUS.md. Пункт 2b добавлен тогда же: без него фото,
    которое перечитывается после авто-ошибки, получало бы ложный SUSPICIOUS,
    если счётчик уже прочитан по другому фото.

    Перенесено из reader.py:process_photo, Шаг 4, включая нормализацию
    account_id через int(float(...)) (pandas читает числовые колонки как
    float → "1300000013.0" вместо "1300000013").
    Golden tests: case6, case7, case8, case24.
    """

    @staticmethod
    def normalize_account_id(value: str) -> str:
        value = value.strip()
        try:
            return str(int(float(value)))
        except (ValueError, OverflowError):
            return value

    def decide(
        self,
        account_id: str,
        table_new_reading_filled: bool,
        processed_accounts: set,
        log_rows: list,
    ) -> DuplicateDecision:
        if account_id in processed_accounts:
            return DuplicateDecision(
                Outcome.REPEAT, f"duplicate in current run: account={account_id}"
            )

        if not table_new_reading_filled:
            return DuplicateDecision(None, None)

        account_norm = self.normalize_account_id(str(account_id))
        is_pre_existing = any(
            r.get("source") == "pre_existing"
            and self.normalize_account_id(str(r.get("account_id", ""))) == account_norm
            for r in log_rows
        )
        if is_pre_existing:
            return DuplicateDecision(
                Outcome.REPEAT, f"pre-existing entry in table (account={account_id})"
            )

        has_reading_in_log = any(
            r.get("outcome") in ("PLUS", "MINUS")
            and self.normalize_account_id(str(r.get("account_id", ""))) == account_norm
            for r in log_rows
        )
        if has_reading_in_log:
            return DuplicateDecision(
                Outcome.REPEAT, f"account already has reading in log (account={account_id})"
            )

        return DuplicateDecision(
            Outcome.SUSPICIOUS,
            "аномалия: данные в таблице есть, фото не найдено в логе — "
            "возможна рассинхронизация",
        )


# ─── Фото уже обрабатывалось? (шаг 0, вариант Г) ─────────────────────────────

@dataclass
class ProcessedPhotoDecision:
    """skip=True — фото уже разобрано, сразу REPEAT, модели не запускаются.
    row — строка лога, по которой принято решение (для имени файла и счёта)."""
    skip: bool
    reason: Optional[str] = None
    row: Optional[dict] = None


class ProcessedPhotoPolicy:
    """
    Нужно ли обрабатывать фото, которое уже встречалось в логе.
    Решение владельца 2026-09-29 (вариант Г, docs/MIGRATION_STATUS.md):

      - в логе нет строк с этим original_filename          → обрабатываем
      - есть строка source="manual" (оператор разобрал фото
        в program2.py: PLUS/MINUS/REPEAT/UNREADABLE/NOT_IN_DB) → REPEAT
      - есть auto-строка с исходом PLUS/MINUS/REPEAT        → REPEAT
      - есть только auto-строки с ошибками (NO_METER,
        NO_SERIAL, SERIAL_LOW_CONF, SERIAL_NOT_FOUND,
        DIGITS_ERROR, SUSPICIOUS)                            → обрабатываем заново

    Зачем: разобранное оператором фото не должно снова попадать в question/
    (иначе его обработают второй раз), а неразобранная авто-ошибка может
    прочитаться, если модели стали лучше.

    auto-REPEAT считается завершённым: повторная обработка такого фото
    (например, второго фото того же счётчика) ничего не даёт и приводила бы
    к ложной "рассинхронизации".

    Идентичность фото — только имя файла (original_filename), как и раньше.
    Два разных фото с одинаковым именем в разных подпапках для лога
    неразличимы — см. "Найденные баги" в MIGRATION_STATUS.md.

    Golden tests: case5, case20-case23, case25.
    """

    DONE_OUTCOMES = ("PLUS", "MINUS", "REPEAT")

    def decide(self, photo_filename: str, log_rows: list) -> ProcessedPhotoDecision:
        rows = [r for r in log_rows if r.get("original_filename") == photo_filename]
        if not rows:
            return ProcessedPhotoDecision(skip=False)

        manual = [r for r in rows if r.get("source") == "manual"]
        if manual:
            return ProcessedPhotoDecision(
                True, f"already in log: {photo_filename} (handled manually)", manual[-1]
            )

        done = [r for r in rows if r.get("outcome") in self.DONE_OUTCOMES]
        if done:
            return ProcessedPhotoDecision(True, f"already in log: {photo_filename}", done[-1])

        return ProcessedPhotoDecision(skip=False)
