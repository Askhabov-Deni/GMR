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
    Порядок проверок ФИКСИРОВАН докс/MIGRATION_TZ.md (раздел 2, кейсы 5-8) и
    не подлежит изменению при рефакторинге:

      1. фото уже есть в логе                              → REPEAT
      2. этот account_id уже обработан в текущем прогоне    → REPEAT
      3. в таблице уже стоит показание:
         3a. и это pre_existing запись (инициализация из таблицы) → REPEAT
         3b. а лога для неё нет                                    → SUSPICIOUS
             (рассинхронизация таблицы и лога — показания НЕ трогаем)
      4. иначе — не дубль, идём читать показания

    Перенесено из reader.py:process_photo, Шаг 4, включая нормализацию
    account_id через int(float(...)) (pandas читает числовые колонки как
    float → "1300000013.0" вместо "1300000013").
    Golden tests: case5, case6, case7, case8.
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
        photo_filename: str,
        account_id: str,
        table_new_reading_filled: bool,
        log_filenames: set,
        processed_accounts: set,
        log_rows: list,
    ) -> DuplicateDecision:
        if photo_filename in log_filenames:
            return DuplicateDecision(Outcome.REPEAT, f"already in log: {photo_filename}")

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

        return DuplicateDecision(
            Outcome.SUSPICIOUS,
            "аномалия: данные в таблице есть, фото не найдено в логе — "
            "возможна рассинхронизация",
        )
