"""
src/gmr/application/recognition.py — распознавание фото через контракты
моделей (Фаза 3, docs/MIGRATION_TZ.md).

Общий код для обеих точек входа:
  - reader.process_photo      (автоматический прогон)
  - program2.ModelBundle      (ручная обработка оператором)
До Фазы 3 program2.py импортировал из reader.py приватные
_read_meter_digits/_find_crop_entry и сам повторял шаги детекции.

read_meter_digits — перенесённый без изменений логики
reader._read_meter_digits: тот же порядок, те же сообщения об ошибках,
те же policy (восстановление пропущенной цифры, прощение хвостовых позиций).
Изменилось только то, как вызываются модели: через DigitDetector /
DigitRecognizer вместо process_array / predict.

Модуль не импортирует torch/cv2/pandas.
"""
from dataclasses import dataclass
from typing import Any, Optional

from src.gmr.domain.config import PipelineConfig
from src.gmr.domain.ml import (
    Detection,
    DigitDetector,
    DigitRecognizer,
    MeterDetector,
    SerialPrediction,
    SerialRecognizer,
)
from src.gmr.domain.policies import DigitForgivenessPolicy, MissingDigitRecoveryPolicy

_recovery_policy    = MissingDigitRecoveryPolicy()
_forgiveness_policy = DigitForgivenessPolicy()


@dataclass
class RecognitionModels:
    """Четыре модели пайплайна, уже обёрнутые в контракты."""
    meter_detector: MeterDetector
    digit_detector: DigitDetector
    digit_recognizer: DigitRecognizer
    serial_recognizer: SerialRecognizer


@dataclass
class DigitReading:
    """
    Результат чтения цифр счётчика (поля — как кортеж _read_meter_digits):

    - number:        int, если все значимые цифры распознаны, иначе None
    - reading_str:   строка вида "56?76" (? — нераспознанная позиция);
                     None только при фатальной ошибке до классификации
    - error:         описание ошибки или None при успехе
    - digit_results: dict по каждой позиции (для debug); None при фатальной ошибке
    - digit_bboxes:  (x1,y1,x2,y2) в координатах кропа по позициям,
                     None для заглушек; None при фатальной ошибке
    """
    number: Optional[int]
    reading_str: Optional[str]
    error: Optional[str]
    digit_results: Optional[list]
    digit_bboxes: Optional[list]

    def as_tuple(self) -> tuple:
        return (self.number, self.reading_str, self.error,
                self.digit_results, self.digit_bboxes)


def find_detection(detections: Optional[list[Detection]], class_name: str) -> Optional[Detection]:
    """Первая детекция нужного класса (весь dict, включая bbox) или None."""
    for c in detections or ():
        if c["class"] == class_name:
            return c
    return None


def read_meter_digits(
    digit_detector: DigitDetector,
    digit_recognizer: DigitRecognizer,
    meter_crop: Any,
    conf_thresh: float,
    expected_digits: int,
    ignore_last_digits: int = 0,
    missing_placeholder: str = "5",
    forgiven_placeholder: str = "0",
) -> DigitReading:
    """
    Прогоняет кроп счётчика через детектор цифр + классификатор цифр.

    missing_placeholder  — символ для восстановленной пропущенной позиции
    forgiven_placeholder — символ для прощённых хвостовых позиций
    ignore_last_digits   — сколько последних позиций прощаем при низком конфидансе

    При ровно expected_digits-1 найденных цифрах пытается восстановить
    пропущенную позицию (только если пропала не одна из двух первых).
    """
    digit_crops = digit_detector.detect(meter_crop)

    if not digit_crops:
        return DigitReading(None, None, "digit detector found nothing", None, None)

    digit_crops_sorted = sorted(digit_crops, key=lambda c: c["bbox"][0])
    n = len(digit_crops_sorted)

    if n == expected_digits:
        pass  # всё хорошо

    elif n == expected_digits - 1:
        centers = [(c["bbox"][0] + c["bbox"][2]) // 2 for c in digit_crops_sorted]
        decision = _recovery_policy.decide(centers, meter_crop.shape[1], expected_digits)

        if decision.reason == "gap not found":
            return DigitReading(
                None, None, f"expected {expected_digits} digits, got {n} (gap not found)", None, None
            )

        if decision.reason == "critical position":
            return DigitReading(None, None, (
                f"expected {expected_digits} digits, got {n} "
                f"(missing digit at critical position {decision.missing_idx})"
            ), None, None)

        digit_crops_sorted.insert(decision.missing_idx, None)  # None = заглушка

    else:
        return DigitReading(None, None, f"expected {expected_digits} digits, got {n}", None, None)

    # ── Классифицируем цифры ──────────────────────────────────────────────────
    forgiven_positions = _forgiveness_policy.forgiven_positions(expected_digits, ignore_last_digits)

    digits        = []
    digit_results = []
    digit_bboxes  = []   # (x1,y1,x2,y2) в координатах кропа, None для заглушек
    has_error     = False

    for pos, dc in enumerate(digit_crops_sorted):
        if dc is None:
            # Заглушка восстановленной позиции
            digits.append(missing_placeholder)
            digit_results.append({
                "position":   pos,
                "digit":      missing_placeholder,
                "confidence": None,
                "ok":         True,
                "note":       "inserted placeholder",
            })
            digit_bboxes.append(None)
            continue

        pred = digit_recognizer.recognize(dc["crop"])
        conf = pred.confidence
        digit_char = pred.digit
        ok = conf >= conf_thresh

        if not ok and _forgiveness_policy.is_forgiven(pos, conf, conf_thresh, forgiven_positions):
            digit_results.append({
                "position":   pos,
                "digit":      digit_char,
                "confidence": round(float(conf), 4),
                "ok":         False,
                "note":       f"forgiven→{forgiven_placeholder}",
            })
            digits.append(forgiven_placeholder)
        else:
            digit_results.append({
                "position":   pos,
                "digit":      digit_char,
                "confidence": round(float(conf), 4),
                "ok":         ok,
            })
            if ok:
                digits.append(digit_char)
            else:
                digits.append("?")
                has_error = True

        digit_bboxes.append(dc["bbox"])

    reading_str = "".join(digits)

    if has_error:
        bad = [
            f"pos{r['position']}(pred={r['digit']},conf={r['confidence']:.3f})"
            for r in digit_results
            if not r["ok"]
            and r.get("note", "").startswith("forgiven") is False
            and r.get("confidence") is not None
        ]
        error_msg = f"low conf digits: {', '.join(bad)}  →  '{reading_str}'"
        return DigitReading(None, reading_str, error_msg, digit_results, digit_bboxes)

    number = int(reading_str)
    return DigitReading(number, reading_str, None, digit_results, digit_bboxes)


SUBSTITUTED_PREFIX = "подставлено:"


def describe_substitutions(digit_results: Optional[list]) -> Optional[str]:
    """
    Какие позиции в показании не прочитаны моделью, а подставлены — для notes
    лога. Например:
      "подставлено: pos2='5' (цифра не найдена), pos4='0' (прочитано 7, conf=0.412)"
    None — если подстановок не было. Позиции — с 0, как в "low conf digits".
    """
    parts = []
    for r in digit_results or []:
        note = r.get("note", "")
        if note == "inserted placeholder":
            parts.append(f"pos{r['position']}='{r['digit']}' (цифра не найдена)")
        elif note.startswith("forgiven→"):
            placeholder = note[len("forgiven→"):]
            parts.append(f"pos{r['position']}='{placeholder}' "
                         f"(прочитано {r['digit']}, conf={r['confidence']:.3f})")
    return f"{SUBSTITUTED_PREFIX} " + ", ".join(parts) if parts else None


def read_meter_digits_for_config(
    models: RecognitionModels, meter_crop: Any, config: PipelineConfig,
) -> DigitReading:
    """read_meter_digits с порогами и заглушками из PipelineConfig."""
    return read_meter_digits(
        models.digit_detector, models.digit_recognizer,
        meter_crop,
        config.digit_conf_thresh,
        config.expected_digits,
        ignore_last_digits=config.ignore_last_digits,
        missing_placeholder=config.missing_digit_placeholder,
        forgiven_placeholder=config.forgiven_digit_placeholder,
    )


def digit_crops_by_position(meter_crop: Any, digit_bboxes: list) -> list:
    """Вырезает кроп каждой цифры из кропа счётчика по bbox; None для заглушек."""
    crops = []
    for bbox in digit_bboxes:
        if bbox is None:
            crops.append(None)
        else:
            x1, y1, x2, y2 = bbox
            crops.append(meter_crop[y1:y2, x1:x2].copy())
    return crops


@dataclass
class PhotoRecognition:
    """
    Всё, что модели увидели на одном фото, без решений по таблице и логу
    (для ручной обработки в program2.py).

    detections — пусто/None, если детектор счётчика ничего не нашёл;
    тогда остальные поля не заполнены.
    """
    detections: Optional[list[Detection]]
    meter: Optional[Detection] = None
    serial: Optional[Detection] = None
    serial_prediction: Optional[SerialPrediction] = None
    digits: Optional[DigitReading] = None


def recognize_photo(
    models: RecognitionModels, photo_path: str, config: PipelineConfig,
) -> PhotoRecognition:
    """
    Детекция → серийник (если найден) → цифры (если найден счётчик).
    Серийник и цифры читаются независимо: отсутствие одного не мешает другому.
    """
    detections = models.meter_detector.detect(photo_path)
    result = PhotoRecognition(detections=detections)
    if not detections:
        return result

    result.meter = find_detection(detections, "gas_meter")
    result.serial = find_detection(detections, "serial_number")

    if result.serial is not None:
        result.serial_prediction = models.serial_recognizer.recognize(result.serial["crop"])

    if result.meter is not None:
        result.digits = read_meter_digits_for_config(models, result.meter["crop"], config)

    return result
