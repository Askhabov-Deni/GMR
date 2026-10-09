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
    AccountRecognizer,
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
    """Модели пайплайна, уже обёрнутые в контракты. account_recognizer —
    надпись маркером, необязательная (None — её нет, PipelineConfig.account_ocr_model)."""
    meter_detector: MeterDetector
    digit_detector: DigitDetector
    digit_recognizer: DigitRecognizer
    serial_recognizer: SerialRecognizer
    account_recognizer: Optional[AccountRecognizer] = None


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


def _gap_bbox(boxes: list, idx: int, crop_shape) -> Optional[tuple]:
    """Рамка пропущенной цифры (пресет «прочитать моделью на этом месте»):
    ширина — медиана найденных, высота — как у соседей, центр — посередине
    между соседями (у крайней позиции — на средний шаг от соседа)."""
    found = [b for b in boxes if b is not None]
    width = sorted(b[2] - b[0] for b in found)[len(found) // 2]
    centers = [(b[0] + b[2]) / 2 for b in found]
    step = (centers[-1] - centers[0]) / (len(found) - 1) if len(found) > 1 else width
    left = boxes[idx - 1] if idx > 0 else None
    right = boxes[idx + 1] if idx + 1 < len(boxes) else None
    if left is not None and right is not None:
        cx, near = ((left[0] + left[2]) / 2 + (right[0] + right[2]) / 2) / 2, [left, right]
    elif left is not None:
        cx, near = (left[0] + left[2]) / 2 + step, [left]
    else:
        cx, near = (right[0] + right[2]) / 2 - step, [right]
    x1 = max(0, int(round(cx - width / 2)))
    x2 = min(int(crop_shape[1]), int(round(cx + width / 2)))
    y1, y2 = min(b[1] for b in near), max(b[3] for b in near)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, int(y1), x2, int(y2))


def _drum_digit(top: list, conf_thresh: float) -> Optional[str]:
    """Правило барабана: две лучшие цифры соседние (d и d+1, 9 и 0) и вместе
    уверенные → барабан между ними, на счётчике ещё d (между 9 и 0 — 9)."""
    if len(top) < 2:
        return None
    (a, pa), (b, pb) = top[0], top[1]
    pair = {int(a), int(b)}
    if pa + pb < conf_thresh:
        return None
    if pair == {0, 9}:
        return "9"
    return str(min(pair)) if max(pair) - min(pair) == 1 else None


def _last_digits(last_reading, expected_digits: int) -> Optional[str]:
    """Прошлое показание цифрами («1234.0» → «01234»); None — не прочитать."""
    try:
        s = str(int(float(str(last_reading).replace(",", ".")))).zfill(expected_digits)
    except (TypeError, ValueError):
        return None
    return s if len(s) == expected_digits else None


def read_meter_digits(
    digit_detector: DigitDetector,
    digit_recognizer: DigitRecognizer,
    meter_crop: Any,
    conf_thresh: float,
    expected_digits: int,
    ignore_last_digits: int = 0,
    missing_placeholder: str = "5",
    forgiven_placeholder: str = "0",
    missing_mode: str = "placeholder",
    forgiven_mode: str = "placeholder",
    drum_rule: bool = False,
    last_reading=None,
) -> DigitReading:
    """
    Прогоняет кроп счётчика через детектор цифр + классификатор цифр.

    missing_placeholder  — символ для восстановленной пропущенной позиции
    forgiven_placeholder — символ для прощённых хвостовых позиций
    ignore_last_digits   — сколько последних позиций прощаем при низком конфидансе
    missing_mode  — "placeholder" (как было) | "model" (прочитать цифру на месте
                    пропуска) | "operator" (позиция «?», DIGITS_ERROR)
    forgiven_mode — "placeholder" (как было) | "model" (ответ модели как есть)
    drum_rule     — правило барабана для неуверенных цифр (_drum_digit)
    last_reading  — прошлое показание: если неуверенна только старшая цифра и
                    его цифра среди двух лучших — берётся она (None — правило выключено)

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
    boxes = [dc["bbox"] if dc is not None else None for dc in digit_crops_sorted]

    digits        = []
    digit_results = []
    digit_bboxes  = []   # (x1,y1,x2,y2) в координатах кропа, None для заглушек
    has_error     = False

    for pos, dc in enumerate(digit_crops_sorted):
        recovered = False
        if dc is None:
            bbox = _gap_bbox(boxes, pos, meter_crop.shape) if missing_mode == "model" else None
            if missing_mode == "operator" or (missing_mode == "model" and bbox is None):
                # пропуск — оператору: позиция «?», показание не пишется
                digits.append("?")
                digit_results.append({"position": pos, "digit": "?", "confidence": None,
                                      "ok": False, "note": "missing→operator"})
                digit_bboxes.append(None)
                has_error = True
                continue
            if bbox is None:
                # Заглушка восстановленной позиции (как было)
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
            x1, y1, x2, y2 = bbox
            dc, recovered = {"crop": meter_crop[y1:y2, x1:x2], "bbox": bbox}, True

        pred = digit_recognizer.recognize(dc["crop"])
        conf = pred.confidence
        digit_char = pred.digit
        ok = conf >= conf_thresh
        top = list(getattr(pred, "top", None) or [])
        drum = _drum_digit(top, conf_thresh) if (drum_rule and not ok) else None

        if drum is not None:
            digit_results.append({
                "position":   pos,
                "digit":      digit_char,
                "confidence": round(float(conf), 4),
                "ok":         True,
                "note":       f"drum→{drum}",
                "top":        top[:2],
            })
            digits.append(drum)
        elif not ok and _forgiveness_policy.is_forgiven(pos, conf, conf_thresh, forgiven_positions):
            by_model = forgiven_mode == "model"
            digit_results.append({
                "position":   pos,
                "digit":      digit_char,
                "confidence": round(float(conf), 4),
                "ok":         False,
                "note":       "forgiven→model" if by_model else f"forgiven→{forgiven_placeholder}",
            })
            digits.append(digit_char if by_model else forgiven_placeholder)
        else:
            r = {
                "position":   pos,
                "digit":      digit_char,
                "confidence": round(float(conf), 4),
                "ok":         ok,
            }
            if recovered:
                r["note"] = "missing→model"
            if not ok and top:
                r["top"] = top[:2]
            digit_results.append(r)
            if ok:
                digits.append(digit_char)
            else:
                digits.append("?")
                has_error = True

        digit_bboxes.append(dc["bbox"])

    # старшая цифра по прошлому показанию: неуверенна только она, и цифра
    # прошлого показания на этой позиции — среди двух лучших
    unsure = [r for r in digit_results if not r["ok"] and not r.get("note", "").startswith("forgiven")]
    prev = _last_digits(last_reading, expected_digits) if last_reading is not None else None
    if (has_error and prev and len(unsure) == 1 and unsure[0]["position"] == 0
            and prev[0] in [d for d, _ in unsure[0].get("top", [])[:2]]):
        unsure[0].update(ok=True, note=f"last→{prev[0]}")
        digits[0], has_error = prev[0], False

    reading_str = "".join(digits)

    if has_error:
        bad = [
            f"pos{r['position']}(pred={r['digit']},conf={r['confidence']:.3f})"
            if r.get("confidence") is not None else f"pos{r['position']}(цифра не найдена)"
            for r in digit_results
            if not r["ok"]
            and r.get("note", "").startswith("forgiven") is False
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
        elif note == "missing→model":
            parts.append(f"pos{r['position']}='{r['digit']}' (цифра не найдена) — прочитано моделью "
                         f"на её месте, conf={r['confidence']:.3f}")
        elif note.startswith("drum→"):
            (a, pa), (b, pb) = r["top"][:2]
            parts.append(f"pos{r['position']}='{note[len('drum→'):]}' "
                         f"(барабан между {a} и {b}, conf={pa:.3f}/{pb:.3f})")
        elif note.startswith("last→"):
            (a, pa), (b, pb) = r["top"][:2]
            parts.append(f"pos{r['position']}='{note[len('last→'):]}' "
                         f"(по прошлому показанию; модель: {a} {pa:.3f} / {b} {pb:.3f})")
        elif note == "forgiven→model":
            parts.append(f"pos{r['position']}='{r['digit']}' "
                         f"(прочитано {r['digit']}, conf={r['confidence']:.3f}) — неуверенно, взято как есть")
        elif note.startswith("forgiven→"):
            placeholder = note[len("forgiven→"):]
            parts.append(f"pos{r['position']}='{placeholder}' "
                         f"(прочитано {r['digit']}, conf={r['confidence']:.3f})")
    return f"{SUBSTITUTED_PREFIX} " + ", ".join(parts) if parts else None


def read_meter_digits_for_config(
    models: RecognitionModels, meter_crop: Any, config: PipelineConfig, last_reading=None,
) -> DigitReading:
    """read_meter_digits с порогами и заглушками из PipelineConfig.
    last_reading — прошлое показание счёта (если известен счёт; нужно для
    first_digit_from_last)."""
    return read_meter_digits(
        models.digit_detector, models.digit_recognizer,
        meter_crop,
        config.digit_conf_thresh,
        config.expected_digits,
        ignore_last_digits=config.ignore_last_digits,
        missing_placeholder=config.missing_digit_placeholder,
        forgiven_placeholder=config.forgiven_digit_placeholder,
        missing_mode=config.missing_digit_mode,
        forgiven_mode=config.forgiven_digit_mode,
        drum_rule=config.drum_rule,
        last_reading=last_reading if config.first_digit_from_last else None,
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
    models: RecognitionModels, photo_path: str, config: PipelineConfig, last_reading=None,
) -> PhotoRecognition:
    """
    Детекция → серийник (если найден) → цифры (если найден счётчик).
    Серийник и цифры читаются независимо: отсутствие одного не мешает другому.
    last_reading — прошлое показание (для first_digit_from_last; эталон).
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
        result.digits = read_meter_digits_for_config(models, result.meter["crop"], config,
                                                     last_reading=last_reading)

    return result
