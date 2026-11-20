"""
Что модели прочитали на фото, ушедшем оператору (этап Е, решение владельца
2026-11-21). Прогон сохраняет это в базе месяца (MonthDB.put_recognition), окно
оператора показывает сразу и само моделей не запускает. Раньше окно читало
фото заново — копию с подписью из question/, — и номер мог выйти другим, чем у
прогона (тест месяца 2026-11-20).

Прогон читает не всё на каждом пути: счётчик не найден — номер не читается,
табличка с номером не найдена — цифры не читаются. Окно же показывало оператору
и то и другое. Недостающее дочитывается здесь по уже найденным рамкам; исход
фото, строка лога и фото с подписью от этого не меняются.

Кропы (номер, каждая цифра) — для тихой разметки: если оператор исправит
номер или цифру, окно сохранит кроп с верным ответом для дообучения моделей.
"""
import logging
from typing import Optional

import cv2

from src.gmr.application import digit_crops_by_position, find_detection, read_meter_digits_for_config
from src.gmr.domain import OUTCOME_FOLDER, Outcome, PhotoResult, PipelineConfig
from src.gmr.domain.datasets import METER, SERIAL
from src.gmr.storage.month import Recognition

JPEG_QUALITY = 95


def for_operator(result: PhotoResult) -> bool:
    """Фото ушло в очередь оператора (question/), и программа на нём не упала."""
    return OUTCOME_FOLDER[result.outcome].startswith("question/") and result.outcome != Outcome.ERROR


def _jpeg(img) -> Optional[bytes]:
    if img is None or getattr(img, "size", 0) == 0:
        return None
    try:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    except cv2.error:
        return None
    return buf.tobytes() if ok else None


def what_models_saw(result: PhotoResult, models, config: PipelineConfig) -> Optional[Recognition]:
    """Запись для окна оператора; None — фото не для оператора или на нём ничего не найдено."""
    if not for_operator(result) or not result.meter_crops_raw:
        return None
    meter = find_detection(result.meter_crops_raw, METER)
    serial = find_detection(result.meter_crops_raw, SERIAL)

    serial_read, serial_conf = result.serial_read, result.serial_conf
    if serial_read is None and serial is not None:          # счётчик не найден — номер не читался
        p = models.serial_recognizer.recognize(serial["crop"])
        serial_read, serial_conf = p.text, p.confidence

    digit_results, digit_bboxes, reading_str = result.digit_results, result.digit_bboxes, result.reading_str
    if digit_results is None and meter is not None:          # номер не найден — цифры не читались
        d = read_meter_digits_for_config(models, meter["crop"], config)
        digit_results, digit_bboxes, reading_str = d.digit_results, d.digit_bboxes, d.reading_str

    crops = {}
    if serial is not None and (b := _jpeg(serial["crop"])):
        crops["serial"] = b
    if meter is not None and digit_bboxes:
        for pos, crop in enumerate(digit_crops_by_position(meter["crop"], digit_bboxes)):
            if b := _jpeg(crop):
                crops[f"digit{pos}"] = b

    return Recognition(
        serial_read=serial_read or "",
        serial_conf=None if serial_conf is None else round(float(serial_conf), 4),
        reading_str=reading_str or "",
        digits_ok=[bool(r.get("ok", True)) for r in digit_results or []],
        crops=crops,
    )


def safe_what_models_saw(result: PhotoResult, models, config: PipelineConfig,
                         log: logging.Logger) -> Optional[Recognition]:
    """То же; ошибка здесь не мешает прогону — окно покажет то, что есть в логе."""
    try:
        return what_models_saw(result, models, config)
    except Exception:                                        # noqa: BLE001
        log.exception("  Не удалось сохранить для окна, что прочитали модели")
        return None
