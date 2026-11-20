"""Модели в окне: загрузка один раз и распознавание открытого фото."""
from pathlib import Path

from src.gmr.application import digit_crops_by_position, recognize_photo
from src.gmr.domain import PipelineConfig
from src.gmr.ml.loader import default_device, load_models

from .common import _BASE, log


def _abs_model(rel_path: str) -> str:
    """Резолвит путь к модели от директории program2.py."""
    p = Path(rel_path)
    return str(p) if p.is_absolute() else str(_BASE / rel_path)

class ModelBundle:
    """Все модели, загруженные один раз при старте."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        device = default_device()
        log.info(f"Загружаем модели на {device}...")
        self.models = load_models(config, device=device, resolve_path=_abs_model)
        log.info("Модели загружены ✓")

    def run_on_photo(self, photo_path: str) -> dict:
        """
        Запускает все модели на фото. Возвращает dict с результатами.
        Вызывается из фонового потока.
        """
        result = {
            "serial_text": None,
            "serial_conf": None,
            "reading_str": None,   # "56?25" — строка с ? на неуверенных позициях
            "digit_preds": None,   # list[dict] с conf по каждой позиции
            "digit_crops": None,   # list[ndarray|None] — кропы по позициям
            "serial_crop": None,   # ndarray кропа таблички с номером (serial)
            "error": None,
        }
        try:
            rec = recognize_photo(self.models, photo_path, self.config)
            if not rec.detections:
                result["error"] = "YOLO: ничего не найдено"
                return result

            # Серийник
            if rec.serial is not None:
                result["serial_crop"] = rec.serial["crop"]
                result["serial_text"] = rec.serial_prediction.text
                result["serial_conf"] = rec.serial_prediction.confidence

            # Цифры
            if rec.digits is not None:
                result["reading_str"]  = rec.digits.reading_str
                result["digit_preds"]  = rec.digits.digit_results

                # Сохраняем кропы цифр для разметки
                if rec.digits.digit_bboxes and rec.digits.digit_results:
                    result["digit_crops"] = digit_crops_by_position(
                        rec.meter["crop"], rec.digits.digit_bboxes
                    )

        except Exception as e:
            result["error"] = str(e)
            log.exception("Ошибка в run_on_photo")

        return result
