"""
src/gmr/domain/config.py — PipelineConfig.

Значения по умолчанию — это настройки обычного прогона (`python gmr.py process
<папка месяца>`, `python reader.py <папка месяца>`): других «настроек прогона»
нет (default_run_config в reader.py убран 2026-10-01). Пути к данным задаёт
только папка месяца (month_dir); переменные окружения GMR_INPUT_DIR,
GMR_OUTPUT_DIR, GMR_TABLE_PATH и путь к таблице убраны вместе со старым
режимом (этап 2.3b).

Внимание: код в run_pipeline() (reader.py) навешивает на экземпляр
PipelineConfig дополнительные динамические атрибуты — `_log_rows_cache`,
`_processed_accounts_cache`, `_processed_hashes_cache` (`_log_filenames_cache`
удалён в Фазе 7: после варианта Г он ни на что не влиял) — которые НЕ являются полями
датакласса (читаются через getattr(..., default) в process_photo). Это
осознанно оставлено как есть в Фазе 2a; полями датакласса их не делаем, чтобы
не расширять публичный контракт конфига без явного решения по этому вопросу.
"""
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # Пути к моделям
    meter_detect_model:       str   = "meter_detect/runs/detect/gas_meter_all_classes_s_v1/weights/best.pt"
    digit_detect_model:       str   = "meter_ocr/runs/yolo/digits_detect_v4/weights/best.pt"
    digit_ocr_model:          str   = "meter_ocr/runs/cnn/runs/v3_platinum/best.pth"
    serial_ocr_model:         str   = "serial_id_ocr/runs/crnn/2026-06-05_01-09/best.pt"

    # Папка месяца (этап 2.2b): фото из <месяц>/фото, результат в
    # <месяц>/результат, абоненты, показания и лог — в базе <месяц>/gmr.sqlite,
    # в конце — выгрузка показания.xlsx (src/gmr/storage/month.py).
    month_dir:                str   = ""
    # <месяц>/фото и <месяц>/результат — заполняет run_pipeline (reader.py)
    input_dir:                str   = ""
    output_base_dir:          str   = ""

    # Имена столбцов в таблице
    col_serial:               str   = "Номер счетчика"
    col_account_id:           str   = "Лицевой счет"
    col_last_reading:         str   = "Последние показания"
    col_new_reading:          str   = "Текущие показания"
    # столбцы реестра компании, которые заполняет выгрузка показания.xlsx (2026-10-01)
    col_date:                 str   = "Дата"
    col_difference:           str   = "Разница"

    # Пороги
    meter_conf_thresh:        float = 0.7
    digit_detect_conf_thresh: float = 0.7
    serial_conf_thresh:       float = 0.6
    digit_conf_thresh:        float = 0.6
    expected_digits:          int   = 5
    delta_threshold:          float = 10_000.0

    # Прочее
    move_photos:              bool  = False  # True=перемещать, False=копировать

    # Заглушки для цифр.
    # missing_digit_placeholder — для восстановленной (пропущенной детектором) позиции.
    #   Детектор иногда пропускает одну цифру посередине — её позиция вычисляется
    #   геометрически, а значение подставляется как placeholder.
    #   '5' — нейтральное среднее для цифры около середины шкалы.
    missing_digit_placeholder:  str = "5"

    # forgiven_digit_placeholder — для прощённых позиций (ignore_last_digits).
    #   Последние цифры часто не имеют веса на счётчике — при низком конфидансе
    #   подставляем '0' (минимальное влияние на итоговое число).
    forgiven_digit_placeholder: str = "0"

    # Режим мягкого чтения: игнорировать ошибки на последних N цифрах счётчика.
    # Последние цифры (десятые/сотые доли) часто не имеют веса на счётчике —
    # при низком конфидансе туда подставляется forgiven_digit_placeholder.
    #   0 — выключено (строгий режим)
    #   1 — игнорируем последнюю цифру
    #   2 — игнорируем последние две  ← рекомендуется
    #   3 — последние три
    #   5 — все цифры (фактически всегда успех, осторожно)
    # Несовместимо с debug_digits по смыслу: при ignore_last_digits > 0
    # ошибки на хвостовых позициях не попадают в digits_error и не будут продебажены.
    ignore_last_digits:         int = 2

    # Дебаг CNN: сохранять кропы цифр и meta.json при ошибке распознавания
    debug_digits:              bool = False

    # Отрисовка боксов на итоговом фото (для визуальной проверки детекций).
    # gas_meter    — зелёный
    # serial_number — синий
    # digit         — оранжевый
    # Подписи классов не рисуются.
    # До 2026-10-01 здесь было True, но обычный прогон (`python reader.py`,
    # `gmr.py process`) всегда переопределял на False — теперь значение одно.
    draw_boxes:                bool = False
