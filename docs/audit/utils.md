# Аудит `utils/` и `meter_ocr/utils/` — Фаза 0

Статус: **13/13 файлов проверены** (прочитаны целиком, проверены импорты и
references по всему репозиторию через `grep -rn`, проверено наличие
CLI/argparse, проверена git history).

Примечание по git history: в репозитории **один коммит** (`Initial commit`),
поэтому пункт "проверить git history" из алгоритма мастер-спеки в данном
случае не даёт дополнительной информации — история отсутствует, а не просто
не изучена. Если в будущем появится больше коммитов, эту графу нужно будет
перепроверить.

Проверка references: ни один из 13 файлов не импортируется из `reader.py`
или `program2.py`, и ни один не импортирует другой файл из этого списка
(кроме исключения, отмеченного отдельно ниже) — то есть черновая гипотеза
"standalone-скрипт" подтверждена для всех 13, это не предположение.

## Расхождения с черновой гипотезой ТЗ (раздел 3), найденные при реальном чтении

1. **`utils/excel_label_tool.py` — НЕ инструмент ручной разметки.**
   Черновая таблица предполагала, что он "пересекается по смыслу" с
   `label_tool.py` и `label_yolo_tool.py`. При чтении содержимого выяснилось:
   он не открывает GUI и не принимает ввод пользователя — это скрипт
   dataset prep, который **автоматически** генерирует `.txt`-лейблы из
   значений столбца Excel-таблицы (`format_to_5_digits`) и раскладывает их
   по файлам, сопоставленным с изображениями по имени. Пересечение с двумя
   другими инструментами не смысловое, а на уровне выходного формата
   (`labels/<name>.txt`), не более. **Решение изменено**: не MERGE,
   отдельный инструмент dataset prep.
2. **`utils/rename.py` — деструктивный скрипт без DRY_RUN, не отмечено в
   черновой ТЗ.** Скрипт переименовывает файлы `os.rename()` в цикле без
   возможности отмены и без флага dry-run (в отличие от `del_lb.py`, где
   защита хотя бы есть, пусть и с опасным дефолтом). Понижаю его в разряд
   "ARCHIVE + пометить как destructive" наравне с `del_lb.py`, а не просто
   "ARCHIVE (хардкод пути)".
3. **`utils/prepare_labeling_dataset.py` тоже деструктивный** — черновая
   ТЗ вообще не отмечала это (таблица §3 в исходном документе не включала
   этот файл в "проблема"). Он удаляет файлы (`file_path.unlink()`), но
   запрашивает подтверждение `y/n`. Итог: в `utils/` реально **три**
   скрипта, которые удаляют или переименовывают файлы без единого
   согласованного механизма защиты (`del_lb.py` — DRY_RUN-флаг с опасным
   дефолтом `False`; `prepare_labeling_dataset.py` — интерактивный confirm;
   `rename.py` — вообще без защиты). Стоит унифицировать паттерн
   (`--dry-run`/`--yes`) при переносе в `tools/diagnostics`, а не решать
   для каждого отдельно.

## Итоговая таблица (13/13)

| # | Путь | Реальное назначение (проверено чтением) | Riski | References/CLI | Финальное решение |
|---|---|---|---|---|---|
| 1 | `utils/check_excel_lb.py` | Фильтрует Excel-таблицу по двум условиям (есть фото + есть текущие показания), сохраняет CSV | Хардкод путей как module-level константы (`EXCEL_FILE_PATH` и т.п.), но пути относительные, не Windows-абсолютные | Нет references, есть `if __name__=="__main__"`, нет argparse | MOVE → `ml/tasks/*/dataset` (или `infrastructure/importers`), константы → CLI-аргументы |
| 2 | `utils/compute_mean_std.py` | Считает mean/std по датасету для нормализации CNN | Нет — уже параметризован | argparse есть, лучший образец в `utils/` (подтверждено) | KEEP, MOVE → `ml/engines/common` без изменений логики |
| 3 | `utils/convert_to_sequence.py` | Конвертирует YOLO-лейблы (bbox+class) в строку цифр по X-координате | Хардкод пути `database/meter_ocr_data/gas_meter1/labels` (module-level, относительный) | Нет references, нет `__main__` guard (выполняется как скрипт целиком на импорт — риск при случайном `import`) | MOVE в dataset pipeline, обернуть в `if __name__ == "__main__":`, путь → CLI-аргумент |
| 4 | `utils/del_lb.py` | Удаляет изображения/лейблы без пары (очистка датасета) | **Деструктивный**: `os.remove()`, `DRY_RUN=False` по умолчанию (подтверждено чтением) | Нет references, есть `__main__` guard | ARCHIVE → `tools/diagnostics`, сменить дефолт на `DRY_RUN=True`, добавить `--yes`-флаг для реального удаления |
| 5 | `utils/excel_label_tool.py` | **Пересмотрено** — авто-генерация лейблов из Excel (не GUI, не разметка руками) | Хардкод путей, суффикс `SUFFIX="_set1"` вручную в коде | Нет references | MOVE → `ml/tasks/serial_recognition/dataset` (dataset prep, НЕ merge с label_tool/label_yolo_tool) |
| 6 | `utils/label_tool.py` | Tkinter-клавиатурный разметчик серийных номеров (CRNN), ввод цифр нумпад-раскладкой | Нет хардкода путей (использует `filedialog`) | Нет references, GUI, ручной запуск | MOVE → `ml/tasks/serial_recognition/crnn/dataset` |
| 7 | `utils/label_yolo_tool.py` | Крупный (1174 строк) полуавтомат разметки bbox для 3 классов (gas_meter/serial_id/marker_id) с EasyOCR-автозаполнением | Самый крупный файл в utils/, вероятно канонический инструмент, но пересекается по функции с `models/yolo_all_detect/labeler_yolo.py` (992 строки) — см. находку 3bis.8 из ТЗ | Нет references к другим файлам utils/ | MOVE → `ml/tasks/digit_detection/yolo/dataset`; решение "нужны ли оба Tkinter-разметчика" — в Фазу 6 (Cleanup), не сейчас |
| 8 | `utils/move_files.py` | Сравнивает две папки с изображениями, переносит уникальные в третью | Хардкод `C:\AD\gas-meter-reader\...` (подтверждено, 3 абсолютных пути) — физически не запустится вне машины автора | Нет references | ARCHIVE как historical tool, не переносить как рабочий (путь физически не существует на других машинах) |
| 9 | `utils/prepare_labeling_dataset.py` | Удаляет файлы-дубликаты с одинаковым именем, но разным расширением (`.jpeg`/`.jpg`/`.JPG`), оставляя один по приоритету | **Деструктивный** (уточнение): вызывает `file_path.unlink()`, но, в отличие от `rename.py`, запрашивает интерактивное подтверждение `y/n` перед стартом — частичная защита, не DRY_RUN | Нет references, `__main__` guard есть | MOVE в dataset pipeline (`tools/diagnostics` или dataset prep), заменить интерактивный confirm на явный `--yes`/`--dry-run` флаг для согласованности с `del_lb.py` |
| 10 | `utils/rename.py` | Массово переименовывает файлы в папке в `file_N.ext` | **Деструктивный** (подтверждено): `os.rename()` без dry-run вообще, хардкод `database/ttt` | Нет references | ARCHIVE как historical/destructive tool (наравне с del_lb.py, но без даже DRY_RUN-заглушки) |
| 11 | `utils/xlx_to_csv.py` | Конвертация xlsx/xls → CSV | Хардкод `D:\РОЗА ФОТО МАЙ 2026...` (подтверждено, абсолютный Windows-путь) | Нет references, 22 строки | MOVE → `infrastructure/importers`, путь → CLI-аргумент/stdin |
| 12 | `meter_ocr/utils/compare_yolo_vs_cnn.py` | Интерактивное сравнение 3 моделей (YOLO/CNN/CRNN) на одном фото | Хардкодит пути к **другим** версиям весов, чем прод (`meter_ocr/yolo/runs/yolov8n_gas_meter_digits_v1` vs прод `meter_ocr/runs/yolo/digits_detect_v4`) — потенциально сравнивает не с той моделью, что реально в проде | Нет references, `__main__` guard | MOVE → `tools/diagnostics`, синхронизировать пути весов с `PipelineConfig` (иначе инструмент вводит в заблуждение) |
| 13 | `meter_ocr/utils/crop_save.py` | Сохраняет кропы цифр в файлы для последующей разметки/анализа | `MODEL_PATH` хардкод, но путь совпадает с прод-моделью цифр (`digits_detect_v4`) — в отличие от файла выше, здесь путь актуальный | Нет references | MOVE → `ml/tasks/digit_detection/yolo/dataset`, параметризовать путь |
| 14 | `meter_ocr/utils/crops_parameters.py` | Считает статистику по размерам кропов (min/max/avg ширина-высота) | Хардкод `C:\AD\GSM\database\...` (подтверждено, абсолютный путь другой машины/пользователя, отличается от `C:\AD\gas-meter-reader\...` в `move_files.py` — возможно другой автор/машина) | Нет references | MOVE → `tools/diagnostics`, путь → CLI-аргумент |

## Открытый вопрос для владельца (не решать в рамках миграции)

`meter_ocr/utils/compare_yolo_vs_cnn.py` (находка #12) сравнивает модели по
путям, которые **не совпадают** с продовыми путями в `PipelineConfig`
(`meter_detect_model`, `digit_detect_model` и т.д.). Это значит, что
человек, отлаживающий качество через этот инструмент, может смотреть не на
ту модель, которая реально работает в проде. Зафиксировано в
`docs/MIGRATION_STATUS.md` как баг, не чинится по ходу миграции.

---

# Аудит `models/` (раздел 3bis ТЗ)

Добавлено 2026-09-29 перед Фазой 3. Все 8 находок раздела 3bis
перепроверены по коду на коммите `edd0762` — подтверждаются. Решения здесь
не принимаются, только фиксируется, в какой фазе их разбирать.

| # | Где | Находка (проверено) | Фаза |
|---|---|---|---|
| 1 | `models/cnn/config_cnn.py:52`, `models/crnn/infer_crnn.py:52,121` | Пороги уверенности standalone-инструментов `0.8`, в проде (`PipelineConfig`) `digit_conf_thresh=0.6`, `serial_conf_thresh=0.6`. Отладка модели через CLI судит по чужому порогу. | 3 (свести к одному источнику; значения прода не менять — правило 4) |
| 2 | `models/crnn/infer_crnn.py:53,86` | `expected_length=5` для серийников, хотя `config_crnn.py:23-24` объявляет длину 4–10. Похоже на копию из логики показаний (5 цифр). | 3 |
| 3 | `models/cnn/model_cnn.py:87` vs `models/crnn/model_crnn.py:78` | CNN грузит чекпоинт с `weights_only=True`, CRNN — без. Плюс разные ключи чекпоинта. | 4 |
| 4 | `models/cnn/extract_digit_crops.py` | Раскладывает кропы в `train/`/`val/`, а `dataset_cnn.py` ждёт плоскую структуру, которую делает `build_dataset_cnn.py`. Кандидат в ARCHIVE, нужно подтверждение владельца, что не используется. | 6 |
| 5 | `models/yolo_all_detect/train_val_split.py:32` | `random.shuffle` без seed — сплит детектора цифр невоспроизводим. Исправить до следующего переобучения. | 4 |
| 6 | `models/yolo_all_detect/train_val_split.py:7-9`, `models/cnn/build_dataset_cnn.py:66-72` | Абсолютные пути `C:\AD\gas-meter-reader\...` (старое расположение проекта). В `models/` это ровно два файла. | 4 |
| 7 | `models/crnn/config_crnn.py:15-16` | Закомментированный альтернативный конфиг `gas_meter_gold` (IMG_W/IMG_H и т.д.) — след эксперимента, сбивает при чтении. | 6 |
| 8 | `models/yolo_all_detect/labeler_yolo.py` (992) + `utils/label_yolo_tool.py` (1174) | Два Tkinter-разметчика bbox, 2166 строк с пересекающейся ролью. | 6 |
