# GMR Migration — статус

Текущая фаза: 2a (Domain extraction) — код готов, ждёт проверки у владельца

## Фаза 0 — Baseline
- [x] golden tests написаны (раздел 2 ТЗ) — сколько из 16 кейсов готово: 16/16
      (`tests/test_golden_reader.py`, `tests/_fixtures.py`, `tests/conftest.py`,
      корневой `conftest.py`). Все проходят: `python3 -m pytest tests/ -v`.
      Мутационно проверено (искусственно сломана forgiven-логика в
      `_read_meter_digits` → тест 14 упал, значит тесты не фиктивные).
- [x] settings.json очищен от личных путей — удалён из git-индекса
      (`git rm --cached settings.json`), добавлен в `.gitignore`, рядом
      создан `settings.example.json` (пустые значения). Хардкод `D:\...` в
      `PipelineConfig.input_dir/output_base_dir/table_path` заменён на
      `os.environ.get("GMR_*", <безопасный относительный дефолт>)`,
      добавлен `.env.example`.
- [x] docs/audit/utils.md заполнен (13/13 файлов) — см. файл. Два
      расхождения с черновой гипотезой ТЗ зафиксированы там же
      (`excel_label_tool.py` — не GUI-разметчик; `rename.py` и
      `prepare_labeling_dataset.py` — деструктивные скрипты, которых не
      было в списке проблемных в черновике).
- [ ] **Не сделано**: физически перенести изменения в реальный git-репозиторий
      `github.com/Askhabov-Deni/GMR` — у агента (Claude в веб-чате) нет push
      credentials в этот репозиторий. Все изменения лежат как файлы/патч,
      которые должен применить и закоммитить владелец репозитория.
- [ ] **Не сделано**: прогон golden tests на "чистом venv" (упомянуто как
      критерий готовности Фазы 1, но частично относится и к Фазе 0) — тесты
      прогнаны в контейнере с вручную доустановленными `torch`, `ultralytics`,
      `albumentations` (их не было в окружении изначально; в проекте до сих
      пор нет `requirements.txt` — это Фаза 1). Список того, что реально
      пришлось поставить, чтобы `reader.py` вообще импортировался:
      `torch`, `torchvision`, `ultralytics`, `albumentations`, `pytest`
      (`cv2`, `pandas`, `numpy`, `PIL` уже были в окружении).

## Фаза 1 — Packaging
- [x] `models/`, `models/cnn/`, `models/crnn/`, `models/yolo_all_detect/` стали
      пакетами (`__init__.py`).
- [x] Голые импорты соседей (`from config_cnn import ...`, `from model_crnn import ...`)
      в 9 файлах `models/cnn/*` и `models/crnn/*` заменены на
      `try: from .x import ... / except ImportError: from x import ...`.
      Так работают оба режима: пакетный (`reader.py`) и запуск скрипта напрямую
      (`python models/crnn/train_crnn.py`) — второй проверен через `--help`.
      `models/yolo_all_detect/*` изменений не потребовал (внутренних импортов нет).
- [x] Два `sys.path.insert` и ставший ненужным `import sys` удалены из `reader.py`.
      `import reader` работает из чужого cwd при одном лишь PYTHONPATH на репозиторий.
- [x] `requirements.txt` (runtime), `requirements-dev.txt` (+pytest),
      `requirements-tools.txt` (easyocr, tqdm — только для отдельных утилит).
- [x] Golden tests 16/16 после изменений (песочница, Python 3.12).
- [ ] **Не проверено**: установка `requirements-dev.txt` в полностью чистый venv —
      в песочнице агента не хватило диска на torch (`No space left on device`).
      Нужно у владельца: новый venv → `python -m pip install -r requirements-dev.txt`
      → `python -m pytest tests/`.
- [ ] Корневой `conftest.py` пока оставлен: он нужен, чтобы локальный `tests/`
      не перекрывался одноимённым пакетом из site-packages (см. решения Фазы 0).
      Это про тесты, не про код приложения.

### Решения Фазы 1
- try/except ImportError вместо чисто относительных импортов: чисто относительные
  сломали бы запуск `python models/cnn/train_cnn.py` ("no known parent package").
  Побочный эффект: `except ImportError` может замаскировать реальную ошибку
  импорта внутри самого модуля — если увидишь странный ImportError про `config_*`,
  ищи причину в первой ветке.
- tqdm не в основных зависимостях: используется только в `utils/compute_mean_std.py`,
  а в окружении агента не установлен, версию проверить было нечем.
- Версии в `requirements.txt` — точные (`==`) те, на которых прошли тесты; у
  владельца стоят Python 3.14 и torch 2.14.0, albumentations был 1.4.18.

## Найденные баги (продолжение)
- 2026-09-28 — **albumentations 1.4.18 молча игнорирует `A.GaussNoise(std_range=...)`**
  (`models/cnn/dataset_cnn.py`, `_train_transform`): выдаёт UserWarning
  "Argument 'std_range' is not valid and will be ignored", а не ошибку. На 2.0.8
  параметр принимается. Проверено в песочнице на обеих версиях. У владельца в
  venv стоял именно 1.4.18. Следствие: если CNN-модель цифр обучалась на 1.4.x,
  шум в аугментации был не тот, что записан в коде. Какая версия использовалась
  при обучении текущих весов — неизвестно, вопрос к владельцу. Веса не трогаем.
  Golden tests этот код не покрывают (аугментации не вызываются).
- 2026-09-28 — venv владельца создан по пути `C:\AD\gas-meter-reader\.venv`, папка
  проекта потом переехала в `C:\AD\GSM\gas-meter-reader`, и `pip.exe` перестал
  запускаться ("Fatal error in launcher"). Тот же класс проблемы, что хардкод
  путей в `utils/`: путь зашивается при создании. Обход: `python -m pip`
  или пересоздать venv.

## Golden tests — расширение после реального прогона (19/19)
- 2026-09-28 — владелец прогнал `reader.py` на 1111 реальных фото (новая CNN
  цифр, новая архитектура) без падений — отчёт сохранён как
  `docs/audit/baseline_run_2026-09-28.txt`. Арифметика отчёта проверена
  (сумма Outcome = 1111, "Успешно"/"На проверку" пересчитаны).
- Сверка отчёта с `Outcome` в коде вскрыла дыру: enum содержит **9** значений
  (`PLUS, MINUS, REPEAT, NO_METER, NO_SERIAL, SERIAL_LOW_CONF,
  SERIAL_NOT_FOUND, DIGITS_ERROR, SUSPICIOUS`), а golden tests Фазы 0 (16
  кейсов из черновой таблицы ТЗ) покрывали только 7 — `NO_METER`/`NO_SERIAL`
  отсутствовали полностью. По реальному отчёту это 45+82=127 фото — 11.5%
  трафика, не мелочь.
- Добавлены кейсы 17-19 (`tests/test_golden_reader.py`), все три условия
  внутри `process_photo` (reader.py:736-756), которые дают эти два Outcome
  с разным `error_detail`:
  - 17: `meter_detector` вообще не вернул кропов ("YOLO returned no crops")
  - 18: кропы есть, но класса `gas_meter` среди них нет
  - 19: `gas_meter` есть, `serial_number` — нет
  19/19 golden tests зелёные. Теперь покрыты все 9 значений Outcome — это и
  есть полный контракт поведения, который обязана сохранить Фаза 2a.

## Фаза 2a — Domain extraction
- [x] `Outcome`, `OUTCOME_FOLDER`, `PhotoResult` перенесены в
      `src/gmr/domain/models.py`; `PipelineConfig` — в
      `src/gmr/domain/config.py`. Поля и дефолты не менялись (прямой перенос).
      Домен не импортирует `torch`/`cv2`/`ultralytics` — проверено
      (`python -c "import src.gmr.domain"` без установки ML-библиотек).
- [x] 4 policy-класса в `src/gmr/domain/policies.py`, каждый — прямая
      пересадка соответствующего блока `reader.py` без изменения значений:
      - `MissingDigitRecoveryPolicy` — восстановление пропущенной цифры
        (gap-эвристика, порог 1.6×, критическая позиция <2)
      - `DigitForgivenessPolicy` — прощение хвостовых позиций
        (`ignore_last_digits`)
      - `DeltaThresholdPolicy` — PLUS/MINUS/SUSPICIOUS по дельте
      - `DuplicatePolicy` — порядок дублей: лог → processed-in-run →
        pre-existing → рассинхронизация (порядок НЕ менялся, кейсы 5-8)
- [x] `reader.py` реэкспортирует `PipelineConfig`/`Outcome`/`OUTCOME_FOLDER`/
      `PhotoResult` из `src.gmr.domain` — `program2.py`, который импортирует
      их напрямую из `reader` (`from reader import PipelineConfig, PhotoResult,
      Outcome, ...`), продолжает работать без изменений. Проверено:
      `import program2` (после доустановки `python3-tk`, см. ниже).
- [x] `reader.py`: 1336 → 1176 строк (-160, ~12%) за счёт делегирования в
      `domain/`. Хранилище (CSV/лог) не тронуто — Фаза 2a изменений в
      `_load_table`/`_save_table`/`_load_log`/`_save_log` не делала.
- [x] Golden tests 19/19 после рефакторинга. Мутационно проверены 2 из 4
      policy: `DuplicatePolicy` (сломал первую проверку лога → упал
      test_case5) и `DeltaThresholdPolicy` (сломал ветку `last_reading is
      None` → упал test_case4) — значит policy реально вызываются, а не
      лежат неиспользуемым кодом рядом со старой логикой.

### Решения Фазы 2a
- `_read_meter_digits` и `process_photo` остались в `reader.py` (сигнатуры,
  имена — без изменений, их напрямую импортирует `program2.py` и вызывают
  golden tests) — они теперь ДЕЛЕГИРУЮТ решения в policy-объекты вместо
  инлайн-логики, но сами функции не переехали. Полный перенос вызовов моделей
  в domain/application слой — это Фаза 3 (ML Interfaces), не 2a.
- Дополнительные динамические атрибуты `PipelineConfig`
  (`_log_filenames_cache`, `_log_rows_cache`, `_processed_accounts_cache`,
  выставляемые в `run_pipeline`) НЕ стали полями датакласса — они как были,
  так и остаются "неофициальным" расширением конфига через `getattr(...,
  default)`. Явно не расширял контракт без отдельного решения по этому вопросу.
- `program2.py` требует системный Tcl/Tk (`python3-tk` через apt), не
  ставится через pip — в песочнице агента отсутствовал изначально, доставлен
  отдельно для проверки импорта. У владельца (Windows) обычно идёт в комплекте
  с python.org-инсталлятором, но если ставился Python через Microsoft Store —
  может понадобиться отдельная установка. Не блокирует Фазу 2a, но стоит
  иметь в виду для Фазы 4/CI, если появится автоматический прогон program2.py.

## Решения, принятые агентом (append-only, не переписывать задним числом)
- 2026-09-27 — golden tests кладём в `tests/` с `tests/__init__.py` и
  корневым `conftest.py`, который принудительно ставит корень репозитория
  первым в `sys.path` — потому что `albumentations`/`ultralytics-thop`
  устанавливают в site-packages свой собственный пакет `tests`, который
  иначе перекрывает локальную папку и ломает `from tests... import`.
- 2026-09-27 — `_read_meter_digits`/`process_photo` в golden tests вызываются
  напрямую с fake-объектами (не Mock из unittest.mock, а простые классы
  с нужными методами) вместо реальных `YOLOInferer`/`CRNNInferer`/`CNNInferer`
  — потому что эти функции и так принимают инференс-объекты как параметры
  (DI уже есть), реальные веса моделей в репозитории отсутствуют (см. .gitignore
  — `*.pt`/`*.pth` не коммитятся), и незачем гонять реальный форвард-пасс
  ради проверки бизнес-логики.
- 2026-09-27 — пути `PipelineConfig` вынесены через `os.environ.get(...)` с
  относительными дефолтами (`data/input`, `data/output`,
  `data/meters_table.csv`), а не через `python-dotenv` — чтобы не добавлять
  новую зависимость в Фазе 0 (это по духу относится к Фазе 1 — Packaging).
  `.env` не подгружается автоматически; переменные должны быть выставлены
  в окружении явно, либо пути передаются в `PipelineConfig(...)` напрямую.
- 2026-09-27 — классификация `utils/excel_label_tool.py` изменена относительно
  черновой таблицы ТЗ: это не разметочный GUI, а автогенерация лейблов из
  Excel — решение "MERGE с label_tool/label_yolo_tool" снято, вместо этого
  MOVE как отдельный dataset-prep скрипт.

## Найденные баги (не чинить в рамках миграции, только фиксировать)
- 2026-09-27 — `meter_ocr/utils/compare_yolo_vs_cnn.py` хардкодит пути к
  весам моделей (`meter_ocr/yolo/runs/yolov8n_gas_meter_digits_v1/...` и
  т.д.), которые **не совпадают** с продовыми путями в `PipelineConfig`
  (прод использует `meter_ocr/runs/yolo/digits_detect_v4/...`). Инструмент
  для ручной проверки качества модели сравнивает не ту версию, что реально
  работает в проде — источник ложных выводов при отладке. См.
  `docs/audit/utils.md`, находка #12.
- 2026-09-27 — `utils/rename.py` — деструктивный скрипт (`os.rename()` без
  возможности отмены), не имеет вообще никакой защиты (ни DRY_RUN, ни
  подтверждения). Черновая версия ТЗ квалифицировала его только по хардкоду
  пути, не по деструктивности.
- 2026-09-27 — `utils/prepare_labeling_dataset.py` тоже удаляет файлы
  (`Path.unlink()`) — не было отмечено в черновой ТЗ вообще. Защита слабее,
  чем у `del_lb.py` (интерактивный `y/n`, а не флаг, который можно
  контролировать программно/в CI).
- 2026-09-27 — три скрипта в `utils/` независимо реализуют разную защиту от
  случайного удаления/переименования файлов (DRY_RUN-флаг с опасным
  дефолтом / интерактивный confirm / вообще ничего) — нет единого паттерна.
  Стоит унифицировать при переносе в `tools/diagnostics` (не в рамках
  текущей фазы — фиксирую как наблюдение для Фазы 6).
