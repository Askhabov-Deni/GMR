import pandas as pd
from pathlib import Path

# ================= НАСТРОЙКИ =================
EXCEL_FILE_PATH = "table.xls"             # Путь к исходному файлу Excel
IMAGE_FOLDER_PATH = "database/raw_photos2"             # Путь к папке с фотографиями
OUTPUT_FILE_PATH = "result_filtered.csv" # Имя нового файла (теперь CSV)

# Столбцы, которые нужно оставить в новой таблице
COLUMNS_TO_KEEP = [
    "Лицевой счет", 
    "Текущие показания", 
    "Последние показания значение", 
    "Номер счетчика"
]
# =============================================

def format_number(val):
    """Убирает .0 у целых чисел для красивого вывода"""
    try:
        num = float(val)
        return str(int(num)) if num.is_integer() else str(num)
    except (ValueError, TypeError):
        return str(val).strip()

def main():
    # 1. Проверка путей
    if not Path(EXCEL_FILE_PATH).exists():
        print(f"❌ Ошибка: Файл Excel не найден: {EXCEL_FILE_PATH}")
        return
    
    img_folder = Path(IMAGE_FOLDER_PATH)
    if not img_folder.exists() or not img_folder.is_dir():
        print(f"❌ Ошибка: Папка с изображениями не найдена: {IMAGE_FOLDER_PATH}")
        return

    # 2. Чтение Excel файла
    print("⏳ Чтение исходного Excel файла...")
    try:
        df = pd.read_excel(EXCEL_FILE_PATH, engine='xlrd')
    except Exception as e:
        print(f"❌ Ошибка при чтении Excel файла: {e}")
        return

    # 3. Очистка названий столбцов от лишних пробелов
    df.columns = df.columns.str.strip()

    # 4. Сбор лицевых счетов из папки с фотографиями
    print("📂 Сканирование папки с фотографиями...")
    valid_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG'}
    accounts_with_photos = set()
    
    for file_path in img_folder.iterdir():
        if file_path.is_file() and file_path.suffix in valid_extensions:
            account = file_path.stem.strip()
            accounts_with_photos.add(account)

    print(f"   Найдено уникальных лицевых счетов в папке: {len(accounts_with_photos)}")

    # 5. Приводим лицевые счета в таблице к такому же виду (строка, без .0)
    df["Лицевой счет"] = (
        df["Лицевой счет"]
        .astype(str)
        .str.replace(r'\.0$', '', regex=True)
        .str.strip()
    )

    # 6. Фильтрация таблицы по двум условиям:
    # Условие А: Лицевой счет есть в папке с фото
    mask_has_photo = df["Лицевой счет"].isin(accounts_with_photos)
    
    # Условие Б: В "Текущие показания" что-то записано (не пусто и не NaN)
    mask_has_readings = df["Текущие показания"].notna() & (df["Текущие показания"].astype(str).str.strip() != "")
    
    # Применяем оба условия (логическое И)
    filtered_df = df[mask_has_photo & mask_has_readings].copy()
    
    print(f"✅ Найдено совпадений (есть фото И есть показания): {len(filtered_df)}")

    if len(filtered_df) == 0:
        print("⚠️ Внимание: Не найдено ни одной строки, удовлетворяющей обоим условиям. Новый файл не создан.")
        return

    # 7. Оставляем только нужные столбцы
    existing_cols = [col for col in COLUMNS_TO_KEEP if col in filtered_df.columns]
    missing_cols = [col for col in COLUMNS_TO_KEEP if col not in filtered_df.columns]
    
    if missing_cols:
        print(f"⚠️ Предупреждение: В исходной таблице не найдены столбцы: {missing_cols}")
        print("   В новый файл будут добавлены только те столбцы, которые существуют.")

    final_df = filtered_df[existing_cols].copy()

    # 8. Форматируем числовые столбцы (убираем .0), если они есть
    for col in existing_cols:
        final_df[col] = final_df[col].apply(format_number)

    # 9. Сохранение в CSV файл
    print(f"💾 Сохранение результата в файл: {OUTPUT_FILE_PATH} ...")
    try:
        # encoding='utf-8-sig' нужен для корректного отображения кириллицы в Excel
        # sep=';' нужен, чтобы Excel в русской локали правильно разбивал данные по столбцам
        final_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig', sep=';')
        print("🎉 Готово! Новая таблица успешно создана в формате CSV.")
    except Exception as e:
        print(f"❌ Ошибка при сохранении файла: {e}")

if __name__ == "__main__":
    main()