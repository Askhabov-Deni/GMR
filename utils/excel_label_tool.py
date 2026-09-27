import pandas as pd
from pathlib import Path

# ================= НАСТРОЙКИ =================
EXCEL_FILE_PATH = "database/5000gold/lb.xls"       # Путь к исходному файлу Excel
IMAGE_FOLDER_PATH = "database/5000gold/crops0_8/serial_number"       # Путь к папке с изображениями
LABELS_FOLDER_PATH = "database/5000gold/labels"      # Путь к папке, где будут созданы .txt файлы

COL_ACCOUNT = "Лицевой счет"
COL_READINGS = "Номер счетчика"

# СУФФИКС для слияния датасетов. 
# Оставьте пустым "", если суффикс не нужен.
# Укажите значение, например "_set2", чтобы файлы стали 1300000060__gas_meter_1_set2.jpg
SUFFIX = "_set1"  
# =============================================

def format_to_5_digits(reading):
    """
    Корректно обрабатывает числа из Excel (включая float типа 86.0),
    превращает их в целое число и дополняет ведущими нулями до 5 символов.
    """
    try:
        if pd.isna(reading):
            return None
            
        clean_int = int(float(reading))
        return str(clean_int).zfill(5)
        
    except (ValueError, TypeError):
        clean_str = "".join(filter(str.isdigit, str(reading)))
        if clean_str:
            return clean_str.zfill(5)
        return None

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

    # 3. Очистка названий столбцов
    df.columns = df.columns.str.strip()

    if COL_ACCOUNT not in df.columns or COL_READINGS not in df.columns:
        print(f"❌ Ошибка: В таблице не найдены столбцы '{COL_ACCOUNT}' или '{COL_READINGS}'.")
        return

    # 4. Создаем словарь соответствия: {лицевой_счет: отформатированные_показания}
    print("🔄 Обработка данных из таблицы...")
    account_to_reading = {}
    
    for _, row in df.iterrows():
        account = str(row[COL_ACCOUNT]).replace('.0', '').strip()
        formatted_reading = format_to_5_digits(row[COL_READINGS])
        
        if formatted_reading is not None:
            account_to_reading[account] = formatted_reading

    print(f"   Загружено корректных записей из таблицы: {len(account_to_reading)}")

    # 5. Создаем папку для лейблов, если её нет
    labels_dir = Path(LABELS_FOLDER_PATH)
    labels_dir.mkdir(exist_ok=True)
    print(f"📁 Папка для лейблов готова: {labels_dir.absolute()}")

    # 6. Обработка изображений, переименование и создание .txt файлов
    valid_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'}
    
    processed_count = 0
    skipped_no_account = 0
    renamed_count = 0

    print("🖼️ Обработка файлов...")
    for img_path in img_folder.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in valid_extensions:
            original_stem = img_path.stem 
            
            # Разделяем имя по "__" и берем первую часть (10-значный лицевой счет)
            parts = original_stem.split('__')
            account_from_name = parts[0].strip()
            
            # Проверяем, есть ли этот счет в нашей таблице
            if account_from_name in account_to_reading:
                reading_value = account_to_reading[account_from_name]
                
                # Логика добавления суффикса
                if SUFFIX and not original_stem.endswith(SUFFIX):
                    new_stem = original_stem + SUFFIX
                    new_img_name = new_stem + img_path.suffix
                    new_img_path = img_path.parent / new_img_name
                    
                    # Переименовываем изображение
                    img_path.rename(new_img_path)
                    renamed_count += 1
                    
                    # Для txt файла используем новое имя с суффиксом
                    txt_stem_to_use = new_stem
                else:
                    # Если суффикс не задан или уже есть, используем оригинальное имя
                    txt_stem_to_use = original_stem

                # Формируем путь к новому .txt файлу
                txt_filename = f"{txt_stem_to_use}.txt"
                txt_path = labels_dir / txt_filename
                
                # Записываем значение в файл
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(reading_value)
                    
                processed_count += 1
            else:
                skipped_no_account += 1

    # 7. Итоговый отчет
    print("\n" + "="*50)
    print("✅ РЕЗУЛЬТАТЫ")
    print("="*50)
    if SUFFIX:
        print(f"🔄 Переименовано изображений (добавлен суффикс '{SUFFIX}'): {renamed_count}")
    print(f"📄 Создано/обновлено файлов .txt в папке '{LABELS_FOLDER_PATH}': {processed_count}")
    print(f"⚠️ Пропущено изображений (лицевого счета нет в таблице): {skipped_no_account}")
    
    if SUFFIX:
        print(f"\n💡 Теперь вы можете безопасно скопировать папки '{IMAGE_FOLDER_PATH}' и '{LABELS_FOLDER_PATH}'")
        print("   в ваш основной датасет. Конфликтов имен не будет.")
    print("="*50)

if __name__ == "__main__":
    main()