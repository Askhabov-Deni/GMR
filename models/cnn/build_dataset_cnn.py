import os
import shutil
from pathlib import Path

def build_cnn_dataset(digits_dir: str, labels_dir: str, output_dir: str):
    digits_path = Path(digits_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)

    # 1. Создаем целевые папки для цифр от 0 до 9
    for i in range(10):
        (output_path / str(i)).mkdir(parents=True, exist_ok=True)
        print(f"Создана/проверена папка: {output_path / str(i)}")

    print("\n--- Начало обработки ---")
    
    # Счетчики для статистики
    success_count = 0
    missing_count = 0

    # 2. Проходим по всем txt файлам с лейблами
    for label_file in labels_path.glob("*.txt"):
        base_name = label_file.stem  # например, "1300000013__gas_meter_1"
        
        # Читаем значение счетчика из файла (например, "51603")
        with open(label_file, 'r', encoding='utf-8') as f:
            label_text = f.read().strip()

        # Оставляем только цифры (игнорируем точки, пробелы или другие символы)
        # Это гарантирует, что 1-я цифра в тексте всегда пойдет в digit_1 и т.д.
        clean_digits = [char for char in label_text if char.isdigit()]

        # 3. Проходим по каждой цифре в очищенном лейбле
        for index, char in enumerate(clean_digits):
            digit_index = index + 1  # Нумерация кропов начинается с 1
            
            # Ищем файл кропа с подходящим расширением (.jpg, .jpeg, .png)
            found_file = None
            for ext in ['.jpeg', '.jpg', '.png']:
                candidate = digits_path / f"{base_name}__digit_{digit_index}{ext}"
                if candidate.exists():
                    found_file = candidate
                    break
            
            if found_file:
                # Копируем файл в соответствующую папку цифры
                target_folder = output_path / char
                target_file = target_folder / found_file.name
                
                shutil.copy2(found_file, target_file)
                success_count += 1
            else:
                print(f"⚠️ ВНИМАНИЕ: Не найден кроп для {base_name}, цифра #{digit_index} (ожидалась '{char}')")
                missing_count += 1

    print("\n--- Обработка завершена ---")
    print(f"✅ Успешно скопировано изображений: {success_count}")
    print(f"❌ Не найдено изображений: {missing_count}")
    print(f"📁 Датасет сохранен в: {output_path.resolve()}")


if __name__ == "__main__":
    # === НАСТРОЙКИ ПУТЕЙ (измените под вашу структуру) ===
    
    # Папка, где лежат вырезанные кропы цифр
    DIGITS_FOLDER = r"C:\AD\gas-meter-reader\database\meter_ocr_data\digit_ocr\images\digit"
    
    # Папка, где лежат .txt файлы с полными показаниями (например, 51603)
    LABELS_FOLDER = r"C:\AD\gas-meter-reader\database\meter_ocr_data\gas_meter_gold\labels" 
    
    # Куда сохранить итоговый датасет для CNN
    OUTPUT_DATASET = r"C:\AD\gas-meter-reader\database\meter_ocr_data\cnn_dataset_gold"

    # Запуск функции
    build_cnn_dataset(DIGITS_FOLDER, LABELS_FOLDER, OUTPUT_DATASET)