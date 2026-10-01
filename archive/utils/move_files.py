import shutil
from pathlib import Path

# ================= НАСТРОЙКИ =================
# Укажите здесь реальные пути к вашим папкам.
# Для Windows используйте префикс r перед строкой, например: r"C:\МоиДокументы\Папка1"
FOLDER_1 = Path(r"C:\AD\gas-meter-reader\database\meter_ocr_data\digits_detect\images_bad")  # Откуда берем файлы для проверки
FOLDER_2 = Path(r"C:\AD\gas-meter-reader\database\meter_ocr_data\digits_detect\dataset\images")  # С чем сравниваем (эталон)
FOLDER_3 = Path(r"C:\AD\gas-meter-reader\database\meter_ocr_data\digits_detect\images_unique")  # Куда переносим уникальные файлы
# =============================================

def main():
    # 1. Создаем Папку 3, если она еще не существует
    FOLDER_3.mkdir(parents=True, exist_ok=True)
    print(f"Папка назначения проверена/создана: {FOLDER_3}")

    # 2. Собираем все имена файлов (БЕЗ расширений) из Папки 2
    # Используем множество (set) для мгновенного поиска и .lower() чтобы игнорировать регистр (File.txt == file.pdf)
    folder2_names = set()
    if FOLDER_2.exists():
        for item in FOLDER_2.iterdir():
            if item.is_file():
                # .stem возвращает имя файла без расширения
                folder2_names.add(item.stem.lower())
    else:
        print(f"⚠️ Внимание: Папка 2 ({FOLDER_2}) не найдена! Сравнение будет со пустым списком.")

    print(f"Найдено уникальных имен (без учета расширения) в Папке 2: {len(folder2_names)}")

    # 3. Проходим по всем файлам в Папке 1
    moved_count = 0
    skipped_count = 0

    for item in FOLDER_1.iterdir():
        if item.is_file():
            file_stem = item.stem.lower()
            
            # Если имени файла нет в Папке 2, значит он уникальный
            if file_stem not in folder2_names:
                destination = FOLDER_3 / item.name
                
                # Дополнительная защита: если такой файл уже есть в Папке 3, не перезаписываем его
                if destination.exists():
                    print(f"⏭️ Пропущен: {item.name} (уже существует в Папке 3)")
                    skipped_count += 1
                    continue
                
                # Переносим файл
                shutil.move(str(item), str(destination))
                print(f"✅ Перенесен: {item.name}")
                moved_count += 1
            else:
                # Если имя совпадает (даже с другим расширением), ничего не делаем
                pass

    print("\n" + "="*40)
    print(f"🏁 Готово!")
    print(f"Перенесено файлов: {moved_count}")
    print(f"Пропущено (уже есть в Папке 3): {skipped_count}")

if __name__ == "__main__":
    main()