import os
from pathlib import Path

def remove_duplicate_extensions(folder_path, priority=['jpeg', 'jpg', 'JPG']):
    """
    Удаляет дубликаты файлов с одинаковым именем, но разными расширениями.
    Оставляет файл с наивысшим приоритетом из списка priority.
    
    Args:
        folder_path (str): путь к папке с фотографиями
        priority (list): порядок приоритета расширений (первый - самый приоритетный)
    """
    
    # Переключаемся в указанную папку
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Ошибка: Папка '{folder_path}' не существует")
        return
    
    # Собираем все файлы с нужными расширениями
    extensions = set(priority)
    files_dict = {}
    
    for ext in extensions:
        for file_path in folder.glob(f"*.{ext}"):
            name_without_ext = file_path.stem  # имя без расширения
            key = name_without_ext
            
            if key not in files_dict:
                files_dict[key] = []
            files_dict[key].append((ext, file_path))
    
    # Обрабатываем дубликаты
    removed_count = 0
    kept_count = 0
    
    for name, files in files_dict.items():
        if len(files) > 1:
            # Сортируем файлы по приоритету расширения
            files_sorted = sorted(files, key=lambda x: priority.index(x[0]) if x[0] in priority else len(priority))
            
            # Оставляем первый (с наивысшим приоритетом)
            keep_ext, keep_path = files_sorted[0]
            
            # Удаляем остальные
            for ext, file_path in files_sorted[1:]:
                try:
                    file_path.unlink()
                    print(f"Удалён: {file_path.name} (оставлен {keep_path.name})")
                    removed_count += 1
                except Exception as e:
                    print(f"Ошибка при удалении {file_path.name}: {e}")
            
            kept_count += 1
    
    print(f"\nГотово! Обработано {kept_count} групп дубликатов. Удалено {removed_count} файлов.")

def main():
    # Текущая папка или укажите свой путь
    folder_path ="database/all_photos"
    
    if not folder_path:
        folder_path = os.getcwd()
    
    print(f"\nРаботаем с папкой: {folder_path}")
    confirm = input("Продолжить? (y/n): ").strip().lower()
    
    if confirm == 'y':
        remove_duplicate_extensions(folder_path, priority=['jpeg', 'jpg', 'JPG'])
    else:
        print("Операция отменена")

if __name__ == "__main__":
    main()