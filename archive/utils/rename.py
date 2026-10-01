import os

# 👇 УКАЖИ ПУТЬ К ПАПКЕ ЗДЕСЬ (буква r перед строкой обязательна для Windows)
folder_path = "database/ttt"

counter = 1

for filename in os.listdir(folder_path):
    full_path = os.path.join(folder_path, filename)
    
    # Проверяем, что это файл, а не папка
    if os.path.isfile(full_path):
        # Получаем расширение (например, .txt, .jpg, .xlsx)
        name, ext = os.path.splitext(filename)
        
        # Формируем новое имя: file_1.jpg, file_2.xlsx и т.д.
        new_name = f"file_{counter}{ext}"
        new_path = os.path.join(folder_path, new_name)
        
        # Переименовываем
        os.rename(full_path, new_path)
        print(f"✅ {filename} -> {new_name}")
        
        counter += 1

print("🎉 Все файлы в папке переименованы!")