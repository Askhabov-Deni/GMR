import os
from pathlib import Path

# Путь к папке с лейблами
labels_dir = Path("database/meter_ocr_data/gas_meter1/labels")

processed_count = 0
skipped_count = 0

for txt_file in labels_dir.glob("*.txt"):
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 1. Проверяем, не является ли файл уже готовой последовательностью из 5 цифр
    if len(lines) == 1 and lines[0].strip().isdigit() and len(lines[0].strip()) == 5:
        skipped_count += 1
        continue
    
    # 2. Парсим YOLO формат
    detections = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            class_id = parts[0]          # Цифра (0-9)
            x_center = float(parts[1])   # Координата центра по X (от 0.0 до 1.0)
            detections.append((x_center, class_id))
    
    if not detections:
        continue
        
    # 3. Сортируем detections по координате X (слева направо)
    detections.sort(key=lambda item: item[0])
    
    # 4. Склеиваем только class_id в одну строку
    result_string = "".join(item[1] for item in detections)
    
    # 5. Перезаписываем файл
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(result_string + "\n")
    
    processed_count += 1
    print(f"✅ {txt_file.name}: {result_string}")

print(f"\n🎉 Готово! Конвертировано файлов: {processed_count}, уже готовых/пропущено: {skipped_count}")