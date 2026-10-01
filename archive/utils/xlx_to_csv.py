import os
import pandas as pd

# 👇 МЕНЯЙ ПУТЬ ЗДЕСЬ 👇
file_path = r"D:\РОЗА ФОТО МАЙ 2026 новый сохр.xls"  # или "таблица.csv"

# Проверяем расширение и делаем обратное
name, ext = os.path.splitext(file_path)

if ext.lower() == '.csv':
    # Если дали CSV -> делаем Excel
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df.to_excel(name + '.xlsx', index=False)
    print(f"✅ Готово: создан {name}.xlsx")

elif ext.lower() in ['.xls', '.xlsx']:
    # Если дали Excel -> делаем CSV
    df = pd.read_excel(file_path)
    df.to_csv(name + '.csv', index=False, encoding='utf-8-sig')
    print(f"✅ Готово: создан {name}.csv")

else:
    print("❌ Файл должен быть .csv, .xls или .xlsx")