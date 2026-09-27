import os

# ================= НАСТРОЙКИ =================
# Укажите пути к вашим папкам (можно использовать относительные или абсолютные пути)
IMAGES_DIR = "database/5000gold/crops0_8/gas_meter"   # Папка с картинками
LABELS_DIR = "database/5000gold/labels"   # Папка с txt файлами

# Если True, скрипт только покажет, что будет удалено, но ничего не удалит.
# Когда убедитесь, что все верно, поменяйте на False.
DRY_RUN = False
# =============================================

def clean_dataset(img_dir, lbl_dir, dry_run):
    # Поддерживаемые расширения изображений
    valid_img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # 1. Собираем списки файлов
    all_img_files = os.listdir(img_dir)
    all_lbl_files = os.listdir(lbl_dir)

    # Фильтруем только нужные форматы
    img_files = [f for f in all_img_files if os.path.splitext(f)[1].lower() in valid_img_exts]
    lbl_files = [f for f in all_lbl_files if f.lower().endswith('.txt')]

    # 2. Получаем имена файлов БЕЗ расширений для сравнения
    img_names = {os.path.splitext(f)[0] for f in img_files}
    lbl_names = {os.path.splitext(f)[0] for f in lbl_files}

    # 3. Находим пересечение (те, у которых есть пара)
    valid_pairs = img_names.intersection(lbl_names)

    print(f"📊 Статистика до очистки:")
    print(f"   Изображений: {len(img_files)}")
    print(f"   Лейблов (.txt): {len(lbl_files)}")
    print(f"   Полных пар (будет сохранено): {len(valid_pairs)}\n")

    removed_imgs = 0
    removed_lbls = 0

    # 4. Удаляем изображения без лейблов
    for f in img_files:
        base_name = os.path.splitext(f)[0]
        if base_name not in valid_pairs:
            file_path = os.path.join(img_dir, f)
            if not dry_run:
                os.remove(file_path)
            print(f"🗑️ Удалено изображение (нет лейбла): {f}")
            removed_imgs += 1

    # 5. Удаляем лейблы без изображений
    for f in lbl_files:
        base_name = os.path.splitext(f)[0]
        if base_name not in valid_pairs:
            file_path = os.path.join(lbl_dir, f)
            if not dry_run:
                os.remove(file_path)
            print(f"🗑️ Удален лейбл (нет изображения): {f}")
            removed_lbls += 1

    print("\n✅ Очистка завершена!")
    print(f"   Удалено изображений: {removed_imgs}")
    print(f"   Удалено лейблов: {removed_lbls}")
    
    if dry_run:
        print("\n⚠️ ВНИМАНИЕ: Это был тестовый режим (DRY_RUN). Файлы НЕ были удалены.")
        print("Чтобы удалить файлы по-настоящему, измените в коде DRY_RUN = False")

if __name__ == "__main__":
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(LABELS_DIR):
        print("❌ Ошибка: Одна из указанных папок не существует. Проверьте пути.")
    else:
        clean_dataset(IMAGES_DIR, LABELS_DIR, DRY_RUN)