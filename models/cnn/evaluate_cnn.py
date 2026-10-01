"""
Полный анализ обученной CNN-модели классификации цифр.
ИСПОЛЬЗОВАНИЕ:
python evaluate_cnn.py --run_dir runs/cnn

ЧТО СОЗДАЁТСЯ в run_dir/eval/:
metrics.json               — все метрики (overall acc, per-class acc)
predictions_test.txt       — таблица pred vs gt для всех примеров
plot_training_history.png  — графики train/val loss и accuracy
plot_confusion_matrix.png  — матрица ошибок 10x10
errors/                    — папка с картинками неправильно распознанных кропов
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .config_cnn import DEVICE, NUM_CLASSES, CROPS_DIR, BATCH_SIZE
    from .dataset_cnn import make_loaders
    from .model_cnn import DigitCNN
except ImportError:  # запуск как отдельный скрипт (python models/cnn/evaluate_cnn.py)
    from config_cnn import DEVICE, NUM_CLASSES, CROPS_DIR, BATCH_SIZE
    from dataset_cnn import make_loaders
    from model_cnn import DigitCNN


# ─── Аргументы ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Оценка CNN классификатора цифр")
    p.add_argument("--run_dir", type=str, required=True, help="Папка с результатами обучения (напр. runs/cnn)")
    p.add_argument("--crops_dir", type=str, default=None, help="Папка с кропами (по умолчанию из config_cnn)")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Размер батча для оценки")
    p.add_argument("--max_errors", type=int, default=50, help="Максимум картинок ошибок для сохранения")
    return p.parse_args()


# ─── Сбор предсказаний ───────────────────────────────────────────────────────
@torch.no_grad()
def collect_predictions(model, test_loader, device):
    model.eval()
    results = []
    dataset = test_loader.dataset
    all_paths = [str(p) for p, _ in dataset.samples]
    global_idx = 0

    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu().numpy()
        labels_np = labels.numpy()
        batch_size = len(imgs)

        for i, (pred, true_label) in enumerate(zip(preds, labels_np)):
            results.append((int(pred), int(true_label), all_paths[global_idx + i]))

        global_idx += batch_size

    return results


# ─── Метрики и Confusion Matrix ──────────────────────────────────────────────
def calculate_metrics(results: list[tuple]) -> tuple[float, np.ndarray, dict]:
    total = len(results)
    correct = sum(1 for p, t, _ in results if p == t)
    overall_acc = correct / total if total > 0 else 0.0
    
    # Инициализация Confusion Matrix 10x10
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for p, t, _ in results:
        cm[t, p] += 1
        
    # Accuracy по каждому классу
    class_acc = {}
    for i in range(NUM_CLASSES):
        total_i = cm[i, :].sum()
        correct_i = cm[i, i]
        class_acc[str(i)] = round(correct_i / total_i, 4) if total_i > 0 else 0.0
        
    return overall_acc, cm, class_acc


# ─── Визуализация: Confusion Matrix ──────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels([str(i) for i in range(NUM_CLASSES)])
    ax.set_yticklabels([str(i) for i in range(NUM_CLASSES)])
    
    ax.set_xlabel("Предсказано (Predicted)", fontsize=12)
    ax.set_ylabel("Истина (Ground Truth)", fontsize=12)
    ax.set_title("Confusion Matrix (Test Set)", fontsize=14, fontweight="bold")
    
    # Добавляем текстовые значения в ячейки
    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color=color, fontsize=10)
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"   ✅ Confusion Matrix → {out_dir}/plot_confusion_matrix.png")


# ─── Визуализация: История обучения ──────────────────────────────────────────
def plot_training_history(run_dir: str, out_dir: str) -> None:
    hist_path = os.path.join(run_dir, "history.json")
    if not os.path.exists(hist_path):
        print("   ⚠️  history.json не найден, графики обучения пропущены.")
        return
        
    with open(hist_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    
    # Фильтруем только эпохи (исключаем запись "final_test")
    epochs_data = [h for h in history if isinstance(h.get("epoch"), int)]
    if not epochs_data:
        return
        
    epochs = [h["epoch"] for h in epochs_data]
    train_acc = [h["train_acc"] for h in epochs_data]
    val_acc = [h["val_acc"] for h in epochs_data]
    train_loss = [h["train_loss"] for h in epochs_data]
    val_loss = [h["val_loss"] for h in epochs_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    best_val_acc = max(val_acc) if val_acc else 0.0
    
    ax1.plot(epochs, train_acc, "--", alpha=0.6, label="Train")
    ax1.plot(epochs, val_acc, linewidth=2, label=f"Val (best={best_val_acc:.4f})")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(alpha=0.3)
    
    ax2.plot(epochs, train_loss, "--", alpha=0.6, label="Train")
    ax2.plot(epochs, val_loss, linewidth=2, label="Val")
    ax2.set_title("Loss"); ax2.legend(); ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_training_history.png"), dpi=150)
    plt.close()
    print(f"   ✅ Графики истории → {out_dir}/plot_training_history.png")


# ─── Сохранение ошибок ───────────────────────────────────────────────────────
def save_error_images(results: list[tuple], out_dir: str, max_errors: int = 50) -> None:
    error_dir = os.path.join(out_dir, "errors")
    os.makedirs(error_dir, exist_ok=True)
    
    errors = [(p, t, path) for p, t, path in results if p != t]
    print(f"   Найдено ошибок: {len(errors)} из {len(results)}")
    
    saved_count = 0
    for pred, true, img_path in errors[:max_errors]:
        try:
            # Открываем и конвертируем в RGB для надежного отображения
            img = Image.open(img_path).convert("RGB")
            
            # Масштабируем до фиксированного размера для удобного просмотра
            # (реальные кропы разного размера, это не отражает исходный размер файла)
            img = img.resize((64, 128), Image.Resampling.LANCZOS)
            
            # Создаем холст с местом для подписи
            canvas = Image.new("RGB", (64, 148), color=(30, 30, 30))
            canvas.paste(img, (0, 0))
            draw = ImageDraw.Draw(canvas)
            
            # Пытаемся использовать стандартный шрифт, иначе fallback
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except IOError:
                font = ImageFont.load_default()
                
            draw.text((4, 130), f"Pred: {pred} | True: {true}", fill=(255, 80, 80), font=font)
            
            safe_name = os.path.basename(img_path).replace(" ", "_").replace(":", "_")
            fname = f"pred_{pred}_true_{true}_{safe_name}"
            canvas.save(os.path.join(error_dir, fname))
            saved_count += 1
        except Exception as e:
            print(f"   ⚠️  Ошибка при обработке {img_path}: {e}")
            
    print(f"   ✅ Сохранено картинок ошибок: {saved_count} → {error_dir}/")


# ─── Текстовый отчет ─────────────────────────────────────────────────────────
def save_predictions_txt(results: list[tuple], out_dir: str) -> None:
    path = os.path.join(out_dir, "predictions_test.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Всего примеров: {len(results)}\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Pred':>6}  {'True':>6}  {'OK':>4}  Файл\n")
        f.write("-" * 40 + "\n")
        
        correct_count = 0
        for pred, true, img_path in results:
            ok = "✓" if pred == true else "✗"
            if pred == true:
                correct_count += 1
            f.write(f"{pred:>6}  {true:>6}  {ok:>4}  {os.path.basename(img_path)}\n")
            
        f.write("-" * 40 + "\n")
        f.write(f"Итого верно: {correct_count}/{len(results)} ({correct_count/len(results):.4f})\n")
    print(f"   ✅ Текстовый отчет → {out_dir}/predictions_test.txt")


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    device = DEVICE
    out_dir = os.path.join(args.run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    
    crops_dir = Path(args.crops_dir) if args.crops_dir else Path(CROPS_DIR)
    
    print(f"🖥️  Устройство  : {device}")
    print(f"📁  Run dir     : {args.run_dir}")
    print(f"📂  Crops dir   : {crops_dir.resolve()}")
    print(f"📁  Output eval : {out_dir}")
    print("─" * 65)

    # 1. Загрузка модели
    ckpt_path = os.path.join(args.run_dir, "best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Не найден best.pth в {args.run_dir}. Сначала запустите train_cnn.py")
        
    print("🔄 Загрузка модели...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = DigitCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    saved_val_acc = ckpt.get("val_acc")
    val_acc_str = f"{saved_val_acc:.4f}" if isinstance(saved_val_acc, (int, float)) else "неизвестно"
    print(f"   ✅ Модель загружена (сохраненная val_acc={val_acc_str})")

    # 2. Загрузка данных (используем только test_loader)
    print("\n📂 Загрузка тестовых данных...")
    _, _, test_loader = make_loaders(crops_dir, batch_size=args.batch_size)

    # 3. Сбор предсказаний
    print("\n🔍 Сбор предсказаний на test-выборке...")
    results = collect_predictions(model, test_loader, device)

    # 4. Расчет метрик
    print("📊 Расчет метрик...")
    overall_acc, cm, class_acc = calculate_metrics(results)
    
    print(f"\n{'=' * 50}")
    print(f"🏆 Overall Test Accuracy : {overall_acc:.4f}")
    print("📊 Accuracy по классам:")
    for cls, acc in sorted(class_acc.items(), key=lambda x: int(x[0])):
        print(f"   Цифра {cls}: {acc:.4f}")
    print(f"{'=' * 50}\n")

    # 5. Сохранение результатов
    print("💾 Сохранение артефактов...")
    
    # Метрики в JSON
    metrics = {
        "overall_accuracy": round(overall_acc, 4),
        "per_class_accuracy": class_acc,
        "confusion_matrix": cm.tolist(),
        "total_samples": len(results)
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Метрики JSON → {out_dir}/metrics.json")

    # Текстовый отчет
    save_predictions_txt(results, out_dir)
    
    # Графики
    plot_training_history(args.run_dir, out_dir)
    plot_confusion_matrix(cm, out_dir)
    
    # Картинки ошибок
    print("\n🖼️  Сохранение картинок ошибок...")
    save_error_images(results, out_dir, args.max_errors)

    print(f"\n{'=' * 65}")
    print("  Оценка завершена успешно!")
    print(f"  Все результаты сохранены в: {os.path.abspath(out_dir)}")
    print(f"{'=' * 65}")
    print("\n➡️  Следующий шаг:")
    print("   python infer_cnn.py --checkpoint runs/cnn/best_model.pt --image path/to/crop.jpg")


if __name__ == "__main__":
    main()