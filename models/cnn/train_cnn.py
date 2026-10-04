"""
Обучение CNN-классификатора цифр (0-9) и экспорт в TorchScript.

ИСПОЛЬЗОВАНИЕ:
Обучение с нуля (из папки проекта; датасет — database/datasets/digits_cnn):
    python models/cnn/train_cnn.py

Продолжение прерванного обучения (resume):
    python train_cnn.py --resume runs/cnn/best.pth

Дообучение (fine-tuning, сброс оптимизатора):
    python train_cnn.py --finetune runs/cnn/best.pth --lr 1e-4

РЕЗУЛЬТАТЫ в --output_dir (по умолчанию новая папка <OUTPUT_DIR>/<дата-время>,
при --resume — папка чекпоинта):
    run_info.json    — датасет (отпечаток), коммит кода, пакеты, деление (этап 5)
    split/*.txt      — какие файлы попали в train / val / test
    best.pth         — чекпоинт (веса модели + оптимизатор + метрики)
    best_model.pt    — TorchScript для продакшена (инференс)
    history.json     — метрики по эпохам + финальный тест
    training_curves.png — графики обучения
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .config_cnn import (
        BATCH_SIZE, CROPS_DIR, IMG_SIZE, IMG_WIDTH,
        EPOCHS, LABEL_SMOOTHING, LR,
        NUM_CLASSES, OUTPUT_DIR, PATIENCE, WEIGHT_DECAY, SEED
    )
    from .dataset_cnn import make_loaders
    from .model_cnn import DigitCNN
except ImportError:  # запуск как отдельный скрипт (python models/cnn/train_cnn.py)
    from config_cnn import (
        BATCH_SIZE, CROPS_DIR, IMG_SIZE, IMG_WIDTH,
        EPOCHS, LABEL_SMOOTHING, LR,
        NUM_CLASSES, OUTPUT_DIR, PATIENCE, WEIGHT_DECAY, SEED
    )
    from dataset_cnn import make_loaders
    from model_cnn import DigitCNN
try:
    from ..datasets import write_run_info, write_split_lists
except ImportError:  # запуск как отдельный скрипт: нужна папка проекта в sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from models.datasets import write_run_info, write_split_lists


# ─── Аргументы командной строки ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение CNN классификатора цифр")
    p.add_argument("--crops_dir", type=str, default=CROPS_DIR, help="Путь к папке с кропами")
    p.add_argument("--output_dir", type=str, default=None,
                   help=f"Папка результата (по умолчанию {OUTPUT_DIR}/<дата-время>; при --resume — папка чекпоинта)")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="Количество эпох")
    p.add_argument("--lr", type=float, default=LR, help="Learning rate")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Размер батча")
    p.add_argument("--seed", type=int, default=SEED, help="Random seed")
    p.add_argument("--resume", type=str, default=None, help="Путь к чекпоинту для продолжения обучения")
    p.add_argument("--finetune", type=str, default=None, help="Путь к чекпоинту для дообучения (сброс оптимизатора)")
    p.add_argument("--patience", type=int, default=PATIENCE, help="Patience для Early Stopping")
    return p.parse_args()


# ─── Циклы обучения и оценки ─────────────────────────────────────────────────
def train_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        logits = model(imgs)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, 
               criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


# ─── Визуализация ────────────────────────────────────────────────────────────
def plot_history(history: list[dict], output_dir: Path) -> None:
    epochs = [e["epoch"] for e in history if isinstance(e["epoch"], int)]
    if not epochs:
        return
        
    train_acc = [e["train_acc"] for e in history if isinstance(e["epoch"], int)]
    val_acc = [e["val_acc"] for e in history if isinstance(e["epoch"], int)]
    train_loss = [e["train_loss"] for e in history if isinstance(e["epoch"], int)]
    val_loss = [e["val_loss"] for e in history if isinstance(e["epoch"], int)]
    
    best_acc = max(val_acc) if val_acc else 0.0

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4))

    ax_acc.plot(epochs, train_acc, "--", alpha=0.6, label="Train")
    ax_acc.plot(epochs, val_acc, linewidth=2, label=f"Val (best={best_acc:.4f})")
    ax_acc.set_title("Accuracy"); ax_acc.legend(); ax_acc.grid(alpha=0.3)

    ax_loss.plot(epochs, train_loss, "--", alpha=0.6, label="Train")
    ax_loss.plot(epochs, val_loss, linewidth=2, label="Val")
    ax_loss.set_title("Loss"); ax_loss.legend(); ax_loss.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()


# ─── Экспорт в TorchScript ───────────────────────────────────────────────────
def export_torchscript(output_dir: Path, device: torch.device) -> None:
    """Загружает лучшие веса и сохраняет трассированную модель."""
    best_ckpt_path = output_dir / "best.pth"
    if not best_ckpt_path.exists():
        print("⚠️  best.pth не найден, экспорт TorchScript пропущен.")
        return

    model = DigitCNN(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_WIDTH, device=device)
    traced = torch.jit.trace(model, dummy)
    ts_path = output_dir / "best_model.pt"
    traced.save(ts_path)

    print(f"💾 TorchScript сохранён: {ts_path.resolve()}")
    print("   (Для инференса класс DigitCNN больше не нужен, загружайте напрямую .pt)")


def run_dir_for(args: argparse.Namespace) -> Path:
    """Куда писать результат: --output_dir; при --resume — папка чекпоинта;
    иначе новая папка <OUTPUT_DIR>/<дата-время> (не затирает прежние прогоны)."""
    if args.output_dir:
        return Path(args.output_dir)
    if args.resume:
        return Path(args.resume).parent
    return Path(OUTPUT_DIR) / datetime.now().strftime("%Y-%m-%d_%H-%M")


def save_split(output_dir: Path, crops_dir: Path, train_loader, val_loader, test_loader,
               args: argparse.Namespace) -> None:
    """split/*.txt и run_info.json: на чём и как обучали (этап 5)."""
    parts = {name: [p for p, _ in loader.dataset.samples]
             for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader))}
    files = [f for v in parts.values() for f in v]
    write_split_lists(output_dir, parts, crops_dir)
    write_run_info(output_dir, crops_dir, files, parts, args.seed, vars(args))


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    
    if args.resume and args.finetune:
        raise ValueError("--resume и --finetune нельзя использовать одновременно.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = run_dir_for(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = Path(args.crops_dir)

    if not crops_dir.exists():
        print(f"❌ Папка с кропами не найдена: {crops_dir.resolve()}")
        print("   Сначала запустите build_dataset_cnn.py")
        return

    print(f"🖥️  Устройство: {device}")
    print(f"📁  Результаты: {output_dir.resolve()}")

    # 1. Данные (теперь распаковываем 3 загрузчика!)
    print("\n📂 Загрузка данных (Train / Val / Test)...")
    train_loader, val_loader, test_loader = make_loaders(
        crops_dir, batch_size=args.batch_size, seed=args.seed,
    )
    save_split(output_dir, crops_dir, train_loader, val_loader, test_loader, args)

    # 2. Модель и оптимизатор
    model = DigitCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    best_acc = 0.0
    no_improve = 0
    history: list[dict] = []
    best_ckpt_path = output_dir / "best.pth"

    # 3. Загрузка чекпоинта (Resume / Finetune)
    if args.finetune:
        print(f"\n🎯 Fine-tuning из: {args.finetune}")
        ckpt = torch.load(args.finetune, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        best_acc = ckpt.get("val_acc", 0.0)
        print(f"   Стартовая val_acc: {best_acc:.4f}")
        
    elif args.resume:
        if not Path(args.resume).exists():
            raise FileNotFoundError(f"Чекпоинт не найден: {args.resume}")
        print(f"\n⏳ Resume из: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("val_acc", 0.0)
        print(f"   Продолжение с эпохи {start_epoch}, best_acc={best_acc:.4f}")

    # 4. Цикл обучения
    print(f"\n{'─' * 65}")
    print(f"  Параметры   : {model.num_parameters:,}")
    print(f"  Эпох        : {args.epochs} (начиная с {start_epoch}) | batch: {args.batch_size} | lr: {args.lr}")
    print(f"{'─' * 65}\n")
    
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, best_ckpt_path)
            marker = " ✓ (best)"
        else:
            no_improve += 1

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })

        if epoch % 5 == 0 or epoch == 1 or marker:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Ep {epoch:3}/{args.epochs} | train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f} | lr: {current_lr:.2e} {marker}")

        if no_improve >= args.patience:
            print(f"\n⏹️  Early Stopping: {args.patience} эпох без улучшения val_acc.")
            break

    elapsed = time.time() - t0
    print(f"\n✅ Обучение завершено за {elapsed:.0f}s. Best val_acc = {best_acc:.4f}")

    # 5. Финальный тест на hold-out выборке (КРИТИЧЕСКИ ВАЖНО для evaluate)
    print("\n" + "═" * 65)
    print("🔄 Загружаю best.pth для финального теста на hold-out (Test) выборке...")
    
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(best_ckpt["model_state_dict"])
    else:
        print("⚠️  best.pth не найден, используется последняя эпоха.")
        best_ckpt = {"model_state_dict": model.state_dict()}

    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"🏁 Final Test Accuracy : {test_acc:.4f} | Loss: {test_loss:.4f}")
    print("═" * 65)

    # Добавляем финальный тест в историю
    history.append({
        "epoch": "final_test",
        "test_loss": round(test_loss, 4),
        "test_acc": round(test_acc, 4),
    })

    # 6. Сохранение артефактов
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"📈 История сохранена: {output_dir / 'history.json'}")

    plot_history(history, output_dir)
    print(f"📊 Графики сохранены: {output_dir / 'training_curves.png'}")

    export_torchscript(output_dir, device)
    
    print(f"\n{'='*65}")
    print(f"  Best val_acc : {best_acc:.4f}")
    print(f"  Final test   : {test_acc:.4f}")
    print(f"  Run dir      : {output_dir.resolve()}")
    print(f"{'='*65}")
    print("\n➡️  Следующие шаги:")
    print("   1. python evaluate_cnn.py --run_dir runs/cnn  (построит Confusion Matrix)")
    print("   2. python infer_cnn.py --checkpoint runs/cnn/best_model.pt --image path/to/crop.jpg")


if __name__ == "__main__":
    main()