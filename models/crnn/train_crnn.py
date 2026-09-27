"""
Обучение, дообучение и экспорт модели CRNN.

ИСПОЛЬЗОВАНИЕ:

1. Обучение с нуля:
    python train_crnn.py --images_dir data/images --labels_dir data/labels

2. Resume (продолжение прерванного обучения):
    python train_crnn.py --images_dir data/images --labels_dir data/labels \\
        --resume runs/2026-06-04_14-25/best.pt

3. Fine-tuning (только веса, оптимизатор заново):
    python train_crnn.py --images_dir data/images --labels_dir data/labels \\
        --finetune runs/2026-06-04_14-25/best.pt --lr 1e-4

РЕЗУЛЬТАТЫ в runs/<дата-время>/:
  best.pt          — чекпоинт (веса + оптимизатор)
  best_model.pt    — TorchScript для продакшена
  history.json     — метрики по эпохам
  config.json      — параметры запуска
  plot_loss.png    — loss каждые 10 эпох (промежуточно)
  plot_summary.png — итоговый summary

После обучения запусти:
  python evaluate_crnn.py --run_dir runs/<дата-время> \\
      --images_dir data/images --labels_dir data/labels
"""

import os
import json
import argparse
import datetime

import torch
from torch.utils.data import DataLoader

from config_crnn import IMG_W, IMG_H
from dataset_crnn import load_dataset, split, MeterDataset, train_transform, val_transform, collate
from metrics_crnn import exact_match, cer
from model_crnn import CRNN, ctc_loss, predict


# ── Аргументы ────────────────────────────────────────────────────────

def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir",   required=True)
    p.add_argument("--labels_dir",   required=True)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--save_dir",     default="runs/crnn")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--resume",       type=str,   default=None)
    p.add_argument("--finetune",     type=str,   default=None)
    p.add_argument("--es_patience",  type=int,   default=15)
    p.add_argument("--es_min_delta", type=float, default=0.0)
    p.add_argument("--num_workers",  type=int,   default=0)
    return p.parse_args()


# ── Train / eval ──────────────────────────────────────────────────────

def train_epoch(model: CRNN, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for imgs, targets, lengths in loader:
        imgs, targets, lengths = imgs.to(device), targets.to(device), lengths.to(device)
        opt.zero_grad()
        loss = ctc_loss(model(imgs), targets, lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: CRNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, list[str], list[str]]:
    model.eval()
    preds_all, gts_all = [], []
    total_loss = 0.0
    for imgs, targets, lengths in loader:
        imgs, targets, lengths = imgs.to(device), targets.to(device), lengths.to(device)
        log_probs   = model(imgs)
        total_loss += ctc_loss(log_probs, targets, lengths).item() * imgs.size(0)
        preds_all.extend(predict(model, imgs))
        offset = 0
        for l in lengths.tolist():
            gts_all.append("".join(str(targets[offset + j].item()) for j in range(l)))
            offset += l
    return total_loss / len(loader.dataset), exact_match(preds_all, gts_all), preds_all, gts_all


# ── Графики ───────────────────────────────────────────────────────────

def save_plots(history: list[dict], save_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("⚠️  matplotlib не найден — pip install matplotlib")
        return

    epochs     = [r["epoch"]       for r in history]
    tr_losses  = [r["train_loss"]  for r in history]
    val_losses = [r["val_loss"]    for r in history]
    val_accs   = [r["val_acc"]     for r in history]
    val_cers   = [r.get("val_cer") for r in history]
    lrs        = [r.get("lr")      for r in history]
    style      = dict(linewidth=1.8)
    best_idx   = max(range(len(val_accs)), key=lambda i: val_accs[i])

    # Loss
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, tr_losses,  label="Train Loss", **style)
    ax.plot(epochs, val_losses, label="Val Loss",   **style)
    ax.set_xlabel("Epoch"); ax.set_ylabel("CTC Loss")
    ax.set_title("Loss"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "plot_loss.png"), dpi=150)
    plt.close(fig)

    # Summary 2×2
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Training Summary", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, tr_losses,  label="Train", **style)
    ax1.plot(epochs, val_losses, label="Val",   **style)
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, val_accs, color="tab:green", **style)
    ax2.set_title("Val Accuracy"); ax2.set_ylim(0, 1); ax2.grid(alpha=0.3)
    ax2.annotate(
        f"best={val_accs[best_idx]:.4f}",
        xy=(epochs[best_idx], val_accs[best_idx]),
        xytext=(10, -15), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8,
    )

    ax3 = fig.add_subplot(gs[1, 0])
    if any(v is not None for v in val_cers):
        ax3.plot(epochs, val_cers, color="tab:orange", **style)
    ax3.set_title("Val CER"); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    if any(v is not None for v in lrs):
        ax4.plot(epochs, lrs, color="tab:red", **style)
        ax4.set_yscale("log")
    ax4.set_title("Learning Rate"); ax4.grid(alpha=0.3)

    fig.savefig(os.path.join(save_dir, "plot_summary.png"), dpi=150)
    plt.close(fig)
    print(f"📊 Графики сохранены: {save_dir}/plot_*.png")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse()

    if args.resume and args.finetune:
        raise ValueError("--resume и --finetune нельзя использовать одновременно.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    if args.finetune:
        run_name = "finetune_" + run_name
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"🖥️  Устройство: {device}")
    print(f"📁  Результаты: {save_dir}")

    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # Данные
    print("📂 Загрузка данных...")
    meta = load_dataset(args.images_dir, args.labels_dir)
    train_meta, val_meta, test_meta = split(meta, seed=args.seed)
    print(f"📊 train={len(train_meta)} | val={len(val_meta)} | test={len(test_meta)}")

    train_dl = DataLoader(
        MeterDataset(train_meta, train_transform()),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=False,
    )
    val_dl = DataLoader(
        MeterDataset(val_meta, val_transform()),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=False,
    )
    test_dl = DataLoader(
        MeterDataset(test_meta, val_transform()),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=False,
    )

    # Модель
    model = CRNN().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)

    start_epoch    = 1
    best_acc       = -1.0
    best_val_loss  = float("inf")
    es_counter     = 0
    history        = []
    best_ckpt_path = os.path.join(save_dir, "best.pt")

    # Загрузка чекпоинта
    if args.finetune:
        print(f"🎯 Fine-tuning из: {args.finetune}")
        ckpt = torch.load(args.finetune, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        best_acc = ckpt.get("val_acc", -1.0)
        print(f"   Стартовая val_acc: {best_acc:.4f}")

    elif args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Чекпоинт не найден: {args.resume}")
        print(f"⏳ Resume из: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc    = ckpt.get("val_acc", -1.0)
        print(f"   Продолжение с эпохи {start_epoch}, best_acc={best_acc:.4f}")

    # Цикл обучения
    print("🚀 Обучение...\n")
    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss                                = train_epoch(model, train_dl, opt, device)
        val_loss, val_acc, val_preds, val_gts  = eval_epoch(model, val_dl, device)
        val_cer_val                            = cer(val_preds, val_gts)
        current_lr                             = opt.param_groups[0]["lr"]
        sched.step(val_loss)

        examples = "  ".join(f"'{p}'→'{g}'" for p, g in zip(val_preds[:3], val_gts[:3]))
        print(
            f"Ep {epoch:3d}/{args.epochs} | "
            f"train={tr_loss:.4f} | val={val_loss:.4f} | "
            f"acc={val_acc:.4f} | cer={val_cer_val:.4f} | "
            f"lr={current_lr:.2e} | [{examples}]"
        )

        history.append({
            "epoch":      epoch,
            "train_loss": tr_loss,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
            "val_cer":    val_cer_val,
            "lr":         current_lr,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "val_acc":         val_acc,
                "val_cer":         val_cer_val,
            }, best_ckpt_path)
            print(f"  💾 best.pt обновлён (acc={val_acc:.4f}, cer={val_cer_val:.4f})")

        if val_loss < best_val_loss - args.es_min_delta:
            best_val_loss = val_loss
            es_counter    = 0
        else:
            es_counter += 1
            if es_counter >= args.es_patience:
                print(f"\n⏹️  Early Stopping на эпохе {epoch} ({args.es_patience} эпох без улучшения val_loss)")
                break

        if epoch % 10 == 0:
            save_plots(history, save_dir)

    # Финальный тест
    print("\n🔄 Загружаю best.pt для финального теста...")
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state"])
    else:
        print("⚠️  best.pt не найден, используется последняя эпоха")
        best_ckpt = {"model_state": model.state_dict()}

    _, test_acc, test_preds, test_gts = eval_epoch(model, test_dl, device)
    test_cer_val = cer(test_preds, test_gts)
    print(f"🏁 Test: Accuracy={test_acc:.4f} | CER={test_cer_val:.4f}")

    # Сохранение истории
    history_path = os.path.join(save_dir, "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "best_val_acc":   best_acc,
                "test_acc":       test_acc,
                "test_cer":       test_cer_val,
                "epochs_trained": len(history),
                "run_dir":        save_dir,
            },
            "epochs": history,
        }, f, indent=2, ensure_ascii=False)
    print(f"📈 История: {history_path}")

    save_plots(history, save_dir)

    # Экспорт TorchScript
    print("\n📦 Экспорт TorchScript...")
    export_model = CRNN().to(device)
    export_model.load_state_dict(best_ckpt["model_state"])
    export_model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(export_model, torch.randn(1, 3, IMG_H, IMG_W, device=device))
    ts_path = os.path.join(save_dir, "best_model.pt")
    traced.save(ts_path)
    print(f"✅ TorchScript: {ts_path}")

    print(f"\n{'='*55}")
    print(f"  Best val_acc : {best_acc:.4f}")
    print(f"  Test acc     : {test_acc:.4f}  |  CER: {test_cer_val:.4f}")
    print(f"  Run dir      : {save_dir}")
    print(f"{'='*55}")
    print(f"\n➡️  Для полного анализа запусти:")
    print(f"   python evaluate_crnn.py --run_dir {save_dir} --images_dir {args.images_dir} --labels_dir {args.labels_dir}")


if __name__ == "__main__":
    main()
