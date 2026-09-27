"""
Полный анализ обученной CRNN модели.

ИСПОЛЬЗОВАНИЕ:
  python evaluate_crnn.py --run_dir runs/2026-06-04_14-25 \\
      --images_dir data/images --labels_dir data/labels

ЧТО СОЗДАЁТСЯ в run_dir/eval/:
  metrics.json               — все метрики
  predictions_test.txt       — таблица pred vs gt для всех примеров
  plot_loss.png              — train/val loss по эпохам
  plot_acc.png               — val accuracy по эпохам
  plot_cer.png               — val CER по эпохам
  plot_summary.png           — все 4 на одном листе
  plot_confusion_digits.png  — confusion matrix по цифрам
  plot_position_acc.png      — accuracy отдельно по каждой позиции
  errors/                    — картинки неправильно распознанных примеров
"""

import os
import json
import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader

from config_crnn import MIN_LABEL_LENGTH, MAX_LABEL_LENGTH
from dataset_crnn import load_dataset, split, MeterDataset, val_transform, collate
from metrics_crnn import exact_match, cer, per_position_accuracy
from model_crnn import CRNN, predict


# ── Аргументы ────────────────────────────────────────────────────────

def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir",     required=True, help="Папка с результатами обучения")
    p.add_argument("--images_dir",  required=True)
    p.add_argument("--labels_dir",  required=True)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--max_errors",  type=int, default=50, help="Максимум картинок ошибок для сохранения")
    p.add_argument("--min_len",     type=int, default=MIN_LABEL_LENGTH)
    p.add_argument("--max_len",     type=int, default=MAX_LABEL_LENGTH)
    return p.parse_args()


# ── Сбор предсказаний ─────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: CRNN,
    metadata: list,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> list[tuple[str, str, str]]:
    dataset = MeterDataset(metadata, val_transform())
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=num_workers, pin_memory=False,
    )
    model.eval()

    preds_all, gts_all = [], []
    for imgs, targets, lengths in loader:
        imgs, targets, lengths = imgs.to(device), targets.to(device), lengths.to(device)
        preds_all.extend(predict(model, imgs))
        offset = 0
        for l in lengths.tolist():
            gts_all.append("".join(str(targets[offset + j].item()) for j in range(l)))
            offset += l

    paths = [str(item["file"]) for item in metadata]
    return list(zip(preds_all, gts_all, paths))


# ── Текстовые отчёты ──────────────────────────────────────────────────

def print_length_distribution(gts: list[str]) -> None:
    counts = Counter(len(g) for g in gts)
    total  = len(gts)
    print("📏 Распределение длин лейблов в датасете:")
    for length in sorted(counts.keys()):
        pct = counts[length] / total * 100
        print(f"   Длина {length}: {counts[length]:5d} примеров ({pct:5.1f}%)")


def save_predictions_txt(
    results: list[tuple],
    path: str,
    max_len: int = 10,
) -> None:
    preds     = [r[0] for r in results]
    gts       = [r[1] for r in results]
    acc       = exact_match(preds, gts)
    cer_v     = cer(preds, gts)
    pos_acc, pos_counts = per_position_accuracy(preds, gts, max_len=max_len)
    col_width = max(12, max_len + 2)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Всего примеров : {len(results)}\n")
        f.write(f"Accuracy       : {acc:.4f}  ({sum(p==g for p,g in zip(preds,gts))}/{len(gts)})\n")
        f.write(f"CER            : {cer_v:.4f}\n")
        f.write("Accuracy по позициям:\n")
        for i in range(max_len):
            f.write(f"  Позиция {i+1}: {pos_acc[i]:.3f}  (n={pos_counts[i]})\n")
        f.write("\n")
        f.write(f"{'Pred':>{col_width}}  {'GT':>{col_width}}  {'OK':>4}  Файл\n")
        f.write("-" * (col_width * 2 + 15) + "\n")
        for pred, gt, img_path in results:
            ok = "✓" if pred == gt else "✗"
            f.write(f"{pred:>{col_width}}  {gt:>{col_width}}  {ok:>4}  {os.path.basename(img_path)}\n")


# ── Визуализация ошибок ───────────────────────────────────────────────

def save_error_images(
    results: list[tuple],
    out_dir: str,
    max_errors: int = 50,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("⚠️  Pillow не найден.")
        return

    os.makedirs(out_dir, exist_ok=True)
    errors = [(p, g, fp) for p, g, fp in results if p != g]
    print(f"   Ошибок: {len(errors)} из {len(results)}")

    for pred, gt, img_path in errors[:max_errors]:
        try:
            img    = Image.open(img_path).convert("RGB")
            width  = max(320, len(max(pred, gt, key=len)) * 25 + 40)
            img    = img.resize((width, 80))
            canvas = Image.new("RGB", (width, 100), color=(30, 30, 30))
            canvas.paste(img, (0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 82), f"pred: {pred}   gt: {gt}", fill=(255, 80, 80))
            safe_name = os.path.basename(img_path).replace(" ", "_")
            fname     = f"pred_{pred}__gt_{gt}__{safe_name}"
            canvas.save(os.path.join(out_dir, fname))
        except Exception:
            pass

    print(f"   Сохранено картинок ошибок: {min(len(errors), max_errors)} → {out_dir}/")


# ── Графики обучения ──────────────────────────────────────────────────

def save_training_plots(history: list[dict], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
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
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, tr_losses,  label="Train Loss", **style)
    ax.plot(epochs, val_losses, label="Val Loss",   **style)
    ax.set_xlabel("Epoch"); ax.set_ylabel("CTC Loss")
    ax.set_title("Loss"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_loss.png"), dpi=150)
    plt.close(fig)

    # Accuracy
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, val_accs, color="tab:green", label="Val Accuracy", **style)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Exact Match Accuracy")
    ax.set_title("Accuracy"); ax.legend(); ax.grid(alpha=0.3)
    ax.annotate(
        f"best={val_accs[best_idx]:.4f}",
        xy=(epochs[best_idx], val_accs[best_idx]),
        xytext=(10, -15), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_acc.png"), dpi=150)
    plt.close(fig)

    # CER
    if any(v is not None for v in val_cers):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(epochs, val_cers, color="tab:orange", label="Val CER", **style)
        ax.set_xlabel("Epoch"); ax.set_ylabel("CER")
        ax.set_title("Character Error Rate"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "plot_cer.png"), dpi=150)
        plt.close(fig)

    # Summary 2×2
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Training Summary", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, tr_losses, label="Train", **style)
    ax1.plot(epochs, val_losses, label="Val",  **style)
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, val_accs, color="tab:green", **style)
    ax2.set_ylim(0, 1); ax2.set_title("Val Accuracy"); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    if any(v is not None for v in val_cers):
        ax3.plot(epochs, val_cers, color="tab:orange", **style)
    ax3.set_title("Val CER"); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    if any(v is not None for v in lrs):
        ax4.plot(epochs, lrs, color="tab:red", **style)
        ax4.set_yscale("log")
    ax4.set_title("Learning Rate"); ax4.grid(alpha=0.3)

    fig.savefig(os.path.join(out_dir, "plot_summary.png"), dpi=150)
    plt.close(fig)


# ── Confusion matrix ──────────────────────────────────────────────────

def save_confusion_matrix(results: list[tuple], out_dir: str) -> None:
    """
    Строит confusion matrix 10×10.
    Учитывает только пары где len(pred) == len(gt) —
    иначе позиции сдвигаются и сравнение некорректно.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    matrix        = [[0] * 10 for _ in range(10)]
    valid_samples = 0

    for pred, gt, _ in results:
        if len(pred) == len(gt):
            valid_samples += 1
            for p_ch, g_ch in zip(pred, gt):
                if p_ch.isdigit() and g_ch.isdigit():
                    matrix[int(g_ch)][int(p_ch)] += 1

    print(f"   ⚠️  Для Confusion Matrix использовано {valid_samples} примеров (только с идеально совпавшей длиной).")

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(list("0123456789"))
    ax.set_yticklabels(list("0123456789"))
    ax.set_xlabel("Предсказано"); ax.set_ylabel("Истина")
    ax.set_title(f"Confusion Matrix (n={valid_samples})")
    plt.colorbar(im, ax=ax)

    max_val = max(max(row) for row in matrix) or 1
    for i in range(10):
        for j in range(10):
            val   = matrix[i][j]
            color = "white" if val > max_val * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_confusion_digits.png"), dpi=150)
    plt.close(fig)
    print(f"   Confusion matrix → {out_dir}/plot_confusion_digits.png")


# ── Accuracy по позициям ──────────────────────────────────────────────

def save_position_accuracy_plot(
    results: list[tuple],
    out_dir: str,
    max_len: int = 10,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    preds = [r[0] for r in results]
    gts   = [r[1] for r in results]
    pos_acc, pos_counts = per_position_accuracy(preds, gts, max_len=max_len)

    fig, ax = plt.subplots(figsize=(max(8, max_len * 1.2), 5))
    bars = ax.bar(range(max_len), pos_acc, color="tab:blue", alpha=0.8)
    ax.set_xticks(range(max_len))
    ax.set_xticklabels([f"Поз. {i+1}" for i in range(max_len)])
    ax.set_ylim(0, 1.15); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy по позициям")
    ax.grid(axis="y", alpha=0.3)

    for bar, val, count in zip(bars, pos_acc, pos_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}\n(n={count})",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_position_acc.png"), dpi=150)
    plt.close(fig)
    print(f"   Accuracy по позициям → {out_dir}/plot_position_acc.png")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    print(f"🖥️  Устройство: {device}")
    print(f"📁  Run dir   : {args.run_dir}")

    # Загрузка модели
    ckpt_path = os.path.join(args.run_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Не найден best.pt в {args.run_dir}")
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = CRNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"✅ Модель загружена (val_acc={ckpt.get('val_acc', '?')})")

    # Данные
    meta = load_dataset(args.images_dir, args.labels_dir)
    _, _, test_meta = split(meta, seed=args.seed)

    # Предсказания
    print("\n🔍 Собираю предсказания на test...")
    test_results           = collect_predictions(model, test_meta, device, args.batch_size, args.num_workers)
    test_preds             = [r[0] for r in test_results]
    test_gts               = [r[1] for r in test_results]
    test_acc               = exact_match(test_preds, test_gts)
    test_cer_v             = cer(test_preds, test_gts)
    test_pos, test_counts  = per_position_accuracy(test_preds, test_gts, max_len=args.max_len)

    print("\n" + "=" * 50)
    print_length_distribution(test_gts)
    print("=" * 50)
    print(f"\n📊 Test — Accuracy: {test_acc:.4f}  |  CER: {test_cer_v:.4f}")
    print("📊 Test позиции:")
    for i in range(args.max_len):
        print(f"   Позиция {i+1}: {test_pos[i]:.3f}  (n={test_counts[i]})")

    # Метрики в JSON
    metrics = {
        "test": {
            "accuracy":            test_acc,
            "cer":                 test_cer_v,
            "per_position_acc":    test_pos,
            "per_position_counts": test_counts,
            "length_distribution": dict(Counter(len(g) for g in test_gts)),
        }
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Отчёты
    print("\n📝 Сохраняю текстовые отчёты...")
    save_predictions_txt(test_results, os.path.join(out_dir, "predictions_test.txt"), max_len=args.max_len)

    # Графики обучения
    history_path = os.path.join(args.run_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path, encoding="utf-8") as f:
            data = json.load(f)
        history = data.get("epochs", data)
        save_training_plots(history, out_dir)

    print("\n🔢 Строю confusion matrix...")
    save_confusion_matrix(test_results, out_dir)

    print("📊 Строю accuracy по позициям...")
    save_position_accuracy_plot(test_results, out_dir, max_len=args.max_len)

    print("\n🖼️  Сохраняю картинки ошибок...")
    save_error_images(test_results, os.path.join(out_dir, "errors"), args.max_errors)

    print(f"\n{'='*55}")
    print(f"  Test acc={test_acc:.4f}  cer={test_cer_v:.4f}")
    print(f"  Всё сохранено → {out_dir}/")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
