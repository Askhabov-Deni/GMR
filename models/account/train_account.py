"""
Обучение модели «лицевой счёт по надписи маркером» (CRNN + CTC).

ИСПОЛЬЗОВАНИЕ (из папки проекта):
  python models/account/train_account.py
      датасет по умолчанию — database/datasets/accounts_crnn/images и labels
  python models/account/train_account.py --finetune account_ocr/runs/crnn/<дата-время>/best.pt --lr 3e-4
      дообучить от прежних весов
  python models/account/train_account.py --device cpu --epochs 40
      без видеокарты (модель маленькая: на процессоре — минуты на эпоху при нескольких тысячах кропов)

РЕЗУЛЬТАТ — новая папка account_ocr/runs/crnn/<дата-время>/:
  best.pt         — веса + размер входа и алфавит (инференс берёт их отсюда)
  history.json    — метрики по эпохам и итог теста
  run_info.json   — датасет (отпечаток), коммит кода, пакеты, деление
  split/*.txt     — какие кропы в обучении / проверке / тесте
  eval/           — отчёт теста (то же, что evaluate_account.py без --table)

Деление — по номеру (метке): кропы одного номера всегда в одной части, и
проверка идёт на номерах, которых модель не видела.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from . import config_account as C
    from .dataset_account import AccountDataset, collate, load_dataset, split
    from .evaluate_account import evaluate, report_text
    from .model_account import AccountCRNN, ctc_loss, load_checkpoint, save_checkpoint
    from ..ctc_lexicon import greedy_decode
    from ..crnn.metrics_crnn import cer, exact_match
    from ..datasets import require_parts, write_run_info, write_split_lists
except ImportError:  # запуск как отдельный скрипт: нужна папка проекта в sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from models.account import config_account as C
    from models.account.dataset_account import AccountDataset, collate, load_dataset, split
    from models.account.evaluate_account import evaluate, report_text
    from models.account.model_account import AccountCRNN, ctc_loss, load_checkpoint, save_checkpoint
    from models.ctc_lexicon import greedy_decode
    from models.crnn.metrics_crnn import cer, exact_match
    from models.datasets import require_parts, write_run_info, write_split_lists


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение модели лицевого счёта по надписи маркером")
    p.add_argument("--images_dir", default=str(Path(C.DATASET_DIR) / "images"))
    p.add_argument("--labels_dir", default=str(Path(C.DATASET_DIR) / "labels"))
    p.add_argument("--output_dir", default=None,
                   help=f"папка результата (по умолчанию {C.OUTPUT_DIR}/<дата-время>)")
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=C.LR)
    p.add_argument("--patience", type=int, default=C.PATIENCE)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--img_h", type=int, default=C.IMG_H)
    p.add_argument("--img_w", type=int, default=C.IMG_W)
    p.add_argument("--finetune", default=None, help="best.pt прежнего обучения: начать с его весов")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default=None, help="cpu или cuda; по умолчанию — что есть")
    return p.parse_args(argv)


def train_epoch(model, loader, opt, sched, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, targets, lengths in loader:
        x, targets, lengths = x.to(device), targets.to(device), lengths.to(device)
        opt.zero_grad()
        loss = ctc_loss(model(x), targets, lengths, model.blank)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, alphabet: str) -> tuple[float, float, float, list]:
    """(loss, весь номер верно, CER, log_probs по примерам)."""
    model.eval()
    total, n, preds, gts, lps = 0.0, 0, [], [], []
    for x, targets, lengths in loader:
        x, targets, lengths = x.to(device), targets.to(device), lengths.to(device)
        lp = model(x)
        total += ctc_loss(lp, targets, lengths, model.blank).item() * x.size(0)
        n += x.size(0)
        lp = lp.cpu()
        offset = 0
        for j, ln in enumerate(lengths.tolist()):
            gts.append("".join(alphabet[i] for i in targets[offset:offset + ln].tolist()))
            offset += ln
            lps.append(lp[:, j, :])
            preds.append(greedy_decode(lp[:, j, :], alphabet, model.blank))
    return total / max(n, 1), exact_match(preds, gts), cer(preds, gts), lps


def main(argv=None) -> int:
    args = parse(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.output_dir or Path(C.OUTPUT_DIR) / datetime.now().strftime("%Y-%m-%d_%H-%M"))
    if (out / "best.pt").exists():
        print(f"ОШИБКА: в {out} уже есть best.pt — выберите другую папку")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    print(f"Устройство: {device}\nРезультат: {out.resolve()}")

    items = load_dataset(args.images_dir, args.labels_dir)
    train_items, val_items, test_items = split(items, C.TRAIN_RATIO, C.VAL_RATIO, args.seed)
    parts = {"train": train_items, "val": val_items, "test": test_items}
    try:
        require_parts(parts, ("train", "val", "test"))
    except ValueError as e:
        print(f"ОШИБКА: {e}")
        return 1
    print(f"Обучение {len(train_items)} | проверка {len(val_items)} | тест {len(test_items)} "
          f"(номеров: {len({i['label'] for i in train_items})} / {len({i['label'] for i in val_items})} / "
          f"{len({i['label'] for i in test_items})})")

    images, labels = Path(args.images_dir).resolve(), Path(args.labels_dir).resolve()
    base = Path(os.path.commonpath([images, labels]))
    files = {k: [Path(i["file"]).resolve() for i in v] for k, v in parts.items()}
    write_split_lists(out, files, base)
    write_run_info(out, base, [f for v in files.values() for f in v]
                   + [labels / (f.stem + ".txt") for v in files.values() for f in v],
                   files, args.seed, vars(args))

    kw = dict(batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers)
    train_dl = DataLoader(AccountDataset(train_items, args.img_h, args.img_w, augment=True),
                          shuffle=True, drop_last=len(train_items) > args.batch_size, **kw)
    val_dl = DataLoader(AccountDataset(val_items, args.img_h, args.img_w), shuffle=False, **kw)
    test_dl = DataLoader(AccountDataset(test_items, args.img_h, args.img_w), shuffle=False, **kw)

    if args.finetune:
        model, meta = load_checkpoint(args.finetune, device)
        if (meta["preprocess"]["img_h"], meta["preprocess"]["img_w"]) != (args.img_h, args.img_w):
            print("ОШИБКА: у дообучаемых весов другой размер входа: "
                  f"{meta['preprocess']} (укажите --img_h/--img_w как у них)")
            return 1
        print(f"Дообучение от {args.finetune}")
    else:
        model = AccountCRNN(img_h=args.img_h, n_symbols=len(C.ALPHABET)).to(device)
    print(f"Параметров: {model.num_parameters:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(train_dl)), pct_start=0.1)
    preprocess = {"img_h": args.img_h, "img_w": args.img_w}

    best, best_key, stale, history = None, (-1.0, 0.0), 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_dl, opt, sched, device)
        val_loss, val_acc, val_cer, _ = eval_epoch(model, val_dl, device, C.ALPHABET)
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4), "val_loss": round(val_loss, 4),
                        "val_acc": round(val_acc, 4), "val_cer": round(val_cer, 4),
                        "lr": opt.param_groups[0]["lr"]})
        key = (val_acc, -val_cer)
        mark = ""
        if key > best_key:
            best_key, best, stale, mark = key, epoch, 0, "  ✓ лучшая"
            save_checkpoint(out / "best.pt", model, preprocess, C.ALPHABET,
                            epoch=epoch, val_acc=val_acc, val_cer=val_cer)
        else:
            stale += 1
        print(f"Эпоха {epoch:3}/{args.epochs} | loss {tr_loss:.4f} / {val_loss:.4f} | "
              f"номер верно {val_acc:.4f} | CER {val_cer:.4f}{mark}")
        if stale >= args.patience:
            print(f"Ранняя остановка: {args.patience} эпох без улучшения")
            break
    print(f"Обучение: {time.time() - t0:.0f} с, лучшая эпоха {best}")

    model, _ = load_checkpoint(out / "best.pt", device)
    test_lps = eval_epoch(model, test_dl, device, C.ALPHABET)[3]
    all_labels = {i["label"]: (i["label"],) for i in items}      # словарь — все номера датасета
    res = evaluate(test_lps, [i["label"] for i in test_items], C.ALPHABET, model.blank, all_labels)
    text = report_text(res, "Тест (словарь — номера датасета; как в работе — evaluate_account.py --table)")
    print("\n" + text)
    (out / "eval").mkdir(exist_ok=True)
    (out / "eval" / "report.txt").write_text(text + "\n", encoding="utf-8")
    summary = {k: v for k, v in res.items() if k != "picks"}
    (out / "history.json").write_text(json.dumps(
        {"summary": {"best_epoch": best, "best_val_acc": best_key[0], "test": summary}, "epochs": history},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nГотово: {out}\nДальше: python models/account/evaluate_account.py --run_dir {out} --table <таблица>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
