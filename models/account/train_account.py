"""
Обучение модели «лицевой счёт по надписи маркером» (CRNN + CTC).

ИСПОЛЬЗОВАНИЕ (из папки проекта):
  python models/account/train_account.py
      датасет по умолчанию — database/datasets/accounts_crnn/images: кропы
      «<лицевой счёт>__….jpg», разметка не нужна (dataset_account.py)
  python models/account/train_account.py --finetune account_ocr/runs/crnn/<дата-время>/best.pt --lr 3e-4
      дообучить от прежних весов (без разминки на синтетике)
  python models/account/train_account.py --device cpu --epochs 40
      без видеокарты (модель маленькая: на процессоре — минуты на эпоху при нескольких тысячах кропов)

РЕЗУЛЬТАТ — новая папка account_ocr/runs/crnn/<дата-время>/:
  best.pt         — веса + размер входа и алфавит (инференс берёт их отсюда)
  history.json    — метрики по эпохам и итог теста
  run_info.json   — датасет (отпечаток), коммит кода, пакеты, деление
  split/*.txt     — какие кропы в обучении / проверке / тесте
  eval/           — отчёт теста (то же, что evaluate_account.py без --table)

Сначала — разминка на синтетических надписях (synthetic.py, --synthetic,
--synthetic_epochs): без неё обучение без разметки не сдвигается.

Деление — по лицевому счёту: кропы одного счёта всегда в одной части, и
проверка идёт на счетах, которых модель не видела. «Прочитано как написано»
— жадное чтение совпало с одним из допустимых написаний счёта (93, 0093, …).
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from . import config_account as C
    from .dataset_account import AccountDataset, collate, load_dataset, split
    from .synthetic import synthetic_items
    from .evaluate_account import evaluate, report_text
    from .model_account import AccountCRNN, load_checkpoint, multi_ctc_loss, save_checkpoint
    from ..ctc_lexicon import greedy_decode
    from ..crnn.metrics_crnn import cer, levenshtein
    from ..datasets import require_parts, write_run_info, write_split_lists
except ImportError:  # запуск как отдельный скрипт: нужна папка проекта в sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from models.account import config_account as C
    from models.account.dataset_account import AccountDataset, collate, load_dataset, split
    from models.account.synthetic import synthetic_items
    from models.account.evaluate_account import evaluate, report_text
    from models.account.model_account import AccountCRNN, load_checkpoint, multi_ctc_loss, save_checkpoint
    from models.ctc_lexicon import greedy_decode
    from models.crnn.metrics_crnn import cer, levenshtein
    from models.datasets import require_parts, write_run_info, write_split_lists


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение модели лицевого счёта по надписи маркером")
    p.add_argument("--images_dir", default=str(Path(C.DATASET_DIR) / "images"))
    p.add_argument("--output_dir", default=None,
                   help=f"папка результата (по умолчанию {C.OUTPUT_DIR}/<дата-время>)")
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=C.LR)
    p.add_argument("--patience", type=int, default=C.PATIENCE)
    p.add_argument("--synthetic", type=int, default=C.SYNTH_COUNT,
                   help="сколько синтетических надписей для разминки (0 — без; с --finetune не нужна)")
    p.add_argument("--synthetic_epochs", type=int, default=C.SYNTH_EPOCHS)
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
    for x, targets, lengths, owners in loader:
        x, targets, lengths, owners = x.to(device), targets.to(device), lengths.to(device), owners.to(device)
        opt.zero_grad()
        loss = multi_ctc_loss(model(x), targets, lengths, owners, model.blank)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def written_metrics(preds: list[str], variants: list) -> tuple[float, float]:
    """(прочитано как написано — чтение есть среди написаний счёта; CER до
    ближайшего написания)."""
    ok = sum(p in vs for p, vs in zip(preds, variants)) / max(len(preds), 1)
    nearest = [min(vs, key=lambda v: levenshtein(p, v)) for p, vs in zip(preds, variants)]
    return ok, cer(preds, nearest)


@torch.no_grad()
def eval_epoch(model, loader, device, alphabet: str, items: list) -> tuple[float, float, float, list]:
    """(loss, прочитано как написано, CER, log_probs по примерам) — items в
    порядке загрузчика (shuffle=False)."""
    model.eval()
    total, n, preds, lps = 0.0, 0, [], []
    for x, targets, lengths, owners in loader:
        x, targets, lengths, owners = x.to(device), targets.to(device), lengths.to(device), owners.to(device)
        lp = model(x)
        total += multi_ctc_loss(lp, targets, lengths, owners, model.blank).item() * x.size(0)
        n += x.size(0)
        lp = lp.cpu()
        for j in range(x.size(0)):
            lps.append(lp[:, j, :])
            preds.append(greedy_decode(lp[:, j, :], alphabet, model.blank))
    ok, err = written_metrics(preds, [it["variants"] for it in items])
    return total / max(n, 1), ok, err, lps


def warm_up(model, args, device) -> None:
    """Разминка на синтетических надписях с точной меткой (synthetic.py):
    модель учится читать цифры, потом на своих кропах сама выбирает
    написание, которое видит."""
    items = synthetic_items(args.synthetic, args.seed)
    dl = DataLoader(AccountDataset(items, args.img_h, args.img_w, augment=True), batch_size=args.batch_size,
                    collate_fn=collate, num_workers=args.num_workers, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.synthetic_epochs * len(dl)), pct_start=0.1)
    for epoch in range(1, args.synthetic_epochs + 1):
        loss = train_epoch(model, dl, opt, sched, device)
        print(f"Разминка на синтетике {epoch:3}/{args.synthetic_epochs} | loss {loss:.4f}")


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

    items = load_dataset(args.images_dir)
    train_items, val_items, test_items = split(items, C.TRAIN_RATIO, C.VAL_RATIO, args.seed)
    parts = {"train": train_items, "val": val_items, "test": test_items}
    try:
        require_parts(parts, ("train", "val", "test"))
    except ValueError as e:
        print(f"ОШИБКА: {e}")
        return 1
    print(f"Обучение {len(train_items)} | проверка {len(val_items)} | тест {len(test_items)} "
          f"(счетов: {len({i['account'] for i in train_items})} / {len({i['account'] for i in val_items})} / "
          f"{len({i['account'] for i in test_items})})")

    base = Path(args.images_dir).resolve().parent
    files = {k: [Path(i["file"]).resolve() for i in v] for k, v in parts.items()}
    write_split_lists(out, files, base)
    write_run_info(out, base, [f for v in files.values() for f in v], files, args.seed, vars(args))

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
    if not args.finetune and args.synthetic > 0:
        warm_up(model, args, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(train_dl)), pct_start=0.1)
    preprocess = {"img_h": args.img_h, "img_w": args.img_w}

    best, best_key, stale, history = None, (-1.0, 0.0), 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_dl, opt, sched, device)
        val_loss, val_acc, val_cer, _ = eval_epoch(model, val_dl, device, C.ALPHABET, val_items)
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
              f"прочитано как написано {val_acc:.4f} | CER {val_cer:.4f}{mark}")
        if stale >= args.patience:
            print(f"Ранняя остановка: {args.patience} эпох без улучшения")
            break
    print(f"Обучение: {time.time() - t0:.0f} с, лучшая эпоха {best}")

    model, _ = load_checkpoint(out / "best.pt", device)
    test_lps = eval_epoch(model, test_dl, device, C.ALPHABET, test_items)[3]
    all_accounts = {i["account"]: i["variants"] for i in items}      # словарь — все счета датасета
    res = evaluate(test_lps, [i["account"] for i in test_items], C.ALPHABET, model.blank, all_accounts)
    text = report_text(res, "Тест (словарь — счета датасета; как в работе — evaluate_account.py --table)")
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
