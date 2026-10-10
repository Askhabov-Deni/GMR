"""
Оценка модели лицевого счёта по надписи маркером.

  python models/account/evaluate_account.py --run_dir account_ocr/runs/crnn/<дата-время>
  python models/account/evaluate_account.py --run_dir … --table <таблица компании .xls/.xlsx/.csv>

Тест — те же файлы, что отложило обучение (split/test.txt в папке результата).

Что считается (eval/metrics.json, eval/report.txt):
  - «как написано»: жадное чтение (без таблицы) — одно из написаний счёта
    из имени кропа (account_match.account_variants: 93, 0093, …), и CER до
    ближайшего написания;
  - «счёт по таблице»: модель выбирает счёт из словаря. Без --table словарь
    — все счета датасета; с --table — все счета таблицы, как в работе;
  - порог → «принято / ошибок среди принятых»: сколько фото программа
    решила бы сама и сколько из них неверно. По этой таблице выбирается
    PipelineConfig.account_conf_thresh: самый низкий порог, при котором
    ошибок среди принятых 0 (или столько, сколько вы готовы терпеть);
  - eval/suspect_labels.txt — кропы всего датасета, где модель уверенно
    выбирает другой счёт, чем в имени файла: часто это неверный кроп
    (не тот счёт, не надпись) — проверьте глазами, переименуйте или удалите;
  - eval/errors/ — картинки ошибок теста (папка очищается при каждой оценке):
    <счёт>__прочитано_<что>__выбран_<счёт по словарю>__<файл>;
  - если счетов теста нет в словаре (таблица не того участка), отчёт об этом
    предупреждает: такие кропы всегда «неверно».
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

try:
    from .config_account import LEN_WINDOW
    from .dataset_account import load_dataset, prepare_input, read_gray
    from .model_account import load_checkpoint
    from ..ctc_lexicon import CompiledLexicon, greedy_decode, match
    from ..crnn.metrics_crnn import cer, levenshtein
except ImportError:  # запуск как отдельный скрипт: нужна папка проекта в sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from models.account.config_account import LEN_WINDOW
    from models.account.dataset_account import load_dataset, prepare_input, read_gray
    from models.account.model_account import load_checkpoint
    from models.ctc_lexicon import CompiledLexicon, greedy_decode, match
    from models.crnn.metrics_crnn import cer, levenshtein
from src.gmr.domain.account_match import account_groups, account_variants

THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
SUSPECT_CONF = 0.95


@torch.no_grad()
def log_probs_for(model, meta: dict, files: Sequence[Path], device="cpu", batch_size: int = 64) -> list[torch.Tensor]:
    """(T, C) на каждый файл — ровно тот же вход, что в работе (prepare_input)."""
    h, w = meta["preprocess"]["img_h"], meta["preprocess"]["img_w"]
    out = []
    for i in range(0, len(files), batch_size):
        x = torch.from_numpy(np.stack([prepare_input(read_gray(f), h, w) for f in files[i:i + batch_size]]))
        lp = model(x.to(device)).cpu()
        out += [lp[:, j, :] for j in range(lp.shape[1])]
    return out


def evaluate(log_probs: Sequence[torch.Tensor], accounts: Sequence[str], alphabet: str, blank: int,
             groups: Optional[Mapping[str, Sequence[str]]] = None) -> dict:
    """Метрики по готовым выходам модели. accounts — верный счёт каждого
    кропа (из имени); groups — словарь {счёт: написания}, None — счета из
    accounts (оптимистично: словарь мал — лучше все счета датасета или
    таблица). Верно — модель выбрала тот самый счёт."""
    greedy = [greedy_decode(lp, alphabet, blank) for lp in log_probs]
    written = [account_variants(a) for a in accounts]
    nearest = [min(vs, key=lambda v: levenshtein(g, v)) if vs else "" for g, vs in zip(greedy, written)]
    res = {"n": len(accounts), "exact_match": sum(g in vs for g, vs in zip(greedy, written)) / max(len(accounts), 1),
           "cer": cer(greedy, nearest)}
    groups = groups if groups is not None else account_groups(accounts)
    res["not_in_dictionary"] = sum(a not in groups for a in accounts)
    lexicon = CompiledLexicon(groups, alphabet)
    picks = []
    for lp, account, g in zip(log_probs, accounts, greedy):
        m = match(lp, lexicon, blank, len_window=LEN_WINDOW)
        picks.append({"account": account, "greedy": g, "group": m.group, "string": m.string,
                      "confidence": m.confidence, "ok": m.group == account})
    n = max(len(picks), 1)
    res["lexicon_size"] = len(lexicon)
    res["lexicon_top1"] = sum(p["ok"] for p in picks) / n
    res["by_threshold"] = []
    for t in THRESHOLDS:
        taken = [p for p in picks if p["confidence"] >= t]
        wrong = sum(not p["ok"] for p in taken)
        res["by_threshold"].append({"threshold": t, "accepted": len(taken) / n, "accepted_n": len(taken),
                                    "wrong_n": wrong, "wrong_share": wrong / len(taken) if taken else 0.0})
    res["picks"] = picks
    return res


def report_text(res: dict, title: str) -> str:
    lines = [title,
             f"  примеров: {res['n']}",
             f"  как написано (без таблицы): прочитано верно {res['exact_match']:.3f}, CER {res['cer']:.3f}",
             f"  счёт по словарю из {res['lexicon_size']}: верно {res['lexicon_top1']:.3f}",
             "  порог  принято        ошибок среди принятых"]
    for r in res["by_threshold"]:
        lines.append(f"  {r['threshold']:<5}  {r['accepted']:6.1%} ({r['accepted_n']:>4})   "
                     f"{r['wrong_n']} ({r['wrong_share']:.2%})")
    missing = res.get("not_in_dictionary", 0)
    if missing:
        lines.append(f"  ⚠ счетов теста нет в словаре: {missing} из {res['n']} — их кропы всегда «неверно»"
                     + (". Похоже, таблица не того участка: возьмите таблицу, из которой эти фото"
                        if missing * 2 > res["n"] else ""))
    return "\n".join(lines)


def table_groups(table: Path) -> dict:
    """{счёт: написания} по таблице компании — как в reader.py."""
    from src.gmr.domain import PipelineConfig
    from src.gmr.storage.register import read_register
    col = PipelineConfig().col_account_id
    reg = read_register(str(table))
    if col not in reg.columns:
        raise ValueError(f"в таблице нет столбца «{col}»")
    return account_groups(r.get(col, "") for r in reg.rows)


def _split_files(run_dir: Path, part: str, base: Path) -> list[Path]:
    path = run_dir / "split" / f"{part}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"нет {path}: это папка результата train_account.py?")
    return [base / line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Оценка модели лицевого счёта по надписи маркером")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--table", help="таблица компании: словарь — все её счета, как в работе")
    p.add_argument("--max_errors", type=int, default=100, help="сколько картинок ошибок сохранить")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    base = Path(info["dataset"])
    targs = info["args"]
    model, meta = load_checkpoint(run_dir / "best.pt")
    items = load_dataset(targs["images_dir"], verbose=False)
    by_file = {Path(it["file"]).resolve(): it["account"] for it in items}

    test_files = [f for f in _split_files(run_dir, "test", base) if f.resolve() in by_file]
    test_accounts = [by_file[f.resolve()] for f in test_files]
    all_accounts = account_groups(it["account"] for it in items)             # все счета датасета
    groups = table_groups(Path(args.table)) if args.table else all_accounts
    lp = log_probs_for(model, meta, test_files)
    res = evaluate(lp, test_accounts, meta["alphabet"], model.blank, groups)

    out = run_dir / "eval"
    (out / "errors").mkdir(parents=True, exist_ok=True)
    for old in (out / "errors").iterdir():                  # прошлая оценка (другой словарь) — не смешивать
        if old.is_file():
            old.unlink()
    title = f"Тест ({'словарь — таблица ' + Path(args.table).name if args.table else 'словарь — счета датасета'})"
    text = report_text(res, title)

    # ошибки теста
    rows, saved = [], 0
    for f, pick in zip(test_files, res["picks"]):
        rows.append(f"{pick['account']:>12} {pick['greedy']:>12} {str(pick['group']):>12} "
                    f"{pick['confidence']:.3f} {'✓' if pick['ok'] else '✗'}  {f.name}")
        if not pick["ok"] and saved < args.max_errors:
            data = f.read_bytes()
            name = (f"{pick['account']}__прочитано_{pick['greedy'] or 'пусто'}__"
                    f"выбран_{pick['group'] or 'нет'}__{f.name}")
            (out / "errors" / name).write_bytes(data)
            saved += 1
    (out / "predictions_test.txt").write_text(
        f"{'счёт':>12} {'прочитано':>12} {'по словарю':>12} увер.  файл\n" + "\n".join(rows) + "\n",
        encoding="utf-8")

    # подозрительные кропы во всём датасете (словарь — счета датасета)
    all_files = [Path(it["file"]) for it in items]
    all_res = evaluate(log_probs_for(model, meta, all_files), [it["account"] for it in items],
                       meta["alphabet"], model.blank, all_accounts)
    suspects = [f"{pk['account']} → модель: {pk['group']} (прочитано {pk['greedy'] or 'пусто'}, "
                f"{pk['confidence']:.2f})  {f.name}"
                for f, pk in zip(all_files, all_res["picks"])
                if not pk["ok"] and pk["confidence"] >= SUSPECT_CONF]
    (out / "suspect_labels.txt").write_text(
        "Счёт в имени → какой счёт уверенно видит модель. Часто это неверный кроп (не тот счёт,\n"
        "не надпись) — проверьте глазами, переименуйте или удалите.\n"
        "(На кропах из обучения модель могла запомнить и неверный счёт: список неполный.)\n\n"
        + "\n".join(suspects) + "\n", encoding="utf-8")
    text += f"\n\nПодозрительных меток во всём датасете: {len(suspects)} — {out / 'suspect_labels.txt'}"

    metrics = {k: v for k, v in res.items() if k != "picks"}
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nПодробно: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
