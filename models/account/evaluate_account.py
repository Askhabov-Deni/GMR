"""
Оценка модели лицевого счёта по надписи маркером.

  python models/account/evaluate_account.py --run_dir account_ocr/runs/crnn/<дата-время>
  python models/account/evaluate_account.py --run_dir … --table <таблица компании .xls/.xlsx/.csv>

Тест — те же файлы, что отложило обучение (split/test.txt в папке результата).

Что считается (eval/metrics.json, eval/report.txt):
  - «как написано»: весь номер верно (жадное чтение, без таблицы) и CER;
  - «счёт по таблице»: модель выбирает счёт из словаря. Без --table словарь
    — все номера датасета (каждый — свой «счёт»); с --table — все счета
    таблицы, как в работе (варианты записи — account_match.account_variants);
  - порог → «принято / ошибок среди принятых»: сколько фото программа
    решила бы сама и сколько из них неверно. По этой таблице выбирается
    PipelineConfig.account_conf_thresh: самый низкий порог, при котором
    ошибок среди принятых 0 (или столько, сколько вы готовы терпеть);
  - eval/suspect_labels.txt — кропы всего датасета, где модель уверенно
    читает не то, что в метке: часто это ошибка разметки, проверьте глазами;
  - eval/errors/ — картинки ошибок теста.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

try:
    from .dataset_account import load_dataset, prepare_input, read_gray
    from .model_account import load_checkpoint
    from ..ctc_lexicon import CompiledLexicon, greedy_decode, match
    from ..crnn.metrics_crnn import cer, exact_match
except ImportError:  # запуск как отдельный скрипт: нужна папка проекта в sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from models.account.dataset_account import load_dataset, prepare_input, read_gray
    from models.account.model_account import load_checkpoint
    from models.ctc_lexicon import CompiledLexicon, greedy_decode, match
    from models.crnn.metrics_crnn import cer, exact_match

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


def evaluate(log_probs: Sequence[torch.Tensor], labels: Sequence[str], alphabet: str, blank: int,
             groups: Optional[Mapping[str, Sequence[str]]] = None) -> dict:
    """Метрики по готовым выходам модели. groups — словарь {счёт: варианты};
    None — каждый номер из labels сам себе счёт (оптимистично: словарь мал —
    лучше передать все номера датасета или таблицу). Верно — если в вариантах
    выбранного счёта есть метка."""
    greedy = [greedy_decode(lp, alphabet, blank) for lp in log_probs]
    res = {"n": len(labels), "exact_match": exact_match(greedy, list(labels)), "cer": cer(greedy, list(labels))}
    groups = groups if groups is not None else {lab: (lab,) for lab in dict.fromkeys(labels)}
    lexicon = CompiledLexicon(groups, alphabet)
    picks = []
    for lp, lab, g in zip(log_probs, labels, greedy):
        m = match(lp, lexicon, blank)
        ok = m.group is not None and lab in groups.get(m.group, ())
        picks.append({"label": lab, "greedy": g, "group": m.group, "string": m.string,
                      "confidence": m.confidence, "ok": ok})
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
             f"  как написано (без таблицы): весь номер верно {res['exact_match']:.3f}, CER {res['cer']:.3f}",
             f"  счёт по словарю из {res['lexicon_size']}: верно {res['lexicon_top1']:.3f}",
             "  порог  принято        ошибок среди принятых"]
    for r in res["by_threshold"]:
        lines.append(f"  {r['threshold']:<5}  {r['accepted']:6.1%} ({r['accepted_n']:>4})   "
                     f"{r['wrong_n']} ({r['wrong_share']:.2%})")
    return "\n".join(lines)


def table_groups(table: Path, digits: int) -> dict:
    """{счёт: варианты} по таблице компании — как в reader.py."""
    from src.gmr.domain import PipelineConfig
    from src.gmr.domain.account_match import account_groups
    from src.gmr.storage.register import read_register
    col = PipelineConfig().col_account_id
    reg = read_register(str(table))
    if col not in reg.columns:
        raise ValueError(f"в таблице нет столбца «{col}»")
    return account_groups((r.get(col, "") for r in reg.rows), digits)


def _split_files(run_dir: Path, part: str, base: Path) -> list[Path]:
    path = run_dir / "split" / f"{part}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"нет {path}: это папка результата train_account.py?")
    return [base / line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None) -> int:
    from src.gmr.domain import PipelineConfig
    p = argparse.ArgumentParser(description="Оценка модели лицевого счёта по надписи маркером")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--table", help="таблица компании: словарь — все её счета, как в работе")
    p.add_argument("--digits", type=int, default=PipelineConfig().account_marker_digits,
                   help="сколько последних цифр счёта пишут на счётчике")
    p.add_argument("--max_errors", type=int, default=100, help="сколько картинок ошибок сохранить")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    base = Path(info["dataset"])
    targs = info["args"]
    model, meta = load_checkpoint(run_dir / "best.pt")
    items = load_dataset(targs["images_dir"], targs["labels_dir"], verbose=False)
    by_file = {Path(it["file"]).resolve(): it["label"] for it in items}

    test_files = [f for f in _split_files(run_dir, "test", base) if f.resolve() in by_file]
    test_labels = [by_file[f.resolve()] for f in test_files]
    groups = (table_groups(Path(args.table), args.digits) if args.table
              else {it["label"]: (it["label"],) for it in items})      # все номера датасета
    lp = log_probs_for(model, meta, test_files)
    res = evaluate(lp, test_labels, meta["alphabet"], model.blank, groups)

    out = run_dir / "eval"
    (out / "errors").mkdir(parents=True, exist_ok=True)
    title = f"Тест ({'словарь — таблица ' + Path(args.table).name if args.table else 'словарь — номера датасета'})"
    text = report_text(res, title)

    # ошибки теста
    rows, saved = [], 0
    for f, pick in zip(test_files, res["picks"]):
        rows.append(f"{pick['label']:>12} {pick['greedy']:>12} {str(pick['string']):>12} "
                    f"{pick['confidence']:.3f} {'✓' if pick['ok'] else '✗'}  {f.name}")
        if (not pick["ok"] or pick["greedy"] != pick["label"]) and saved < args.max_errors:
            data = f.read_bytes()
            (out / "errors" / f"{pick['label']}__read_{pick['greedy'] or 'пусто'}__{f.name}").write_bytes(data)
            saved += 1
    (out / "predictions_test.txt").write_text(
        f"{'метка':>12} {'прочитано':>12} {'по словарю':>12} увер.  файл\n" + "\n".join(rows) + "\n",
        encoding="utf-8")

    # подозрительные метки во всём датасете (словарь — номера датасета)
    all_files = [Path(it["file"]) for it in items]
    all_labels = [it["label"] for it in items]
    all_res = evaluate(log_probs_for(model, meta, all_files), all_labels, meta["alphabet"], model.blank)
    suspects = [f"{pk['label']} → модель: {pk['string'] or pk['greedy']} ({pk['confidence']:.2f})  {f.name}"
                for f, pk in zip(all_files, all_res["picks"])
                if not pk["ok"] and pk["confidence"] >= SUSPECT_CONF]
    (out / "suspect_labels.txt").write_text(
        "Метка → что уверенно читает модель. Часто это ошибка разметки — проверьте кроп.\n"
        "(На кропах из обучения модель могла запомнить и неверную метку: список неполный.)\n\n"
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
