"""
Обучение модели надписи маркером без разметки (решение владельца
2026-10-12): кроп «<лицевой счёт>__….jpg», что именно написано (93, 0093,
13-0093, …) — неизвестно; модель учится на любом из написаний счёта
(model_account.multi_ctc_loss), словарь в работе — все написания
(account_match.account_variants).
"""
import math

import cv2
import numpy as np
import pytest
import torch

from models.account import dataset_account as ds
from models.account.model_account import multi_ctc_loss
from models.ctc_lexicon import CompiledLexicon, match, string_log_likelihoods
from src.gmr.domain.account_match import account_groups, account_variants
from tests.test_account_model import BLANK, DIGITS, peaky


def _t(s):
    return torch.tensor([int(c) for c in s], dtype=torch.long)


def _loss(lps, variants):
    """multi_ctc_loss для пачки: lps — (T, C) на картинку, variants — строки на картинку."""
    flat = [_t(v) for vs in variants for v in vs]
    owners = torch.tensor([i for i, vs in enumerate(variants) for _ in vs])
    return multi_ctc_loss(torch.stack(lps, 1), torch.cat(flat), torch.tensor([len(t) for t in flat]),
                          owners, BLANK)


def test_multi_ctc_loss_is_minus_log_of_summed_probabilities():
    lps = [peaky("00093"), peaky("00082")]                      # одинаковое T
    variants = [("93", "093", "0093", "00093"), ("82", "00082")]
    expected = []
    for lp, vs in zip(lps, variants):
        ll = string_log_likelihoods(lp, [_t(v) for v in vs], BLANK)
        expected.append(-float(torch.logsumexp(ll, 0)))
    mean_len = sum(len(v) for vs in variants for v in vs) / 6
    assert float(_loss(lps, variants)) == pytest.approx(sum(expected) / 2 / mean_len, rel=1e-4)


def test_multi_ctc_loss_does_not_punish_unknown_writing():
    lp = peaky("0093")
    all_ways = float(_loss([lp], [account_variants("1300000093")]))
    only_written = float(_loss([lp], [("0093",)]))
    only_other = float(_loss([lp], [("93",)]))
    # «0093» на картинке: все написания вместе — почти как одно верное, а «93» — штраф
    norm_all = sum(map(len, account_variants("1300000093"))) / 14
    assert all_ways * norm_all == pytest.approx(only_written * 4, abs=0.05)
    assert only_other > 10 * only_written


def test_multi_ctc_loss_survives_impossible_variant():
    lp = peaky("93", steps_per_char=1)                       # T = 5: «1300000093» не влезает
    loss = _loss([lp], [("93", "1300000093")])
    assert torch.isfinite(loss)
    assert float(loss) * 6 == pytest.approx(float(_loss([lp], [("93",)])) * 2, abs=1e-3)
    assert torch.isfinite(_loss([lp], [("1300000093",)]))   # даже если не влезает ничего


def test_multi_ctc_loss_gradient_flows():
    lp = torch.log_softmax(torch.randn(20, 2, 11), 2).detach().requires_grad_()
    flat = [_t("93"), _t("0093"), _t("82")]
    loss = multi_ctc_loss(lp, torch.cat(flat), torch.tensor([2, 4, 2]), torch.tensor([0, 0, 1]), BLANK)
    loss.backward()
    assert torch.isfinite(lp.grad).all() and lp.grad.abs().sum() > 0


def _jpg(path, text="0093"):
    img = np.full((40, 120, 3), 220, np.uint8)
    cv2.putText(img, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    cv2.imwrite(str(path), img)


def test_dense_groups():
    from models.account.evaluate_account import dense_groups
    g = dense_groups(["1300000093", "1300000090", "1300000095"])
    assert list(g) == [f"13000000{n}" for n in range(90, 96)] and g["1300000093"] == account_variants("1300000093")
    with pytest.raises(ValueError, match="слишком много"):
        dense_groups(["1300000000", "1301000000"])
    with pytest.raises(ValueError):
        dense_groups([])


def test_written_metrics():
    from models.account.train_account import written_metrics
    vs = account_variants("1300000093")
    ok, err = written_metrics(["0093", "93", "5"], [vs, vs, vs])
    assert ok == pytest.approx(2 / 3) and err == pytest.approx(2 / 8)      # «5» → ближе всех «93»


def test_dataset_from_names(tmp_path, capsys):
    images = tmp_path / "images"
    images.mkdir()
    for n in ("1300000093__marker_id_1.jpeg", "1300000093__marker_id_2.jpg", "1300000082 (2).jpg",
              "IMG_1.jpg", "12__x.jpg"):
        _jpg(images / n)
    items = ds.load_dataset(images)
    assert "Загружено: 3  |  в имени нет лицевого счёта: 2" in capsys.readouterr().out
    assert [(it["file"].name, it["account"]) for it in items] == [
        ("1300000082 (2).jpg", "1300000082"), ("1300000093__marker_id_1.jpeg", "1300000093"),
        ("1300000093__marker_id_2.jpg", "1300000093")]
    assert items[1]["variants"] == account_variants("1300000093")
    assert ds.account_of("1300000093__marker_id_1.jpeg") == "1300000093" and ds.account_of("IMG_1.jpg") is None


def test_split_keeps_account_together(tmp_path):
    items = [{"file": tmp_path / f"{a} ({i}).jpg", "account": a, "variants": account_variants(a)}
             for a in (f"13{n:08d}" for n in range(1, 200)) for i in range(3)]       # имена без «__»
    tr, va, te = ds.split(items, 0.8, 0.1, 42)
    part = {}
    for name, rows in (("tr", tr), ("va", va), ("te", te)):
        for it in rows:
            assert part.setdefault(it["account"], name) == name
    assert tr and va and te


def test_dataset_items_and_collate(tmp_path):
    _jpg(tmp_path / "1300000093__a.jpg")
    _jpg(tmp_path / "1312345678__b.jpg", "12345678")
    items = ds.load_dataset(tmp_path, verbose=False)
    data = ds.AccountDataset(items, 48, 224)
    x, targets = data[0]
    assert x.shape == (1, 48, 224) and len(targets) == 14 and targets[0].tolist() == [9, 3]
    xs, flat, lengths, owners = ds.collate([data[0], data[1]])
    assert xs.shape == (2, 1, 48, 224)
    assert owners.tolist() == [0] * 14 + [1] * 2 and lengths.tolist()[-2:] == [8, 10]
    assert flat.shape[0] == int(lengths.sum())


def test_lexicon_length_window_same_answer_faster():
    groups = account_groups(f"13{n:08d}" for n in range(1, 400))
    lexicon = CompiledLexicon(groups, DIGITS)
    lp = peaky("0093")
    full, near = match(lp, lexicon, BLANK), match(lp, lexicon, BLANK, len_window=2)
    assert full.group == near.group == "1300000093" and near.string == "0093"
    assert near.confidence == pytest.approx(full.confidence, abs=1e-6)
    unsure = peaky("0093", alt={3: ("4", 0.5)})                      # 93 или 94
    a, b = match(unsure, lexicon, BLANK), match(unsure, lexicon, BLANK, len_window=2)
    assert {a.group, b.group} <= {"1300000093", "1300000094"} and b.confidence == pytest.approx(a.confidence, abs=1e-4)
    far = match(peaky("12345678"), CompiledLexicon({"1300000093": account_variants("1300000093")}, DIGITS),
                BLANK, len_window=1)
    assert far.confidence < 1e-6                                     # ничего похожего по длине


def test_length_window_not_used_when_nothing_read():
    lp = torch.log(torch.tensor([[0.0889] * 10 + [0.111]])).repeat(30, 1)    # чаще всего — пусто
    lexicon = CompiledLexicon({"A": ("93",), "B": ("1234",)}, DIGITS)
    full, near = match(lp, lexicon, BLANK), match(lp, lexicon, BLANK, len_window=1)
    assert full.greedy == near.greedy == ""
    assert near.top == pytest.approx(full.top) and dict(full.top)["B"] > 1e-4      # длинная строка тоже в счёте


def test_inferer_uses_length_window(tmp_path, monkeypatch):
    from models.account import infer_account
    from models.account.model_account import AccountCRNN, save_checkpoint
    save_checkpoint(tmp_path / "best.pt", AccountCRNN(img_h=48), {"img_h": 48, "img_w": 224}, DIGITS)
    seen = []
    real = infer_account.match
    monkeypatch.setattr(infer_account, "match", lambda *a, **k: seen.append(k) or real(*a, **k))
    inf = infer_account.AccountInferer(str(tmp_path / "best.pt"))
    inf.predict(np.zeros((40, 120, 3), np.uint8), account_groups(["1300000093"]))
    assert seen == [{"len_window": 2}]


def test_synthetic_items():
    from models.account.synthetic import synthetic_items, written_text
    items = synthetic_items(200, 1)
    assert [it["variants"] for it in items[:5]] == [it["variants"] for it in synthetic_items(5, 1)]
    for it in items:
        (label,) = it["variants"]
        assert label.isdigit() and 1 <= len(label) <= 10
        assert it["image"].ndim == 2 and it["image"].dtype == np.uint8 and min(it["image"].shape) > 10
    rng = np.random.default_rng(0)
    texts = [written_text(rng) for _ in range(2000)]
    assert any(t.startswith("13-") for t in texts) and any(t.startswith("00") for t in texts)
    assert any(t.isdigit() and len(t) == 10 and t.startswith("13") for t in texts)
    assert any(not t.startswith("0") for t in texts)
    for t in texts:                                                 # каждое — одно из написаний своего счёта
        digits = t.replace("-", "")
        core = digits[2:].lstrip("0") if t.startswith("13-") or len(digits) == 10 else digits.lstrip("0")
        account = "13" + (core or "0").zfill(8)
        assert digits in account_variants(account), t


def test_synthetic_goes_through_dataset():
    from models.account.synthetic import synthetic_items
    x, targets = ds.AccountDataset(synthetic_items(1, 0), 48, 224, augment=True)[0]
    assert x.shape == (1, 48, 224) and len(targets) == 1


class _Stop(Exception):
    pass


def test_warm_up_only_when_training_from_scratch(tmp_path, monkeypatch):
    from models.account import train_account
    from models.account.model_account import AccountCRNN, save_checkpoint
    items = [{"file": tmp_path / f"{a}__1.jpg", "account": a, "variants": account_variants(a)}
             for a in (f"13{n:08d}" for n in range(1, 300))]
    monkeypatch.setattr(train_account, "load_dataset", lambda *a, **k: items)
    monkeypatch.setattr(train_account, "write_run_info", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(train_account, "warm_up", lambda model, args, device: calls.append(args.synthetic))

    def stop(*a, **k):
        raise _Stop
    monkeypatch.setattr(train_account, "train_epoch", stop)
    ckpt = tmp_path / "old.pt"
    save_checkpoint(ckpt, AccountCRNN(img_h=48), {"img_h": 48, "img_w": 224}, DIGITS)
    for k, extra in enumerate(([], ["--synthetic", "0"], ["--finetune", str(ckpt)])):
        with pytest.raises(_Stop):
            train_account.main(["--images_dir", str(tmp_path), "--output_dir", str(tmp_path / f"r{k}"),
                                "--device", "cpu", *extra])
    assert calls == [4000]                                          # только с нуля и по умолчанию


def test_datasets_check(tmp_path):
    from tools import datasets
    root = tmp_path / "accounts_crnn"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for i in range(30):
        _jpg(root / "images" / f"13{i:08d}__marker_id_1.jpg", str(i))
    _jpg(root / "images" / "photo.jpg")
    rep = datasets.Report()
    datasets.check_account(root, rep)
    text = "\n".join(rep.lines + rep.problems)
    assert "кропов: 31, с лицевым счётом в имени: 30, разных счетов: 30" in text
    assert "в имени нет лицевого счёта" in text and "photo.jpg" in text
    assert "labels\\ не нужна" in text and "деление по счёту: обучение" in text


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_train_account_smoke(tmp_path):
    from models.account import train_account
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(0)
    for i in range(60):
        a = f"13{int(rng.integers(1, 99)):08d}"
        _jpg(images / f"{a}__marker_id_{i}.jpg", a[-4:])
    out = tmp_path / "run"
    seen = []
    real = train_account.synthetic_items
    train_account.synthetic_items = lambda n, seed: seen.append((n, seed)) or real(n, seed)
    try:
        assert train_account.main(["--images_dir", str(images), "--output_dir", str(out), "--epochs", "1",
                                   "--device", "cpu", "--batch_size", "16", "--synthetic", "32",
                                   "--synthetic_epochs", "1"]) == 0
    finally:
        train_account.synthetic_items = real
    assert seen == [(32, 42)]                                       # разминка была
    assert (out / "best.pt").is_file() and "счёт по словарю" in (out / "eval" / "report.txt").read_text(
        encoding="utf-8")
    import json
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    assert "labels_dir" not in info["args"] and math.isfinite(
        json.loads((out / "history.json").read_text(encoding="utf-8"))["epochs"][0]["train_loss"])

    # оценка с таблицей другого участка: предупреждение, папка ошибок — только эта оценка
    from models.account import evaluate_account
    (out / "eval" / "errors").mkdir(exist_ok=True)
    (out / "eval" / "errors" / "старое.jpg").write_bytes(b"x")
    table = tmp_path / "t.csv"
    table.write_text("Лицевой счет;Номер счетчика\n1300026377;1\n1300015718;2\n", encoding="utf-8")
    assert evaluate_account.main(["--run_dir", str(out), "--table", str(table)]) == 0
    report = (out / "eval" / "report.txt").read_text(encoding="utf-8")
    assert "Похоже, таблица не того участка" in report
    errors = sorted(p.name for p in (out / "eval" / "errors").iterdir())
    assert "старое.jpg" not in errors and errors and all("__прочитано_" in e and "__выбран_" in e for e in errors)

    # таблицы нет: все номера подряд
    assert evaluate_account.main(["--run_dir", str(out), "--dense"]) == 0
    report = (out / "eval" / "report.txt").read_text(encoding="utf-8")
    assert "словарь — все номера подряд" in report and "нет в словаре" not in report
    with pytest.raises(SystemExit):
        evaluate_account.main(["--run_dir", str(out), "--dense", "--table", str(table)])

    # другие кропы (новый месяц) вместо теста обучения
    month = tmp_path / "month_crops"
    month.mkdir()
    for i, a in enumerate(("1300026377", "1300015718", "1300015718")):
        _jpg(month / f"{a}__marker_id_{i}.jpg", a[-5:])
    _jpg(month / "photo.jpg")
    assert evaluate_account.main(["--run_dir", str(out), "--images", str(month), "--table", str(table)]) == 0
    report = (out / "eval" / "report.txt").read_text(encoding="utf-8")
    assert f"Кропы {month}" in report and "примеров: 3" in report and "нет в словаре" not in report
    assert "Подозрительных меток в этих кропах" in report
    (month / "1300026377__marker_id_0.jpg").unlink()
    for f in month.glob("1300015718*"):
        f.unlink()
    assert evaluate_account.main(["--run_dir", str(out), "--images", str(month)]) == 1
