"""
Кропы на проверку (gmr.py datasets --review / --review-done,
tools/dataset_review.py): модель не согласна с ответом или не уверена —
кроп переносится в <датасет>/проверить/, после просмотра — обратно.
«Модель» — подделка: что прочитано, задаётся цветом картинки.
"""
import csv
from pathlib import Path

import numpy as np
import pytest

import gmr
from src.gmr.render.image_io import write_image
from tools import dataset_review as rv
from tools import datasets as check_ds

# цвет картинки → (что прочитала модель, уверенность)
READS = {10: ("0070351", 0.99), 20: ("0070357", 0.98), 30: ("0070351", 0.5), 40: ("1234567", 0.95),
         50: ("93", 0.97), 60: ("0095", 0.99), 70: ("7", 0.99), 80: ("3", 0.99), 90: ("7", 0.4)}


def read(img):
    return READS[int(img[0, 0, 0])]


def _img(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_image(path, np.full((10, 20, 3), value, np.uint8))
    return path


def _datasets(root: Path):
    s = root / "serials_crnn" / "images"
    _img(s / "1300000001__serial1__0070351.jpg", 10)                # согласна
    _img(s / "1300000002__serial1__0070351.jpg", 20)                # прочитала другое
    _img(s / "1300000003__serial1__0070351.jpg", 30)                # то же, но неуверенно
    _img(s / "старое.jpg", 40)                                      # не по стандарту — не трогается
    a = root / "accounts_crnn" / "images"
    _img(a / "1300000093__account1.jpg", 50)                        # «93» — написание счёта
    _img(a / "1300000094__account1.jpg", 60)                        # «0095» — не его счёт
    d = root / "digits_cnn"
    _img(d / "7" / "1300000001__digit1.jpg", 70)
    _img(d / "7" / "1300000002__digit1.jpg", 80)                    # модель видит 3
    _img(d / "7" / "1300000003__digit1.jpg", 90)                    # неуверенно
    (d / "3").mkdir()


def _report(folder: Path) -> list[dict]:
    with open(folder / rv.REPORT, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=";"))


def test_review_serials_and_back(tmp_path):
    root = tmp_path / "ds"
    _datasets(root)
    (root / "serials_crnn" / "images" / "1300000009__serial1__1234567.jpg").write_bytes(b"x")   # битая
    out = rv.review(root, "serials", read=read)
    assert out.splitlines()[0].startswith("Модель: serial_id_ocr/") and "из настроек программы" in out
    assert out.splitlines()[1:3] == [
        "Проверка serials_crnn: кропов 4, на проверку — 2 (не совпадает — 1, не уверена (< 0.9) — 1)",
        "Не открылись как картинка (оставлены): 1"]
    review = root / "serials_crnn" / "проверить"
    assert sorted(p.name for p in review.glob("*.jpg")) == ["1300000002__serial1__0070351.jpg",
                                                           "1300000003__serial1__0070351.jpg"]
    assert sorted(p.name for p in (root / "serials_crnn" / "images").iterdir()) == [
        "1300000001__serial1__0070351.jpg", "1300000009__serial1__1234567.jpg", "старое.jpg"]
    rows = _report(review)
    assert rows[0] == {"кроп": "1300000003__serial1__0070351.jpg", "ответ": "0070351", "модель": "0070351",
                       "уверенность": "0,500", "почему": "не уверена"}            # самые неуверенные — сверху
    assert rows[1]["модель"] == "0070357" and rows[1]["почему"] == "не совпадает"
    text = check_ds.run(root)
    assert "⚠ на проверке (проверить\\, обучение не читает): 2" in text
    with pytest.raises(ValueError, match="ещё лежат кропы прошлой проверки"):
        rv.review(root, "serials", read=read)
    # проверил: один номер был неверный — переименовал
    (review / "1300000002__serial1__0070351.jpg").rename(review / "1300000002__serial1__0070357.jpg")
    assert rv.done(root, "serials") == f"Возвращено в {root / 'serials_crnn'}: 2"
    assert not review.exists()
    assert sorted(p.name for p in (root / "serials_crnn" / "images").iterdir()) == [
        "1300000001__serial1__0070351.jpg", "1300000002__serial1__0070357.jpg",
        "1300000003__serial1__0070351.jpg", "1300000009__serial1__1234567.jpg", "старое.jpg"]
    with pytest.raises(ValueError, match="проверка не запускалась"):
        rv.done(root, "serials")


def test_review_accounts_any_written_form(tmp_path):
    root = tmp_path / "ds"
    _datasets(root)
    out = rv.review(root, "accounts", read=read)
    assert "кропов 2, на проверку — 1 (не совпадает — 1" in out
    assert [p.name for p in (root / "accounts_crnn" / "проверить").glob("*.jpg")] == ["1300000094__account1.jpg"]


def test_review_digits_by_folder(tmp_path):
    root = tmp_path / "ds"
    _datasets(root)
    out = rv.review(root, "digits", read=read)
    assert "кропов 3, на проверку — 2 (не совпадает — 1, не уверена (< 0.9) — 1)" in out
    review = root / "digits_cnn" / "проверить"
    assert sorted(p.relative_to(review).as_posix() for p in review.rglob("*.jpg")) == [
        "7/1300000002__digit1.jpg", "7/1300000003__digit1.jpg"]           # ответ — папка
    assert "лишние папки" not in check_ds.run(root)                       # проверить\ — не лишняя
    # проверил: это тройка — перенёс в проверить\3\; ещё один кроп положил в проверить\ без папки
    (review / "3").mkdir()
    (review / "7" / "1300000002__digit1.jpg").rename(review / "3" / "1300000002__digit1.jpg")
    _img(review / "без_папки.jpg", 70)
    _img(review / "мусор" / "y.jpg", 70)
    _img(root / "digits_cnn" / "7" / "занято.jpg", 70)
    _img(review / "7" / "занято.jpg", 70)
    out = rv.done(root, "digits")
    assert out == f"Возвращено в {root / 'digits_cnn'}: 2; осталось в {review} (имя уже занято или цифра не в папке 0–9): 3"
    assert not (root / "digits_cnn" / "мусор").exists()
    assert (root / "digits_cnn" / "3" / "1300000002__digit1.jpg").is_file()
    assert (root / "digits_cnn" / "7" / "1300000003__digit1.jpg").is_file()
    assert review.is_dir()                                                  # не всё вернулось — папка остаётся


def test_review_conf_and_empty(tmp_path):
    root = tmp_path / "ds"
    _datasets(root)
    out = rv.review(root, "serials", 0.3, weights="w/best.pt", read=read)
    assert "на проверку — 1 (не совпадает — 1, не уверена (< 0.3) — 0)" in out
    assert out.splitlines()[0] == "Модель: w/best.pt"
    with pytest.raises(ValueError, match="нет кропов с ответом"):
        rv.review(tmp_path / "пусто", "serials", read=read)
    assert rv.agrees("accounts", "1300000093", "13-0093") is False         # черта не читается моделью
    assert rv.agrees("accounts", "1300000093", "130093") and rv.agrees("serials", "1", "1")


def test_review_command(tmp_path, monkeypatch, capsys):
    root = tmp_path / "ds"
    _datasets(root)
    monkeypatch.setattr(rv, "load_reader", lambda kind, weights=None: read)
    assert gmr.main(["datasets", "--root", str(root), "--review", "serials", "--conf", "0.3"]) == 0
    assert "на проверку — 1" in capsys.readouterr().out
    assert gmr.main(["datasets", "--root", str(root), "--review", "serials"]) == 1
    assert "ОШИБКА: в " in capsys.readouterr().out
    assert gmr.main(["datasets", "--root", str(root), "--review-done", "serials"]) == 0
    assert "Возвращено" in capsys.readouterr().out


def test_load_reader_account_needs_weights():
    with pytest.raises(ValueError, match="модель надписи не подключена"):
        rv.load_reader("accounts")


def test_load_reader_uses_weights(monkeypatch):
    from types import SimpleNamespace

    from src.gmr.ml import loader
    seen = []
    monkeypatch.setattr(loader, "load_serial_recognizer", lambda cfg: seen.append(cfg.serial_ocr_model) or
                        SimpleNamespace(recognize=lambda img: SimpleNamespace(text="12345", confidence=0.7)))
    monkeypatch.setattr(loader, "load_digit_recognizer", lambda cfg: seen.append(cfg.digit_ocr_model) or
                        SimpleNamespace(recognize=lambda img: SimpleNamespace(digit="4", confidence=0.8)))
    monkeypatch.setattr(loader, "load_account_recognizer", lambda cfg: seen.append(cfg.account_ocr_model) or
                        SimpleNamespace(recognize=lambda img: SimpleNamespace(text="93", text_conf=0.9)))
    assert rv.load_reader("serials", "s.pt")(None) == ("12345", 0.7)
    assert rv.load_reader("digits")(None) == ("4", 0.8)
    assert rv.load_reader("accounts", "a.pt")(None) == ("93", 0.9)
    from src.gmr.domain import PipelineConfig
    assert seen == ["s.pt", PipelineConfig().digit_ocr_model, "a.pt"]
