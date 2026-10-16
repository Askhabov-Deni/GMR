"""
tools/dataset_review.py — кропы на проверку глазами (шаг 3 стандарта
датасетов, docs/DATASETS.md; решение владельца 2026-10-15).

  python gmr.py datasets --review serials|accounts|digits [--conf 0.9] [--weights <best>]
  python gmr.py datasets --review-done serials|accounts|digits

Модель читает каждый кроп датасета. Где она не согласна с ответом (имя
файла, у цифр — папка) или не уверена (ниже --conf), кроп ПЕРЕНОСИТСЯ в
<датасет>/проверить/ (у цифр — проверить/<цифра>/): обучение эту папку не
читает. Там вы листаете картинки: неверный ответ — переименуйте файл (цифру —
перенесите в папку нужной цифры внутри проверить/), мусор — удалите.
--review-done возвращает всё из проверить/ в датасет. Что прочитала модель —
проверить/проверить.csv (самые неуверенные — сверху).

Модель, которая училась на этих же кропах, часто «соглашается» и с неверным
ответом — поэтому несогласие на них особенно подозрительно, а пропустить
ошибку она может. Работу программы не меняет.
"""
import csv
import shutil
from pathlib import Path
from typing import Callable, Optional

from models.datasets import DIGITS, images_in
from src.gmr.domain import PipelineConfig
from src.gmr.domain.account_match import account_variants
from src.gmr.render.image_io import read_image

REVIEW = "проверить"
REPORT = "проверить.csv"
FOLDERS = {"serials": "serials_crnn", "accounts": "accounts_crnn", "digits": "digits_cnn"}


def items(root: Path, kind: str) -> list[tuple[Path, str]]:
    """(кроп, ответ) датасета: у серийника и надписи — из имени, у цифры — папка."""
    ds = Path(root) / FOLDERS[kind]
    if kind == "digits":
        return [(p, d) for d in DIGITS for p in images_in(ds / d)]
    if kind == "serials":
        from models.crnn.dataset_crnn import label_of
        answer_of = label_of
    else:
        from models.account.dataset_account import account_of
        answer_of = account_of
    return [(p, a) for p in images_in(ds / "images") if (a := answer_of(p.name)) is not None]


def agrees(kind: str, answer: str, read: str) -> bool:
    """Модель согласна с ответом: у надписи — прочитано любое написание счёта."""
    if kind == "accounts":
        return read in account_variants(answer)
    return read == answer


def model_path(kind: str, weights: Optional[str] = None) -> str:
    """Чьи веса читают: --weights или модель из настроек программы."""
    cfg = PipelineConfig()
    default = {"serials": cfg.serial_ocr_model, "digits": cfg.digit_ocr_model,
               "accounts": cfg.account_ocr_model}[kind]
    return weights or default


def load_reader(kind: str, weights: Optional[str] = None) -> Callable:
    """Картинка → (что прочитано, уверенность) моделью из настроек или --weights."""
    from src.gmr.ml import loader
    cfg = PipelineConfig()
    if kind == "serials":
        rec = loader.load_serial_recognizer(PipelineConfig(serial_ocr_model=weights or cfg.serial_ocr_model))
        return lambda img: (lambda p: (p.text, p.confidence))(rec.recognize(img))
    if kind == "digits":
        rec = loader.load_digit_recognizer(PipelineConfig(digit_ocr_model=weights or cfg.digit_ocr_model))
        return lambda img: (lambda p: (p.digit, p.confidence))(rec.recognize(img))
    rec = loader.load_account_recognizer(PipelineConfig(account_ocr_model=weights or cfg.account_ocr_model))
    if rec is None:
        raise ValueError("модель надписи не подключена — укажите её веса: --weights account_ocr\\runs\\crnn\\<…>\\best.pt")
    return lambda img: (lambda p: (p.text, p.text_conf))(rec.recognize(img))


def _review_dir(root: Path, kind: str) -> Path:
    return Path(root) / FOLDERS[kind] / REVIEW


def review(root: Path, kind: str, conf: float = 0.9, weights: Optional[str] = None,
           read: Optional[Callable] = None) -> str:
    out = _review_dir(root, kind)
    if out.is_dir() and any(p.is_file() for p in out.rglob("*") if p.name != REPORT):
        raise ValueError(f"в {out} ещё лежат кропы прошлой проверки — сначала верните их: "
                         f"python gmr.py datasets --review-done {kind}")
    crops = items(root, kind)
    if not crops:
        raise ValueError(f"в {Path(root) / FOLDERS[kind]} нет кропов с ответом по стандарту")
    read = read or load_reader(kind, weights)
    rows, unread = [], 0
    for i, (p, answer) in enumerate(crops, 1):
        img = read_image(p)
        if img is None:
            unread += 1
            continue
        text, c = read(img)
        why = "не совпадает" if not agrees(kind, answer, text) else "не уверена" if c < conf else ""
        if why:
            dst = out / answer / p.name if kind == "digits" else out / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dst))
            rows.append({"кроп": dst.relative_to(out).as_posix(), "ответ": answer, "модель": text,
                         "уверенность": f"{c:.3f}".replace(".", ","), "почему": why, "_c": c})
        if i % 200 == 0 or i == len(crops):
            print(f"Кроп {i} из {len(crops)}", flush=True)
    rows.sort(key=lambda r: r.pop("_c"))
    lines = [f"Модель: {model_path(kind, weights)}" + ("" if weights else " (из настроек программы; другую — --weights)"),
             f"Проверка {FOLDERS[kind]}: кропов {len(crops)}, на проверку — {len(rows)} "
             f"(не совпадает — {sum(r['почему'] == 'не совпадает' for r in rows)}, "
             f"не уверена (< {conf}) — {sum(r['почему'] == 'не уверена' for r in rows)})"]
    if unread:
        lines.append(f"Не открылись как картинка (оставлены): {unread}")
    if rows:
        with open(out / REPORT, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter=";")
            w.writeheader()
            w.writerows(rows)
        lines += [f"Кропы — в {out} (обучение их не читает), что прочитала модель — {out / REPORT}.",
                  "Неверный ответ — переименуйте" + (" (цифру — перенесите в папку нужной цифры внутри "
                                                     "проверить\\)" if kind == "digits" else "")
                  + f", мусор — удалите; потом: python gmr.py datasets --review-done {kind}"]
    return "\n".join(lines)


def done(root: Path, kind: str) -> str:
    """Всё из проверить/ — обратно в датасет (имя уже занято — кроп остаётся)."""
    src = _review_dir(root, kind)
    if not src.is_dir():
        raise ValueError(f"нет {src}: проверка не запускалась или уже возвращена")
    ds = Path(root) / FOLDERS[kind]
    back, left = 0, []
    for p in sorted(f for f in src.rglob("*") if f.is_file() and f.name != REPORT):
        if kind == "digits":
            digit = p.parent.name if p.parent != src else ""
            if digit not in DIGITS:
                left.append(p.relative_to(src).as_posix())
                continue
            dst = ds / digit / p.name
        else:
            dst = ds / "images" / p.name
        if dst.exists():
            left.append(p.relative_to(src).as_posix())
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dst))
        back += 1
    if not left:
        shutil.rmtree(src)
    return (f"Возвращено в {ds}: {back}"
            + (f"; осталось в {src} (имя уже занято или цифра не в папке 0–9): {len(left)}" if left else ""))
