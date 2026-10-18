"""
Сборка кропов для датасетов из папки фото и таблицы (gmr.py datasets --build,
tools/dataset_build.py). Детекторы — подделки: что «найдено», задаётся по
имени фото; кроп — картинка, по цвету которой видно, откуда она.
"""
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gmr
from src.gmr.render.image_io import read_image, write_image
from tools import dataset_build as build

# имя фото → что находит детектор счётчика (S — табличка, A — надпись, M — показания)
FOUND = {"1300000001": "SAM", "1300000001 (2)": "SAM", "1300000002": "SAM", "1300000003": "SAM",
         "1300000004": "SAM", "1300000006": "SAM", "1300000007": "", "IMG_1": "SAM"}
FOUR_DIGITS = {"1300000003"}                      # детектор цифр найдёт 4


def _crop(value):
    return np.full((10, 20, 3), value, np.uint8)


class Meter:
    def __init__(self):
        self.seen = []

    def detect(self, path):
        stem = Path(path).stem
        self.seen.append(stem)
        what = FOUND[stem]
        out = []
        if "S" in what:
            out.append({"class": "serial_number", "crop": _crop(10)})
        if "A" in what:
            out.append({"class": "marker_id", "crop": _crop(20)})
        if "M" in what:
            out.append({"class": "gas_meter", "crop": _crop(30), "stem": stem})
        return out


class Digits:
    fail = ()                                     # на этих фото детектор цифр «падает»

    def detect(self, meter_crop):
        if self.stem in self.fail:
            raise RuntimeError("сбой модели")
        n = 4 if self.stem in FOUR_DIGITS else 5
        xs = [40, 0, 30, 10, 20][:n]                    # не по порядку: сборка сортирует слева направо
        return [{"bbox": (x, 0, x + 8, 10), "crop": _crop(100 + x)} for x in xs]


class Models(SimpleNamespace):
    pass


def _models(fail=()):
    digits = Digits()
    digits.fail = fail
    meter = Meter()

    class Wrap:
        def detect(self, path):
            digits.stem = Path(path).stem
            return meter.detect(path)
    return Models(meter_detector=Wrap(), digit_detector=digits)


def _photos(tmp_path):
    folder = tmp_path / "фото"
    folder.mkdir(exist_ok=True)
    for stem in FOUND:
        write_image(folder / f"{stem}.jpg", np.zeros((30, 40, 3), np.uint8))
    (folder / "notes.txt").write_text("x", encoding="utf-8")
    return folder


def _table(tmp_path):
    t = tmp_path / "таблица.csv"
    t.write_text("Лицевой счет;Номер счетчика;Текущие показания\n"
                 "1300000001;0070351;2353\n"
                 "1300000003;111111;12,0\n1300000003;222222;12,0\n"      # два номера
                 "1300000004;12;\n"                                      # номер короткий, показаний нет
                 "1300000006;12-34;2353.5\n"
                 "1300000007;0012345;123456\n", encoding="utf-8")
    return t


def _names(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.jpg"))


def _rows(folder: Path) -> dict:
    with open(folder / "сборка.csv", encoding="utf-8-sig", newline="") as fh:
        return {r["фото"]: r for r in csv.DictReader(fh, delimiter=";")}


def test_build_all(tmp_path, capsys):
    ds_root = tmp_path / "ds"
    (ds_root / "accounts_crnn" / "images").mkdir(parents=True)
    write_image(ds_root / "accounts_crnn" / "images" / "1300000001__account1.jpg", _crop(1))   # уже в датасете
    out = build.run(_photos(tmp_path), _table(tmp_path), dataset_root=ds_root, models=_models())
    target = tmp_path / "фото_датасет"
    assert _names(target) == sorted([
        "serials_crnn/images/1300000001__serial1__0070351.jpg",
        "serials_crnn/images/1300000001-2__serial1__0070351.jpg",        # «(2)» — тот же счёт ещё раз
        "accounts_crnn/images/1300000001-2__account1.jpg",               # 1300000001 уже есть в датасете
        "accounts_crnn/images/1300000001-3__account1.jpg",
        "accounts_crnn/images/1300000002__account1.jpg",                 # счёта нет в таблице — надпись есть
        "accounts_crnn/images/1300000003__account1.jpg",
        "accounts_crnn/images/1300000004__account1.jpg",
        "accounts_crnn/images/1300000006__account1.jpg",
        "digits_cnn/0/1300000001__digit1.jpg", "digits_cnn/2/1300000001__digit2.jpg",
        "digits_cnn/3/1300000001__digit3.jpg", "digits_cnn/5/1300000001__digit4.jpg",
        "digits_cnn/3/1300000001__digit5.jpg",
        "digits_cnn/0/1300000001-2__digit1.jpg", "digits_cnn/2/1300000001-2__digit2.jpg",
        "digits_cnn/3/1300000001-2__digit3.jpg", "digits_cnn/5/1300000001-2__digit4.jpg",
        "digits_cnn/3/1300000001-2__digit5.jpg",
    ])
    # цифры — слева направо: у первой x=0 (кроп 100), у пятой x=40 (кроп 140)
    assert int(read_image(target / "digits_cnn" / "0" / "1300000001__digit1.jpg")[0, 0, 0]) == 100
    assert int(read_image(target / "digits_cnn" / "3" / "1300000001__digit5.jpg")[0, 0, 0]) == 140
    assert int(read_image(target / "serials_crnn" / "images" / "1300000001__serial1__0070351.jpg")[0, 0, 0]) == 10
    lines = out.splitlines()
    assert lines[0] == f"Сборка датасета: 8 фото ({tmp_path / 'фото'}) → {target}"
    assert lines[1] == ("serials_crnn: сделан — 2, номер в таблице не из 4–10 цифр — 2, "
                        "счёта нет в таблице (или номер пустой) — 1, у счёта несколько номеров — 1, "
                        "детектор не нашёл табличку — 1, в имени фото нет лицевого счёта — 1")
    assert lines[2] == ("accounts_crnn: сделан — 6, детектор не нашёл надпись — 1, "
                        "в имени фото нет лицевого счёта — 1")
    assert lines[3] == ("digits_cnn: сделаны — 2, показаний в таблице нет — 2, "
                        "показание не число из ≤5 цифр — 2, найдено цифр не 5 — 1, "
                        "в имени фото нет лицевого счёта — 1")
    rows = _rows(target)
    assert rows["1300000001.jpg"] == {"фото": "1300000001.jpg", "счёт": "1300000001",
                                      "номер": "1300000001-2__serial1__0070351.jpg",   # «… (2).jpg» — раньше по имени
                                      "надпись": "1300000001-3__account1.jpg", "цифры": "02353"}
    assert rows["IMG_1.jpg"]["счёт"] == "" and rows["1300000003.jpg"]["цифры"] == "найдено цифр не 5"
    assert not (ds_root / "serials_crnn").exists()                       # датасет не тронут
    # тот же запуск ещё раз — продолжение: всё уже собрано, ничего не добавлено
    again = build.run(_photos(tmp_path), _table(tmp_path), models=_models())
    assert "Продолжение: уже были в сборка.csv — 8, сейчас — 0" in again
    assert len(_names(target)) == 18 and len(_rows(target)) == 8


def test_build_resume_after_crash_and_failed_photo(tmp_path):
    photos, table, out = _photos(tmp_path), _table(tmp_path), tmp_path / "out"
    (photos / "минус").mkdir()
    write_image(photos / "минус" / "1300000002.jpg", np.zeros((30, 40, 3), np.uint8))   # тот же счёт в подпапке
    # детектор цифр падает на «1300000001 (2)»: от этого фото — ни одного кропа, номер не занят
    text = build.run(photos, table, out, models=_models(fail={"1300000001 (2)"}))
    assert ("⚠ программа упала на фото: 1 (кропов от них нет; при продолжении — снова), первое: "
            "1300000001 (2).jpg: RuntimeError: сбой модели") in text
    assert ", сделан — 1, " in text.splitlines()[1]                          # счётчики упавшего фото не в счёт
    assert "1300000001 (2).jpg" not in _rows(out) and "минус/1300000002.jpg" in _rows(out)
    assert "serials_crnn/images/1300000001__serial1__0070351.jpg" in _names(out)
    # обрыв посреди сборки: последней строки отчёта нет
    report = out / "сборка.csv"
    lines = report.read_text(encoding="utf-8-sig").splitlines()
    report.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8-sig")
    lost = lines[-1].split(";")[0]
    text = build.run(photos, table, out, models=_models())
    assert "Продолжение: уже были в сборка.csv — 7, сейчас — 2" in text
    rows = _rows(out)
    assert len(rows) == 9 and lost in rows and "1300000001 (2).jpg" in rows
    assert rows["1300000001 (2).jpg"]["номер"] == "1300000001-2__serial1__0070351.jpg"
    with pytest.raises(ValueError, match="собирались кропы: accounts digits serials"):
        build.run(photos, table, out, only=["serials"], models=_models())
    (tmp_path / "занято").mkdir()
    (tmp_path / "занято" / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="уже есть и не пустая"):
        build.run(photos, table, tmp_path / "занято", models=_models())


def test_left_time(monkeypatch):
    monkeypatch.setattr(build.time, "monotonic", lambda: 600.0)
    assert build._left(0.0, 100, 300) == ", осталось ~20 мин"
    assert build._left(590.0, 100, 101) == ", осталось меньше минуты"
    assert build._left(0.0, 0, 5) == "" and build._left(0.0, 5, 5) == ""


def test_build_only_serials(tmp_path):
    models = _models()
    out = build.run(_photos(tmp_path), _table(tmp_path), tmp_path / "out", only=["serials"], models=models)
    assert _names(tmp_path / "out") == ["serials_crnn/images/1300000001-2__serial1__0070351.jpg",
                                        "serials_crnn/images/1300000001__serial1__0070351.jpg"]
    assert "accounts_crnn" not in out and "digits_cnn" not in out


@pytest.mark.parametrize("text, digits", [("2353", "02353"), ("12,0", "00012"), ("99999", "99999"),
                                          ("0", "00000"), ("123456", None), ("2353.5", None), ("-1", None),
                                          ("abc", None), ("", None)])
def test_reading_digits(text, digits):
    assert build.reading_digits(text, 5) == digits


def test_errors(tmp_path):
    with pytest.raises(ValueError, match="нет папки"):
        build.run(tmp_path / "нет", _table(tmp_path), models=_models())
    (tmp_path / "пусто").mkdir()
    with pytest.raises(ValueError, match="нет фото"):
        build.run(tmp_path / "пусто", _table(tmp_path), models=_models())
    bad = tmp_path / "bad.csv"
    bad.write_text("Лицевой счет;Абонент\n1;2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="нет столбцов: Номер счетчика"):
        build.run(_photos(tmp_path), bad, models=_models())


def test_build_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(build, "_load_detectors", lambda cfg: _models())
    photos, table = _photos(tmp_path), _table(tmp_path)
    assert gmr.main(["datasets", "--root", str(tmp_path / "ds"), "--build", str(photos), "--table", str(table),
                     "--only", "accounts", "--out", str(tmp_path / "o")]) == 0
    assert "accounts_crnn: сделан — 6" in capsys.readouterr().out
    assert gmr.main(["datasets", "--root", str(tmp_path / "ds"), "--build", str(photos), "--table", str(table),
                     "--out", str(tmp_path / "o")]) == 1
    assert "ОШИБКА: в " in capsys.readouterr().out                        # продолжать — тем же --only
    with pytest.raises(SystemExit):
        gmr.main(["datasets", "--build", str(photos)])


def test_account_only_from_start_of_name_and_serial_cleaned(tmp_path, monkeypatch):
    monkeypatch.setitem(FOUND, "IMG_1300000001", "SAM")
    folder = tmp_path / "ф"
    folder.mkdir()
    for stem in ("1300000001", "IMG_1300000001"):
        write_image(folder / f"{stem}.jpg", np.zeros((30, 40, 3), np.uint8))
    t = tmp_path / "t.csv"
    t.write_text("Лицевой счет;Номер счетчика;Текущие показания\n1300000001; 0070351. ;\n", encoding="utf-8")
    out = build.run(folder, t, tmp_path / "o", only=["serials"], models=_models())
    assert _names(tmp_path / "o") == ["serials_crnn/images/1300000001__serial1__0070351.jpg"]   # «.» по краю — опечатка
    assert "в имени фото нет лицевого счёта — 1" in out                     # счёт — только в начале имени
