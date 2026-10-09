"""
Этап 5a: датасеты моделей (models/datasets.py, tools/datasets.py), деление
по фото, обучение YOLO без папок train/val, предразметка. Обучения здесь
нет: вместо ultralytics — подделка, которая записывает, что ей передали.
"""
import json
import sys
import types
from pathlib import Path

import pytest

import gmr
from models import datasets as ds
from models.yolo_all_detect import prelabel, train_yolo
from tools import datasets as check_ds


def touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ─── Раскладка ───────────────────────────────────────────────────────────────

def test_layout_created_and_nothing_touched(tmp_path):
    keep = touch(tmp_path / "digits_cnn" / "3" / "a.jpg", b"crop")
    created = ds.ensure_layout(tmp_path)
    for d in ("meter_yolo/images", "meter_yolo/labels", "meter_yolo/new", "digits_yolo/labels",
              "digits_cnn/0", "digits_cnn/new", "serials_crnn/new", "etalon"):
        assert (tmp_path / d).is_dir(), d
    assert (tmp_path / "digits_yolo" / "classes.txt").read_text(encoding="utf-8") == "digit\n"
    assert keep.read_bytes() == b"crop" and tmp_path / "digits_cnn" / "3" not in created
    (tmp_path / "digits_yolo" / "classes.txt").write_text("цифра\n", encoding="utf-8")
    assert ds.ensure_layout(tmp_path) == []                       # второй раз — ничего
    assert (tmp_path / "digits_yolo" / "classes.txt").read_text(encoding="utf-8") == "цифра\n"


# ─── Деление по фото ─────────────────────────────────────────────────────────

def test_photo_key():
    assert ds.photo_key("1300000013__gas_meter_1__digit_3.jpg") == "1300000013"
    assert ds.photo_key("IMG-20260915-WA0001.jpg") == "IMG-20260915-WA0001"
    assert ds.photo_key("12345678.jpeg") == "12345678"


def test_split_keeps_photo_together_and_old_photos_in_place():
    crops = [f"p{i:04d}__gas_meter_1__digit_{k}.jpg" for i in range(600) for k in range(1, 6)]
    parts = ds.split_by_photo(crops, lambda n: n, 67, 0.7, 0.15)
    where = {}
    for part, names in parts.items():
        for n in names:
            assert where.setdefault(ds.photo_key(n), part) == part     # все цифры фото — в одной части
    n = len(crops)
    assert abs(len(parts["train"]) / n - 0.7) < 0.06 and abs(len(parts["val"]) / n - 0.15) < 0.05
    more = crops + [f"q{i:04d}__gas_meter_1__digit_1.jpg" for i in range(300)]
    again = ds.split_by_photo(list(reversed(more)), lambda n: n, 67, 0.7, 0.15)
    for part in parts:                                             # добавили фото — старые на месте
        assert set(parts[part]) <= set(again[part])
    assert parts == ds.split_by_photo(crops, lambda n: n, 67, 0.7, 0.15)
    assert parts != ds.split_by_photo(crops, lambda n: n, 1, 0.7, 0.15)   # seed меняет деление


def test_split_of_bounds():
    assert ds.split_of("a", 1, 1.0, 0.0) == "train"
    assert ds.split_of("a", 1, 0.0, 1.0) == "val"
    assert ds.split_of("a", 1, 0.0, 0.0) == "test"


def test_cnn_split_by_photo(tmp_path):
    from models.cnn import dataset_cnn
    for i in range(80):
        for k, d in enumerate("01234", 1):
            touch(tmp_path / d / f"p{i}__gas_meter_1__digit_{k}.jpg")
    samples = dataset_cnn._collect_samples(tmp_path)
    tr, va, te = dataset_cnn._train_val_test_split(samples, seed=67)
    keys = [{ds.photo_key(p.name) for p, _ in part} for part in (tr, va, te)]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    assert len(tr) + len(va) + len(te) == 400 and va and te
    assert (tr, va, te) == dataset_cnn._train_val_test_split(list(reversed(samples)), seed=67)


def test_too_few_photos_clear_error(tmp_path):
    from models.cnn import dataset_cnn
    for i in range(2):
        touch(tmp_path / "1" / f"p{i}.jpg")
    with pytest.raises(ValueError, match="фото слишком мало для деления: обучение"):
        dataset_cnn.make_loaders(tmp_path)
    ds.require_parts({"train": [1], "val": [], "test": [1]}, ("train", "test"))   # val не нужна — ок


def test_cnn_new_folder_not_trained(tmp_path):
    from models.cnn import dataset_cnn
    touch(tmp_path / "1" / "a.jpg")
    touch(tmp_path / "new" / "1" / "b.jpg")
    assert [p.name for p, _ in dataset_cnn._collect_samples(tmp_path)] == ["a.jpg"]


def test_crnn_split_by_photo(tmp_path):
    from models.crnn import dataset_crnn
    meta = [{"file": tmp_path / f"p{i}__serial_number_{k}.jpg", "label": "123456"}
            for i in range(100) for k in (1, 2)]
    tr, va, te = dataset_crnn.split(meta, seed=42)
    keys = [{ds.photo_key(m["file"].name) for m in part} for part in (tr, va, te)]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    assert len(tr) + len(va) + len(te) == 200 and tr and va and te


# ─── run_info.json ───────────────────────────────────────────────────────────

def test_run_info_and_split_lists(tmp_path, monkeypatch):
    data = tmp_path / "data"
    files = [touch(data / "images" / f"p{i}.jpg", b"x" * i) for i in range(1, 5)]
    parts = {"train": files[:3], "val": files[3:]}
    monkeypatch.setattr(ds, "_git", lambda *a: "abc123" if a[0] == "rev-parse" else " M reader.py")
    ds.write_split_lists(tmp_path / "run", parts, data)
    info = json.loads(ds.write_run_info(tmp_path / "run", data, files, parts, 67, {"epochs": 5})
                      .read_text(encoding="utf-8"))
    assert (tmp_path / "run" / "split" / "val.txt").read_text(encoding="utf-8") == "images/p4.jpg\n"
    assert info["dataset_files"] == 4 and info["split"]["counts"] == {"train": 3, "val": 1}
    assert info["split"]["seed"] == 67 and info["args"] == {"epochs": 5}
    assert info["code"] == "abc123 + незакоммиченные изменения" and "python" in info["packages"]
    fp = info["dataset_fingerprint"]
    files[0].write_bytes(b"changed")                  # файл заменили — отпечаток другой
    assert ds.fingerprint(files, data) != fp
    monkeypatch.setattr(ds, "_git", lambda *a: "")
    assert ds.code_version().startswith("неизвестно")


# ─── Проверка датасетов ──────────────────────────────────────────────────────

def _yolo(root: Path):
    (root / "classes.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("display\nserial\naccount\n", encoding="utf-8")
    for i in range(10):
        touch(root / "images" / f"p{i}.jpg", f"photo{i}".encode())
        touch(root / "labels" / f"p{i}.txt", b"0 0.5 0.5 0.2 0.1\n1 0.3 0.3 0.1 0.1\n")
    touch(root / "images" / "nolabel.jpg", b"n")
    touch(root / "images" / "copy.jpg", b"photo0")                 # тот же файл, что p0.jpg
    touch(root / "labels" / "copy.txt", b"")                        # пустая — «ничего нет»
    touch(root / "labels" / "orphan.txt", b"0 0.5 0.5 0.1 0.1\n")
    for n, line in (("bad1", b"7 0.5 0.5 0.1 0.1\n"),               # класса нет в classes.txt
                    ("bad2", b"0 1.5 0.5 0.1 0.1\n"),               # рамка за краем фото
                    ("bad3", b"0 0.5 0.5 0.1\n")):                  # не 5 чисел
        touch(root / "images" / f"{n}.jpg", n.encode())
        touch(root / "labels" / f"{n}.txt", line)
    touch(root / "images" / "readme.txt", b"not a photo")
    touch(root / "new" / "2026-10" / "n1__2026-10.jpg")


def test_check_yolo(tmp_path):
    _yolo(tmp_path / "meter_yolo")
    out = check_ds.run(tmp_path, list_problems=True)
    block = out.split("meter_yolo")[1].split("digits_yolo")[0]
    lines = block.splitlines()
    assert "фото: 15, с разметкой: 14" in block
    assert "0 — display, 1 — serial, 2 — account" in block
    assert "объектов по классам: 0 — display: 10, 1 — serial: 10" in block
    assert "пустая разметка (на фото ничего нет — так тоже учат): 1" in block
    for line in ("фото без разметки (в обучение не пойдут): 1", "разметка без фото: 1",
                 "ошибки в разметке (не «класс cx cy w h», числа вне 0…1, класс вне classes.txt): 3",
                 "одинаковые фото: 1"):
        assert f"  ⚠ {line}" in lines, line
    assert "new/ (на разметку): 1 — 2026-10: 1" in block and "деление при обучении: обучение" in block
    problems = (tmp_path / "datasets_problems.txt").read_text(encoding="utf-8")
    assert "nolabel.jpg" in problems and "orphan.txt" in problems and "bad3.txt" in problems
    assert "p0.jpg = copy.jpg" in problems or "copy.jpg = p0.jpg" in problems
    assert "nolabel" not in out                                     # имена — только в файле


def test_check_cnn_and_crnn(tmp_path):
    cnn = tmp_path / "digits_cnn"
    for i in range(5):
        touch(cnn / "3" / f"p{i}__gas_meter_1__digit_1.jpg", f"c{i}".encode())
    touch(cnn / "8" / "same.jpg", b"c0")                           # тот же кроп с другой меткой
    touch(cnn / "3" / "again.jpg", b"c1")                          # повтор в той же цифре
    touch(cnn / "8" / "notes.txt", b"?")
    touch(cnn / "new" / "2026-10" / "4" / "x.jpg")
    touch(cnn / "старое" / "y.jpg")
    crnn = tmp_path / "serials_crnn"
    touch(crnn / "images" / "s1.jpg", b"1")
    touch(crnn / "labels" / "s1.txt", b"1234567")
    touch(crnn / "images" / "s2.jpg", b"2")
    touch(crnn / "labels" / "s2.txt", b"12")                        # коротко
    touch(crnn / "images" / "s3.jpg", b"3")
    touch(crnn / "new" / "2026-10" / "images" / "1234567__h1__2026-10.jpg")   # исправил оператор (окно)
    touch(crnn / "new" / "2026-11" / "images" / "1234567__h2__2026-11.jpg")
    out = check_ds.run(tmp_path)
    c = out.split("digits_cnn —")[1].split("serials_crnn —")[0]
    lines = c.splitlines()
    assert "кропов: 7 — 0: 0, 1: 0, 2: 0, 3: 6, 4: 0" in c and "8: 1" in c
    assert "кропов без «__» в имени: 2" in c
    assert "  ⚠ одинаковые кропы в разных цифрах (одна из меток неверна): 1" in lines
    assert "  ⚠ одинаковые кропы в одной цифре: 1" in lines
    assert "не картинки в папках цифр: 1" in c and "лишние папки (обучение их не читает): 1" in c
    assert "new/ (исправил оператор): 1 — 2026-10: 1" in c and "по цифрам: 0: 0, 1: 0, 2: 0, 3: 0, 4: 1" in c
    s = out.split("serials_crnn —")[1].split("etalon —")[0]
    assert "фото: 3, с годным номером: 1" in s and "длина номера: 7 цифр — 1" in s
    assert "фото без разметки: 1" in s and "номер не годится" in s
    assert "new/ (исправил оператор): 2 — 2026-10: 1, 2026-11: 1" in s
    assert "Имена проблемных файлов — с --list" in out


def test_datasets_command(tmp_path, capsys):
    assert gmr.main(["datasets", "--root", str(tmp_path / "ds")]) == 0
    out = capsys.readouterr().out
    assert "Создано:" in out and (tmp_path / "ds" / "serials_crnn" / "new").is_dir()
    assert "Проблем не найдено." in out


# ─── Обучение YOLO без train/val ─────────────────────────────────────────────

def test_ultralytics_finds_labels_for_our_layout(tmp_path):
    utils = pytest.importorskip("ultralytics.data.utils")
    img = tmp_path / "meter_yolo" / "images" / "p1.jpg"
    assert Path(utils.img2label_paths([str(img)])[0]) == tmp_path / "meter_yolo" / "labels" / "p1.txt"


def test_prepare_yolo_split(tmp_path):
    data = tmp_path / "meter_yolo"
    _yolo(data)
    yaml = train_yolo.prepare(data, tmp_path / "runs" / "v2", 0.2, 67, {"epochs": 3})
    split = tmp_path / "runs" / "v2" / "split"
    train = (split / "train.txt").read_text(encoding="utf-8").split()
    val = (split / "val.txt").read_text(encoding="utf-8").split()
    assert sorted(train + val) == sorted(f"images/{n}" for n in
                                         [f"p{i}.jpg" for i in range(10)] + ["copy.jpg", "bad1.jpg", "bad2.jpg", "bad3.jpg"])
    assert not set(train) & set(val) and "images/nolabel.jpg" not in train + val
    abs_train = (split / "train_abs.txt").read_text(encoding="utf-8").split("\n")[0]
    assert Path(abs_train).is_file() and Path(abs_train).parent == data.resolve() / "images"
    text = yaml.read_text(encoding="utf-8")
    assert "nc: 3" in text and "0: 'display'" in text and "2: 'account'" in text
    assert f"train: {(split / 'train_abs.txt').as_posix()}" in text
    info = json.loads((tmp_path / "runs" / "v2" / "run_info.json").read_text(encoding="utf-8"))
    assert info["args"]["classes"] == ["display", "serial", "account"]
    assert set(info["split"]["counts"]) == {"train", "val"}           # у YOLO теста нет
    assert sorted(p.name for p in split.iterdir()) == ["data.yaml", "train.txt", "train_abs.txt",
                                                       "val.txt", "val_abs.txt"]
    assert info["args"]["unlabelled_images"] == 1 and info["dataset_files"] == 28
    assert sorted(p.name for p in data.iterdir()) == ["classes.txt", "images", "labels", "new"]


def test_prepare_yolo_refuses(tmp_path):
    data = tmp_path / "d"
    touch(data / "images" / "a.jpg")
    touch(data / "labels" / "a.txt", b"0 0.5 0.5 0.1 0.1\n")
    with pytest.raises(train_yolo.DatasetError, match="classes.txt"):
        train_yolo.prepare(data, tmp_path / "r", 0.2, 67, {})
    (data / "classes.txt").write_text("\n", encoding="utf-8")
    with pytest.raises(train_yolo.DatasetError, match="пустой"):
        train_yolo.prepare(data, tmp_path / "r", 0.2, 67, {})
    (data / "classes.txt").write_text("display\n", encoding="utf-8")
    with pytest.raises(train_yolo.DatasetError, match="слишком мало"):
        train_yolo.prepare(data, tmp_path / "r", 0.2, 67, {})
    (data / "labels" / "a.txt").unlink()
    with pytest.raises(train_yolo.DatasetError, match="нет фото с разметкой"):
        train_yolo.prepare(data, tmp_path / "r", 0.2, 67, {})


@pytest.fixture
def fake_ultralytics(monkeypatch):
    calls = _Calls()

    class YOLO:
        names = {0: "display", 1: "serial", 2: "account"}

        def __init__(self, weights):
            calls.append(("init", weights))

        def train(self, **kw):
            calls.append(("train", kw))
            assert Path(kw["data"]).is_file()
            cache = Path(calls.dataset) / "labels.cache"
            cache.write_text("cache", encoding="utf-8")            # как ultralytics: рядом с labels/

        def predict(self, path, conf, verbose):
            calls.append(("predict", Path(path).name, conf))
            found = Path(path).stem != "empty"
            boxes = types.SimpleNamespace(cls=_T([0.0, 1.0] if found else []),
                                          xywhn=_T([[0.5, 0.5, 0.2, 0.1], [0.25, 0.75, 0.1, 0.05]]
                                                   if found else []))
            return [types.SimpleNamespace(boxes=boxes)]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=YOLO))
    return calls


class _Calls(list):
    dataset = None        # папка датасета: туда «ultralytics» кладёт labels.cache


class _T(list):
    def tolist(self):
        return list(self)


def test_train_yolo_main(tmp_path, fake_ultralytics, capsys):
    data = tmp_path / "meter_yolo"
    _yolo(data)
    fake_ultralytics.dataset = data
    argv = ["--data", str(data), "--model", "yolov8s.pt", "--project", str(tmp_path / "runs"),
            "--name", "meter_v2", "--epochs", "3", "--device", "cpu"]
    assert train_yolo.main(argv) == 0
    (_, w), (_, kw) = fake_ultralytics
    assert w == "yolov8s.pt" and kw["name"] == "meter_v2" and kw["exist_ok"] is True
    assert Path(kw["project"]) == (tmp_path / "runs").resolve() and kw["epochs"] == 3
    assert kw["seed"] == 67 and kw["device"] == "cpu" and kw["data"].endswith("data.yaml")
    assert kw["fliplr"] == 0.0 and kw["imgsz"] == 640           # зеркала выключены (этап 6c)
    assert not (data / "labels.cache").exists()                    # в папке датасета ничего не осталось
    assert train_yolo.main(argv) == 1                              # такой прогон уже есть
    assert "уже есть" in capsys.readouterr().out
    (data / "labels.cache").write_text("старый", encoding="utf-8")  # был до обучения — не трогаем
    assert train_yolo.main(argv[:-5] + ["meter_v3", "--fliplr", "0.5", "--imgsz", "960"]) == 0
    assert (data / "labels.cache").read_text(encoding="utf-8") == "cache"
    assert (fake_ultralytics[-1][1]["fliplr"], fake_ultralytics[-1][1]["imgsz"]) == (0.5, 960)


def test_train_yolo_main_dataset_error(tmp_path, fake_ultralytics, capsys):
    touch(tmp_path / "d" / "images" / "a.jpg")
    assert train_yolo.main(["--data", str(tmp_path / "d"), "--model", "m.pt", "--project",
                            str(tmp_path / "r"), "--name", "x"]) == 1
    assert "ОШИБКА" in capsys.readouterr().out and fake_ultralytics == []


# ─── Предразметка ────────────────────────────────────────────────────────────

def test_prelabel_writes_only_missing(tmp_path):
    images = tmp_path / "meter_yolo" / "images"
    for n in ("a", "b", "empty"):
        touch(images / f"{n}.jpg")
    labels = prelabel.labels_dir_for(images)
    assert labels == tmp_path / "meter_yolo" / "labels"
    touch(labels / "b.txt", b"2 0.1 0.1 0.1 0.1\n")               # ручная разметка — не трогать
    seen = []

    def predict(path):
        seen.append(path.name)
        return ([], []) if path.stem == "empty" else ([0.0, 2.0], [[0.5, 0.5, 0.25, 0.125], [0.1, 0.2, 0.3, 0.4]])

    s = prelabel.prelabel(images, labels, predict, ["display", "serial", "account"])
    assert s == {"written": 1, "had_labels": 1, "nothing_found": 1} and seen == ["a.jpg", "empty.jpg"]
    assert (labels / "a.txt").read_text(encoding="utf-8") == (
        "0 0.500000 0.500000 0.250000 0.125000\n2 0.100000 0.200000 0.300000 0.400000\n")
    assert (labels / "b.txt").read_text(encoding="utf-8") == "2 0.1 0.1 0.1 0.1\n"
    assert not (labels / "empty.txt").exists()
    for folder in (labels, tmp_path / "meter_yolo"):
        assert (folder / "classes.txt").read_text(encoding="utf-8") == "display\nserial\naccount\n"
    assert prelabel.labels_dir_for(tmp_path / "new") == tmp_path / "new"


def test_prelabel_keeps_existing_classes(tmp_path):
    (tmp_path / "classes.txt").write_text("мои\n", encoding="utf-8")
    assert prelabel.write_classes(tmp_path, ["display"]) is False
    assert (tmp_path / "classes.txt").read_text(encoding="utf-8") == "мои\n"


def test_prelabel_command(tmp_path, fake_ultralytics, capsys):
    new = tmp_path / "new"
    touch(new / "a.jpg")
    touch(new / "empty.jpg")
    assert gmr.main(["prelabel", str(new), "--weights", "w.pt", "--conf", "0.3"]) == 0
    out = capsys.readouterr().out
    assert fake_ultralytics[0] == ("init", "w.pt") and ("predict", "a.jpg", 0.3) in fake_ultralytics
    assert "0 — display, 1 — serial, 2 — account" in out and "Разметка записана: 1" in out
    assert "Модель ничего не нашла (разметьте вручную): 1" in out
    assert (new / "a.txt").read_text(encoding="utf-8").startswith("0 0.500000")
    assert gmr.main(["prelabel", str(tmp_path / "нет")]) == 1


def test_prelabel_default_weights_are_prod_meter_detector(tmp_path, fake_ultralytics):
    from src.gmr.domain import PipelineConfig
    assert prelabel.main([str(tmp_path)]) == 0
    assert fake_ultralytics[0] == ("init", PipelineConfig().meter_detect_model)


# ─── Обучение CNN / CRNN: папка результата и run_info ───────────────────────

def test_cnn_run_dir_and_split_saved(tmp_path, monkeypatch):
    from argparse import Namespace

    from models.cnn import train_cnn
    a = Namespace(output_dir=None, resume=None, seed=67)
    d = train_cnn.run_dir_for(a)
    assert d.parent == Path(train_cnn.OUTPUT_DIR) and d.name[:4].isdigit()   # новая папка с датой
    assert train_cnn.run_dir_for(Namespace(output_dir=None, resume="x/r1/best.pth")) == Path("x/r1")
    assert train_cnn.run_dir_for(Namespace(output_dir="o", resume="x/best.pth")) == Path("o")
    crops = tmp_path / "crops"
    files = [touch(crops / "1" / f"p{i}__digit_1.jpg") for i in range(3)]

    def loader(fs):
        return types.SimpleNamespace(dataset=types.SimpleNamespace(samples=[(f, 1) for f in fs]))

    train_cnn.save_split(tmp_path / "run", crops, loader(files[:1]), loader(files[1:2]), loader(files[2:]), a)
    info = json.loads((tmp_path / "run" / "run_info.json").read_text(encoding="utf-8"))
    assert info["split"]["counts"] == {"train": 1, "val": 1, "test": 1} and info["dataset_files"] == 3
    assert (tmp_path / "run" / "split" / "test.txt").read_text(encoding="utf-8") == "1/p2__digit_1.jpg\n"


def test_cnn_crnn_default_dataset_is_new_layout():
    from models.cnn import config_cnn
    from models.crnn import train_crnn
    assert config_cnn.CROPS_DIR == "database/datasets/digits_cnn"
    sys_argv, sys.argv = sys.argv, ["train_crnn.py"]
    try:
        a = train_crnn.parse()
    finally:
        sys.argv = sys_argv
    assert Path(a.images_dir) == Path("database/datasets/serials_crnn/images")
    assert Path(a.labels_dir) == Path("database/datasets/serials_crnn/labels")
    assert a.save_dir == "serial_id_ocr/runs/crnn"


def test_crnn_split_saved(tmp_path):
    from argparse import Namespace

    from models.crnn import train_crnn
    data = tmp_path / "serials_crnn"
    meta = [{"file": touch(data / "images" / f"s{i}.jpg"), "label": "1234567"} for i in range(3)]
    for i in range(3):
        touch(data / "labels" / f"s{i}.txt", b"1234567")
    a = Namespace(images_dir=str(data / "images"), labels_dir=str(data / "labels"), seed=42)
    train_crnn.save_split(str(tmp_path / "run"), a, meta, meta[:2], meta[2:], [])
    info = json.loads((tmp_path / "run" / "run_info.json").read_text(encoding="utf-8"))
    assert info["dataset_files"] == 6 and info["split"]["counts"] == {"train": 2, "val": 1, "test": 0}
    assert (tmp_path / "run" / "split" / "val.txt").read_text(encoding="utf-8") == "images/s2.jpg\n"


# ─── 2026-10-11: разметка лежит в папке месяца, --collect забирает её в датасеты ─

def test_collect_markup_from_month(tmp_path, capsys):
    from src.gmr.storage.month import MonthDB
    from tests._month import make_month
    f = make_month(tmp_path, [("100000", "A1", "1", "")], name="Октябрь_2026")
    with MonthDB(f.db) as db, db.transaction():
        db.set_meta("created_at", "2026-09-28 09:00:00")      # месяц создан в сентябре
    m = f.markup
    assert m == f.root / "разметка"
    touch(m / "digits_cnn" / "7" / "h1__2026-10__digit_3.jpg")
    touch(m / "serials_crnn" / "images" / "123__h2__2026-10.jpg")
    touch(m / "serials_crnn" / "labels" / "123__h2__2026-10.txt", b"123")
    touch(m / "meter_yolo" / "h3__2026-10.jpg")
    touch(m / "чужое" / "x.jpg")                                    # не датасет — пропускается
    root = tmp_path / "ds"
    assert check_ds.collect(f.root, root) == (
        "Разметка месяца Октябрь_2026 (2026-09): digits_cnn — 1, meter_yolo — 1, serials_crnn — 1")
    new = root / "serials_crnn" / "new" / "2026-09"
    assert (new / "labels" / "123__h2__2026-10.txt").read_text() == "123"
    assert (root / "digits_cnn" / "new" / "2026-09" / "7" / "h1__2026-10__digit_3.jpg").is_file()
    assert (root / "meter_yolo" / "new" / "2026-09" / "h3__2026-10.jpg").is_file()
    assert (m / "meter_yolo" / "h3__2026-10.jpg").is_file()            # в месяце остаётся
    assert check_ds.collect(f.root, root) == "Разметка месяца Октябрь_2026 (2026-09): новых нет; уже были: 3"
    assert gmr.main(["datasets", "--root", str(root), "--collect", str(f.root)]) == 0
    out = capsys.readouterr().out
    assert "новых нет" in out and "new/ (исправил оператор): 1 — 2026-09: 1" in out
    assert gmr.main(["datasets", "--root", str(root), "--collect", str(tmp_path / "нет")]) == 1
    assert "не папка месяца" in capsys.readouterr().out


def test_collect_month_without_markup(tmp_path):
    from tests._month import make_month
    f = make_month(tmp_path, [("100000", "A1", "1", "")])
    assert check_ds.collect(f.root, tmp_path / "ds").endswith("новых нет")
