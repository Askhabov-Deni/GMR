"""
models/datasets.py — датасеты моделей и «виртуальное» деление (этап 5,
решения владельца 2026-10-04).

Раскладка (`python gmr.py datasets` создаёт недостающие папки, ничего не
переносит и не удаляет):

  database/datasets/
    meter_yolo/    images/ labels/ classes.txt   детектор на фото (YOLO)
                   new/                          фото на ручную разметку
    digits_yolo/   images/ labels/ classes.txt   детектор цифр на кропе показаний
    digits_cnn/    0/ … 9/                       цифры (CNN)
                   new/0/ … new/9/               цифры, исправленные оператором
    serials_crnn/  images/ labels/               серийники (CRNN), labels/<имя>.txt — номер
                   new/                          серийники, исправленные оператором
    etalon/                                      эталон для сравнения моделей (в обучение не идёт)

Датасет — одна папка без train/val: «туда всегда можно что-то докинуть».
Обучение само делит его при запуске («виртуальное деление») и сохраняет
списки в папке результата обучения; в папке датасета ничего не появляется.
Папку new/ обучение не читает: туда программа кладёт трудные случаи, владелец
просматривает их и переносит в датасет сам.

Деление — по фото: кропы одного фото (`<фото>__…`) всегда в одной части,
иначе модель проверяется на кропах фото, которые видела в обучении, и
точность завышена. Часть фото определяется отпечатком его имени, а не
случайным перемешиванием: когда в датасет добавляют новые фото, старые
остаются в своей части.

Модуль без torch/cv2 — его импортируют и скрипты обучения, и проверка
датасетов (tools/datasets.py), и тесты.
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from src.gmr.domain.datasets import DATASETS_DIR, to_etalon  # noqa: F401 (общие с окном оператора)

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DIGITS = [str(d) for d in range(10)]

# датасет → папки внутри него
LAYOUT: dict[str, list[str]] = {
    "meter_yolo":   ["images", "labels", "new"],
    "digits_yolo":  ["images", "labels"],
    "digits_cnn":   DIGITS + [f"new/{d}" for d in DIGITS],
    "serials_crnn": ["images", "labels", "new/images", "new/labels"],
    "etalon":       [],
}
# classes.txt, который известен заранее (детектор цифр — один класс, как в
# прежнем data.yaml). Для meter_yolo его пишет предразметка из модели.
KNOWN_CLASSES = {"digits_yolo": ["digit"]}


def ensure_layout(root: Path) -> list[Path]:
    """Создаёт недостающие папки раскладки (и известный заранее classes.txt).
    Ничего не переносит и не перезаписывает. Возвращает созданное."""
    created = []
    for name, subdirs in LAYOUT.items():
        for d in [Path(root) / name] + [Path(root) / name / s for s in subdirs]:
            if not d.is_dir():
                d.mkdir(parents=True)
                created.append(d)
    for name, classes in KNOWN_CLASSES.items():
        f = Path(root) / name / "classes.txt"
        if not f.exists():
            f.write_text("".join(f"{c}\n" for c in classes), encoding="utf-8")
            created.append(f)
    return created


def photo_key(name: str) -> str:
    """Исходное фото по имени файла: всё до первого «__»
    (`1300000013__gas_meter_1__digit_3.jpg` → `1300000013`). Без «__» —
    имя без расширения: файл сам себе фото."""
    return Path(name).stem.split("__", 1)[0]


def split_of(key: str, seed: int, train: float, val: float) -> str:
    """"train" / "val" / "test" для фото с ключом key: доля по отпечатку
    sha256("<seed>:<key>"), одна и та же при любом составе датасета."""
    x = int(hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()[:12], 16) / 16 ** 12
    if x < train:
        return "train"
    return "val" if x < train + val else "test"


def split_by_photo(items: Iterable, name_of, seed: int, train: float, val: float) -> dict[str, list]:
    """Делит items по фото: {"train": [...], "val": [...], "test": [...]}.
    name_of(item) — имя файла. Порядок внутри части — по имени (не зависит
    от порядка файлов на диске)."""
    parts: dict[str, list] = {"train": [], "val": [], "test": []}
    for item in sorted(items, key=lambda it: str(name_of(it))):
        parts[split_of(photo_key(name_of(item)), seed, train, val)].append(item)
    return parts


def require_parts(parts: dict[str, list], needed: Iterable[str]) -> None:
    """ValueError, если после деления какая-то нужная часть пуста (фото мало):
    иначе обучение падает непонятной ошибкой посреди эпохи."""
    names = {"train": "обучение", "val": "проверка", "test": "тест"}
    if any(not parts[p] for p in needed):
        raise ValueError("фото слишком мало для деления: " + ", ".join(
            f"{names[p]} {len(parts[p])}" for p in needed) + " — добавьте фото в датасет")


def images_in(folder: Path) -> list[Path]:
    """Картинки прямо в папке (без подпапок), по имени."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def fingerprint(files: Iterable[Path], base: Path) -> str:
    """Отпечаток состава датасета: имена (относительно base) и размеры файлов.
    Добавили, убрали или заменили файл — отпечаток другой."""
    h = hashlib.sha256()
    for p in sorted(Path(f) for f in files):
        h.update(f"{p.relative_to(base).as_posix()}\t{p.stat().st_size}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def _git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def code_version() -> str:
    """Коммит кода, которым обучали (и пометка, если есть незакоммиченные правки)."""
    commit = _git("rev-parse", "--short", "HEAD")
    if not commit:
        return "неизвестно (нет git)"
    return commit + (" + незакоммиченные изменения" if _git("status", "--porcelain") else "")


def package_versions(names=("torch", "torchvision", "ultralytics", "albumentations", "numpy")) -> dict:
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            pass
    return out


def write_split_lists(run_dir: Path, parts: dict[str, list[Path]], base: Path) -> None:
    """split/<часть>.txt в папке результата: какие файлы попали в какую часть
    (пути относительно папки датасета)."""
    d = Path(run_dir) / "split"
    d.mkdir(parents=True, exist_ok=True)
    for part, files in parts.items():
        (d / f"{part}.txt").write_text(
            "".join(f"{Path(f).relative_to(base).as_posix()}\n" for f in files), encoding="utf-8")


def write_run_info(run_dir: Path, dataset: Path, files: list[Path], parts: dict[str, list],
                   seed: int, args: Optional[dict] = None) -> Path:
    """run_info.json в папке результата: на каком датасете, каким кодом и
    какими пакетами обучали, как поделили. Персональных данных нет: только
    путь, числа и отпечаток."""
    info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(Path(dataset).resolve()),
        "dataset_files": len(files),
        "dataset_fingerprint": fingerprint(files, Path(dataset)),
        "split": {
            "rule": "по фото (имя до «__»), часть — по sha256(seed:фото); списки — split/*.txt",
            "seed": seed,
            "counts": {k: len(v) for k, v in parts.items()},
            "photos": {k: len({photo_key(Path(f).name) for f in v}) for k, v in parts.items()},
        },
        "code": code_version(),
        "packages": package_versions(),
        "args": args or {},
    }
    path = Path(run_dir) / "run_info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
