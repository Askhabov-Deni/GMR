"""
src/gmr/domain/datasets.py — где лежат датасеты моделей, какое трудное фото
идёт в эталон (этап 5, решения владельца 2026-10-04) и как называются кропы
(стандарт датасетов, docs/DATASETS.md). Без тяжёлых библиотек:
это нужно и окну оператора (тихая разметка), и инструментам обучения
(models/datasets.py, tools/etalon.py).
"""
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DATASETS_DIR = Path("database") / "datasets"
ETALON_SHARE = 1 / 3
ARCHIVE_PERCENT = 5


def _share(key: str) -> float:
    """Число 0…1 по отпечатку key: одно и то же при каждом запуске."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16) / 16 ** 12


def to_etalon(photo_hash: str) -> bool:
    """Трудное фото — в эталон или в обучение: каждое третье по отпечатку
    фото. Одно и то же фото — всегда в одно место: эталон (tools/etalon.py)
    и папки new/ (окно оператора) не пересекаются."""
    return _share(f"etalon:{photo_hash}") < ETALON_SHARE


def in_archive_etalon(account: str, percent: float = ARCHIVE_PERCENT) -> bool:
    """Лицевой счёт — в эталоне архива (etalon from-table --percent): отбор
    по отпечатку счёта, всегда одни и те же счета; с бо́льшим процентом
    прежние остаются и добавляются новые."""
    return _share(f"archive:{account}") < percent / 100


# ── Имена кропов (стандарт датасетов, docs/DATASETS.md) ──────────────────────
# <фото>__<что><№>[__<ответ>].jpg
#   фото  — имя исходного фото, обычно лицевой счёт (1300000375); тот же счёт,
#           снятый ещё раз, — «1300000375-2»;
#   что   — serial (номер счётчика), account (лицевой счёт маркером),
#           digit (цифра показаний); № — какой по счёту на фото (у цифры —
#           место в показаниях, 1–5);
#   ответ — что на кропе: у serial — номер с нулями; у account — счёт, если
#           он не совпадает с фото (обычно его нет); у digit ответа в имени
#           нет — ответ папка (digits_cnn/<цифра>/).
SERIAL, ACCOUNT, DIGIT, METER = "serial", "account", "digit", "meter"
KINDS = (SERIAL, ACCOUNT, DIGIT)
# Классы детектора счётчика — те же слова (meter — рамка показаний). У
# обученного детектора они зашиты в весах по-старому: программа понимает оба.
_OLD_CLASSES = {"gas_meter": METER, "serial_number": SERIAL, "marker_id": ACCOUNT}


def standard_class(name: str) -> str:
    """Имя класса детектора одним словом: gas_meter → meter, serial_number →
    serial, marker_id → account; остальные — как есть."""
    return _OLD_CLASSES.get(name, name)
_CROP = re.compile(r"^(?P<photo>[^_]+(?:_[^_]+)*)__(?P<kind>serial|account|digit)(?P<n>\d+)"
                   r"(?:__(?P<answer>[0-9A-Za-z]+))?$")
_REPEAT = re.compile(r"^(\d+)-\d+$")           # 1300000375-2 — тот же счёт ещё раз
_NOT_IN_PHOTO = re.compile(r"[^0-9A-Za-zА-Яа-яЁё.-]+")


@dataclass(frozen=True)
class CropName:
    photo: str
    kind: str
    n: int
    answer: str = ""

    @property
    def source(self) -> str:
        """К чему относится кроп: счёт без «-2» (кропы одного счётчика —
        в одной части при делении датасета)."""
        m = _REPEAT.match(self.photo)
        return m.group(1) if m else self.photo

    @property
    def account(self) -> str:
        """Лицевой счёт надписи: ответ, а без него — счёт фото."""
        return self.answer or self.source


def parse_crop_name(name: str) -> Optional[CropName]:
    """Имя файла (с расширением или без) → CropName; None — не по стандарту."""
    m = _CROP.match(Path(name).stem)
    if m is None:
        return None
    return CropName(m["photo"], m["kind"], int(m["n"]), m["answer"] or "")


def photo_id(text: str) -> str:
    """Имя исходного фото для имени кропа: без «__» и знаков, которые
    мешают в имени файла («IMG 12(1)» → «IMG_12_1»)."""
    return _NOT_IN_PHOTO.sub("_", str(text)).strip("_") or "photo"


def crop_name(photo: str, kind: str, n: int, answer: str = "", ext: str = ".jpg") -> str:
    if kind not in KINDS:
        raise ValueError(f"неизвестный кроп «{kind}»: {', '.join(KINDS)}")
    return f"{photo_id(photo)}__{kind}{n}" + (f"__{answer}" if answer else "") + ext


def free_crop_name(taken: set, photo: str, kind: str, n: int, answer: str = "", ext: str = ".jpg") -> str:
    """Имя, которого ещё нет среди taken (имена файлов в нижнем регистре):
    если занято — фото «-2», «-3», … (тот же счёт ещё раз, другой месяц)."""
    base, k = photo_id(photo), 1
    while True:
        name = crop_name(base if k == 1 else f"{base}-{k}", kind, n, answer, ext)
        if name.lower() not in taken:
            return name
        k += 1
