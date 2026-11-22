"""
src/gmr/application/add_photos.py — «Добавить фото» (1.5.0, просьба владельца
2026-11-22): папку с фото контролёра — в папку месяца, фото/<контролёр>/.

Фото КОПИРУЮТСЯ (оригинал остаётся, где был). Папка контролёра уже есть —
новые фото добавляются к ней (объединение): тот же файл (то же имя и то же
содержимое) второй раз не копируется, другое фото с тем же именем получает
имя «имя (2).jpg». Обработка читает только фото прямо в папке контролёра,
поэтому фото из вложенных папок кладутся туда же, рядом.

Что можно перетащить или выбрать:
  - папку с фото одного контролёра (имя папки — имя контролёра, его можно
    сменить);
  - папку, в которой папки контролёров (Аюб/, Фатима Я/…) — каждая к своему;
  - несколько фото — к контролёру, которого выберут.
"""
import filecmp
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

PHOTO_EXTS = (".jpg", ".jpeg", ".png")


def is_photo(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in PHOTO_EXTS


def photos_in(folder: Path) -> list[Path]:
    """Все фото папки, и во вложенных папках тоже, по порядку имён."""
    return sorted((p for p in folder.rglob("*") if is_photo(p)), key=lambda p: str(p).lower())


@dataclass
class PhotoGroup:
    """Фото для одного контролёра. name — имя папки контролёра в месяце."""
    name: str
    files: list = field(default_factory=list)
    source: Optional[Path] = None


def find_groups(paths: Iterable[str]) -> list[PhotoGroup]:
    """Что перетащили (или выбрали) → группы по контролёрам. Пусто — фото нет.

    Папка, в которой фото лежат прямо в ней, — один контролёр (имя папки).
    Папка без своих фото, но с папками, где есть фото, — по контролёру на
    каждую. Отдельные фото — одна группа без имени (имя выберут в окне)."""
    groups: list[PhotoGroup] = []
    loose: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            if any(is_photo(c) for c in p.iterdir()):
                groups.append(PhotoGroup(p.name, photos_in(p), p))
            else:
                for sub in sorted((c for c in p.iterdir() if c.is_dir()), key=lambda c: c.name.lower()):
                    files = photos_in(sub)
                    if files:
                        groups.append(PhotoGroup(sub.name, files, sub))
        elif is_photo(p):
            loose.append(p)
    if loose:
        groups.append(PhotoGroup("", sorted(loose, key=lambda p: p.name.lower()), loose[0].parent))
    return groups


def match_controller(name: str, existing: Iterable[str]) -> str:
    """Имя как у папки контролёра, которая уже есть в месяце (без учёта
    регистра и пробелов по краям), иначе — как есть."""
    key = name.strip().casefold()
    return next((e for e in existing if e.strip().casefold() == key), name.strip())


def controllers(photos_dir: Path) -> list[str]:
    """Папки контролёров месяца (фото/<контролёр>)."""
    if not photos_dir.is_dir():
        return []
    return sorted((p.name for p in photos_dir.iterdir() if p.is_dir()), key=str.lower)


@dataclass
class AddReport:
    copied: dict = field(default_factory=dict)      # контролёр → сколько скопировано
    same: int = 0                                   # уже были (то же имя и содержимое)
    renamed: int = 0                                # имя занято другим фото — «имя (2)»
    new_folders: list = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.copied.values())

    def text(self) -> str:
        parts = [f"{name} — {n}" for name, n in self.copied.items() if n]
        out = (f"Добавлено фото: {self.total}" + (f" ({', '.join(parts)})" if parts else "")) if self.total \
            else "Новых фото нет"
        if self.same:
            out += f"; уже были в месяце: {self.same}"
        return out


def _same(a: Path, b: Path) -> bool:
    return b.is_file() and a.stat().st_size == b.stat().st_size and filecmp.cmp(a, b, shallow=False)


def _name_variants(folder: Path, name: str) -> list:
    """name и «stem (2)», «stem (3)»… — куда это фото могло лечь раньше."""
    stem, suffix = Path(name).stem, Path(name).suffix
    out, n = [folder / name], 2
    while (folder / f"{stem} ({n}){suffix}").exists():
        out.append(folder / f"{stem} ({n}){suffix}")
        n += 1
    return out


def _free_name(folder: Path, name: str) -> Path:
    stem, suffix = Path(name).stem, Path(name).suffix
    n = 2
    while (folder / f"{stem} ({n}){suffix}").exists():
        n += 1
    return folder / f"{stem} ({n}){suffix}"


def add_photos(photos_dir: Path, groups: Iterable[PhotoGroup]) -> AddReport:
    """Копирует фото групп в photos_dir/<группа.name>/. Имя группы — не пустое,
    без «\\ / : * ? " < > |» (проверяет окно: check_name)."""
    rep = AddReport()
    for g in groups:
        error = check_name(g.name)
        if error:
            raise ValueError(error)
        dst = photos_dir / g.name.strip()
        if not dst.is_dir():
            rep.new_folders.append(g.name.strip())
        dst.mkdir(parents=True, exist_ok=True)
        copied = 0
        for src in g.files:
            target = dst / src.name
            if target.exists():
                if any(_same(src, c) for c in _name_variants(dst, src.name)):
                    rep.same += 1                  # уже было — под этим же именем или «имя (2)»
                    continue
                target = _free_name(dst, src.name)
                rep.renamed += 1
            if src.resolve() == target.resolve():
                continue
            tmp = target.with_name(target.name + ".копируется")
            shutil.copy2(src, tmp)
            tmp.replace(target)            # обработка не увидит недокопированное фото
            copied += 1
        rep.copied[g.name.strip()] = rep.copied.get(g.name.strip(), 0) + copied
    return rep


BAD_CHARS = set('\\/:*?"<>|')


def check_name(name: str) -> Optional[str]:
    """Имя папки контролёра: None — годится, иначе — что не так."""
    name = name.strip()
    if not name:
        return "Укажите контролёра — чьи это фото."
    if BAD_CHARS & set(name) or name in (".", ".."):
        return "В имени контролёра нельзя \\ / : * ? \" < > |"
    return None
