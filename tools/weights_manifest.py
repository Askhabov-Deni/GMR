"""
tools/weights_manifest.py — какие файлы весов сейчас в работе (Фаза 4).

  python -m tools.weights_manifest            показать таблицу
  python -m tools.weights_manifest --write    записать её в docs/models.md

Для каждой модели из PipelineConfig: путь, размер, дата изменения,
SHA-256 (первые 16 символов). SHA-256 — «отпечаток» файла: если он совпал,
это ровно тот же файл весов, байт в байт. Так можно проверить, что копия
в резерве — та же модель, и узнать, какая модель прочитала показания.
"""
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.gmr.domain import PipelineConfig

ROOT = Path(__file__).resolve().parent.parent
MODELS_MD = ROOT / "docs" / "models.md"
BEGIN, END = "<!-- weights:begin -->", "<!-- weights:end -->"

# поле PipelineConfig → роль модели
FIELDS = [
    ("meter_detect_model", "детектор счётчика и серийника (YOLO)"),
    ("digit_detect_model", "детектор цифр (YOLO)"),
    ("digit_ocr_model",    "распознавание цифры (CNN)"),
    ("serial_ocr_model",   "распознавание серийника (CRNN)"),
    ("account_ocr_model",  "лицевой счёт по надписи маркером (CRNN)"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(config: Optional[PipelineConfig] = None, root: Path = ROOT) -> list[dict]:
    config = config or PipelineConfig()
    rows = []
    for field, role in FIELDS:
        rel = getattr(config, field)
        if not rel:             # необязательная модель не подключена (account_ocr_model = "")
            continue
        path = Path(rel) if Path(rel).is_absolute() else root / rel
        row = {"field": field, "role": role, "path": rel.replace("\\", "/")}
        if path.is_file():
            st = path.stat()
            row.update(size_mb=f"{st.st_size / 2**20:.1f}",
                       modified=datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
                       sha256=sha256(path)[:16])
        else:
            row.update(size_mb="—", modified="—", sha256="ФАЙЛ НЕ НАЙДЕН")
        rows.append(row)
    return rows


def to_markdown(rows: list[dict]) -> str:
    lines = [
        f"Снято: {datetime.now():%Y-%m-%d %H:%M} (`python -m tools.weights_manifest --write`)",
        "",
        "| Модель | Путь (PipelineConfig) | МБ | Изменён | SHA-256 (16) |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {r['role']} | `{r['path']}` | {r['size_mb']} | {r['modified']} | `{r['sha256']}` |"
              for r in rows]
    return "\n".join(lines)


def write_into(md_path: Path, table: str) -> None:
    # newline="" — читать и писать как есть: на Windows write_text иначе
    # переводит весь файл в CRLF, и git видит изменённым каждую строку.
    with open(md_path, encoding="utf-8", newline="") as f:
        text = f.read()
    nl = "\r\n" if "\r\n" in text else "\n"
    a, b = text.index(BEGIN) + len(BEGIN), text.index(END)
    new = text[:a] + nl + table.replace("\n", nl) + nl + text[b:]
    with open(md_path, "w", encoding="utf-8", newline="") as f:
        f.write(new)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tools.weights_manifest")
    p.add_argument("--write", action="store_true", help=f"записать таблицу в {MODELS_MD.relative_to(ROOT)}")
    args = p.parse_args(argv)
    table = to_markdown(manifest())
    print(table)
    if args.write:
        write_into(MODELS_MD, table)
        print(f"\nЗаписано в {MODELS_MD}")


if __name__ == "__main__":
    main()
