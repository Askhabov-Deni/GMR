"""
tools/manual_html.py — инструкция оператора одним файлом HTML (для папки
программы, docs/BUILD.md): docs/OPERATOR.md → «Инструкция оператора.html»,
снимки экрана — внутри файла. Открывается любым браузером, интернет не нужен.

  python -m tools.manual_html [куда]

Разметка — только то, что есть в docs/OPERATOR.md: заголовки, абзацы, списки
(нумерованные и вложенные), таблицы, картинки, **жирный**, `код`, ссылки.
"""
import base64
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "OPERATOR.md"

CSS = """
body{font-family:"Segoe UI",Arial,sans-serif;max-width:900px;margin:32px auto;padding:0 20px;
     color:#1d2733;line-height:1.55;background:#f6f8fb}
h1{font-size:28px;border-bottom:3px solid #1f6feb;padding-bottom:8px}
h2{font-size:21px;margin-top:36px;color:#16324f}
code{background:#e8edf3;padding:1px 5px;border-radius:4px;font-size:92%}
img{max-width:100%;border:1px solid #cfd8e3;border-radius:6px;margin:10px 0;box-shadow:0 2px 8px #0001}
table{border-collapse:collapse;margin:12px 0;background:#fff}
td,th{border:1px solid #cfd8e3;padding:6px 10px;vertical-align:top;text-align:left}
th{background:#e8edf3}
li{margin:4px 0}
.ver{color:#6b7a8c;font-size:13px}
"""


def inline(text: str, base: Path) -> str:
    """**жирный**, `код`, [ссылка](url) — остальное экранируется."""
    out, pos = [], 0
    for m in re.finditer(r"`([^`]+)`|\*\*(.+?)\*\*|\[([^\]]+)\]\(([^)]+)\)", text):
        out.append(html.escape(text[pos:m.start()]))
        if m.group(1) is not None:
            out.append(f"<code>{html.escape(m.group(1))}</code>")
        elif m.group(2) is not None:
            out.append(f"<b>{inline(m.group(2), base)}</b>")
        else:
            out.append(f'<a href="{html.escape(m.group(4))}">{html.escape(m.group(3))}</a>')
        pos = m.end()
    out.append(html.escape(text[pos:]))
    return "".join(out)


def image(alt: str, src: str, base: Path) -> str:
    path = base / src
    if not path.is_file():
        return f"<p>[{html.escape(alt)}]</p>"
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{html.escape(alt)}" src="data:{mime};base64,{data}">'


def to_html(md: str, base: Path, title: str = "Инструкция оператора", version: str = "") -> str:
    lines = md.splitlines()
    out: list[str] = []
    lists: list[tuple[int, str]] = []        # открытые списки: (отступ, ul|ol)
    para: list[str] = []
    i = 0

    def flush_para():
        if para:
            out.append(f"<p>{inline(' '.join(para), base)}</p>")
            para.clear()

    def close_lists(indent=-1):
        while lists and lists[-1][0] > indent:
            out.append(f"</li></{lists.pop()[1]}>")

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        img = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        item = re.match(r"^(\s*)(\d+\.|[-*])\s+(.*)$", line)
        if not stripped:
            flush_para()
            i += 1
            continue
        if stripped.startswith("#"):
            flush_para()
            close_lists()
            level = len(stripped) - len(stripped.lstrip("#"))
            out.append(f"<h{level}>{inline(stripped[level:].strip(), base)}</h{level}>")
        elif stripped.startswith("|"):
            flush_para()
            close_lists()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            out.append("<table>")
            for n, cells in enumerate(rows):
                tag = "th" if n == 0 else "td"
                out.append("<tr>" + "".join(f"<{tag}>{inline(c, base)}</{tag}>" for c in cells) + "</tr>")
            out.append("</table>")
            continue
        elif img:
            flush_para()
            out.append(image(img.group(1), img.group(2), base))
        elif item:
            flush_para()
            indent, kind = len(item.group(1)), "ol" if item.group(2)[0].isdigit() else "ul"
            if lists and lists[-1][0] > indent:
                close_lists(indent)
            if lists and lists[-1][0] == indent:
                out.append("</li><li>")
            else:
                out.append(f"<{kind}><li>")
                lists.append((indent, kind))
            out.append(inline(item.group(3), base))
        elif lists and line.startswith(" "):
            flush_para()
            out.append(" " + inline(stripped, base))     # продолжение пункта списка
        else:
            close_lists()
            para.append(stripped)
        i += 1
    flush_para()
    close_lists()
    ver = f'<p class="ver">GMR {html.escape(version)}</p>' if version else ""
    return (f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
            f"<style>{CSS}</style></head><body>{ver}{''.join(out)}</body></html>\n")


def write_manual(dest: Path, version: str = "") -> Path:
    dest = Path(dest)
    dest.write_text(to_html(SOURCE.read_text(encoding="utf-8"), SOURCE.parent, version=version), encoding="utf-8")
    return dest


if __name__ == "__main__":
    print(write_manual(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Инструкция оператора.html"))
