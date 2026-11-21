"""
src/gmr/application/quality.py — качество распознавания месяца по работе
оператора (просьба владельца 2026-11-21: «если оператор нажал „Исправить“ —
чтобы процент ошибок хранился»).

Ничего нового не записывается: всё уже лежит в логе месяца.
  - «Проверка»: «Верно» — в авто-строке PLUS/MINUS заполняется verified_by;
    «Исправить» — ещё и пометка «исправлено при проверке» в notes (при смене
    абонента — «(абонент X → Y)»), исходное чтение модели — в model_reading_str.
  - «Очередь»: решение оператора — строка source=manual.

Процент ошибок программы — среди показаний, которые она записала сама и
которые оператор проверил: исправлено / проверено. Это выборка, поэтому — с
пределами (95%, по Уилсону): при 50 проверенных и 1 ошибке это «2% (0,4–11%)».
Чем больше проверено, тем точнее. Список «Проверки» идёт в случайном (но
постоянном) порядке, чтобы проверенные фото были честной выборкой.
"""
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

CORRECTED_NOTE = "исправлено при проверке"     # src/gmr/application/operator.py
WRITTEN = ("PLUS", "MINUS")
MIN_CHECKED = 30                               # меньше — оценка на весь месяц не показывается

OPERATOR_WORDS = {"PLUS": "принято", "MINUS": "принято", "SUSPICIOUS": "принято",
                  "UNREADABLE": "нечитаемо", "NOT_IN_DB": "нет в базе", "REPEAT": "дубль",
                  "DUPLICATE": "дубль"}


@dataclass
class Quality:
    photos: int = 0              # фото, прочитанные программой (каждое — один раз)
    written: int = 0             # показание программа записала сама
    to_operator: int = 0         # ушло оператору
    checked: int = 0             # из записанных сама — проверено («Верно» или «Исправить»)
    corrected: int = 0           # из проверенных — исправлено
    corrected_account: int = 0   # из исправленных — другой абонент (номер/счёт)
    decided: Counter = field(default_factory=Counter)   # решения оператора по очереди
    model_full: int = 0          # фото очереди, принятые оператором, где модель прочла все 5 цифр
    model_full_ok: int = 0       # …и прочла так же, как оператор

    @property
    def corrected_reading(self) -> int:
        return self.corrected - self.corrected_account

    def error_rate(self) -> Optional[tuple[float, float, float]]:
        """(доля ошибок, нижний, верхний предел 95%) или None — не проверено ничего."""
        if not self.checked:
            return None
        return (self.corrected / self.checked, *wilson(self.corrected, self.checked))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Пределы 95% для доли k/n (метод Уилсона — честный и при 0 ошибок)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _int(v) -> Optional[int]:
    s = str(v or "").strip()
    return int(s) if s.isdigit() else None


def month_quality(log_rows: Iterable[dict]) -> Quality:
    rows = list(log_rows)
    auto: dict[str, dict] = {}
    for r in rows:                                     # последнее авто-чтение каждого фото
        if r.get("source") == "auto" and r.get("outcome") != "REPEAT" and r.get("photo_hash"):
            auto[r["photo_hash"]] = r
    q = Quality(photos=len(auto))
    for r in auto.values():
        if r["outcome"] in WRITTEN:
            q.written += 1
            if r.get("verified_by"):
                q.checked += 1
                if CORRECTED_NOTE in (r.get("notes") or ""):
                    q.corrected += 1
                    q.corrected_account += "(абонент" in (r.get("notes") or "")
        else:
            q.to_operator += 1
    manual = {}
    for r in rows:                                     # последнее решение оператора по фото очереди
        if r.get("source") == "manual" and r.get("photo_hash") in auto:
            manual[r["photo_hash"]] = r
    for h, m in manual.items():
        if auto[h]["outcome"] in WRITTEN:
            continue
        q.decided[OPERATOR_WORDS.get(m.get("outcome", ""), m.get("outcome", "") or "другое")] += 1
        model = (m.get("model_reading_str") or "").strip()
        if m.get("outcome") in WRITTEN and len(model) == 5 and model.isdigit():
            q.model_full += 1
            q.model_full_ok += _int(model) == _int(m.get("reading"))
    # фото очереди, закрытые без оператора: у счёта появилось показание с
    # другого фото — программа пишет строку REPEAT с тем же отпечатком
    closed = {r.get("photo_hash") for r in rows if r.get("outcome") == "REPEAT" and r.get("photo_hash")}
    for h, r in auto.items():
        if r["outcome"] not in WRITTEN and h not in manual and h in closed:
            q.decided["закрыто: у счёта уже есть показание"] += 1
    return q


def _pct(x: float) -> str:
    return f"{x * 100:.1f}".replace(".", ",") + "%"


def quality_lines(q: Quality) -> list[str]:
    """Для «Итога месяца»."""
    if not q.photos:
        return ["  качество распознавания: фото ещё не читались"]
    lines = ["  качество распознавания (по работе оператора):",
             f"    программа записала сама: {q.written} из {q.photos} фото ({_pct(q.written / q.photos)}), "
             f"оператору — {q.to_operator}"]
    rate = q.error_rate()
    if rate is None:
        lines.append("    проверено на «Проверке»: 0 — процент ошибок появится, когда оператор проверит фото")
    else:
        p, lo, hi = rate
        kinds = (f" (показание {q.corrected_reading}, абонент {q.corrected_account})" if q.corrected else "")
        lines.append(f"    проверено на «Проверке»: {q.checked}, исправлено {q.corrected}{kinds} — "
                     f"ошибок {_pct(p)} (вероятно от {_pct(lo)} до {_pct(hi)})")
        if q.checked >= MIN_CHECKED:
            lines.append(f"    значит, среди {q.written} записанных программой неверных — примерно "
                         f"{round(p * q.written)} (от {round(lo * q.written)} до {round(hi * q.written)})")
        else:
            lines.append(f"    для оценки по всему месяцу проверьте хотя бы {MIN_CHECKED} фото")
    done = sum(q.decided.values())
    if q.to_operator:
        lines.append(f"    очередь разобрана: {done} из {q.to_operator}"
                     + (" — " + ", ".join(f"{k} {v}" for k, v in q.decided.most_common()) if done else ""))
    if q.model_full:
        lines.append(f"    в очереди модель прочла все 5 цифр, но не была уверена: {q.model_full}, "
                     f"из них верно {q.model_full_ok} ({_pct(q.model_full_ok / q.model_full)})")
    return lines


def quality_row(name: str, q: Quality) -> dict:
    """Строка для сравнения месяцев (python gmr.py quality)."""
    rate = q.error_rate()
    return {"месяц": name, "фото": q.photos, "сам": q.written,
            "сам %": _pct(q.written / q.photos) if q.photos else "",
            "оператору": q.to_operator, "проверено": q.checked, "исправлено": q.corrected,
            "ошибок %": _pct(rate[0]) if rate else "",
            "пределы 95%": f"{_pct(rate[1])}–{_pct(rate[2])}" if rate else ""}
