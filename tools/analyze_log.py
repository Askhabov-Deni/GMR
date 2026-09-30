"""
tools/analyze_log.py — качество чтения reader.py по логу обработки (Фаза 5).
Запускать из папки проекта:

  python -m tools.analyze_log <лог.csv>
  python -m tools.analyze_log <лог.csv> --table <таблица.csv|xlsx>   + разбор SERIAL_NOT_FOUND
  python -m tools.analyze_log <лог.csv> --out docs/audit/quality.txt

Модели не запускаются — только чтение лога (и таблицы, если указана).
В отчёт попадают только числа, без серийников и лицевых счетов.

Разделы отчёта:
  1. Фото — итог по каждому фото (лог копит строки всех прогонов; для фото
     берётся последний результат, отличный от REPEAT).
  2. Цифры — на каких позициях модель не уверена, что ей мешает.
  3. Серийники — чем не найденные в таблице отличаются от найденных;
     с --table — на что похож каждый не найденный (обрезан / ошибка в
     одном символе / похожего в таблице нет).
  4. Точность по проверке оператора — считается, только если в логе есть
     строки, проверенные (verified_by) или разобранные вручную (source=manual)
     в program2.py. Это единственный источник правильных ответов.
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from statistics import quantiles
from typing import Optional

from src.gmr.domain import PipelineConfig

READ_OK = {"PLUS", "MINUS", "SUSPICIOUS"}          # серийник найден, цифры прочитаны
SERIAL_FOUND = READ_OK | {"DIGITS_ERROR"}          # серийник найден в таблице
_LOW_CONF = re.compile(r"pos(\d+)\(pred=(\d),conf=([\d.]+)\)")
CORRECTED_NOTE = "исправлено при проверке"         # program2.VerifyScreen._save_edit


def load_log(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ─── 1. Фото ─────────────────────────────────────────────────────────────────

def photo_key(row: dict, folder_by_name: dict) -> tuple:
    """
    Фото = (подпапка, имя исходного файла). У строк до 2026-09-29 подпапки
    нет — берём её из более новых строк с тем же именем, если она однозначна.
    """
    name = row.get("original_filename", "")
    folder = row.get("source_folder") or folder_by_name.get(name, "")
    return folder, name


def last_state_per_photo(rows: list[dict]) -> dict[tuple, dict]:
    """Для каждого фото — последняя авто-строка не REPEAT, иначе последняя REPEAT."""
    auto = [r for r in rows if r.get("source") == "auto"]
    folders: dict[str, set] = {}
    for r in auto:
        if r.get("source_folder"):
            folders.setdefault(r["original_filename"], set()).add(r["source_folder"])
    folder_by_name = {n: next(iter(f)) for n, f in folders.items() if len(f) == 1}

    state: dict[tuple, dict] = {}
    for r in auto:                          # строки лога — в порядке записи
        key = photo_key(r, folder_by_name)
        prev = state.get(key)
        if prev is None or r["outcome"] != "REPEAT" or prev["outcome"] == "REPEAT":
            state[key] = r
    return state


# ─── 2. Цифры ────────────────────────────────────────────────────────────────

def digit_stats(photos: list[dict], cfg: PipelineConfig) -> dict:
    readings = [r["model_reading_str"] for r in photos if r.get("model_reading_str")]
    unsure_pos = Counter(i for s in readings for i, c in enumerate(s) if c == "?")
    low = [(int(p), d, float(c)) for r in photos for p, d, c in _LOW_CONF.findall(r.get("notes", ""))]
    errors = Counter()
    for r in photos:
        if r["outcome"] == "DIGITS_ERROR":
            note = r.get("notes", "")
            errors["неуверенные цифры" if note.startswith("low conf")
                   else "не 5 цифр / не найдены" if note else "без пояснения"] += 1
    forgiven_from = cfg.expected_digits - cfg.ignore_last_digits
    return {
        "with_reading": len(readings),
        "with_unsure": sum("?" in s for s in readings),
        "only_first_unsure": sum(s.startswith("?") and "?" not in s[1:] for s in readings),
        "unsure_by_pos": dict(sorted(unsure_pos.items())),
        "low_conf_by_pos": dict(sorted(Counter(p for p, _, _ in low).items())),
        "low_conf_pos0_pred": dict(Counter(d for p, d, _ in low if p == 0).most_common()),
        "errors": dict(errors),
        "first_digit_read": dict(sorted(Counter(
            r["model_reading_str"][0] for r in photos
            if r["outcome"] in READ_OK and r.get("model_reading_str")).items())),
        "forgiven_positions": list(range(forgiven_from, cfg.expected_digits)),
    }


# ─── 3. Серийники ────────────────────────────────────────────────────────────

def _q(values: list[float]) -> str:
    if len(values) < 4:
        return "мало данных"
    q1, q2, q3 = quantiles(values, n=4)
    return f"25% ниже {q1:.3f}, медиана {q2:.3f}, 75% ниже {q3:.3f}"


def _conf(r: dict) -> Optional[float]:
    try:
        return float(r.get("model_serial_conf") or "")
    except ValueError:
        return None


def one_edit_neighbors(s: str, alphabet: str) -> set[str]:
    """Все строки, отличающиеся от s одной заменой, вставкой или удалением."""
    out = set()
    for i in range(len(s) + 1):
        if i < len(s):
            out.add(s[:i] + s[i + 1:])                         # удаление
            out.update(s[:i] + c + s[i + 1:] for c in alphabet if c != s[i])   # замена
        out.update(s[:i] + c + s[i:] for c in alphabet)       # вставка
    return out


def _alphabet(table_serials: set[str]) -> str:
    return "".join(sorted(set("".join(table_serials)) | set("0123456789")))


def one_char_matches(serial: str, table_serials: set[str], alphabet: str = "") -> list[tuple[str, str]]:
    """
    Пары (прочитанный вариант, номер из таблицы), отличающиеся одним символом.
    Варианты — как в reader.py: сам номер, '0'+номер, '00'+номер.
    """
    alphabet = alphabet or _alphabet(table_serials)
    pairs = set()
    for v in (serial, "0" + serial, "00" + serial):
        pairs.update((v, t) for t in one_edit_neighbors(v, alphabet) & table_serials)
    return sorted(pairs)


def describe_edit(read: str, true: str) -> str:
    """Чем прочитанное отличается от номера в таблице (ровно одна правка)."""
    if len(read) == len(true):
        i = next(k for k in range(len(read)) if read[k] != true[k])
        return f"замена {true[i]}→{read[i]}"
    if len(read) < len(true):
        i = next((k for k in range(len(read)) if read[k] != true[k]), len(read))
        where = "первый" if i == 0 else "последний" if i == len(true) - 1 else "в середине"
        return f"пропущен символ ({where})"
    i = next((k for k in range(len(true)) if read[k] != true[k]), len(true))
    where = "первый" if i == 0 else "последний" if i == len(read) - 1 else "в середине"
    return f"лишний символ ({where})"


def classify_not_found(serial: str, table_serials: set[str], alphabet: str = "") -> str:
    """На что похож серийник, которого нет в таблице (reader уже пробовал '0'/'00' спереди)."""
    if one_char_matches(serial, table_serials, alphabet):
        return "ошибка в одном символе"
    if len(serial) >= 3 and any(serial in t for t in table_serials):
        return "обрезан (часть номера из таблицы)"
    return "похожего в таблице нет"


def chance_one_char_rate(lengths: list[int], table_serials: set[str], n: int = 1000, seed: int = 67) -> float:
    """
    Контроль: доля СЛУЧАЙНЫХ номеров тех же длин, у которых нашёлся «похожий в
    одном символе» в таблице. Если она мала, совпадения у SERIAL_NOT_FOUND —
    не случайность, а ошибки чтения.
    """
    import random
    rnd = random.Random(seed)
    alphabet = _alphabet(table_serials)
    hits = 0
    for _ in range(n):
        s = "".join(rnd.choice("0123456789") for _ in range(rnd.choice(lengths)))
        hits += bool(one_char_matches(s, table_serials, alphabet))
    return hits / n


def _float(v) -> Optional[float]:
    try:
        return float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return None


def plausible_delta_range(photos: list[dict]) -> Optional[tuple[float, float]]:
    """
    Обычный расход между показаниями — по фото, где счётчик найден и прочитан
    (PLUS/MINUS): от 5-го до 95-го процентиля delta. Нужен, чтобы понять,
    подходит ли кандидат из таблицы по показанию.
    """
    deltas = sorted(d for r in photos if r["outcome"] in ("PLUS", "MINUS")
                    if (d := _float(r.get("delta"))) is not None)
    if len(deltas) < 20:
        return None
    q = quantiles(deltas, n=20)
    return q[0], q[-1]


def reading_check(missing: list[dict], matches: list[list], last_readings: dict,
                  rng: tuple[float, float]) -> dict:
    """
    Для SERIAL_NOT_FOUND с похожими в одном символе номерами и полностью
    прочитанным показанием: у скольких кандидатов показание «подходит»
    (показание − последнее показание кандидата — в обычном диапазоне).
    """
    lo, hi = rng
    res = Counter()
    for r, m in zip(missing, matches):
        cands = sorted({t for _, t in m})
        reading = r.get("model_reading_str", "")
        if not cands or not reading.isdigit():
            continue
        fits = [t for t in cands
                if (last := last_readings.get(t)) is not None and lo <= int(reading) - last <= hi]
        kind = "один похожий" if len(cands) == 1 else "несколько похожих"
        res[f"{kind}: подходит по показанию ровно один"] += len(fits) == 1
        res[f"{kind}: не подходит ни один"] += len(fits) == 0
        res[f"{kind}: подходят несколько"] += len(fits) > 1
        res["проверено"] += 1
    return {k: v for k, v in res.items() if v}


def serial_stats(photos: list[dict], table_serials: Optional[set[str]] = None,
                 last_readings: Optional[dict] = None) -> dict:
    found = [r for r in photos if r["outcome"] in SERIAL_FOUND]
    missing = [r for r in photos if r["outcome"] == "SERIAL_NOT_FOUND"]
    typical = {n for n, _ in Counter(len(r["serial_id"]) for r in found).most_common(3)}
    res = {
        "found": len(found), "not_found": len(missing),
        "len_found": dict(sorted(Counter(len(r["serial_id"]) for r in found).items())),
        "len_not_found": dict(sorted(Counter(len(r["serial_id"]) for r in missing).items())),
        "not_found_untypical_len": sum(len(r["serial_id"]) not in typical for r in missing),
        "typical_len": sorted(typical),
        "conf_found": _q([c for r in found if (c := _conf(r)) is not None]),
        "conf_not_found": _q([c for r in missing if (c := _conf(r)) is not None]),
    }
    if table_serials is not None:
        alphabet = _alphabet(table_serials)
        res["not_found_kinds"] = dict(Counter(
            classify_not_found(r["serial_id"], table_serials, alphabet) for r in missing).most_common())
        matches = [one_char_matches(r["serial_id"], table_serials, alphabet) for r in missing]
        matched = [m for m in matches if m]
        res["one_char_unique"] = sum(len({t for _, t in m}) == 1 for m in matched)
        res["one_char_several"] = sum(len({t for _, t in m}) > 1 for m in matched)
        res["one_char_edits"] = dict(Counter(
            describe_edit(*m[0]) for m in matched if len({t for _, t in m}) == 1).most_common(12))
        lengths = [len(r["serial_id"]) for r in found] or [7]
        res["chance_rate"] = chance_one_char_rate(lengths, table_serials)
        res["table_size"] = len(table_serials)
        rng = plausible_delta_range(photos)
        if last_readings is not None and rng is not None:
            res["delta_range"] = rng
            res["reading_check"] = reading_check(missing, matches, last_readings, rng)
    return res


# ─── 4. Точность по проверке оператора ───────────────────────────────────────

def _as_int(s: str) -> Optional[int]:
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def operator_accuracy(rows: list[dict]) -> dict:
    """
    Правильные ответы в логе появляются только из program2.py:
      - проверка (вкладка «Проверка»): у авто-строки PLUS/MINUS заполняется
        verified_by; при исправлении reading меняется, model_reading_str — нет,
        в notes дописывается «исправлено при проверке»;
      - ручная обработка фото из question/: новая строка source=manual;
        её model_reading_str — то, что прочитала модель (program2 запускает
        те же модели), reading/serial_id — ответ оператора.
    """
    verified = [r for r in rows if r.get("source") == "auto" and r.get("verified_by")]
    v_changed_reading = [r for r in verified
                         if _as_int(r.get("reading")) != _as_int(r.get("model_reading_str"))]
    v_corrected = [r for r in verified if CORRECTED_NOTE in r.get("notes", "")]

    manual = [r for r in rows if r.get("source") == "manual"]
    m_read = [r for r in manual if r.get("reading") and r.get("model_reading_str")
              and "?" not in r["model_reading_str"]]
    m_read_ok = [r for r in m_read if _as_int(r["reading"]) == _as_int(r["model_reading_str"])]

    auto_by_hash = {r["photo_hash"]: r for r in rows
                    if r.get("source") == "auto" and r.get("photo_hash")}
    serial_pairs = [(auto_by_hash[m["photo_hash"]], m) for m in manual
                    if m.get("photo_hash") in auto_by_hash and m.get("serial_id")]
    snf = [(a, m) for a, m in serial_pairs if a["outcome"] == "SERIAL_NOT_FOUND"]
    return {
        "verified": len(verified),
        "verified_reading_ok": len(verified) - len(v_changed_reading),
        "verified_corrected": len(v_corrected),
        "manual": len(manual),
        "manual_outcomes": dict(Counter(r["outcome"] for r in manual).most_common()),
        "manual_full_model_reading": len(m_read),
        "manual_model_reading_ok": len(m_read_ok),
        "snf_with_answer": len(snf),
        "snf_model_serial_ok": sum(a["serial_id"].strip() == m["serial_id"].strip() for a, m in snf),
    }


# ─── Отчёт ───────────────────────────────────────────────────────────────────

def _pct(a: int, b: int) -> str:
    return f"{a} из {b} ({100 * a / b:.1f}%)" if b else f"{a} из 0"


def build_report(rows: list[dict], cfg: PipelineConfig, table_serials: Optional[set[str]] = None,
                 last_readings: Optional[dict] = None,
                 title: str = "") -> str:
    state = last_state_per_photo(rows)
    photos = list(state.values())
    by_outcome = Counter(r["outcome"] for r in photos)
    d = digit_stats(photos, cfg)
    s = serial_stats(photos, table_serials, last_readings)
    a = operator_accuracy(rows)
    L = []
    w = L.append
    w("=" * 64)
    w(f"КАЧЕСТВО ЧТЕНИЯ ПО ЛОГУ {title}".rstrip())
    w("=" * 64)
    w(f"Строк в логе: {len(rows)} (авто {sum(r.get('source') == 'auto' for r in rows)}, "
      f"ручных {a['manual']}, проверенных {a['verified']})")
    w(f"Фото (последний результат каждого): {len(photos)}")
    for o, n in by_outcome.most_common():
        w(f"  {o:<18} {n:>6}  {100 * n / len(photos):5.1f}%")

    w("")
    w("── Цифры показаний " + "─" * 45)
    w(f"Фото с прочитанной строкой цифр: {d['with_reading']}; "
      f"с неуверенной цифрой ('?'): {_pct(d['with_unsure'], d['with_reading'])}")
    w(f"  из них неуверенна только первая цифра: {d['only_first_unsure']}")
    w(f"'?' по позициям (0 — первая слева): {d['unsure_by_pos']}")
    w(f"Неуверенные цифры по позициям (из notes): {d['low_conf_by_pos']}")
    w(f"  что модель видела на позиции 0: {d['low_conf_pos0_pred']}")
    w(f"Ошибки DIGITS_ERROR: {d['errors']}")
    w(f"Первая цифра у прочитанных показаний: {d['first_digit_read']}")
    w(f"Позиции {d['forgiven_positions']} при низкой уверенности не дают '?' — туда молча "
      f"подставляется '{cfg.forgiven_digit_placeholder}' (ignore_last_digits={cfg.ignore_last_digits}); "
      f"по логу не видно, как часто.")

    w("")
    w("── Серийные номера " + "─" * 45)
    w(f"Найдены в таблице: {s['found']}, не найдены (SERIAL_NOT_FOUND): {s['not_found']}")
    w(f"Длина у найденных: {s['len_found']}")
    w(f"Длина у не найденных: {s['len_not_found']}")
    w(f"Не найденные с нетипичной длиной (не {s['typical_len']}): "
      f"{_pct(s['not_found_untypical_len'], s['not_found'])} — скорее всего, модель прочитала номер не целиком")
    w(f"Уверенность у найденных: {s['conf_found']}")
    w(f"Уверенность у не найденных: {s['conf_not_found']}")
    if "not_found_kinds" in s:
        w(f"Сравнение не найденных с таблицей ({s['table_size']} номеров): {s['not_found_kinds']}")
        w(f"  «в одном символе»: похожий номер один — {s['one_char_unique']}, "
          f"несколько — {s['one_char_several']}")
        w(f"  контроль: у случайного номера той же длины похожий в одном символе "
          f"нашёлся бы в {100 * s['chance_rate']:.1f}% случаев")
        w(f"  что модель путает (только однозначные): {s['one_char_edits']}")
    if "reading_check" in s:
        lo, hi = s["delta_range"]
        w(f"  проверка кандидатов по показанию (обычный расход {lo:+.0f}…{hi:+.0f}, "
          f"5–95% у найденных): {s['reading_check']}")
    else:
        w("Сравнение с таблицей: не делалось (добавьте --table)")

    w("")
    w("── Точность по проверке оператора " + "─" * 30)
    if not a["verified"] and not a["manual"]:
        w("Проверенных (verified_by) и ручных (source=manual) строк нет — точность")
        w("посчитать не по чему. Правильные ответы появляются, когда оператор")
        w("проверяет фото на вкладке «Проверка» и разбирает question/ в program2.py.")
    else:
        w(f"Проверено авто-показаний: {a['verified']}; модель прочитала верно: "
          f"{_pct(a['verified_reading_ok'], a['verified'])}; исправлено при проверке: {a['verified_corrected']}")
        w(f"Ручных строк: {a['manual']} {a['manual_outcomes']}")
        w(f"  модель прочитала все цифры: {a['manual_full_model_reading']}; из них верно: "
          f"{_pct(a['manual_model_reading_ok'], a['manual_full_model_reading'])}")
        w(f"  SERIAL_NOT_FOUND с ответом оператора: {a['snf_with_answer']}; модель прочитала "
          f"серийник верно (значит, его нет в таблице): "
          f"{_pct(a['snf_model_serial_ok'], a['snf_with_answer'])}")
    w("=" * 64)
    return "\n".join(L)


def load_table(path: Path, cfg: PipelineConfig) -> tuple[set[str], dict[str, Optional[float]]]:
    """Серийники таблицы и последнее показание по каждому (первая строка с этим номером)."""
    from reader import _load_table, _normalize_serial   # те же правила чтения, что в reader.py
    df = _load_table(str(path))
    last: dict[str, Optional[float]] = {}
    lasts = df[cfg.col_last_reading] if cfg.col_last_reading in df.columns else [""] * len(df)
    for serial, value in zip(df[cfg.col_serial], lasts):
        key = _normalize_serial(str(serial))
        if key and key != "nan" and key not in last:
            last[key] = _float(value) if str(value).strip() not in ("", "nan") else None
    return set(last), last


def load_table_serials(path: Path, cfg: PipelineConfig) -> set[str]:
    return load_table(path, cfg)[0]


def main(argv=None) -> str:
    p = argparse.ArgumentParser(prog="python -m tools.analyze_log",
                                description="Качество чтения reader.py по логу обработки.")
    p.add_argument("log", help="лог обработки (<таблица>_log.csv)")
    p.add_argument("--table", default=None, help="таблица счётчиков — для разбора SERIAL_NOT_FOUND")
    p.add_argument("--out", default=None, help="сохранить отчёт в файл")
    args = p.parse_args(argv)

    cfg = PipelineConfig()
    rows = load_log(Path(args.log))
    serials, last = load_table(Path(args.table), cfg) if args.table else (None, None)
    report = build_report(rows, cfg, serials, last)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"\nСохранено: {args.out}")
    return report


if __name__ == "__main__":
    main()
