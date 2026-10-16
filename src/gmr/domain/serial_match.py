"""
src/gmr/domain/serial_match.py — поиск номеров из таблицы, отличающихся от
прочитанного серийника одним символом (одна замена, вставка или удаление).

Зачем (Фаза 5, docs/MIGRATION_STATUS.md): у SERIAL_NOT_FOUND почти всегда в
таблице есть номер «в одном символе» — либо модель ошиблась при чтении,
либо в базе опечатка; в обоих случаях это тот же счётчик. Используется:
  - tools/analyze_log.py — оценка по логу;
  - program2.py — подсказка оператору («похожие номера в базе»);
  - серийник по таблице (этап 6b) — словарь номеров build_serial_groups.

Только строки, без pandas/torch.
"""


def normalize_serial(s: str) -> str:
    """
    Приводит серийный номер к каноническому виду для сравнения.

    Убирает только пробелы по краям — внутренние символы (включая ведущие
    нули) остаются нетронутыми. Намеренно НЕ вызываем int() / lstrip('0'),
    чтобы не потерять ведущие нули у номеров вроде "007123".
    (Перенесено из reader._normalize_serial, Фаза 6.)
    """
    return s.strip()


def lookup_variants(serial: str) -> tuple[str, str, str]:
    """Как номер ищется в таблице: сам номер, '0'+номер, '00'+номер — на
    табличке ведущие нули бывает не видно, а модель, учившаяся на старой
    разметке без нулей, их теряла. Единственное место этого правила:
    reader.py, program2.py, account_match.py, one_char_matches."""
    return (serial, "0" + serial, "00" + serial)


def one_edit_neighbors(s: str, alphabet: str) -> set[str]:
    """Все строки, отличающиеся от s одной заменой, вставкой или удалением."""
    out = set()
    for i in range(len(s) + 1):
        if i < len(s):
            out.add(s[:i] + s[i + 1:])                                        # удаление
            out.update(s[:i] + c + s[i + 1:] for c in alphabet if c != s[i])  # замена
        out.update(s[:i] + c + s[i:] for c in alphabet)                      # вставка
    return out


def alphabet_for(table_serials: set[str]) -> str:
    """Символы, из которых состоят номера таблицы (плюс все цифры)."""
    return "".join(sorted(set("".join(table_serials)) | set("0123456789")))


def one_char_matches(serial: str, table_serials: set[str], alphabet: str = "") -> list[tuple[str, str]]:
    """
    Пары (прочитанный вариант, номер из таблицы), отличающиеся одним символом.
    Варианты — как ищет reader.py (lookup_variants).
    """
    alphabet = alphabet or alphabet_for(table_serials)
    pairs = set()
    for v in lookup_variants(serial):
        pairs.update((v, t) for t in one_edit_neighbors(v, alphabet) & table_serials)
    return sorted(pairs)


def build_serial_groups(serials) -> dict[str, tuple]:
    """Словарь для серийника по таблице (этап 6b): {номер: варианты записи}.
    Только номера из цифр; варианты — как ищет reader.py: номер, без одного
    и без двух ведущих нулей (на фото их часто не видно)."""
    groups: dict[str, tuple] = {}
    for raw in serials:
        s = normalize_serial(str(raw))
        if s.isdigit() and s not in groups:
            groups[s] = tuple(dict.fromkeys(
                v for v in (s, s[1:] if s.startswith("0") else "", s[2:] if s.startswith("00") else "") if v))
    return groups
