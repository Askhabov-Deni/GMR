"""
src/gmr/domain/serial_match.py — поиск номеров из таблицы, отличающихся от
прочитанного серийника одним символом (одна замена, вставка или удаление).

Зачем (Фаза 5, docs/MIGRATION_STATUS.md): у SERIAL_NOT_FOUND почти всегда в
таблице есть номер «в одном символе» — либо модель ошиблась при чтении,
либо в базе опечатка; в обоих случаях это тот же счётчик. Используется:
  - tools/analyze_log.py — оценка по логу;
  - program2.py — подсказка оператору («похожие номера в базе»).

Только строки, без pandas/torch.
"""


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
    Варианты — как в reader.py: сам номер, '0'+номер, '00'+номер.
    """
    alphabet = alphabet or alphabet_for(table_serials)
    pairs = set()
    for v in (serial, "0" + serial, "00" + serial):
        pairs.update((v, t) for t in one_edit_neighbors(v, alphabet) & table_serials)
    return sorted(pairs)
