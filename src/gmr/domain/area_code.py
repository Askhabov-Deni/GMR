"""
src/gmr/domain/area_code.py — код участка контролёра.

Код участка — первые цифры лицевого счёта (1300…, 0429…). У контролёра один
участок, значит и один код (наблюдение владельца 2026-11-21): надпись маркером
с его фото сравнивается только со счетами его кода — короткая надпись
(«0674», «503») не уходит в счёт чужого участка.

Код выводится сам, без файлов и новых столбцов таблицы: по счетам таблицы,
которые уже найдены по фото этого контролёра в этом месяце (строки лога с его
папкой). Мало счетов или коды смешаны — кода нет, всё как без него.
"""
from collections import Counter
from typing import Iterable, Optional

ACCOUNT_LEN = 10      # лицевой счёт — 10 цифр (у части таблиц Excel съедает ведущий ноль)
CODE_LEN = 4          # код участка — первые 4 цифры
MIN_ACCOUNTS = 10     # меньше найденных счетов — код ещё не известен
MIN_SHARE = 0.95      # доля счетов одного кода, чтобы считать его кодом контролёра


def account_code(account) -> Optional[str]:
    """Код участка счёта: «1300024317» → «1300», «429000123» → «0429»;
    None — не номер счёта."""
    a = str(account).strip()
    if not a.isdigit() or len(a) > ACCOUNT_LEN:
        return None
    return a.zfill(ACCOUNT_LEN)[:CODE_LEN]


def controller_code(accounts: Iterable[str]) -> Optional[str]:
    """Код участка по найденным счетам контролёра (каждый счёт — один раз):
    не меньше MIN_ACCOUNTS счетов и доля одного кода не меньше MIN_SHARE."""
    codes = Counter(c for c in map(account_code, {str(a).strip() for a in accounts}) if c)
    total = sum(codes.values())
    if total < MIN_ACCOUNTS:
        return None
    code, n = codes.most_common(1)[0]
    return code if n / total >= MIN_SHARE else None


def accounts_by_controller(log_rows: Iterable[dict], table_accounts: set) -> dict[str, set]:
    """{папка контролёра: счета таблицы из его строк лога}. Счёт не из таблицы
    (например, номер «Нет в базе») не считается."""
    out: dict[str, set] = {}
    for r in log_rows:
        folder = r.get("source_folder") or ""
        account = str(r.get("account_id") or "").strip()
        if folder and account in table_accounts:
            out.setdefault(folder, set()).add(account)
    return out


def only_code(groups: dict, code: Optional[str]) -> dict:
    """Словарь модели надписи только из счетов этого кода (и счетов, у которых
    кода не понять). code None — словарь как есть."""
    if code is None:
        return groups
    return {a: v for a, v in groups.items() if account_code(a) in (code, None)}


def other_code_note(account, code: Optional[str]) -> Optional[str]:
    """Пояснение, если счёт не с участка контролёра; None — с участка или код неизвестен."""
    own = account_code(account)
    if code is None or own is None or own == code:
        return None
    return f"счёт не с участка контролёра: код {own}, у контролёра {code}"
