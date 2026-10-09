"""
src/gmr/domain/account_match.py — лицевой счёт по надписи маркером: какие
записи номера ищем в таблице и как надпись сочетается с серийником.

Контролёр пишет на корпусе счётчика лицевой счёт (на фото — «00065»,
«00082»: последние 5 цифр счёта вида 1300000065). Это второй, независимый от
серийника ключ к строке таблицы. Модель (models/account) выбирает счёт из
таблицы по вероятности (models/ctc_lexicon.py); здесь — правила без моделей:

  1. Серийник нашёл ровно один счёт X:
     - надпись уверенно указывает на другой счёт Y → SUSPICIOUS: показание
       чуть не записалось не тому абоненту (или у одного из них в базе
       ошибка) — решает оператор;
     - иначе — как без надписи.
  2. Серийник счёт не нашёл (нет таблички, прочитан неуверенно, нет в
     таблице, номер у нескольких абонентов), а надпись уверенно указывает на
     счёт Y:
     - номер у нескольких абонентов и Y среди них → дальше со счётом Y;
     - таблички на фото нет → дальше со счётом Y (надпись — единственный
       ключ, ради этого её и пишут);
     - серийник прочитан, и он совпадает с номером Y в таблице или
       отличается от него одним символом (ошибка чтения или опечатка в базе)
       → дальше со счётом Y;
     - серийник прочитан и совсем не похож на номер Y → как без надписи
       (возможно, счётчик заменён — решает оператор), в notes — подсказка.

«Уверенно» — confidence ≥ PipelineConfig.account_conf_thresh.
Только строки, без pandas/torch.
"""
from dataclasses import dataclass
from typing import Iterable, Optional

from .ml import AccountPrediction
from .models import Outcome
from .serial_match import normalize_serial


def account_variants(account_id: str, digits: int) -> tuple[str, ...]:
    """Как может быть написан счёт: последние `digits` цифр (с ведущими
    нулями) и они же без ведущих нулей: 1300000065, 5 → ('00065', '65').
    digits = 0 — пишут счёт целиком: ('1300000065',). Счёт не из цифр —
    пусто (модель пишет только цифры).

    Полный номер при digits > 0 в словарь не входит: строк было бы вдвое
    больше по длине, а чтение — вдвое медленнее. Если контролёр написал
    счёт целиком, надпись просто не найдётся в словаре (уверенность низкая,
    решает серийник или оператор)."""
    a = "".join(str(account_id).split())
    if not a.isdigit():
        return ()
    if digits <= 0:
        return (a,)
    tail = a[-digits:] if len(a) >= digits else a.zfill(digits)
    return tuple(dict.fromkeys([tail, tail.lstrip("0") or "0"]))


def account_groups(accounts: Iterable[str], digits: int) -> dict[str, tuple[str, ...]]:
    """{счёт: варианты записи} для всех счетов таблицы — словарь модели."""
    groups = {}
    for a in accounts:
        a = str(a).strip()
        if a and a not in groups:
            v = account_variants(a, digits)
            if v:
                groups[a] = v
    return groups


def within_one_edit(a: str, b: str) -> bool:
    """Строки равны или отличаются одной заменой, вставкой или удалением."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:] or (len(a) == len(b) and a[i + 1:] == b[i + 1:])


def serial_agrees(read: str, table_serials: Iterable[str]) -> bool:
    """Прочитанный серийник — это номер из таблицы с точностью до одного
    символа (с учётом ведущих нулей, как ищет reader.py)."""
    read = normalize_serial(read or "")
    if not read:
        return False
    variants = (read, "0" + read, "00" + read)
    return any(within_one_edit(v, normalize_serial(str(t))) for t in table_serials for v in variants)


def _short(pred: AccountPrediction) -> str:
    return f"маркер '{pred.text}' → л/с {pred.account} ({pred.confidence:.2f})"


@dataclass
class AccountDecision:
    """account — с каким счётом продолжать (None — не продолжать: исход
    остаётся прежним); note — пометка в notes лога."""
    account: Optional[str]
    note: Optional[str]


class AccountMarkerPolicy:
    def confident(self, pred: Optional[AccountPrediction], thresh: float) -> bool:
        return pred is not None and pred.account is not None and pred.confidence >= thresh

    def conflict(self, serial_account: str, pred: Optional[AccountPrediction], thresh: float) -> Optional[str]:
        """Правило 1: текст для error_detail, если надпись уверенно указывает
        на другой счёт, иначе None."""
        if not self.confident(pred, thresh) or pred.account == str(serial_account):
            return None
        return f"на счётчике {_short(pred)}, по серийнику — л/с {serial_account}"

    def rescue(
        self,
        outcome: Outcome,
        pred: Optional[AccountPrediction],
        thresh: float,
        serial_text: Optional[str],
        account_serials: Iterable[str],
        ambiguous_accounts: Iterable[str] = (),
    ) -> AccountDecision:
        """Правило 2. outcome — исход по серийнику (NO_SERIAL,
        SERIAL_LOW_CONF, SERIAL_NOT_FOUND, SERIAL_AMBIGUOUS);
        account_serials — номера счётчика счёта pred.account в таблице."""
        if pred is None:
            return AccountDecision(None, None)
        if not self.confident(pred, thresh):
            hint = (f"{_short(pred)}?" if pred.account is not None
                    else f"маркер '{pred.text}' — нет в таблице")
            return AccountDecision(None, hint)

        if outcome == Outcome.SERIAL_AMBIGUOUS:
            if pred.account in {str(a) for a in ambiguous_accounts}:
                return AccountDecision(pred.account, f"номер у нескольких абонентов, выбран по надписи: {_short(pred)}")
            return AccountDecision(None, f"{_short(pred)} — не из абонентов с этим номером")
        if outcome == Outcome.NO_SERIAL:
            return AccountDecision(pred.account, f"таблички нет, л/с по надписи: {_short(pred)}")
        if outcome in (Outcome.SERIAL_LOW_CONF, Outcome.SERIAL_NOT_FOUND):
            serials = [str(s) for s in account_serials]
            if serial_agrees(serial_text or "", serials):
                return AccountDecision(pred.account, (
                    f"л/с по надписи: {_short(pred)}; серийник прочитан '{serial_text}', "
                    f"в базе '{', '.join(serials)}'"))
            return AccountDecision(None, f"{_short(pred)}, но серийник '{serial_text}' не похож "
                                         f"на номер в базе '{', '.join(serials)}'")
        return AccountDecision(None, None)
