"""
src/gmr/domain/note_match.py — надпись контролёра поверх фото (строки,
прочитанные src/gmr/overlay.py) → лицевой счёт и показание по таблице. Без
картинок и моделей: только строки и таблица.

Контролёры пишут как придётся: «24317», «лс17777», «пок51234», «17743пок»,
«ПКЗ…» (показания), «лс0429» / «000013» (число перенесено на вторую
строку), счёт с любым числом нулей. Обрывки и мусор («лс0», «1525») не
угадываются: надпись используется, только если понята целиком.

  1. Прочтения (hypotheses): каждая строка сама по себе или вместе с
     соседней, если число перенесено (wrapped); «З» вместо «3» после «к»
     (spellings).
  2. Строка → числа с припиской (parse): «лс» — счёт, «пок»/«пк»/«пкз» —
     показание, без приписки — решает таблица.
  3. Прочтение понято, если есть счёт, которым объясняется КАЖДОЕ число:
     это запись счёта (AccountIndex) или показание, правдоподобное для него
     (READING_WINDOW от прошлого). Хоть одно число лишнее — не понято.
  4. Счёт по табличке известен: надпись его подтверждает и даёт показание;
     явно указывает на другой счёт («лс…» или число длиннее показания) —
     спор. Счёт не известен: счёт по надписи — если надпись объясняет ровно
     один счёт и показание с барабана для него правдоподобно.
  5. Прочтения дали разные счета или разные показания — не используется
     (разные счета — спор); ни одно не понято — «не понять»: фото идёт как
     без надписи.
"""
import re
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Optional

from .account_match import account_variants

ACCOUNT, READING = "account", "reading"
# Конец счёта, записанный без начала («9005357», «005357» — 0429005357), —
# от 6 цифр (docs/BACKLOG.md): короче — у многих счетов, и так пишет обрывок
# («2051» из «22051»), а не контролёр.
MIN_TAIL = 6
# Показание правдоподобно для счёта, если от прошлого отличается на −1500…+3000:
# реестр архива 2026-10 (16 245 счетов) — так у 98.4% настоящих пар, у
# случайного чужого счёта — у 8.5%.
READING_WINDOW = (-1500, 3000)
MAX_READING_DIGITS = 6      # число длиннее — не показание, только счёт
MAX_HYPOTHESES = 256        # прочтений больше — надпись не разобрать
_RUNS = re.compile(r"\d+|\D+")
_ZE = re.compile(r"(?<=к)3")


# ─── Текст ───────────────────────────────────────────────────────────────────

def wrapped(upper: str, lower: str) -> bool:
    """Число перенесено на следующую строку: верхняя кончается цифрой, нижняя
    начинается цифрой («лс0429» / «000013» — счёт 0429000013)."""
    return upper[-1:].isdigit() and lower[:1].isdigit()


def spellings(text: str) -> list[str]:
    """Как ещё могла быть написана строка: «З» и «3» в шрифте почти одинаковы,
    и «ПКЗ» (показания) читается как «пк3». Вариант с «з» вместо «3» сразу
    после «к» — какой верный, решает таблица."""
    alt = _ZE.sub("з", text)
    return [text] if alt == text else [text, alt]


def _kind(letters: str) -> str:
    return ACCOUNT if set(letters) & set("лс") else READING if set(letters) & set("пок") else ""


def parse(text: str) -> list[tuple[str, str]]:
    """Строка → [(что, число)]: что — ACCOUNT («лс»), READING («пок», «пк»,
    «пкз») или "" (приписки нет: решает таблица). Приписка обычно перед
    числом; в конце строки после числа без своей приписки — его («17743пок»)."""
    runs = _RUNS.findall(text.lower())
    out = []
    for n, run in enumerate(runs):
        if not run.isdigit():
            continue
        before = runs[n - 1] if n else ""
        after = runs[n + 1] if n + 2 == len(runs) else ""
        out.append((_kind(before) or (_kind(after) if not before else ""), run))
    return out


def hypotheses(lines: list[str]) -> Optional[list[list[str]]]:
    """Как можно понять строки (сверху вниз): каждая сама по себе или с
    соседней, если число перенесено; и «З» вместо «3». None — прочтений
    больше MAX_HYPOTHESES."""
    texts = [t for t in lines if t]
    seen: dict = {}
    joins = [i for i in range(len(texts) - 1) if wrapped(texts[i], texts[i + 1])]
    for mask in range(2 ** len(joins)) if texts else ():
        glued = {joins[j] for j in range(len(joins)) if mask >> j & 1}
        merged, cur = [], texts[0]
        for i in range(1, len(texts)):
            if i - 1 in glued:
                cur += texts[i]
            else:
                merged.append(cur)
                cur = texts[i]
        merged.append(cur)
        for h in product(*(spellings(t) for t in merged)):
            seen.setdefault(h, None)
            if len(seen) > MAX_HYPOTHESES:
                return None
    return [list(h) for h in seen]


# ─── Счета ───────────────────────────────────────────────────────────────────

class AccountIndex:
    """Как можно записать каждый счёт таблицы → счета. Способы записи:
      - без нулей, как маркером на корпусе (account_variants: «17777»);
      - начало счёта как есть, остаток — с любым числом нулей («0429» +
        «00013» — счёт 0429000013);
      - конец счёта как есть, от MIN_TAIL цифр («9005357», «005357» —
        0429005357)."""

    def __init__(self, accounts: Iterable[str]):
        self.variants: dict[str, set] = {}
        self.tails: dict[str, set] = {}
        self.heads: dict[tuple, set] = {}
        for a in {str(a).strip() for a in accounts}:
            if not a.isdigit():
                continue
            for v in account_variants(a):
                self.variants.setdefault(v, set()).add(a)
            for k in range(len(a) - MIN_TAIL + 1):
                self.tails.setdefault(a[k:], set()).add(a)
            for k in range(1, len(a)):
                if int(a[k:]):
                    self.heads.setdefault((a[:k], int(a[k:])), set()).add(a)

    def find(self, number: str) -> set:
        """Счета, которые так можно записать (пусто — ни один)."""
        if not number.isdigit():
            return set()
        out = set(self.variants.get(number, ()))
        out |= self.tails.get(number, set())
        for k in range(1, len(number)):
            out |= self.heads.get((number[:k], int(number[k:])), set())
        return out


def is_account_form(number: str, account: str) -> bool:
    """Так можно записать этот счёт (AccountIndex)."""
    return account in AccountIndex([account]).find(number)


def plausible(reading: Optional[int], last: Optional[int]) -> bool:
    """Показание подходит счёту: от прошлого — в READING_WINDOW. Нет показания
    или прошлого — нечем проверить: нет."""
    return reading is not None and last is not None and READING_WINDOW[0] <= reading - last <= READING_WINDOW[1]


# ─── Ответ ───────────────────────────────────────────────────────────────────

@dataclass
class NoteAnswer:
    account: Optional[str] = None   # счёт по надписи (если по табличке не известен)
    reading: Optional[int] = None   # показание по надписи
    conflict: str = ""              # спор: надпись указывает на другой счёт — к оператору
    why: str = ""                   # что понято или почему нет — для лога и разбора
    understood: bool = False        # надпись понята целиком (может только подтверждать счёт)


def _show(tokens: list) -> str:
    return ", ".join({ACCOUNT: "лс", READING: "пок"}.get(k, "") + n for k, n in tokens)


def _explain(tokens: list, target: str, index: AccountIndex, last: Optional[int]) -> tuple[set, list]:
    """Надпись — про счёт target: (показания, лишние числа). Число — запись
    этого счёта или показание, правдоподобное для него (прошлого нет — только
    с припиской «пок»); остальное — лишнее."""
    readings, extra = set(), []
    for kind, number in tokens:
        if kind != READING and target in index.find(number):
            continue
        fits = kind == READING if last is None else plausible(int(number), last)
        if kind != ACCOUNT and len(number) <= MAX_READING_DIGITS and fits:
            readings.add(int(number))
        else:
            extra.append((kind, number))
    return readings, extra


def _unique(index: AccountIndex, number: str) -> set:
    """Счёт, который так можно записать, если он один; иначе пусто."""
    found = index.find(number)
    return found if len(found) == 1 else set()


def _strong(kind: str, number: str) -> bool:
    """Явно счёт: с припиской «лс» или длиннее показания."""
    return kind == ACCOUNT or (kind == "" and len(number) > MAX_READING_DIGITS)


def _part_of(number: str, barrel: Optional[int]) -> bool:
    """Число — показание с барабана или его кусок (обрывок строки «1525» из
    «15253»): это не запись счёта."""
    return barrel is not None and number.lstrip("0") != "" and number.lstrip("0") in str(barrel)


def _with_reading(ans: NoteAnswer, readings: set) -> NoteAnswer:
    if len(readings) == 1:
        ans.reading = next(iter(readings))
        ans.why += f", показание {ans.reading}"
    elif readings:
        ans.why += f", показаний несколько: {', '.join(map(str, sorted(readings)))}"
    return ans


def _known(tokens: list, index: AccountIndex, lasts: dict, account: str) -> NoteAnswer:
    """Счёт известен по табличке: надпись его подтверждает, даёт показание или спорит."""
    readings, extra = _explain(tokens, account, index, lasts.get(account))
    others = sorted(a for kind, n in extra if _strong(kind, n) for a in _unique(index, n))
    if others:
        return NoteAnswer(conflict=f"надпись указывает на л/с {', '.join(others)}, по табличке — {account}")
    if extra:
        return NoteAnswer(why=f"лишнее: {_show(extra)}")
    return _with_reading(NoteAnswer(understood=True, why=f"подтверждает л/с {account}"), readings)


def _unknown(tokens: list, index: AccountIndex, lasts: dict, barrel: Optional[int]) -> list[NoteAnswer]:
    """Счёт не известен: счета, которыми надпись объясняется целиком и
    которым подходит показание с барабана (нет барабана — по надписи)."""
    candidates = sorted({a for kind, n in tokens if kind != READING and not (kind == "" and _part_of(n, barrel))
                         for a in _unique(index, n)})
    out, why = [], []
    for t in candidates:
        last = lasts.get(t)
        readings, extra = _explain(tokens, t, index, last)
        seen = barrel if barrel is not None else next(iter(readings)) if len(readings) == 1 else None
        if extra:
            why.append(f"л/с {t}? лишнее: {_show(extra)}")
        elif not plausible(seen, last):
            why.append(f"л/с {t}? показание {seen if seen is not None else '—'} ему не подходит "
                       f"(прошлое {last if last is not None else '—'})")
        else:
            out.append(_with_reading(NoteAnswer(account=t, understood=True, why=f"л/с {t} по надписи"), readings))
    return out or [NoteAnswer(why="; ".join(why) or f"не запись счёта: {_show(tokens)}")]


def interpret(lines: list[str], index: AccountIndex, lasts: dict, account: Optional[str] = None,
              barrel: Optional[int] = None, unread: int = 0) -> NoteAnswer:
    """Надпись (строки сверху вниз) → ответ.
    lasts — {счёт: прошлое показание}; account — счёт, уже известный по
    табличке; barrel — показание с барабана, если прочитано (подтверждает
    счёт по надписи); unread — сколько строк похожи на часть надписи, но не
    прочитаны (overlay.Dropped.text_like): тогда надпись понята не целиком."""
    if unread:
        return NoteAnswer(why="часть надписи не прочитана")
    hyps = hypotheses(lines)
    if hyps is None:
        return NoteAnswer(why="слишком много способов прочитать")
    answers = []
    for h in hyps:
        tokens = [tok for t in h for tok in parse(t)]
        if not tokens:
            continue
        if account is not None:
            answers.append(_known(tokens, index, lasts, str(account)))
        else:
            answers += _unknown(tokens, index, lasts, barrel)
    if not answers:
        return NoteAnswer(why="чисел нет")
    ok = [a for a in answers if a.understood]
    if not ok:                                      # спор — если ни одно прочтение не согласно с табличкой
        return next((a for a in answers if a.conflict), answers[0])
    accounts = sorted({a.account for a in ok if a.account})
    if len(accounts) > 1:
        return NoteAnswer(conflict=f"надпись можно понять как л/с {' или '.join(accounts)}")
    readings = sorted({a.reading for a in ok if a.reading is not None})
    best = next((a for a in ok if a.reading is not None), ok[0])
    if len(readings) > 1:
        return NoteAnswer(account=best.account, understood=True,
                          why=f"показание можно понять как {' или '.join(map(str, readings))}")
    return best
