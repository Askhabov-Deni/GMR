"""
models/ctc_lexicon.py — чтение по закрытому списку (словарю) для CTC-моделей.

Модель CRNN+CTC отдаёт не одну строку, а вероятности символов по шагам.
Обычное чтение («жадное») берёт на каждом шаге самый вероятный символ — так
одна ошибка в символе даёт номер, которого нет в таблице. Но правильный
ответ всегда один из номеров таблицы абонентов. CTC умеет посчитать
вероятность ЛЮБОЙ заданной строки: P(строка | картинка) — ровно то, чему
модель училась. Поэтому:

  1. считаем log P(s | картинка) для каждой строки s словаря (это несколько
     тысяч строк — миллисекунды на процессоре);
  2. строки собраны в группы (группа = лицевой счёт, строки — варианты его
     записи, например 00065 и 65); вероятность группы — сумма вероятностей
     её строк;
  3. уверенность группы — её доля среди всех групп. Если модель уверенно
     видит строку, которой нет в словаре (новый абонент, написано не то),
     она тоже участвует в знаменателе с весом OOV_WEIGHT — уверенность
     номера из таблицы тогда падает, и номер не «натягивается» на таблицу.

Так ошибка в одном символе не мешает найти счёт, а уверенность честная:
если два счёта похожи и модель не различает их — уверенность ~0.5.

Модуль общий: подходит и для серийника (models/crnn), и для лицевого счёта
по надписи маркером (models/account). Не зависит от таблицы и pandas.
"""
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# Вес гипотезы «написано то, чего нет в словаре» относительно одной группы
# словаря (см. 3 выше). 0.05: строка вне словаря перевешивает номер из
# таблицы, только если модель видит её в 20+ раз вероятнее.
OOV_WEIGHT = 0.05
_CHUNK = 8192            # строк за один вызов ctc_loss (ограничивает память)


@dataclass
class LexiconMatch:
    """Результат чтения по словарю.

    greedy      — что прочитано без словаря (жадно);
    group       — лучшая группа (например, лицевой счёт) или None;
    string      — строка лучшей группы, которая лучше всего подходит к картинке;
    confidence  — доля лучшей группы (0..1), см. докстринг модуля;
    top         — [(группа, доля), ...] — несколько лучших, по убыванию.
    """
    greedy: str
    group: Optional[str]
    string: Optional[str]
    confidence: float
    top: list


class CompiledLexicon:
    """Словарь, подготовленный для быстрого подсчёта: строки → тензоры.

    groups — {группа: [строки]}. Одна строка может входить в несколько групп
    (например, у двух счетов одинаковые последние 5 цифр) — тогда эти группы
    делят её вероятность и ни одна не будет уверенной.
    """

    def __init__(self, groups: Mapping[str, Sequence[str]], alphabet: str):
        self.alphabet = alphabet
        self._ids: dict[str, int] = {}                  # строка → её номер
        self.strings: list[str] = []
        self._targets: list[torch.Tensor] = []
        self.group_names: list[str] = []
        self._membership: list[tuple[int, int]] = []    # (строка, группа)
        self._add(groups)
        self._freeze()

    def _add(self, groups: Mapping[str, Sequence[str]]) -> None:
        index = {c: i for i, c in enumerate(self.alphabet)}
        for g, variants in groups.items():
            gi = len(self.group_names)
            self.group_names.append(str(g))
            for s in dict.fromkeys(variants):
                if not s or any(c not in index for c in s):
                    continue
                si = self._ids.get(s)
                if si is None:
                    si = self._ids[s] = len(self.strings)
                    self.strings.append(s)
                    self._targets.append(torch.tensor([index[c] for c in s], dtype=torch.long))
                self._membership.append((si, gi))

    def _freeze(self) -> None:
        self.string_set = frozenset(self._ids)
        self._lengths = torch.tensor([len(s) for s in self.strings], dtype=torch.long)
        self._member_string = torch.tensor([m[0] for m in self._membership], dtype=torch.long)
        self._member_group = torch.tensor([m[1] for m in self._membership], dtype=torch.long)

    def extended(self, extra: Mapping[str, Sequence[str]]) -> "CompiledLexicon":
        """Словарь с ещё несколькими группами в конце — то же, что
        CompiledLexicon({**groups, **extra}) (групп extra в словаре ещё нет), но без пересборки всех строк
        (1.6.0: надпись маркером сверяется со счётом серийника другого кода —
        раньше на это уходило полсекунды на фото). Сам словарь не меняется."""
        new = object.__new__(CompiledLexicon)
        new.alphabet = self.alphabet
        new._ids, new.strings, new._targets = dict(self._ids), list(self.strings), list(self._targets)
        new.group_names, new._membership = list(self.group_names), list(self._membership)
        new._add(extra)
        new._freeze()
        return new

    def __len__(self) -> int:
        return len(self.group_names)

    def edit_distances(self, text: str, rows: np.ndarray) -> np.ndarray:
        """Сколько правок (вставка, удаление, замена символа) от text до
        каждой строки rows (номера строк) — сразу для всех, numpy."""
        if getattr(self, "_padded", None) is None:
            index = {c: i for i, c in enumerate(self.alphabet)}
            width = max((len(s) for s in self.strings), default=0)
            self._padded = np.full((len(self.strings), width), -1, np.int16)
            for i, s in enumerate(self.strings):
                self._padded[i, :len(s)] = [index[c] for c in s]
            self._np_lengths = np.array([len(s) for s in self.strings], np.int64)
        index = {c: i for i, c in enumerate(self.alphabet)}
        strings, width = self._padded[rows], self._padded.shape[1]
        prev = np.tile(np.arange(width + 1, dtype=np.int16), (len(rows), 1))     # от "" до префикса строки
        for i, ch in enumerate(text, 1):
            sub = prev[:, :-1] + (strings != index.get(ch, -2))
            cur = np.empty_like(prev)
            cur[:, 0] = i
            for j in range(1, width + 1):
                cur[:, j] = np.minimum(np.minimum(prev[:, j], cur[:, j - 1]) + 1, sub[:, j - 1])
            prev = cur
        return prev[np.arange(len(rows)), self._np_lengths[rows]]


def string_log_likelihoods(log_probs: torch.Tensor, targets: Sequence[torch.Tensor], blank: int) -> torch.Tensor:
    """log P(строка | картинка) по CTC для каждой строки.

    log_probs — (T, C) log-softmax выхода модели для ОДНОЙ картинки;
    targets   — строки как тензоры индексов (непустые).
    Строка длиннее, чем может уместиться в T шагов, получает -inf.
    """
    T, C = log_probs.shape
    out = []
    for start in range(0, len(targets), _CHUNK):
        chunk = targets[start:start + _CHUNK]
        n = len(chunk)
        lengths = torch.tensor([len(t) for t in chunk], dtype=torch.long)
        flat = torch.cat(list(chunk)) if n else torch.empty(0, dtype=torch.long)
        lp = log_probs.unsqueeze(1).expand(T, n, C).contiguous()
        nll = F.ctc_loss(lp, flat, torch.full((n,), T, dtype=torch.long), lengths,
                         blank=blank, reduction="none", zero_infinity=False)
        out.append(-nll)
    return torch.cat(out) if out else torch.empty(0)


def greedy_decode(log_probs: torch.Tensor, alphabet: str, blank: int) -> str:
    """(T, C) → строка: самый вероятный символ на шаге, повторы и blank убраны."""
    chars, prev = [], -1
    for idx in log_probs.argmax(1).tolist():
        if idx != prev and idx != blank:
            chars.append(alphabet[idx])
        prev = idx
    return "".join(chars)


def _greedy_log_likelihood(log_probs: torch.Tensor, text: str, alphabet: str, blank: int) -> float:
    if not text:
        return float(log_probs[:, blank].sum())
    target = torch.tensor([alphabet.index(c) for c in text], dtype=torch.long)
    return float(string_log_likelihoods(log_probs, [target], blank)[0])


def _group_logsumexp(values: torch.Tensor, groups: torch.Tensor, n_groups: int) -> torch.Tensor:
    """logsumexp значений внутри каждой группы (без цикла по группам)."""
    peak = torch.full((n_groups,), float("-inf")).scatter_reduce(
        0, groups, values, reduce="amax", include_self=True)
    safe_peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
    total = torch.zeros(n_groups).index_add(0, groups, torch.exp(values - safe_peak[groups]))
    return torch.log(total) + safe_peak


@torch.no_grad()
def match(log_probs: torch.Tensor, lexicon: CompiledLexicon, blank: int,
          oov_weight: float = OOV_WEIGHT, top_k: int = 3, len_window: Optional[int] = None,
          max_edits: Optional[int] = None) -> LexiconMatch:
    """Лучшая группа словаря для одной картинки. log_probs — (T, C).

    len_window — считать только строки, длина которых отличается от жадного
    чтения не больше чем на столько символов (у остальных вероятность
    ничтожна: каждая лишняя или пропущенная цифра — в разы меньше). Ускоряет
    большие словари (надпись маркером — до 14 вариантов на счёт); None — все.

    max_edits — и только строки, до которых от жадного чтения не больше
    стольких правок (1.6.0: надпись маркером — 3; сверка на 1442 надписях:
    тот же счёт и то же решение у всех, уверенность — до 0.05 иначе, в 6 раз
    быстрее). У далёких строк вероятность ничтожна. None — все."""
    log_probs = log_probs.detach().float().cpu()
    greedy = greedy_decode(log_probs, lexicon.alphabet, blank)
    if not lexicon.strings:
        return LexiconMatch(greedy, None, None, 0.0, [])

    if len_window is not None and greedy:
        near = torch.nonzero((lexicon._lengths - len(greedy)).abs() <= len_window).flatten().tolist()
    else:
        near = range(len(lexicon.strings))
    if max_edits is not None and greedy and len(near):
        rows = np.asarray(list(near), np.int64)
        close = rows[lexicon.edit_distances(greedy, rows) <= max_edits].tolist()
        near = close or near                 # близких нет — как без отбора: ответ не хуже прежнего
    ll = torch.full((len(lexicon.strings),), float("-inf"))
    if near:
        ll[list(near)] = string_log_likelihoods(log_probs, [lexicon._targets[i] for i in near], blank)
    # группа = logsumexp по её строкам
    group_ll = _group_logsumexp(ll[lexicon._member_string], lexicon._member_group,
                                len(lexicon.group_names))

    terms = [group_ll]
    if greedy not in lexicon.string_set and oov_weight > 0:
        g_ll = _greedy_log_likelihood(log_probs, greedy, lexicon.alphabet, blank)
        terms.append(torch.tensor([g_ll + float(torch.log(torch.tensor(oov_weight)))]))
    norm = torch.logsumexp(torch.cat(terms), dim=0)
    if not torch.isfinite(norm):
        return LexiconMatch(greedy, None, None, 0.0, [])

    share = torch.exp(group_ll - norm)
    order = torch.argsort(share, descending=True)[:top_k].tolist()
    best = order[0]
    best_strings = lexicon._member_string[lexicon._member_group == best]
    string = lexicon.strings[int(best_strings[torch.argmax(ll[best_strings])])]
    top = [(lexicon.group_names[i], float(share[i])) for i in order]
    return LexiconMatch(greedy, lexicon.group_names[best], string, float(share[best]), top)
