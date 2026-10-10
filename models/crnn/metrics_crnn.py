"""
Метрики качества OCR: exact match и CER.
Используются в train_crnn.py и evaluate.py.
"""


def exact_match(preds: list[str], gts: list[str]) -> float:
    """Доля полностью правильно распознанных строк."""
    if not gts:
        return 0.0
    return sum(p == g for p, g in zip(preds, gts)) / len(gts)


def levenshtein(a: str, b: str) -> int:
    """Сколько замен, вставок и удалений символов отделяют a от b."""
    dp = list(range(len(b) + 1))
    for ca in a:
        ndp = [dp[0] + 1]
        for j, cb in enumerate(b):
            ndp.append(min(dp[j] + (ca != cb), dp[j + 1] + 1, ndp[-1] + 1))
        dp = ndp
    return dp[-1]


def cer(preds: list[str], gts: list[str]) -> float:
    """Character Error Rate (расстояние Левенштейна / суммарная длина GT)."""
    total_err   = sum(levenshtein(p, g) for p, g in zip(preds, gts))
    total_chars = sum(max(len(g), 1) for g in gts)
    return total_err / total_chars if total_chars > 0 else 0.0


def per_position_accuracy(
    preds: list[str],
    gts: list[str],
    max_len: int = 10,
) -> tuple[list[float], list[int]]:
    """
    Accuracy отдельно для каждой позиции символа.

    Возвращает:
        accs   — список точностей по позициям (длина max_len)
        counts — число примеров, вошедших в подсчёт каждой позиции
    """
    pos_correct = [0] * max_len
    pos_total   = [0] * max_len

    for p, g in zip(preds, gts):
        for i in range(min(len(g), max_len)):
            pos_total[i] += 1
            if i < len(p) and p[i] == g[i]:
                pos_correct[i] += 1

    accs = [
        pos_correct[i] / pos_total[i] if pos_total[i] > 0 else 0.0
        for i in range(max_len)
    ]
    return accs, pos_total
