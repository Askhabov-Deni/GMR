"""Качество распознавания по работе оператора (src/gmr/application/quality.py)."""
from src.gmr.application.quality import month_quality, quality_lines, quality_row, wilson


def auto(h, outcome, **kw):
    return {"source": "auto", "photo_hash": h, "outcome": outcome, **kw}


def manual(h, outcome, **kw):
    return {"source": "manual", "photo_hash": h, "outcome": outcome, **kw}


ROWS = [
    auto("a", "PLUS", verified_by="Оп"),
    auto("b", "MINUS", verified_by="Оп", notes="исправлено при проверке"),
    auto("c", "PLUS", verified_by="Оп", notes="исправлено при проверке (абонент 1 → 2)"),
    auto("d", "PLUS"),                                       # не проверено
    auto("e", "DIGITS_ERROR", model_reading_str="0123?"),
    manual("e", "PLUS", reading="01234", model_reading_str="01234"),
    auto("f", "SERIAL_NOT_FOUND"),
    manual("f", "PLUS", reading="00500", model_reading_str="00600"),
    auto("g", "NO_METER"),
    manual("g", "UNREADABLE"),
    auto("h", "NO_SERIAL"),                                  # ещё в очереди
    auto("a", "REPEAT"),                                     # повтор — не считается
]


def test_month_quality_counts():
    q = month_quality(ROWS)
    assert (q.photos, q.written, q.to_operator) == (8, 4, 4)
    assert (q.checked, q.corrected, q.corrected_account, q.corrected_reading) == (3, 2, 1, 1)
    assert dict(q.decided) == {"принято": 2, "нечитаемо": 1}
    assert (q.model_full, q.model_full_ok) == (2, 1)            # «0123?» — не все цифры, «00600» ≠ «00500»
    rate, lo, hi = q.error_rate()
    assert round(rate, 3) == 0.667 and lo < rate < hi


def test_last_auto_reading_counts_after_reread():
    q = month_quality([auto("a", "DIGITS_ERROR"), auto("a", "PLUS", verified_by="Оп")])
    assert (q.photos, q.written, q.to_operator, q.checked) == (1, 1, 0, 1)


def test_wilson_is_honest_with_zero_errors():
    lo, hi = wilson(0, 60)
    assert lo == 0 and 0.05 < hi < 0.07                          # 0 из 60 — это «до 6%», а не «0%»
    assert wilson(0, 0) == (0.0, 1.0)


def test_lines_say_how_much_to_check():
    text = "\n".join(quality_lines(month_quality(ROWS)))
    assert "записала сама: 4 из 8 фото (50,0%)" in text
    assert "исправлено 2 (показание 1, абонент 1) — ошибок 66,7%" in text
    assert "проверьте хотя бы 30 фото" in text
    many = [auto(str(i), "PLUS", verified_by="Оп") for i in range(99)] + [
        auto("x", "PLUS", verified_by="Оп", notes="исправлено при проверке")] + [
        auto(f"u{i}", "PLUS") for i in range(100)]
    text = "\n".join(quality_lines(month_quality(many)))
    assert "ошибок 1,0%" in text and "среди 200 записанных программой неверных — примерно 2" in text
    assert "фото ещё не читались" in quality_lines(month_quality([]))[0]


def test_quality_row_for_months():
    row = quality_row("Май", month_quality(ROWS))
    assert row["месяц"] == "Май" and row["сам %"] == "50,0%" and row["ошибок %"] == "66,7%"


def test_queue_photo_closed_by_other_photo_counts_as_decided():
    rows = [auto("q", "DIGITS_ERROR", account_id="A"), auto("q", "REPEAT", processed_by="Оп")]
    q = month_quality(rows)
    assert q.to_operator == 1 and dict(q.decided) == {"закрыто: у счёта уже есть показание": 1}
