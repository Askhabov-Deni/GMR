"""
Известные ошибки (аудит 2026-10-01, docs/BACKLOG.md → «Дорожная карта»).

Тест известной ошибки описывает, как ДОЛЖНО быть, и сейчас падает — он
помечен `known_issue` (xfail, strict). Когда ошибку исправят, тест пройдёт,
и pytest сообщит об этом как об ошибке (XPASS strict): снимите пометку —
тест станет обычным.

Сейчас известных ошибок нет:
  - этапа 2 исправлены в папке месяца (этапы 2.2b, 2.3a): tests/test_month_run.py,
    tests/test_operator.py, tests/test_program2_verify.py; старый режим с его
    ошибками убран (2.3b);
  - этапа 3 («неудачное фото блокирует хорошее», «фото в корне фото\\ прячет
    папки контролёров») исправлены месячным циклом: tests/test_month_cycle.py.

Новую найденную ошибку записывайте сюда так:

    @known_issue("этап N", "что не так")
    def test_...(tmp_path): ...
"""
import pytest


def known_issue(stage: str, what: str):
    return pytest.mark.xfail(strict=True, raises=AssertionError,
                             reason=f"известная ошибка, {stage}: {what}")
