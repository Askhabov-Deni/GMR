"""Код участка контролёра (src/gmr/domain/area_code.py)."""
from src.gmr.domain.area_code import (
    MIN_ACCOUNTS,
    account_code,
    accounts_by_controller,
    controller_code,
    only_code,
    other_code_note,
)


def test_account_code():
    assert account_code("1300024317") == "1300"
    assert account_code(" 0429000123 ") == "0429"
    assert account_code("429000123") == "0429"        # Excel съел ведущий ноль
    assert account_code("A-1") is None
    assert account_code("13000243170") is None        # длиннее счёта
    assert account_code("") is None


def _accounts(code, n, start=0):
    return [f"{code}{i:06d}" for i in range(start, start + n)]


def test_controller_code_needs_enough_accounts():
    assert controller_code(_accounts("1300", MIN_ACCOUNTS - 1)) is None
    assert controller_code(_accounts("1300", MIN_ACCOUNTS)) == "1300"
    # один и тот же счёт несколько раз — один счёт
    assert controller_code(_accounts("1300", 1) * 50) is None


def test_controller_code_tolerates_a_few_others_but_not_a_mix():
    assert controller_code(_accounts("1300", 160) + _accounts("1303", 2)) == "1300"   # как в мае
    assert controller_code(_accounts("1300", 20) + _accounts("0429", 20)) is None


def test_accounts_by_controller_counts_only_table_accounts():
    table = {"1300000001", "1300000002"}
    rows = [
        {"source_folder": "Аюб", "account_id": "1300000001"},
        {"source_folder": "Аюб", "account_id": "0123456"},        # номер «Нет в базе» — не счёт таблицы
        {"source_folder": "Фатима Я", "account_id": "1300000002"},
        {"source_folder": "", "account_id": "1300000002"},          # pre_existing — без папки
        {"source_folder": "Аюб", "account_id": ""},
    ]
    assert accounts_by_controller(rows, table) == {"Аюб": {"1300000001"}, "Фатима Я": {"1300000002"}}


def test_only_code_keeps_accounts_without_code():
    groups = {"1300000001": ("1",), "0429000001": ("04291",), "A-1": ("A1",)}
    assert only_code(groups, "1300") == {"1300000001": ("1",), "A-1": ("A1",)}
    assert only_code(groups, None) is groups


def test_other_code_note():
    assert other_code_note("1303000001", "1300") == "счёт не с участка контролёра: код 1303, у контролёра 1300"
    assert other_code_note("1300000001", "1300") is None
    assert other_code_note("1303000001", None) is None
    assert other_code_note("A-1", "1300") is None
