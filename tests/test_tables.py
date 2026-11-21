"""Сортировка списков окна по заголовкам (src/gmr/operator/tables.py)."""
from src.gmr.operator.tables import number, sort_items


def test_number_reads_signs_spaces_and_commas():
    assert number("+362") == 362 and number("-12") == -12 and number("−12") == -12
    assert number("1 284") == 1284 and number("12,5") == 12.5 and number("00123") == 123
    assert number("") is None and number("abc") is None and number(None) is None


def test_sort_numbers_desc_and_empty_last():
    items = ["+5", "", "-10", "300", "7"]
    assert sort_items(items, lambda v: v, "number", True) == ["300", "7", "+5", "-10", ""]
    assert sort_items(items, lambda v: v, "number", False) == ["-10", "+5", "7", "300", ""]


def test_sort_text_ignores_case_and_keeps_ties_stable():
    items = [("б", 1), ("А", 2), ("а", 3), ("", 4)]
    assert sort_items(items, lambda it: it[0], "text", False) == [("А", 2), ("а", 3), ("б", 1), ("", 4)]
