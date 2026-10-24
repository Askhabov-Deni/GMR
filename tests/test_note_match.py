"""
Надпись контролёра → счёт и показание по таблице (src/gmr/domain/note_match.py).
Только строки и таблица; счета и показания выдуманные.
"""
import pytest

from src.gmr.domain import note_match as nm
from src.gmr.domain.note_match import AccountIndex, interpret, is_account_form

ACCOUNTS = ["1300017777", "1300024317", "0429005357", "0429000013", "1300000429",
            "1300001525", "1300000024", "1300051234", "1300004330", "1300014330"]
LASTS = {"1300017777": 51000, "1300024317": 4300, "0429005357": 1, "0429000013": 100,
         "1300000429": 7000, "1300001525": 50500, "1300000024": 10, "1300051234": 90000,
         "1300004330": 37000, "1300014330": 20000}
INDEX = AccountIndex(ACCOUNTS)


def known(lines, account="1300017777", unread=0):
    return interpret(lines, INDEX, LASTS, account=account, unread=unread)


def unknown(lines, barrel=51234):
    return interpret(lines, INDEX, LASTS, barrel=barrel)


# ─── Текст ───────────────────────────────────────────────────────────────────

def test_wrapped():
    assert nm.wrapped("лс04290", "05357") and nm.wrapped("пк3", "4567")
    assert not nm.wrapped("лс17777", "пок51234") and not nm.wrapped("24317", "")
    assert not nm.wrapped("пок", "123")


def test_attached():
    """Одна запись на двух строках: перенос числа или приписка своей строкой."""
    assert nm.attached("лс04290", "05357")
    assert nm.attached("51234", "пк3") and nm.attached("17777", "лц")     # приписка под числом
    assert nm.attached("лс", "17777") and nm.attached("пкз", "51234")     # над числом
    assert not nm.attached("пок51234", "лс17777") and not nm.attached("лс", "пок")
    assert not nm.attached("лс17777", "") and not nm.attached("", "17777")


def test_ze_read_as_three():
    """«ПКЗ12345» читается как «пк312345»: есть и вариант с «з»."""
    assert nm.spellings("пк312345") == ["пк312345", "пкз12345"]
    assert nm.spellings("24317") == ["24317"] and nm.spellings("лс3") == ["лс3"] and nm.spellings("п3") == ["п3"]
    assert nm.parse("пкз12345") == [("reading", "12345")]


@pytest.mark.parametrize("text, tokens", [
    ("пок51234", [("reading", "51234")]), ("лс17777", [("account", "17777")]),
    ("24317", [("", "24317")]), ("пк123лс45", [("reading", "123"), ("account", "45")]),
    ("ПОК7", [("reading", "7")]), ("лсок", []), ("с0042", [("", "0042")]),           # «с» — не приписка
    ("л/с17777", [("account", "17777")]),
    ("17743пок", [("reading", "17743")]), ("17743лс", [("account", "17743")]),        # приписка после числа
    ("123пок456", [("", "123"), ("reading", "456")]),                                 # «пок» — к числу после
    ("лс5пок", [("account", "5")]), ("лц17777", [("account", "17777")]),
    ("пск13297", [("", "13297")]), ("пкз5", [("reading", "5")]),                      # смешанная — не приписка
])
def test_parse(text, tokens):
    assert nm.parse(text) == tokens


def test_hypotheses():
    assert nm.hypotheses(["лс04290", "05357"]) == [["лс04290", "05357"], ["лс0429005357"]]
    assert nm.hypotheses(["пк3", "пок5"]) == [["пк3", "пок5"], ["пкз", "пок5"]]
    assert nm.hypotheses([]) == [] and nm.hypotheses([""]) == []
    assert nm.hypotheses(["лс1к3"] * 8) is not None and nm.hypotheses(["лс1к3"] * 9) is None   # 2⁹ = 512


# ─── Счета ───────────────────────────────────────────────────────────────────

def test_account_forms():
    assert is_account_form("24317", "1300024317") and is_account_form("0024317", "1300024317")
    assert is_account_form("9005357", "0429005357") and is_account_form("005357", "0429005357")
    assert is_account_form("5357", "0429005357") and is_account_form("4317", "1300024317")   # конец — от 4 цифр
    assert not is_account_form("357", "0429005357")
    assert not is_account_form("0005357", "0429005357")                  # конец — как записан в счёте
    assert not is_account_form("24315", "1300024317") and not is_account_form("314", "1300024317")
    assert is_account_form("042900013", "0429000013")                    # «0429» + «00013»
    assert is_account_form("13000017", "1300000017") and is_account_form("130017", "1300000017")
    assert not is_account_form("042900014", "0429000013")
    assert not is_account_form("013", "0429000013") and not is_account_form("0000", "1300010000")
    assert not is_account_form("04290", "0429000000")                    # остаток счёта — нули
    assert not is_account_form("лс", "1300024317")


def test_index_finds_all_accounts_written_so():
    assert INDEX.find("17777") == {"1300017777"}
    assert INDEX.find("004330") == {"1300004330"}                        # не 1300014330: там «014330»
    assert INDEX.find("4330") == {"1300004330", "1300014330"}           # и ядро, и конец другого счёта
    assert INDEX.find("0") == set() and INDEX.find("x") == set()
    assert AccountIndex(["1300017777", " 1300017777 ", "абв"]).find("17777") == {"1300017777"}


def test_plausible_window():
    lo, hi = nm.READING_WINDOW
    assert nm.plausible(51000 + lo, 51000) and nm.plausible(51000 + hi, 51000)
    assert not nm.plausible(51000 + lo - 1, 51000) and not nm.plausible(51000 + hi + 1, 51000)
    assert not nm.plausible(5, None) and not nm.plausible(None, 5)


# ─── Счёт известен по табличке ───────────────────────────────────────────────

def test_known_account_reading_from_note():
    a = known(["пок51234", "лс17777"])
    assert (a.reading, a.account, a.conflict, a.understood) == (51234, None, "", True)
    assert a.why == "подтверждает л/с 1300017777, показание 51234"
    assert known(["51234"]).reading == 51234                              # без приписки — подходит к прошлому
    assert known(["17743пок"], "1300024317").reading is None              # не подходит к прошлому 4300


def test_known_account_without_last_reading():
    """Прошлого показания нет — проверить нечем: показание только с «пок»."""
    idx = AccountIndex(["1300000001"])
    assert interpret(["пок5100"], idx, {}, account="1300000001").reading == 5100
    a = interpret(["5100"], idx, {}, account="1300000001")
    assert a.reading is None and a.why == "лишнее: 5100"


def test_known_account_tail_and_labels_below():
    """Как на фото замера 2026-10-24: «4317» у счёта 1300024317 — конец счёта,
    а не показание (хоть и подходит к прошлому 4300); «51234» / «ПКЗ» /
    «17777» / «ЛЦ» — приписки строкой ниже."""
    a = known(["4317"], "1300024317")
    assert a.understood and a.reading is None and a.why == "подтверждает л/с 1300024317"
    a = known(["51234", "пк3", "17777", "лц"])
    assert (a.reading, a.understood) == (51234, True)
    assert unknown(["17777", "лц", "51234", "пк3"]).account == "1300017777"


def test_strange_letters_mean_misread():
    """Буквы, которые не приписка: «2431о» — цифра прочитана как «о»; «пск» —
    «пок» с ошибкой. Надпись прочитана не так — не используется."""
    a = known(["2431о", "51234"])
    assert (a.reading, a.understood, a.why) == (None, False, "непонятные буквы «о»")
    assert unknown(["пск51234", "17777"]).account is None
    assert known(["кк", "пок51234"]).reading is None                   # и строка из одних непонятных букв
    assert known(["л/с17777", "пк3", "51234"]).reading == 51234         # приписки — не «непонятные»


def test_label_needs_its_number():
    """Приписка своей строкой — к соседнему числу: «214» / «ЛЦ» (обрывок
    «21438») — запись счёта 214, а не показание 214."""
    a = known(["214", "лц"], "1300000024")
    assert a.reading is None and not a.understood
    assert known(["пкз"]).why == "чисел нет"


def test_known_account_ze_and_wrap():
    """«ПКЗ» / «51234»: переносом «пк3»+«51234» и «з» — одно показание 51234."""
    assert known(["пк3", "51234"]).reading == 51234
    assert known(["лс0429", "005357"], "0429005357").understood


def test_known_account_extra_number_not_understood():
    """Обрывок или лишнее число — надпись не используется. «пок» — только
    показание (даже если так пишется счёт), число длиннее 6 цифр — не показание."""
    for lines in (["лс0"], ["1525"], ["лс042"], ["пок51234", "1525"], ["пок99999"], ["пок17777"]):
        a = known(lines)
        assert (a.reading, a.conflict, a.understood) == (None, "", False), lines
        assert a.why.startswith("лишнее")
    assert known(["пкз"]).why == "чисел нет" and known([]).why == "чисел нет"
    a = interpret(["1234567"], AccountIndex(["1300000001"]), {"1300000001": 1234000}, account="1300000001")
    assert a.reading is None and a.why == "лишнее: 1234567"


def test_known_account_conflict():
    """Явно другой счёт: «лс…» или число длиннее показания — спор."""
    assert known(["лс24317"]).conflict == "надпись указывает на л/с 1300024317, по табличке — 1300017777"
    assert known(["1300024317"]).conflict
    assert not known(["24317"]).conflict                  # без «лс», 5 цифр — может быть и показанием
    assert known(["лс24317", "лс17777"]).conflict
    assert known(["лс51234"]).conflict                    # «лс» — счёт, хоть и подошло бы как показание
    assert known(["лс2", "4317"]).conflict                # спор — только если число перенесено


def test_known_account_two_readings():
    a = known(["51234", "51300"])
    assert a.reading is None and a.understood and "51234, 51300" in a.why
    a = known(["пк3", "2"], "1300000024")                 # «пк32» или «пкз» + «2» — оба подходят к прошлому 10
    assert a.reading is None and a.why == "показание можно понять как 2 или 32"


def test_unread_part_of_note():
    a = known(["пок51234"], unread=1)
    assert (a.reading, a.understood, a.why) == (None, False, "часть надписи не прочитана")


# ─── Счёт не известен ────────────────────────────────────────────────────────

def test_account_from_note_confirmed_by_reading():
    a = unknown(["пок51234", "лс17777"])
    assert (a.account, a.reading, a.why) == ("1300017777", 51234, "л/с 1300017777 по надписи, показание 51234")
    assert unknown(["17777", "51234"], barrel=None).account == "1300017777"   # без барабана — по надписи
    assert unknown(["лс0429", "000013"], barrel=150).account == "0429000013"
    assert unknown(["л/с17777"]).account == "1300017777"


def test_account_from_note_needs_fitting_reading():
    a = unknown(["24317"], barrel=99999)
    assert a.account is None and "показание 99999 ему не подходит (прошлое 4300)" in a.why
    assert unknown(["24317"], barrel=None).account is None                # нечем подтвердить
    assert interpret(["24317"], INDEX, {}, barrel=4400).account is None   # прошлого нет


def test_short_number_of_several_accounts_picked_by_reading():
    """«4330» — ядро 1300004330 и конец 1300014330: тот, к прошлому которого
    подходит показание с барабана (решение владельца 2026-10-26)."""
    a = unknown(["4330"], barrel=37200)
    assert (a.account, a.why) == ("1300004330", "л/с 1300004330 по надписи")
    assert unknown(["лс4330"], barrel=20100).account == "1300014330"
    assert unknown(["4330"], barrel=60000).account is None              # ни к одному не подходит
    assert unknown(["4330"], barrel=43305).why == "не запись счёта: 4330"  # кусок показания — не счёт
    idx = AccountIndex(["1300004330", "1300014330"])
    a = interpret(["4330"], idx, {"1300004330": 37000, "1300014330": 37100}, barrel=37200)
    assert a.account is None and a.conflict == "надпись можно понять как л/с 1300004330 или 1300014330"


def test_short_number_of_too_many_accounts_is_not_account():
    accounts = ["1300004330", "1300014330", "1300024330", "0429004330"]
    idx = AccountIndex(accounts)
    a = interpret(["4330"], idx, dict.fromkeys(accounts, 37000), barrel=37200)
    assert a.account is None and a.why == "4330 — так записывается 4 счетов: без таблички не понять какой"
    assert interpret(["4330"], AccountIndex(accounts[:3]), dict.fromkeys(accounts, 37000), barrel=37200).conflict


def test_number_from_barrel_is_not_account():
    """«1525» при показании 15253 — кусок показания, не счёт 1300001525."""
    assert unknown(["1525"], barrel=51525).account is None
    assert unknown(["лс1525"], barrel=51525).account == "1300001525"      # с «лс» — счёт


def test_fragments_give_nothing():
    for lines in (["лс0"], ["лс042"], ["пкз"], ["5"]):
        assert unknown(lines).account is None and not unknown(lines).conflict, lines


def test_two_accounts_fit():
    """Оба числа — записи счетов, и каждое подходит как показание другого."""
    idx = AccountIndex(["1300051000", "1300051100"])
    a = interpret(["51000", "51100"], idx, {"1300051000": 51050, "1300051100": 50990}, barrel=51040)
    assert a.account is None and a.conflict.startswith("надпись можно понять как л/с")


def test_too_many_readings():
    assert known(["пк3"] * 9).why == "слишком много способов прочитать"
