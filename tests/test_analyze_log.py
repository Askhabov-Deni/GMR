"""
Фаза 5: tools/analyze_log.py — качество чтения по логу обработки.
"""
import csv

from src.gmr.domain import PipelineConfig
from src.gmr.storage import LOG_COLUMNS
from src.gmr.domain.serial_match import one_edit_neighbors
from tools import analyze_log as al


def row(name, outcome, source="auto", folder="", **kw):
    r = {c: "" for c in LOG_COLUMNS}
    r.update(original_filename=name, outcome=outcome, source=source, source_folder=folder, **kw)
    return r


CFG = PipelineConfig()


# ─── 1. Фото ─────────────────────────────────────────────────────────────────

def test_last_state_skips_repeat_and_joins_old_rows_without_folder():
    rows = [
        # прогон 1 (до подпапок в логе): ошибка
        row("a.jpg", "NO_METER"),
        # прогон 2: то же фото перечитано → PLUS, теперь с подпапкой
        row("a.jpg", "PLUS", folder="Аюб"),
        # прогон 3: REPEAT — не перетирает PLUS
        row("a.jpg", "REPEAT", folder="Аюб"),
        # фото, у которого только REPEAT (копия)
        row("b.jpg", "REPEAT", folder="Аюб"),
        # pre_existing из таблицы — не фото
        row("__pre_existing__", "PLUS", source="pre_existing"),
    ]
    state = al.last_state_per_photo(rows)
    assert {k: v["outcome"] for k, v in state.items()} == {
        ("Аюб", "a.jpg"): "PLUS", ("Аюб", "b.jpg"): "REPEAT"}


def test_same_name_in_two_folders_is_two_photos():
    rows = [row("IMG_1.jpg", "PLUS", folder="A"), row("IMG_1.jpg", "NO_METER", folder="B")]
    assert len(al.last_state_per_photo(rows)) == 2


# ─── 2. Цифры ────────────────────────────────────────────────────────────────

def test_digit_stats():
    photos = [
        row("1", "DIGITS_ERROR", model_reading_str="?0040",
            notes="low conf digits: pos0(pred=1,conf=0.417)  →  '?0040'"),
        row("2", "DIGITS_ERROR", model_reading_str="??010",
            notes="low conf digits: pos0(pred=6,conf=0.413), pos1(pred=8,conf=0.279)  →  '??010'"),
        row("3", "DIGITS_ERROR", notes="expected 5 digits, got 3"),
        row("4", "PLUS", model_reading_str="12345"),
        row("5", "SERIAL_NOT_FOUND", model_reading_str="3?196"),
    ]
    d = al.digit_stats(photos, CFG)
    assert d["with_reading"] == 4 and d["with_unsure"] == 3 and d["only_first_unsure"] == 1
    assert d["unsure_by_pos"] == {0: 2, 1: 2}
    assert d["low_conf_by_pos"] == {0: 2, 1: 1}
    assert d["low_conf_pos0_pred"] == {"1": 1, "6": 1}
    assert d["errors"] == {"неуверенные цифры": 2, "не 5 цифр / не найдены": 1}
    assert d["first_digit_read"] == {"1": 1}
    assert d["forgiven_positions"] == [3, 4]


# ─── 3. Серийники ────────────────────────────────────────────────────────────

def _dist_le1(a, b):
    """Эталон для проверки: расстояние Левенштейна ≤ 1 (динамика, медленно, но очевидно)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] <= 1


def test_one_edit_neighbors_match_levenshtein():
    import random
    rnd = random.Random(1)
    for _ in range(300):
        a = "".join(rnd.choice("0123") for _ in range(rnd.randint(1, 5)))
        b = "".join(rnd.choice("0123") for _ in range(rnd.randint(1, 5)))
        assert (b in one_edit_neighbors(a, "0123") or a == b) == _dist_le1(a, b), (a, b)


def test_describe_edit():
    assert al.describe_edit("1284567", "1234567") == "замена 3→8"
    assert al.describe_edit("234567", "1234567") == "пропущен символ (первый)"
    assert al.describe_edit("123456", "1234567") == "пропущен символ (последний)"
    assert al.describe_edit("123567", "1234567") == "пропущен символ (в середине)"
    assert al.describe_edit("12345678", "1234567") == "лишний символ (последний)"


def test_one_char_matches_counts_candidates():
    table = {"1234567", "1234568", "7654321"}
    assert {t for _, t in al.one_char_matches("1234569", table)} == {"1234567", "1234568"}
    assert al.one_char_matches("7654320", table) == [("7654320", "7654321")]


def test_chance_rate_small_for_sparse_table():
    table = {f"{i:07d}" for i in range(0, 10_000_000, 99_991)}   # ~100 номеров
    assert al.chance_one_char_rate([7], table, n=300) < 0.05


def test_classify_not_found():
    table = {"0045618", "9076647", "1012688"}
    assert al.classify_not_found("9076617", table) == "ошибка в одном символе"
    assert al.classify_not_found("45617", table) == "ошибка в одном символе"   # '00'+… на 1 символ
    assert al.classify_not_found("7664", table) == "обрезан (часть номера из таблицы)"
    assert al.classify_not_found("5555555", table) == "похожего в таблице нет"


def test_serial_stats():
    photos = ([row(str(i), "PLUS", serial_id="1234567", model_serial_conf="0.99") for i in range(5)]
              + [row("x", "SERIAL_NOT_FOUND", serial_id="12345", model_serial_conf="0.7"),
                 row("y", "SERIAL_NOT_FOUND", serial_id="7654321", model_serial_conf="0.95")])
    s = al.serial_stats(photos, {"1234567", "7654329", "7654328"})
    assert s["found"] == 5 and s["not_found"] == 2
    assert s["typical_len"] == [7] and s["not_found_untypical_len"] == 1
    assert s["not_found_kinds"] == {"обрезан (часть номера из таблицы)": 1, "ошибка в одном символе": 1}
    assert s["one_char_unique"] == 0 and s["one_char_several"] == 1   # 7654321 → …29 или …28
    assert s["table_size"] == 3 and 0 <= s["chance_rate"] <= 1


# ─── 4. Точность по проверке оператора ───────────────────────────────────────

def test_operator_accuracy():
    rows = [
        # проверено, верно
        row("a", "PLUS", reading="1200", model_reading_str="01200", verified_by="Оператор"),
        # проверено, исправлено (program2 меняет reading и дописывает notes)
        row("b", "PLUS", reading="1350", model_reading_str="01300", verified_by="Оператор",
            notes=f"{al.CORRECTED_NOTE}"),
        # не проверено — не считается
        row("c", "MINUS", reading="1", model_reading_str="00001"),
        # SERIAL_NOT_FOUND: модель прочитала серийник, оператор нашёл счётчик по другому номеру
        row("d", "SERIAL_NOT_FOUND", serial_id="123456", photo_hash="h1"),
        row("d", "PLUS", source="manual", serial_id="1234567", reading="500",
            model_reading_str="00500", photo_hash="h1"),
        # SERIAL_NOT_FOUND: номер прочитан верно, но в таблице его нет
        row("e", "SERIAL_NOT_FOUND", serial_id="7777777", photo_hash="h2"),
        row("e", "NOT_IN_DB", source="manual", serial_id="7777777", model_reading_str="0?100",
            photo_hash="h2"),
        # DIGITS_ERROR, оператор ввёл показание; модель прочитала не все цифры —
        # в точность «модель прочитала все цифры» не входит
        row("f", "PLUS", source="manual", reading="700", model_reading_str="?0700"),
        # отметка «проверено» попала в строку REPEAT (ошибка program2) — не учитывается
        row("g", "REPEAT", verified_by="Оператор"),
    ]
    a = al.operator_accuracy(rows)
    assert a["verified"] == 2 and a["verified_reading_ok"] == 1 and a["verified_corrected"] == 1
    assert a["verified_other"] == 1
    assert a["manual"] == 3 and a["manual_outcomes"] == {"PLUS": 2, "NOT_IN_DB": 1}
    assert a["manual_full_model_reading"] == 1 and a["manual_model_reading_ok"] == 1
    assert a["snf_with_answer"] == 2 and a["snf_found"] == 1 and a["snf_not_in_db"] == 1
    assert "guess" not in a   # без таблицы угадывание не проверяется


def test_guess_serial():
    table = {"1234567", "1234568", "7654321"}
    last = {"1234567": 1000.0, "1234568": 5000.0, "7654321": 300.0}
    rng = (0.0, 200.0)
    assert al.guess_serial("1234569", "01050", table, last, rng) == "1234567"   # из двух подходит один
    assert al.guess_serial("7654320", "09000", table, last, rng) is None       # расход не подходит
    assert al.guess_serial("7654320", "0?300", table, last, rng) is None       # показание не дочитано
    assert al.guess_serial("5555555", "00100", table, last, rng) is None       # похожих нет


def test_guess_checked_against_operator_answers():
    table = {"1234567", "1234568", "7654321", "1111111"}
    last = {"1234567": 1000.0, "1234568": 5000.0, "7654321": 300.0, "1111111": 0.0}
    rows = [
        # оператор нашёл 1234567 — догадка совпала
        row("a", "SERIAL_NOT_FOUND", serial_id="1234569", model_reading_str="01050", photo_hash="h1"),
        row("a", "PLUS", source="manual", serial_id="1234567", reading="1050", photo_hash="h1"),
        # оператор нашёл 1234568 — догадка была бы 1234567 (ошибка)
        row("b", "SERIAL_NOT_FOUND", serial_id="1234569", model_reading_str="01050", photo_hash="h2"),
        row("b", "MINUS", source="manual", serial_id="1234568", reading="1050", photo_hash="h2"),
        # оператор нашёл, а догадки нет (расход не подходит)
        row("c", "SERIAL_NOT_FOUND", serial_id="7654320", model_reading_str="09000", photo_hash="h3"),
        row("c", "PLUS", source="manual", serial_id="7654321", reading="9000", photo_hash="h3"),
        # «нет в базе», но догадка была бы — ложная
        row("d", "SERIAL_NOT_FOUND", serial_id="7654320", model_reading_str="00400", photo_hash="h4"),
        row("d", "NOT_IN_DB", source="manual", serial_id="7654320", photo_hash="h4"),
    ]
    a = al.operator_accuracy(rows, table, last, (0.0, 200.0))
    assert a["snf_found"] == 3 and a["snf_not_in_db"] == 1
    assert a["guess"] == {"right": 1, "wrong": 1, "no_guess": 1, "false_on_not_in_db": 1}


def test_report_without_operator_rows_says_so():
    text = al.build_report([row("a", "PLUS", model_reading_str="12345", serial_id="1234567")], CFG)
    assert "Проверенных (verified_by) и ручных (source=manual) строк нет" in text
    assert "--table" in text


def test_report_contains_no_serials_or_accounts():
    rows = [row("a.jpg", "SERIAL_NOT_FOUND", serial_id="9988776", account_id="1300014021",
                model_reading_str="12345")]
    text = al.build_report(rows, CFG, {"9988770"})
    assert "9988776" not in text and "1300014021" not in text and "a.jpg" not in text


def test_main_reads_csv_with_bom_and_writes_report(tmp_path):
    log = tmp_path / "meters_table_log.csv"
    with open(log, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        w.writeheader()
        w.writerow(row("a.jpg", "PLUS", model_reading_str="12345", serial_id="1234567"))
    out = tmp_path / "report.txt"
    text = al.main([str(log), "--out", str(out)])
    assert out.read_text(encoding="utf-8").startswith(text[:40])
    assert "PLUS" in text


def test_table_serials_keep_leading_zeros(tmp_path):
    cfg = PipelineConfig()
    table = tmp_path / "t.csv"
    # таблица читается, как при создании месяца: «;» или «,», номер без знаков по краям
    table.write_text(f"{cfg.col_serial};{cfg.col_account_id}\n0045618;1\n 9076647. ;2\n;3\n", encoding="utf-8")
    assert al.load_table_serials(table, cfg) == {"0045618", "9076647"}
    table.write_text(f"{cfg.col_serial},{cfg.col_account_id}\n0045618,1\n 9076647 ,2\n,3\n", encoding="utf-8")
    assert al.load_table_serials(table, cfg) == {"0045618", "9076647"}


def test_load_table_reads_xlsx_like_month(tmp_path):
    # выгрузка показания.xlsx и таблица компании в .xlsx читаются одинаково
    import openpyxl
    cfg = PipelineConfig()
    table = tmp_path / "показания.xlsx"
    wb = openpyxl.Workbook()
    wb.active.append([cfg.col_serial, cfg.col_account_id, cfg.col_last_reading])
    wb.active.append(["0045618", 1300000013, 1000])
    wb.save(table)
    assert al.load_table(table, cfg) == ({"0045618"}, {"0045618": 1000.0}, {"0045618": "1300000013"})


# ─── Проверка кандидатов по показанию ────────────────────────────────────────

def _found_with_deltas(deltas):
    return [row(f"f{i}", "PLUS" if d >= 0 else "MINUS", delta=str(d), serial_id="1111111")
            for i, d in enumerate(deltas)]


def test_plausible_delta_range():
    photos = _found_with_deltas(list(range(0, 200, 2)))   # 100 фото, расход 0..198
    lo, hi = al.plausible_delta_range(photos)
    assert 0 < lo < 20 and 180 < hi < 198      # края отрезаны: 5-й и 95-й процентили
    assert al.plausible_delta_range(_found_with_deltas([1, 2, 3])) is None   # мало данных


def test_reading_check():
    table_last = {"1234567": 1000.0, "1234568": 5000.0, "7654321": 300.0, "2222222": None}
    missing = [
        # два похожих, по показанию подходит только 1234567 (1050 − 1000 = 50)
        row("a", "SERIAL_NOT_FOUND", serial_id="1234569", model_reading_str="01050"),
        # один похожий и он подходит
        row("b", "SERIAL_NOT_FOUND", serial_id="7654320", model_reading_str="00320"),
        # один похожий, но расход не подходит (9000)
        row("c", "SERIAL_NOT_FOUND", serial_id="7654329", model_reading_str="09300"),
        # показание не дочитано — не проверяется
        row("d", "SERIAL_NOT_FOUND", serial_id="7654322", model_reading_str="0?300"),
        # у кандидата нет последнего показания — не подходит
        row("e", "SERIAL_NOT_FOUND", serial_id="2222223", model_reading_str="00100"),
    ]
    table = set(table_last)
    matches = [al.one_char_matches(r["serial_id"], table) for r in missing]
    res = al.reading_check(missing, matches, table_last, (0.0, 200.0))
    assert res == {
        "несколько похожих: подходит по показанию ровно один": 1,
        "один похожий: подходит по показанию ровно один": 1,
        "один похожий: не подходит ни один": 2,
        "проверено": 4,
    }


def test_report_has_reading_check_line():
    photos = _found_with_deltas(list(range(0, 200, 2))) + [
        row("a", "SERIAL_NOT_FOUND", serial_id="1234569", model_reading_str="01050")]
    text = al.build_report(photos, CFG, {"1234567", "1111111"}, {"1234567": 1000.0, "1111111": 0.0})
    assert "проверка кандидатов по показанию" in text
    assert "1234567" not in text and "1234569" not in text


def test_load_table_last_readings(tmp_path):
    cfg = PipelineConfig()
    table = tmp_path / "t.csv"
    table.write_text(
        f"{cfg.col_serial},{cfg.col_account_id},{cfg.col_last_reading},{cfg.col_new_reading}\n"
        "0045618,1,\"12345,5\",\n9076647,2,,\n0045618,3,1,\n", encoding="utf-8")
    serials, last, accounts = al.load_table(table, cfg)
    assert accounts == {"0045618": "1", "9076647": "2"}
    assert serials == {"0045618", "9076647"}
    assert last == {"0045618": 12345.5, "9076647": None}


def test_guess_details():
    table = {"1234567", "7654321"}
    last = {"1234567": 1000.0, "7654321": 300.0}
    rows = [
        row("a.jpg", "SERIAL_NOT_FOUND", serial_id="1234569", model_reading_str="01050",
            photo_hash="h1", folder="Аюб"),
        row("a.jpg", "PLUS", source="manual", serial_id="1234567", photo_hash="h1"),
        row("b.jpg", "SERIAL_NOT_FOUND", serial_id="7654320", model_reading_str="00400", photo_hash="h2"),
        row("b.jpg", "NOT_IN_DB", source="manual", serial_id="7654320", photo_hash="h2"),
    ]
    d = al.guess_details(rows, table, last, (0.0, 200.0))
    assert [(x["file"], x["guess"], x["guess_delta"], x["verdict"]) for x in d] == [
        ("Аюб/a.jpg", "1234567", "+50", "ВЕРНО"),
        ("b.jpg", "7654321", "+100", "ЛОЖНАЯ (оператор: нет в базе)"),
    ]


def test_db_serial_fix_counts_as_found_and_compared_by_account():
    # в базе 1234567 (опечатка), на фото 1234569; программа догадалась 1234567 → тот же абонент
    table = {"1234567", "7654321"}
    last = {"1234567": 1000.0, "7654321": 300.0}
    accounts = {"1234567": "A1", "7654321": "A2"}
    rows = [
        row("a.jpg", "SERIAL_NOT_FOUND", serial_id="1234569", model_reading_str="01050", photo_hash="h1"),
        row("a.jpg", "DB_SERIAL_FIX", source="manual", serial_id="1234569", account_id="A1", photo_hash="h1"),
    ]
    a = al.operator_accuracy(rows, table, last, (0.0, 200.0), accounts)
    assert a["snf_found"] == 1 and a["guess"]["right"] == 1 and a["guess"]["wrong"] == 0
    d = al.guess_details(rows, table, last, (0.0, 200.0), accounts)
    assert d[0]["verdict"] == "ВЕРНО"
    # без лицевых счетов — сравнение по серийнику, опечатка выглядит как «другой номер»
    assert al.operator_accuracy(rows, table, last, (0.0, 200.0))["guess"]["wrong"] == 1


def test_hint_usage_counted():
    rows = [row("a", "PLUS", source="manual", notes="serial_not_found | подсказка: 1234567"),
            row("b", "DB_SERIAL_FIX", source="manual",
                notes="серийник в базе с ошибкой: в базе 1, на фото 2 | подсказка: 1"),
            row("c", "PLUS", source="manual", notes="digits_error")]
    assert al.operator_accuracy(rows)["hint_used"] == {"PLUS": 1, "DB_SERIAL_FIX": 1}
    # с этапа 2.3 «Серийник в базе с ошибкой» пишет показание: строка PLUS/MINUS с пометкой
    rows.append(row("d", "MINUS", source="manual",
                    notes="серийник в базе с ошибкой: в базе 1, на фото 2 | подсказка: 1"))
    a = al.operator_accuracy(rows)
    assert a["hint_used"] == {"DB_SERIAL_FIX": 2, "PLUS": 1}
    assert a["manual_outcomes"] == {"PLUS": 2, "DB_SERIAL_FIX": 2}
    assert "подсказка «похожие номера в базе» использована" in al.build_report(rows, CFG)
