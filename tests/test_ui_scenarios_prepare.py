"""1.6.0: tools.ui_scenarios --from-month — копию месяца и «новые» фото
программа готовит сама; сам месяц не меняется."""
import sqlite3

from tools import ui_scenarios


def _month(root):
    month = root / "Май"
    for ctrl, n in (("Аюб", 15), ("Фатима", 3)):
        d = month / "фото" / ctrl
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"p{i:02d}.jpg").write_bytes(f"{ctrl}{i}".encode())
    con = sqlite3.connect(month / "gmr.sqlite")
    con.execute("CREATE TABLE processing_log (original_filename TEXT, source_folder TEXT, photo_hash TEXT)")
    con.executemany("INSERT INTO processing_log VALUES (?, ?, ?)",
                    [(f"p{i:02d}.jpg", "Аюб", "") for i in range(15)] + [("p00.jpg", "Фатима", "")])
    con.commit()
    con.close()
    return month


def _rows(db):
    con = sqlite3.connect(db)
    try:
        return con.execute("SELECT original_filename, source_folder FROM processing_log").fetchall()
    finally:
        con.close()


def test_prepare_copies_month_and_takes_last_photos(tmp_path):
    month = _month(tmp_path)
    copy, new = ui_scenarios.prepare(month, n=12)
    assert copy.parent == tmp_path and copy.name.startswith("_ui_check_")
    assert new.name == "Аюб" and new.parent.name.startswith("_ui_new_photos_")
    assert sorted(f.name for f in new.iterdir()) == [f"p{i:02d}.jpg" for i in range(3, 15)]
    assert sorted(f.name for f in (copy / "фото" / "Аюб").iterdir()) == ["p00.jpg", "p01.jpg", "p02.jpg"]
    assert sorted(_rows(copy / "gmr.sqlite")) == [("p00.jpg", "Аюб"), ("p00.jpg", "Фатима"),
                                                  ("p01.jpg", "Аюб"), ("p02.jpg", "Аюб")]
    assert len(_rows(month / "gmr.sqlite")) == 16                       # сам месяц — как был
    assert len(list((month / "фото" / "Аюб").iterdir())) == 15


def test_from_month_cleans_up(tmp_path, monkeypatch):
    month = _month(tmp_path)
    seen = []
    monkeypatch.setattr(ui_scenarios, "main_run", lambda m, p, exe, only: seen.append((m, p)) or 0)
    assert ui_scenarios.main(["--from-month", str(month)]) == 0
    assert seen and sorted(d.name for d in tmp_path.iterdir()) == ["Май"]   # копия и новые фото убраны
