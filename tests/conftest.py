"""pytest-фикстуры для golden tests. Хелперы/fake-классы — в tests/_fixtures.py."""
import pytest
from tests._fixtures import reader


@pytest.fixture
def base_config():
    """PipelineConfig с дефолтными порогами (см. ТЗ миграции (история git) §1, п.4 — не менять молча)."""
    return reader.PipelineConfig()


@pytest.fixture
def state():
    """Состояние прогона для process_photo: строки лога, счета и отпечатки,
    уже прочитанные в этом прогоне (src/gmr/pipeline/photo.py, RunState)."""
    return reader.RunState()



@pytest.fixture(autouse=True)
def _no_drop_panels(monkeypatch):
    """Поля «перетащите сюда» (1.5.0) ждут человека — в тестах таблица
    берётся из filedialog.askopenfilename (тесты его подменяют), а
    «Добавить фото» без подмены ничего не добавляет."""
    try:
        from tkinter import filedialog

        from src.gmr.operator import window as op_window
    except ImportError:
        return
    monkeypatch.setattr(op_window, "pick_table",
                        lambda window, title, subtitle: filedialog.askopenfilename(parent=window) or None)
    monkeypatch.setattr(op_window, "pick_photos", lambda window, existing, paths=None: None)
