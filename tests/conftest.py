"""pytest-фикстуры для golden tests. Хелперы/fake-классы — в tests/_fixtures.py."""
import sys

import pytest
from tests._fixtures import reader


@pytest.fixture
def base_config():
    """PipelineConfig с дефолтными порогами (см. MIGRATION_TZ.md §1, п.4 — не менять молча)."""
    cfg = reader.PipelineConfig()
    cfg._log_rows_cache = []
    cfg._processed_accounts_cache = set()
    return cfg


@pytest.fixture(autouse=True)
def _datasets_in_tmp(tmp_path, monkeypatch):
    """Тихая разметка окна (program2.DATASETS_ROOT) в тестах — во временную
    папку, а не в database/datasets проекта."""
    mod = sys.modules.get("program2")
    if mod is not None:
        monkeypatch.setattr(mod, "DATASETS_ROOT", tmp_path / "datasets")
