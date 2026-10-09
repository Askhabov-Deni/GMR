"""pytest-фикстуры для golden tests. Хелперы/fake-классы — в tests/_fixtures.py."""
import pytest
from tests._fixtures import reader


@pytest.fixture
def base_config():
    """PipelineConfig с дефолтными порогами (см. MIGRATION_TZ.md §1, п.4 — не менять молча)."""
    cfg = reader.PipelineConfig()
    cfg._log_rows_cache = []
    cfg._processed_accounts_cache = set()
    return cfg

