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

