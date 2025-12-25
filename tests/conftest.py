import pytest
from sqlalchemy import create_engine

from beaverschoice.config import DEFAULT_SEED, START_DATE
from beaverschoice.db import init_database


@pytest.fixture()
def engine():
    engine = create_engine("sqlite:///:memory:")
    init_database(engine, seed=DEFAULT_SEED)
    yield engine
    engine.dispose()


@pytest.fixture()
def start_date():
    return START_DATE.isoformat()
