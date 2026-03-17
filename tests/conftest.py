"""Shared test fixtures — isolate every test from the real ~/.lattice DB."""

import pytest

from lattice.storage.sqlite import DEFAULT_DB_PATH
from lattice.storage.store import configure


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path):
    """Point the store at a throw-away temp DB for every test."""
    db = tmp_path / "test_traces.db"
    configure(db_path=db)
    yield
    configure(db_path=DEFAULT_DB_PATH)
