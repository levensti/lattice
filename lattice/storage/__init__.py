"""Pluggable storage backends for lattice traces.

The :class:`Store` protocol defines the interface every backend must
implement.  :class:`SQLiteStore` ships as the zero-config default.

Bring your own backend::

    import lattice
    from my_package import PostgresBackend

    lattice.configure(backend=PostgresBackend(os.environ["DATABASE_URL"]))
"""

from .base import Store
from .sqlite import SQLiteStore

__all__ = ["Store", "SQLiteStore"]
