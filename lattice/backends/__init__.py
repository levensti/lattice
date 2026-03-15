"""Pluggable storage backends for lattice traces.

The :class:`StorageBackend` protocol defines the interface every backend must
implement.  :class:`SQLiteBackend` ships as the zero-config default.

Bring your own backend::

    import lattice
    from my_package import PostgresBackend

    lattice.configure(backend=PostgresBackend(os.environ["DATABASE_URL"]))
"""

from .base import StorageBackend
from .sqlite import SQLiteBackend

__all__ = ["StorageBackend", "SQLiteBackend"]
