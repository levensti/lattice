"""Console logging helpers for Lattice."""

from __future__ import annotations

import logging
import sys

from .context import TraceSession

logger = logging.getLogger("lattice")

if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s \u2014 %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def print_trace_summary(session: TraceSession) -> None:
    """Print a human-readable summary of a completed trace session.

    .. deprecated::
        Use ``print(session)`` instead.  ``TraceSession.__str__`` produces
        the same output.
    """
    print(session)
