"""CLI entry point for Lattice.

Usage::

    python -m lattice dashboard
    python -m lattice dashboard --port 8080
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="lattice", description="Lattice CLI")
    subparsers = parser.add_subparsers(dest="command")

    dash = subparsers.add_parser("dashboard", help="Launch the local trace dashboard")
    dash.add_argument("--host", default="localhost", help="Bind address (default: localhost)")
    dash.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")

    args = parser.parse_args()

    if args.command == "dashboard":
        from .dashboard import serve
        serve(host=args.host, port=args.port)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
