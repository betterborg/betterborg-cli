"""Executable module used by the standalone Borg binary."""

from __future__ import annotations

from betterborg_cli.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
