"""Executable module used by the standalone Borg binary."""

from __future__ import annotations

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()

    from betterborg_cli.cli import main

    raise SystemExit(main())
