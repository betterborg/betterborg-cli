"""Filesystem helpers shared by plugin lifecycle tests."""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any


def executable(path: Path, body: str) -> Path:
    """Create a minimal executable host fixture."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def copy_resource(source: Any, destination: Path) -> None:
    """Copy an importlib resource tree into an editable test directory."""

    destination.mkdir()
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            copy_resource(child, target)
        else:
            target.write_bytes(child.read_bytes())
