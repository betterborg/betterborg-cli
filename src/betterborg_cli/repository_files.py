"""Portable names and contained publication for repository-owned files."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

_WINDOWS_RESERVED_BASENAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


class RepositoryPathError(ValueError):
    """Raised when a repository-owned destination resolves outside its root."""


def is_windows_reserved_filename(name: str) -> bool:
    """Return whether ``name`` uses a reserved Windows device basename."""
    return name.casefold().split(".", 1)[0] in _WINDOWS_RESERVED_BASENAMES


def publish_repository_text(
    path: Path,
    body: str,
    *,
    root: Path,
    overwrite: bool,
) -> None:
    """Atomically publish UTF-8 text within ``root``.

    ``overwrite=False`` atomically claims a new destination and fails if any
    file or symlink already occupies it. ``overwrite=True`` atomically replaces
    the destination without following a destination symlink.
    """
    parent = path.parent
    if not parent.resolve().is_relative_to(root):
        raise RepositoryPathError(f"output directory escapes repository: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = parent.resolve(strict=True)
    if not resolved_parent.is_relative_to(root):
        raise RepositoryPathError(f"output directory escapes repository: {parent}")

    destination = resolved_parent / path.name
    temporary = resolved_parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as output:
            output.write(body)
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
