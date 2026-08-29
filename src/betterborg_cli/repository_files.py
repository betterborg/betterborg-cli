"""Portable names and contained publication for repository-owned files."""

from __future__ import annotations

import os
import subprocess
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


class RepositoryGitVisibilityError(RuntimeError):
    """Raised when a repository-owned path is ignored or cannot be checked."""


def is_windows_reserved_filename(name: str) -> bool:
    """Return whether ``name`` uses a reserved Windows device basename."""
    return name.casefold().split(".", 1)[0] in _WINDOWS_RESERVED_BASENAMES


def require_git_trackable(path: Path, *, root: Path) -> None:
    """Require ``path`` to resolve within ``root`` and be visible to Git."""
    resolved_root = root.resolve(strict=True)
    candidate = path if path.is_absolute() else resolved_root / path
    try:
        relative = candidate.resolve(strict=False).relative_to(resolved_root)
    except ValueError as error:
        raise RepositoryPathError(f"repository path escapes root: {path}") from error

    relative_text = relative.as_posix()
    result = subprocess.run(
        [
            "git",
            "-C",
            str(resolved_root),
            "check-ignore",
            "--quiet",
            "--",
            relative_text,
        ],
        check=False,
    )
    if result.returncode == 0:
        raise RepositoryGitVisibilityError(
            f"repository path is ignored by Git: {relative_text}"
        )
    if result.returncode != 1:
        raise RepositoryGitVisibilityError(
            f"could not check Git visibility for {relative_text}"
        )


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
