"""Filesystem locations owned by Betterborg for a Git repository."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from betterborg_cli.agent_runtime.base import CancellationToken
from betterborg_cli.agent_runtime.process import run_captured
from betterborg_cli.repository_files import publish_repository_text

TRACKED_DIR_NAME = ".betterborg"
HOME_VARIABLE = "BETTERBORG_HOME"
MANAGED_IGNORE_BEGIN = "# >>> Betterborg managed ignores >>>"
MANAGED_IGNORE_END = "# <<< Betterborg managed ignores <<<"
MANAGED_IGNORE_RULE = f"{TRACKED_DIR_NAME}/state/"


class BetterborgHomeError(ValueError):
    """Raised when the operator declares a home Betterborg cannot honour."""


@dataclass(frozen=True)
class RepoPaths:
    """Resolved repository, Borg, and sibling-worktree locations."""

    root: Path
    tracked_dir: Path
    state_dir: Path
    artifacts_dir: Path
    worktrees_dir: Path

    @property
    def gitignore(self) -> Path:
        """Return the repository's managed ignore file."""
        return self.root / ".gitignore"

    @property
    def tracked_in_repository(self) -> bool:
        """Return whether Betterborg's own files live inside the repository."""
        return self.tracked_dir.is_relative_to(self.root)

    @property
    def tracked_root(self) -> Path:
        """Return the root containing everything Betterborg owns for itself.

        Containment checks on Betterborg's own files ask whether a path
        escaped what Betterborg owns. That is the repository while the tracked
        directory lives inside it, and the tracked directory itself once it
        does not.
        """
        return self.root if self.tracked_in_repository else self.tracked_dir

    @property
    def prompts_dir(self) -> Path:
        """Return the tracked directory containing stable generated prompts."""
        return self.tracked_dir / "prompts"

    @property
    def prds_dir(self) -> Path:
        """Return the tracked directory containing confirmed Borg PRDs."""
        return self.tracked_dir / "prds"

    @property
    def improvement_prds_dir(self) -> Path:
        """Return the tracked directory containing generated improvement PRDs."""
        return self.prds_dir / "improvements"

    @property
    def tasks_dir(self) -> Path:
        """Return the tracked root for immutable published task generations."""
        return self.tracked_dir / "tasks"

    @property
    def task_staging_dir(self) -> Path:
        """Return the ignored same-repository task publication staging root."""
        return self.state_dir / "task-staging"

    @property
    def plans_dir(self) -> Path:
        """Return the tracked directory containing approved plan Markdown."""
        return self.tracked_dir / "plans"

    @property
    def score_report(self) -> Path:
        """Return the tracked repository score report path."""
        return self.tracked_dir / "score.md"

    def in_checkout(self, path: Path) -> Path:
        """Name a tracked file by the path it takes inside a checkout.

        A checkout Betterborg prepares always carries its context under
        ``.betterborg``, whether or not this repository's own tracked
        directory lives there.
        """
        return Path(TRACKED_DIR_NAME) / Path(path).relative_to(self.tracked_dir)

    def label(self, path: Path) -> str:
        """Return ``path`` as an operator reads it.

        Paths inside the repository are named relative to it, the way every
        other repository path is quoted. Betterborg's own files can sit
        outside the repository, where only an absolute path names them.
        """
        resolved = Path(path).resolve()
        if resolved.is_relative_to(self.root):
            return resolved.relative_to(self.root).as_posix()
        return resolved.as_posix()

    def manages(self, path: Path) -> bool:
        """Return whether ``path`` lies in the worktrees this repository mints.

        Both sides are resolved so a symlink cannot smuggle a foreign checkout
        in, and so a sibling whose name merely begins with the worktrees
        directory's own name stays outside.
        """
        return Path(path).resolve().is_relative_to(self.worktrees_dir.resolve())

    @classmethod
    def discover(
        cls,
        start: Path | None = None,
        *,
        cancel: CancellationToken | None = None,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = run_captured,
    ) -> RepoPaths:
        """Discover the nearest containing Git repository from ``start``."""
        candidate = (start or Path.cwd()).resolve()
        if candidate.is_file():
            candidate = candidate.parent

        command = ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"]
        try:
            result = command_runner(
                command,
                check=True,
                cancel=cancel,
            )
        except subprocess.CalledProcessError as error:
            raise ValueError(f"not inside a Git repository: {candidate}") from error
        if result.returncode != 0:
            error = subprocess.CalledProcessError(
                result.returncode,
                command,
                output=result.stdout,
                stderr=result.stderr,
            )
            raise ValueError(f"not inside a Git repository: {candidate}") from error

        root = Path(result.stdout.strip()).resolve()
        tracked_dir = _tracked_dir(root)
        state_dir = tracked_dir / "state"
        return cls(
            root=root,
            tracked_dir=tracked_dir,
            state_dir=state_dir,
            artifacts_dir=state_dir / "artifacts",
            worktrees_dir=root.parent / ".betterborg-worktrees" / root.name,
        )


def _tracked_dir(root: Path) -> Path:
    """Locate the directory holding Betterborg's own files for ``root``.

    Where Betterborg keeps its files belongs to whoever started it and never
    to the repository being worked on, so the declaration arrives by
    environment variable rather than as tracked configuration.

    This reads the Betterborg process environment deliberately. A repository
    drives the environments Betterborg later builds for its agents, and a
    repository that could move this directory could move the configuration
    that governs how it is worked on.

    A home inside the repository is refused rather than accepted quietly: it
    would reintroduce exactly the scaffolding in the working tree that the
    declaration exists to keep out. A home that contains the repository is
    refused for the mirror reason: every containment check Betterborg makes
    against what it owns would then admit the whole working tree.
    """
    declared = os.environ.get(HOME_VARIABLE, "")
    home = declared.strip()
    if not home:
        return root / TRACKED_DIR_NAME

    candidate = Path(home).expanduser()
    if not candidate.is_absolute():
        raise BetterborgHomeError(
            f"{HOME_VARIABLE}={declared!r} must name an absolute path"
        )
    resolved = candidate.resolve()
    if resolved.is_relative_to(root):
        raise BetterborgHomeError(
            f"{HOME_VARIABLE}={declared!r} resolves inside the repository "
            f"{root}; it must name a directory outside it"
        )
    if root.is_relative_to(resolved):
        raise BetterborgHomeError(
            f"{HOME_VARIABLE}={declared!r} contains the repository {root}; it "
            "must name a directory outside it"
        )
    return resolved


def ensure_managed_gitignore(paths: RepoPaths) -> None:
    """Write one canonical Betterborg block while preserving other ignore rules."""
    if not paths.tracked_in_repository:
        # Betterborg owns nothing inside the repository to ignore, so its
        # ignore file is left exactly as Betterborg found it.
        return

    try:
        existing = paths.gitignore.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = ""

    lines = existing.splitlines()
    retained: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line == MANAGED_IGNORE_END:
            raise ValueError(
                f"managed ignore end marker has no beginning in {paths.gitignore}"
            )
        if line != MANAGED_IGNORE_BEGIN:
            retained.append(line)
            index += 1
            continue

        try:
            index = lines.index(MANAGED_IGNORE_END, index + 1) + 1
        except ValueError as error:
            raise ValueError(
                f"managed ignore block is incomplete in {paths.gitignore}"
            ) from error

    while retained and retained[-1] == "":
        retained.pop()
    if retained:
        retained.append("")
    retained.extend([MANAGED_IGNORE_BEGIN, MANAGED_IGNORE_RULE, MANAGED_IGNORE_END])
    publish_repository_text(
        paths.gitignore,
        "\n".join(retained) + "\n",
        root=paths.root,
        overwrite=True,
    )
