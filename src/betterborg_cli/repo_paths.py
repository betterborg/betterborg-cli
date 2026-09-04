"""Filesystem locations owned by Betterborg inside a Git repository."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from betterborg_cli.agent_runtime.base import CancellationToken
from betterborg_cli.agent_runtime.process import run_captured
from betterborg_cli.repository_files import publish_repository_text

MANAGED_IGNORE_BEGIN = "# >>> Betterborg managed ignores >>>"
MANAGED_IGNORE_END = "# <<< Betterborg managed ignores <<<"
MANAGED_IGNORE_RULE = ".betterborg/state/"


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
    def prompts_dir(self) -> Path:
        """Return the tracked directory containing stable generated prompts."""
        return self.tracked_dir / "prompts"

    @property
    def improvement_prds_dir(self) -> Path:
        """Return the tracked directory containing generated improvement PRDs."""
        return self.tracked_dir / "prds" / "improvements"

    @property
    def tasks_dir(self) -> Path:
        """Return the tracked root for immutable published task generations."""
        return self.tracked_dir / "tasks"

    @property
    def task_staging_dir(self) -> Path:
        """Return the ignored same-repository task publication staging root."""
        return self.state_dir / "task-staging"

    @property
    def score_report(self) -> Path:
        """Return the tracked repository score report path."""
        return self.tracked_dir / "score.md"

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
        tracked_dir = root / ".betterborg"
        state_dir = tracked_dir / "state"
        return cls(
            root=root,
            tracked_dir=tracked_dir,
            state_dir=state_dir,
            artifacts_dir=state_dir / "artifacts",
            worktrees_dir=root.parent / ".betterborg-worktrees" / root.name,
        )


def ensure_managed_gitignore(paths: RepoPaths) -> None:
    """Write one canonical Betterborg block while preserving other ignore rules."""
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
