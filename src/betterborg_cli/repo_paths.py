"""Filesystem locations owned by BetterBorg inside a Git repository."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

MANAGED_IGNORE_BEGIN = "# >>> BetterBorg managed ignores >>>"
MANAGED_IGNORE_END = "# <<< BetterBorg managed ignores <<<"
MANAGED_IGNORE_RULE = ".borg/state/"


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

    @classmethod
    def discover(cls, start: Path | None = None) -> RepoPaths:
        """Discover the nearest containing Git repository from ``start``."""
        candidate = (start or Path.cwd()).resolve()
        if candidate.is_file():
            candidate = candidate.parent

        try:
            result = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            raise ValueError(f"not inside a Git repository: {candidate}") from error

        root = Path(result.stdout.strip()).resolve()
        tracked_dir = root / ".borg"
        state_dir = tracked_dir / "state"
        return cls(
            root=root,
            tracked_dir=tracked_dir,
            state_dir=state_dir,
            artifacts_dir=state_dir / "artifacts",
            worktrees_dir=root.parent / ".betterborg-worktrees" / root.name,
        )


def ensure_managed_gitignore(paths: RepoPaths) -> None:
    """Write one canonical BetterBorg block while preserving other ignore rules."""
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
    paths.gitignore.write_text("\n".join(retained) + "\n", encoding="utf-8")
