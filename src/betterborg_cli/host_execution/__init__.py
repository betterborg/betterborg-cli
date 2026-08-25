"""Guarded host Git and sibling-worktree lifecycle."""

from betterborg_cli.host_execution.git import SafeGit, UnsafeGitError
from betterborg_cli.host_execution.guard import (
    PrimaryCheckoutContaminationError,
    PrimaryCheckoutGuard,
)
from betterborg_cli.host_execution.worktrees import (
    HostWorktreeManager,
    WorktreeError,
    WorktreeSpec,
)

__all__ = [
    "HostWorktreeManager",
    "PrimaryCheckoutContaminationError",
    "PrimaryCheckoutGuard",
    "SafeGit",
    "UnsafeGitError",
    "WorktreeError",
    "WorktreeSpec",
]
