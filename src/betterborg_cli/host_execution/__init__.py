"""Guarded host Git and sibling-worktree lifecycle."""

from betterborg_cli.host_execution.git import SafeGit, UnsafeGitError
from betterborg_cli.host_execution.guard import (
    PrimaryCheckoutContaminationError,
    PrimaryCheckoutGuard,
)
from betterborg_cli.host_execution.preflight import (
    HostCommand,
    HostExecutable,
    HostPreflight,
    HostPreflightBlock,
    HostPreflightFailure,
    HostPreflightPlan,
    HostPreflightResult,
    HostService,
)
from betterborg_cli.host_execution.worktrees import (
    HostWorktreeManager,
    WorktreeError,
    WorktreeSpec,
)

__all__ = [
    "HostWorktreeManager",
    "HostCommand",
    "HostExecutable",
    "HostPreflight",
    "HostPreflightBlock",
    "HostPreflightFailure",
    "HostPreflightPlan",
    "HostPreflightResult",
    "HostService",
    "PrimaryCheckoutContaminationError",
    "PrimaryCheckoutGuard",
    "SafeGit",
    "UnsafeGitError",
    "WorktreeError",
    "WorktreeSpec",
]
