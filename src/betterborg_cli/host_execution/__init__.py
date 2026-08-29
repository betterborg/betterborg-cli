"""Guarded host Git and sibling-worktree lifecycle."""

from betterborg_cli.host_execution.environment import (
    EnvironmentMaterialization,
    EnvironmentMaterializationError,
    HostEnvironmentManager,
    environment_fingerprint,
    package_manager_cache_environment,
)
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
    HostSecret,
    HostService,
)
from betterborg_cli.host_execution.worktrees import (
    HostWorktreeManager,
    WorktreeError,
    WorktreeSpec,
)

__all__ = [
    "EnvironmentMaterialization",
    "EnvironmentMaterializationError",
    "HostEnvironmentManager",
    "HostWorktreeManager",
    "HostCommand",
    "HostExecutable",
    "HostPreflight",
    "HostPreflightBlock",
    "HostPreflightFailure",
    "HostPreflightPlan",
    "HostPreflightResult",
    "HostSecret",
    "HostService",
    "PrimaryCheckoutContaminationError",
    "PrimaryCheckoutGuard",
    "SafeGit",
    "UnsafeGitError",
    "WorktreeError",
    "WorktreeSpec",
    "environment_fingerprint",
    "package_manager_cache_environment",
]
