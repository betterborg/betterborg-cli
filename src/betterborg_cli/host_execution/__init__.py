"""Guarded host Git and sibling-worktree lifecycle."""

from betterborg_cli.host_execution.coding import (
    CodingPhaseError,
    HostCodingConfig,
    HostCodingPhase,
)
from betterborg_cli.host_execution.environment import (
    EnvironmentMaterialization,
    EnvironmentMaterializationError,
    HostEnvironmentManager,
    redacted_dropped_command_summary,
)
from betterborg_cli.host_execution.git import SafeGit, UnsafeGitError
from betterborg_cli.host_execution.guard import (
    PrimaryCheckoutContaminationError,
    PrimaryCheckoutGuard,
)
from betterborg_cli.host_execution.merge import (
    MERGE_RESULT_SCHEMA,
    HostMergeConfig,
    HostMergePhase,
    HostMergeResult,
    MergePhaseError,
    MergeTip,
)
from betterborg_cli.host_execution.preflight import (
    HostCommand,
    HostDroppedCommand,
    HostPreflight,
    HostPreflightBlock,
    HostPreflightFailure,
    HostPreflightPlan,
    HostPreflightResult,
    HostSecret,
)
from betterborg_cli.host_execution.review import (
    REVIEW_RESULT_SCHEMA,
    HostReviewFixConfig,
    HostReviewFixPhase,
    ReviewFixPhaseError,
)
from betterborg_cli.host_execution.sanity import (
    HostSanityPhase,
    HostSanityResult,
    SanityCommandResult,
    SanityPhaseError,
)
from betterborg_cli.host_execution.scheduler import (
    ActivitySink,
    HostSchedulerConfig,
    HostSchedulerResult,
    HostTaskBehavior,
    HostTaskScheduler,
    ScheduledTaskContext,
    TaskActivitySink,
)
from betterborg_cli.host_execution.service import (
    HostExecutionError,
    HostExecutionResult,
    HostExecutionService,
    HostTaskRuntime,
)
from betterborg_cli.host_execution.worktrees import (
    HostWorktreeManager,
    WorktreeError,
    WorktreeSpec,
)

__all__ = [
    "ActivitySink",
    "CodingPhaseError",
    "EnvironmentMaterialization",
    "EnvironmentMaterializationError",
    "HostEnvironmentManager",
    "HostExecutionError",
    "HostExecutionResult",
    "HostExecutionService",
    "HostMergeConfig",
    "HostMergePhase",
    "HostMergeResult",
    "HostCodingConfig",
    "HostCodingPhase",
    "HostWorktreeManager",
    "HostCommand",
    "HostDroppedCommand",
    "HostPreflight",
    "HostPreflightBlock",
    "HostPreflightFailure",
    "HostPreflightPlan",
    "HostPreflightResult",
    "HostReviewFixConfig",
    "HostReviewFixPhase",
    "HostSchedulerConfig",
    "HostSchedulerResult",
    "HostSanityPhase",
    "HostSanityResult",
    "HostSecret",
    "HostTaskBehavior",
    "HostTaskRuntime",
    "HostTaskScheduler",
    "PrimaryCheckoutContaminationError",
    "PrimaryCheckoutGuard",
    "MERGE_RESULT_SCHEMA",
    "REVIEW_RESULT_SCHEMA",
    "ReviewFixPhaseError",
    "MergePhaseError",
    "MergeTip",
    "SafeGit",
    "ScheduledTaskContext",
    "TaskActivitySink",
    "SanityCommandResult",
    "SanityPhaseError",
    "UnsafeGitError",
    "WorktreeError",
    "WorktreeSpec",
    "redacted_dropped_command_summary",
]
