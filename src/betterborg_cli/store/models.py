"""Domain records persisted in the local BetterBorg store."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from secrets import token_urlsafe
from typing import Any
from uuid import UUID, uuid4

from betterborg_cli.agent_runtime.base import AgentStatus, AgentUsage, BillingMode

_PROMPT_ROLES = frozenset({"coding", "review", "merge"})


class BorgState(str, Enum):
    """Durable lifecycle states used by the Borg planning pipeline."""

    DRAFT = "draft"
    ARCHITECT_WORKING = "architect_working"
    ARCHITECT_AWAITING_ANSWERS = "architect_awaiting_answers"
    TECH_REVIEW_WORKING = "tech_review_working"
    PLAN_APPROVAL_PENDING = "plan_approval_pending"
    PM_WORKING = "pm_working"
    SUPERVISOR_WORKING = "supervisor_working"
    # Keep the persisted value compatible with schema-004 databases while
    # naming the public lifecycle outcome for what it now means: task
    # publication is complete and no second human gate remains.
    READY_TO_EXECUTE = "tasks_approval_pending"
    TASKS_APPROVAL_PENDING = READY_TO_EXECUTE
    EXECUTING = "executing"
    DONE = "done"
    BLOCKED = "blocked"


class PlanningAttemptStatus(str, Enum):
    """Completion state for one planning-agent invocation."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskComplexity(str, Enum):
    """Coarse implementation estimate attached to an immutable task record."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class TaskGenerationStatus(str, Enum):
    """Publication lifecycle for one immutable task-generation snapshot."""

    PREPARING = "preparing"
    CURRENT = "current"
    SUPERSEDED = "superseded"


class ExecutionRunStatus(str, Enum):
    """Lease-backed lifecycle for one execution of a task generation."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionAttemptStatus(str, Enum):
    """Lifecycle state for one host-execution attempt."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskRuntimeStatus(str, Enum):
    """Durable scheduling state for one generated task."""

    PENDING = "pending"
    CLAIMED = "claimed"
    ENVIRONMENT = "environment"
    CODING = "coding"
    REVIEW = "review"
    FIX = "fix"
    MERGING = "merging"
    DONE = "done"
    BLOCKED = "blocked"
    FAILED = "failed"


def _new_ownership_token() -> str:
    """Return a URL-safe token carrying 256 bits of randomness."""
    return token_urlsafe(32)


def utcnow() -> datetime:
    """Return a timezone-aware timestamp in UTC."""
    return datetime.now(UTC)


def _validate_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("timestamps must be timezone-aware UTC values")


@dataclass(frozen=True, slots=True)
class Repository:
    """One Git repository managed by this local BetterBorg store."""

    root: Path
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())
        if not isinstance(self.id, UUID):
            raise TypeError("repository ID must be a UUID")
        _validate_utc(self.created_at)


@dataclass(frozen=True, slots=True)
class Operation:
    """An immutable entry in a repository's local operation ledger."""

    repository_id: UUID
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if not isinstance(self.id, UUID):
            raise TypeError("operation ID must be a UUID")
        if not isinstance(self.repository_id, UUID):
            raise TypeError("repository ID must be a UUID")
        if not self.kind:
            raise ValueError("operation kind must not be empty")
        _validate_utc(self.created_at)


@dataclass(frozen=True, slots=True)
class RepositoryAnalysis:
    """One immutable, successful analyzer run for a repository."""

    repository_id: UUID
    head_sha: str
    summary: str
    primary_language: str
    is_monorepo: bool
    overall_score: float
    analysis_json: dict[str, Any]
    prior_analysis_id: UUID | None = None
    score_delta: float | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "repository_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"analysis {name} must be a UUID")
        if self.prior_analysis_id is not None and not isinstance(
            self.prior_analysis_id, UUID
        ):
            raise TypeError("prior analysis ID must be a UUID")
        if not self.head_sha:
            raise ValueError("analysis Git HEAD must not be empty")
        if not self.summary:
            raise ValueError("analysis summary must not be empty")
        if not self.primary_language:
            raise ValueError("analysis primary language must not be empty")
        if not 0 <= self.overall_score <= 5:
            raise ValueError("analysis overall score must be between 0 and 5")
        if (self.prior_analysis_id is None) != (self.score_delta is None):
            raise ValueError(
                "analysis score delta and prior analysis ID must be set together"
            )
        _validate_utc(self.created_at)


@dataclass(frozen=True, slots=True)
class RepositoryPackage:
    """One immutable package score belonging to an analysis run."""

    repository_id: UUID
    analysis_id: UUID
    package_path: str
    package_name: str
    primary_language: str
    rubric: dict[str, Any]
    overall_score: float
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        for name in ("id", "repository_id", "analysis_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"package {name} must be a UUID")
        if not self.package_path:
            raise ValueError("package path must not be empty")
        if not self.package_name:
            raise ValueError("package name must not be empty")
        if not self.primary_language:
            raise ValueError("package primary language must not be empty")
        if not 0 <= self.overall_score <= 5:
            raise ValueError("package overall score must be between 0 and 5")


@dataclass(frozen=True, slots=True)
class GeneratedPrompt:
    """One immutable, versioned role prompt generated from an analysis."""

    repository_id: UUID
    analysis_id: UUID
    role: str
    version: int
    body_md: str
    id: UUID = field(default_factory=uuid4)
    generated_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "repository_id", "analysis_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"generated prompt {name} must be a UUID")
        if self.role not in _PROMPT_ROLES:
            raise ValueError(f"unknown generated prompt role: {self.role!r}")
        if self.version < 1:
            raise ValueError("generated prompt version must be positive")
        if not self.body_md:
            raise ValueError("generated prompt body must not be empty")
        _validate_utc(self.generated_at)


@dataclass(frozen=True, slots=True)
class Borg:
    """One named Borg identity belonging to a repository."""

    repository_id: UUID
    name: str
    state: BorgState = BorgState.DRAFT
    state_version: int = 0
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "repository_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"Borg {name} must be a UUID")
        if not self.name.strip():
            raise ValueError("Borg name must not be empty")
        if not isinstance(self.state, BorgState):
            raise TypeError("Borg state must be a BorgState")
        if self.state_version < 0:
            raise ValueError("Borg state version must not be negative")
        _validate_utc(self.created_at)


@dataclass(frozen=True, slots=True)
class PrdSession:
    """A durable conversation whose confirmed PRD remains tracked Markdown."""

    repository_id: UUID
    borg_id: UUID
    prd_path: Path
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "repository_id", "borg_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"PRD session {name} must be a UUID")
        path = Path(self.prd_path)
        if path.is_absolute() or path == Path(".") or ".." in path.parts:
            raise ValueError("PRD path must be repository-relative")
        if path.suffix.casefold() != ".md":
            raise ValueError("PRD path must identify a Markdown file")
        object.__setattr__(self, "prd_path", path)
        _validate_utc(self.created_at)


@dataclass(frozen=True, slots=True)
class PrdTurn:
    """One immutable turn in a PRD onboarding session."""

    session_id: UUID
    position: int
    role: str
    content: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "session_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"PRD turn {name} must be a UUID")
        if self.position < 1:
            raise ValueError("PRD turn position must be positive")
        if not self.role.strip():
            raise ValueError("PRD turn role must not be empty")
        if not self.content:
            raise ValueError("PRD turn content must not be empty")
        _validate_utc(self.created_at)


@dataclass(frozen=True, slots=True)
class PlanningAttempt:
    """One durable invocation in a Borg's planning pipeline."""

    borg_id: UUID
    phase: str
    round: int
    adapter: str
    model: str
    request: dict[str, Any] = field(default_factory=dict)
    status: PlanningAttemptStatus = PlanningAttemptStatus.RUNNING
    result: dict[str, Any] | None = None
    summary: str | None = None
    id: UUID = field(default_factory=uuid4)
    started_at: datetime = field(default_factory=utcnow)
    finished_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("id", "borg_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"planning attempt {name} must be a UUID")
        if not self.phase.strip():
            raise ValueError("planning attempt phase must not be empty")
        if self.round < 1:
            raise ValueError("planning attempt round must be positive")
        if not self.adapter.strip() or not self.model.strip():
            raise ValueError("planning attempt adapter and model must not be empty")
        if not isinstance(self.status, PlanningAttemptStatus):
            raise TypeError("planning attempt status must be a PlanningAttemptStatus")
        if (self.status is PlanningAttemptStatus.RUNNING) != (
            self.finished_at is None
        ):
            raise ValueError("only running planning attempts may be unfinished")
        _validate_utc(self.started_at)
        if self.finished_at is not None:
            _validate_utc(self.finished_at)


@dataclass(frozen=True, slots=True)
class PlanningQuestion:
    """One durable architect Q&A round, optionally awaiting answers."""

    borg_id: UUID
    round: int
    questions: list[dict[str, Any]]
    attempt_id: UUID | None = None
    answers: list[dict[str, Any]] | None = None
    id: UUID = field(default_factory=uuid4)
    asked_at: datetime = field(default_factory=utcnow)
    answered_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("id", "borg_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"planning question {name} must be a UUID")
        if self.attempt_id is not None and not isinstance(self.attempt_id, UUID):
            raise TypeError("planning question attempt ID must be a UUID")
        if self.round < 1:
            raise ValueError("planning question round must be positive")
        if not self.questions:
            raise ValueError("planning question round must contain questions")
        if (self.answers is None) != (self.answered_at is None):
            raise ValueError(
                "planning answers and answered timestamp must be set together"
            )
        _validate_utc(self.asked_at)
        if self.answered_at is not None:
            _validate_utc(self.answered_at)


@dataclass(frozen=True, slots=True)
class PlanningFinding:
    """One immutable tech-lead finding from a planning attempt."""

    borg_id: UUID
    attempt_id: UUID
    round: int
    severity: str
    message: str
    suggestion: str | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "borg_id", "attempt_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"planning finding {name} must be a UUID")
        if self.round < 1:
            raise ValueError("planning finding round must be positive")
        if not self.severity.strip() or not self.message.strip():
            raise ValueError("planning finding severity and message must not be empty")
        _validate_utc(self.created_at)


@dataclass(frozen=True, slots=True)
class PlanChangeRequest:
    """One immutable human request in a Borg's plan-revision thread."""

    borg_id: UUID
    round: int
    note: str
    decided_by: str | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "borg_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"plan change request {name} must be a UUID")
        if self.round < 1:
            raise ValueError("plan change request round must be positive")
        if not self.note.strip():
            raise ValueError("plan change request note must not be empty")
        _validate_utc(self.created_at)


@dataclass(frozen=True, slots=True)
class PlanApproval:
    """One immutable approval of a digest-bound plan manifest."""

    borg_id: UUID
    plan_digest: str
    manifest: dict[str, Any]
    attempt_id: UUID | None = None
    approved_by: str | None = None
    id: UUID = field(default_factory=uuid4)
    approved_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "borg_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"plan approval {name} must be a UUID")
        if self.attempt_id is not None and not isinstance(self.attempt_id, UUID):
            raise TypeError("plan approval attempt ID must be a UUID")
        if not self.plan_digest.strip():
            raise ValueError("plan approval digest must not be empty")
        if not isinstance(self.manifest, dict):
            raise TypeError("plan approval manifest must be a dictionary")
        _validate_utc(self.approved_at)


@dataclass(frozen=True, slots=True)
class TaskBatch:
    """One immutable PM-produced task batch bound to an approved plan."""

    borg_id: UUID
    plan_approval_id: UUID
    round: int
    digest: str
    manifest: dict[str, Any]
    summary: str = ""
    attempt_id: UUID | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "borg_id", "plan_approval_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"task batch {name} must be a UUID")
        if self.attempt_id is not None and not isinstance(self.attempt_id, UUID):
            raise TypeError("task batch attempt ID must be a UUID")
        if self.round < 1:
            raise ValueError("task batch round must be positive")
        if not self.digest.strip():
            raise ValueError("task batch digest must not be empty")
        if not isinstance(self.manifest, dict):
            raise TypeError("task batch manifest must be a dictionary")
        _validate_utc(self.created_at)


@dataclass(frozen=True, slots=True)
class TaskFinding:
    """One immutable supervisor finding against a task batch."""

    borg_id: UUID
    batch_id: UUID
    round: int
    severity: str
    message: str
    attempt_id: UUID | None = None
    suggestion: str | None = None
    task_ref: str | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "borg_id", "batch_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"task finding {name} must be a UUID")
        if self.attempt_id is not None and not isinstance(self.attempt_id, UUID):
            raise TypeError("task finding attempt ID must be a UUID")
        if self.round < 1:
            raise ValueError("task finding round must be positive")
        if not self.severity.strip() or not self.message.strip():
            raise ValueError("task finding severity and message must not be empty")
        if self.task_ref is not None and not self.task_ref.strip():
            raise ValueError("task finding task ref must not be empty")
        _validate_utc(self.created_at)


@dataclass(frozen=True, slots=True)
class TaskGeneration:
    """One digest-bound task set whose metadata never changes after creation."""

    borg_id: UUID
    plan_approval_id: UUID
    batch_id: UUID
    digest: str
    manifest: dict[str, Any]
    status: TaskGenerationStatus = TaskGenerationStatus.PREPARING
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)
    current_at: datetime | None = None
    superseded_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("id", "borg_id", "plan_approval_id", "batch_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"task generation {name} must be a UUID")
        if not isinstance(self.status, TaskGenerationStatus):
            raise TypeError("task generation status must be a TaskGenerationStatus")
        if not self.digest.strip():
            raise ValueError("task generation digest must not be empty")
        if not isinstance(self.manifest, dict):
            raise TypeError("task generation manifest must be a dictionary")
        _validate_utc(self.created_at)
        if self.current_at is not None:
            _validate_utc(self.current_at)
        if self.superseded_at is not None:
            _validate_utc(self.superseded_at)
        timestamps_are_valid = (
            self.status is TaskGenerationStatus.PREPARING
            and self.current_at is None
            and self.superseded_at is None
        ) or (
            self.status is TaskGenerationStatus.CURRENT
            and self.current_at is not None
            and self.superseded_at is None
        ) or (
            self.status is TaskGenerationStatus.SUPERSEDED
            and self.current_at is not None
            and self.superseded_at is not None
        )
        if not timestamps_are_valid:
            raise ValueError("task generation timestamps do not match its status")


@dataclass(frozen=True, slots=True)
class ExecutionDecision:
    """One immutable estimate decision bound to an exact task generation."""

    borg_id: UUID
    generation_id: UUID
    approved_plan_digest: str
    task_batch_digest: str
    estimate_version: str
    source: str
    snapshot: dict[str, Any]
    decision: str
    id: UUID = field(default_factory=uuid4)
    decided_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "borg_id", "generation_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"execution decision {name} must be a UUID")
        for name in (
            "approved_plan_digest",
            "task_batch_digest",
            "estimate_version",
            "source",
            "decision",
        ):
            if not getattr(self, name).strip():
                raise ValueError(
                    f"execution decision {name.replace('_', ' ')} must not be empty"
                )
        if not isinstance(self.snapshot, dict):
            raise TypeError("execution decision snapshot must be a dictionary")
        _validate_utc(self.decided_at)


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """Immutable structured metadata for one task in a generation."""

    generation_id: UUID
    borg_id: UUID
    task_ref: str
    stage: str
    stem: str
    position: int
    title: str
    complexity: TaskComplexity
    digest: str
    task: dict[str, Any]
    manifest: dict[str, Any]
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "generation_id", "borg_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"task record {name} must be a UUID")
        for name in ("task_ref", "stage", "stem", "title", "digest"):
            if not getattr(self, name).strip():
                raise ValueError(f"task record {name} must not be empty")
        if self.position < 1:
            raise ValueError("task record position must be positive")
        if not isinstance(self.complexity, TaskComplexity):
            raise TypeError("task record complexity must be a TaskComplexity")
        if not isinstance(self.task, dict) or not isinstance(self.manifest, dict):
            raise TypeError("task record task and manifest must be dictionaries")
        _validate_utc(self.created_at)


@dataclass(frozen=True, slots=True)
class TaskDependency:
    """One immutable directed edge between tasks in the same generation."""

    generation_id: UUID
    task_id: UUID
    depends_on_task_id: UUID
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "generation_id", "task_id", "depends_on_task_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"task dependency {name} must be a UUID")
        if self.task_id == self.depends_on_task_id:
            raise ValueError("a task cannot depend on itself")
        _validate_utc(self.created_at)


@dataclass(frozen=True, slots=True)
class ExecutionRun:
    """One lease-owned execution of an immutable task generation."""

    borg_id: UUID
    generation_id: UUID
    lease_expires_at: datetime
    owner_token: str = field(default_factory=_new_ownership_token, repr=False)
    status: ExecutionRunStatus = ExecutionRunStatus.RUNNING
    id: UUID = field(default_factory=uuid4)
    started_at: datetime = field(default_factory=utcnow)
    heartbeat_at: datetime | None = None
    finished_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("id", "borg_id", "generation_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"execution run {name} must be a UUID")
        if len(self.owner_token) < 32:
            raise ValueError("execution run owner token must be unguessable")
        if not isinstance(self.status, ExecutionRunStatus):
            raise TypeError("execution run status must be an ExecutionRunStatus")
        for value in (self.started_at, self.lease_expires_at):
            _validate_utc(value)
        if self.heartbeat_at is not None:
            _validate_utc(self.heartbeat_at)
        if self.finished_at is not None:
            _validate_utc(self.finished_at)
        if self.lease_expires_at <= self.started_at:
            raise ValueError("execution run lease must expire after it starts")
        if (self.status is ExecutionRunStatus.RUNNING) != (
            self.finished_at is None
        ):
            raise ValueError("only running execution runs may be unfinished")


@dataclass(frozen=True, slots=True)
class ExecutionRunAcquisition:
    """Result of contending for one Borg execution operation.

    Every contender learns the durable operation ID, but only the caller that
    created the run receives its ownership token.
    """

    operation_id: UUID
    owner_token: str | None
    acquired: bool

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, UUID):
            raise TypeError("execution operation ID must be a UUID")
        if self.acquired != (self.owner_token is not None):
            raise ValueError("only an acquired execution run has an owner token")
        if self.owner_token is not None and len(self.owner_token) < 32:
            raise ValueError("execution run owner token must be unguessable")

    @property
    def run_id(self) -> UUID:
        """Return the operation ID using the execution-store name."""
        return self.operation_id


@dataclass(frozen=True, slots=True)
class TaskRuntime:
    """Mutable execution projection for one immutable generated task."""

    generation_id: UUID
    task_id: UUID
    status: TaskRuntimeStatus = TaskRuntimeStatus.PENDING
    resume_phase: str = "environment"
    review_round: int = 0
    state_reason: str | None = None
    branch: str | None = None
    worktree_path: str | None = None
    last_run_id: UUID | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "generation_id", "task_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"task runtime {name} must be a UUID")
        if self.last_run_id is not None and not isinstance(self.last_run_id, UUID):
            raise TypeError("task runtime last run ID must be a UUID")
        if not isinstance(self.status, TaskRuntimeStatus):
            raise TypeError("task runtime status must be a TaskRuntimeStatus")
        if not self.resume_phase.strip():
            raise ValueError("task runtime resume phase must not be empty")
        if self.review_round < 0:
            raise ValueError("task runtime review round must not be negative")
        if (self.branch is None) != (self.worktree_path is None):
            raise ValueError(
                "task runtime branch and worktree path must be assigned together"
            )
        _validate_utc(self.created_at)
        _validate_utc(self.updated_at)
        if self.updated_at < self.created_at:
            raise ValueError("task runtime cannot be updated before it is created")


@dataclass(frozen=True, slots=True)
class TaskRuntimeCost:
    """Billing-aware cost summary for a task's persisted agent attempts."""

    api_spend_usd: float | None
    api_spend_unknown: bool
    subscription_included: bool

    def __post_init__(self) -> None:
        if self.api_spend_usd is not None and self.api_spend_usd < 0:
            raise ValueError("task runtime API spend must not be negative")
        if self.api_spend_unknown and self.api_spend_usd is not None:
            raise ValueError("unknown API spend cannot have a USD value")


@dataclass(frozen=True, slots=True)
class TaskRuntimeRow:
    """Current-generation task metadata enriched with execution status."""

    generation_id: UUID
    task_id: UUID
    task_ref: str
    stage: str
    stem: str
    position: int
    title: str
    complexity: TaskComplexity
    status: TaskRuntimeStatus
    state_reason: str | None
    review_round: int
    attempt_count: int
    duration_seconds: float | None
    cost: TaskRuntimeCost

    def __post_init__(self) -> None:
        for name in ("generation_id", "task_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"task runtime row {name} must be a UUID")
        for name in ("task_ref", "stage", "stem", "title"):
            if not getattr(self, name).strip():
                raise ValueError(f"task runtime row {name} must not be empty")
        if self.position < 1:
            raise ValueError("task runtime row position must be positive")
        if not isinstance(self.complexity, TaskComplexity):
            raise TypeError("task runtime row complexity must be a TaskComplexity")
        if not isinstance(self.status, TaskRuntimeStatus):
            raise TypeError("task runtime row status must be a TaskRuntimeStatus")
        if self.review_round < 0:
            raise ValueError("task runtime row review round must not be negative")
        if self.attempt_count < 0:
            raise ValueError("task runtime row attempt count must not be negative")
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ValueError("task runtime row duration must not be negative")
        if not isinstance(self.cost, TaskRuntimeCost):
            raise TypeError("task runtime row cost must be a TaskRuntimeCost")


@dataclass(frozen=True, slots=True)
class TaskCompletionSample:
    """Measured agent work from one locally completed generated task."""

    generation_id: UUID
    task_id: UUID
    complexity: TaskComplexity
    duration_seconds: float | None
    coding_usage: AgentUsage | None
    review_usage: AgentUsage | None
    merge_usage: AgentUsage | None

    def __post_init__(self) -> None:
        for name in ("generation_id", "task_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"task completion sample {name} must be a UUID")
        if not isinstance(self.complexity, TaskComplexity):
            raise TypeError(
                "task completion sample complexity must be a TaskComplexity"
            )
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ValueError("task completion sample duration must not be negative")
        for name in ("coding_usage", "review_usage", "merge_usage"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, AgentUsage):
                raise TypeError(f"task completion sample {name} must be AgentUsage")


@dataclass(frozen=True, slots=True)
class TaskClaim:
    """A lease-owned claim granting one run authority over one task."""

    run_id: UUID
    task_id: UUID
    lease_expires_at: datetime
    resume_phase: str
    claim_token: str = field(default_factory=_new_ownership_token, repr=False)
    id: UUID = field(default_factory=uuid4)
    claimed_at: datetime = field(default_factory=utcnow)
    released_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("id", "run_id", "task_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"task claim {name} must be a UUID")
        if len(self.claim_token) < 32:
            raise ValueError("task claim token must be unguessable")
        if not self.resume_phase.strip():
            raise ValueError("task claim resume phase must not be empty")
        _validate_utc(self.claimed_at)
        _validate_utc(self.lease_expires_at)
        if self.lease_expires_at <= self.claimed_at:
            raise ValueError("task claim lease must expire after it is claimed")
        if self.released_at is not None:
            _validate_utc(self.released_at)
            if self.released_at < self.claimed_at:
                raise ValueError("task claim cannot be released before it is claimed")


@dataclass(frozen=True, slots=True)
class EnvironmentAttempt:
    """Immutable record of one environment preparation or materialization.

    Reusable preparation may be owned by the execution run before a task is
    claimed. Checkout-local materialization always remains claim-owned.
    """

    run_id: UUID
    claim_id: UUID | None
    task_id: UUID
    kind: str
    attempt_number: int
    fingerprint: str
    status: ExecutionAttemptStatus | AgentStatus
    commands: list[list[str]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    duration_seconds: float | None = None
    id: UUID = field(default_factory=uuid4)
    started_at: datetime = field(default_factory=utcnow)
    finished_at: datetime | None = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "run_id", "task_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"environment attempt {name} must be a UUID")
        if self.claim_id is not None and not isinstance(self.claim_id, UUID):
            raise TypeError("environment attempt claim_id must be a UUID or None")
        for name in ("kind", "fingerprint"):
            if not getattr(self, name).strip():
                raise ValueError(f"environment attempt {name} must not be empty")
        if self.claim_id is None and self.kind != "prepare":
            raise ValueError("only preparation may be owned directly by a run")
        if self.attempt_number < 1:
            raise ValueError("environment attempt number must be positive")
        if not isinstance(self.status, ExecutionAttemptStatus | AgentStatus):
            raise TypeError(
                "environment attempt status must be an ExecutionAttemptStatus"
            )
        object.__setattr__(
            self, "status", ExecutionAttemptStatus(self.status.value)
        )
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ValueError("environment attempt duration must not be negative")
        if any(
            not command or any(not arg for arg in command)
            for command in self.commands
        ):
            raise ValueError("environment attempt commands must contain non-empty argv")
        _validate_utc(self.started_at)
        if (self.status is ExecutionAttemptStatus.RUNNING) != (
            self.finished_at is None
        ):
            raise ValueError(
                "only a running environment attempt has no finish timestamp"
            )
        if self.finished_at is not None:
            _validate_utc(self.finished_at)
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("environment attempt cannot finish before it starts")


@dataclass(frozen=True, slots=True)
class AgentAttempt:
    """Immutable, billing-aware record of one execution-agent invocation."""

    run_id: UUID
    claim_id: UUID
    task_id: UUID
    phase: str
    attempt_number: int
    adapter: str
    model: str
    billing_mode: BillingMode
    status: ExecutionAttemptStatus | AgentStatus
    log_path: str
    review_round: int = 0
    result_path: str | None = None
    result: dict[str, Any] | None = None
    summary: str | None = None
    duration_seconds: float | None = None
    usage: AgentUsage | None = None
    id: UUID = field(default_factory=uuid4)
    started_at: datetime = field(default_factory=utcnow)
    finished_at: datetime | None = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "run_id", "claim_id", "task_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"agent attempt {name} must be a UUID")
        for name in ("phase", "adapter", "model", "log_path"):
            if not getattr(self, name).strip():
                raise ValueError(f"agent attempt {name} must not be empty")
        if self.attempt_number < 1:
            raise ValueError("agent attempt number must be positive")
        if self.review_round < 0:
            raise ValueError("agent attempt review round must not be negative")
        if not isinstance(self.billing_mode, BillingMode):
            raise TypeError("agent attempt billing mode must be a BillingMode")
        if not isinstance(self.status, ExecutionAttemptStatus | AgentStatus):
            raise TypeError("agent attempt status must be an ExecutionAttemptStatus")
        object.__setattr__(
            self, "status", ExecutionAttemptStatus(self.status.value)
        )
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ValueError("agent attempt duration must not be negative")
        if self.usage is not None and not isinstance(self.usage, AgentUsage):
            raise TypeError("agent attempt usage must be an AgentUsage")
        _validate_utc(self.started_at)
        if (self.status is ExecutionAttemptStatus.RUNNING) != (
            self.finished_at is None
        ):
            raise ValueError("only a running agent attempt has no finish timestamp")
        if self.finished_at is not None:
            _validate_utc(self.finished_at)
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("agent attempt cannot finish before it starts")


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """One immutable event emitted by the host-execution pipeline."""

    run_id: UUID
    kind: str
    task_id: UUID | None = None
    attempt_id: UUID | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "run_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"execution event {name} must be a UUID")
        for name in ("task_id", "attempt_id"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, UUID):
                raise TypeError(f"execution event {name} must be a UUID")
        if not self.kind.strip():
            raise ValueError("execution event kind must not be empty")
        if not isinstance(self.payload, dict):
            raise TypeError("execution event payload must be a dictionary")
        _validate_utc(self.created_at)


@dataclass(frozen=True, slots=True)
class ComposeResource:
    """Durable identity for a Compose resource owned by one claimed task."""

    run_id: UUID
    claim_id: UUID
    task_id: UUID
    project_name: str
    resource_type: str
    resource_name: str
    labels: dict[str, str] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "run_id", "claim_id", "task_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"Compose resource {name} must be a UUID")
        for name in ("project_name", "resource_type", "resource_name"):
            if not getattr(self, name).strip():
                raise ValueError(f"Compose resource {name} must not be empty")
        if not isinstance(self.labels, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.labels.items()
        ):
            raise TypeError("Compose resource labels must map strings to strings")
        _validate_utc(self.created_at)
