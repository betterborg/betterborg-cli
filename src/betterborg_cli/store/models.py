"""Domain records persisted in the local BetterBorg store."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

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
    TASKS_APPROVAL_PENDING = "tasks_approval_pending"
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
