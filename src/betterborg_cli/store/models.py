"""Domain records persisted in the local BetterBorg store."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

_PROMPT_ROLES = frozenset({"coding", "review", "merge"})


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
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for name in ("id", "repository_id"):
            if not isinstance(getattr(self, name), UUID):
                raise TypeError(f"Borg {name} must be a UUID")
        if not self.name.strip():
            raise ValueError("Borg name must not be empty")
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
