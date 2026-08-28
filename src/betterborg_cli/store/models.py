"""Domain records persisted in the local BetterBorg store."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


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
