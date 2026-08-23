"""Domain records persisted in the local BetterBorg store."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utcnow() -> datetime:
    """Return a timezone-aware timestamp in UTC."""
    return datetime.now(UTC)


def _new_id() -> str:
    return secrets.token_hex(16)


def _validate_id(value: str) -> None:
    if len(value) != 32 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("IDs must contain exactly 32 lowercase hexadecimal characters")


def _validate_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("timestamps must be timezone-aware UTC values")


@dataclass(frozen=True, slots=True)
class Repository:
    """One Git repository managed by this local BetterBorg store."""

    root: Path
    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())
        _validate_id(self.id)
        _validate_utc(self.created_at)


@dataclass(frozen=True, slots=True)
class Operation:
    """An immutable entry in a repository's local operation ledger."""

    repository_id: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        _validate_id(self.id)
        _validate_id(self.repository_id)
        if not self.kind:
            raise ValueError("operation kind must not be empty")
        _validate_utc(self.created_at)
