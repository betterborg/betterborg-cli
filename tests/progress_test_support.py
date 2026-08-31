"""Shared deterministic renderer scaffolding for progress tests."""

from __future__ import annotations

from io import StringIO


class FakeClock:
    """A manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TTYStringIO(StringIO):
    """An in-memory stream that exercises Rich's interactive renderer."""

    def isatty(self) -> bool:
        return True
