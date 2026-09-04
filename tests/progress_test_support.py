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


class ASCIIOnlyStringIO(StringIO):
    """An in-memory stream that rejects every non-ASCII write."""

    def __init__(self, *, interactive: bool = False) -> None:
        super().__init__()
        self._interactive = interactive

    @property
    def encoding(self) -> str:
        return "ascii"

    def isatty(self) -> bool:
        return self._interactive

    def write(self, value: str) -> int:
        value.encode(self.encoding)
        return super().write(value)
