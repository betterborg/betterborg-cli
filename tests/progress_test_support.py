"""Shared deterministic renderer scaffolding for progress tests."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
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


class WaitableStringIO(StringIO):
    """A thread-safe capture whose contents can drive deterministic waits."""

    def __init__(self, *, interactive: bool = False) -> None:
        super().__init__()
        self._interactive = interactive
        self._condition = threading.Condition(threading.RLock())
        self.write_count = 0

    def isatty(self) -> bool:
        return self._interactive

    def write(self, value: str) -> int:
        with self._condition:
            written = super().write(value)
            self.write_count += 1
            self._condition.notify_all()
            return written

    def flush(self) -> None:
        with self._condition:
            super().flush()
            self._condition.notify_all()

    def getvalue(self) -> str:
        with self._condition:
            return super().getvalue()

    def wait_for(
        self, predicate: Callable[[str], bool], *, timeout: float = 2.0
    ) -> str:
        """Return captured text once ``predicate`` matches or fail on timeout."""

        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                value = super().getvalue()
                if predicate(value):
                    return value
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for renderer output")
                self._condition.wait(remaining)


class BarrierStringIO(WaitableStringIO):
    """A capture that can hold one renderer write until a test releases it."""

    def __init__(self, *, interactive: bool = True) -> None:
        super().__init__(interactive=interactive)
        self._armed = threading.Event()
        self.entered = threading.Event()
        self.release = threading.Event()

    def hold_next_write(self) -> None:
        self.entered.clear()
        self.release.clear()
        self._armed.set()

    def write(self, value: str) -> int:
        if self._armed.is_set():
            self._armed.clear()
            self.entered.set()
            if not self.release.wait(timeout=2):
                raise TimeoutError("render barrier was not released")
        return super().write(value)


class FailingStringIO(WaitableStringIO):
    """A capture that raises the canonical renderer failure on demand."""

    def __init__(self, *, interactive: bool = False) -> None:
        super().__init__(interactive=interactive)
        self._fail_write = threading.Event()
        self._fail_flush = threading.Event()

    def fail_next_write(self) -> None:
        self._fail_write.set()

    def fail_next_flush(self) -> None:
        self._fail_flush.set()

    def write(self, value: str) -> int:
        if self._fail_write.is_set():
            self._fail_write.clear()
            raise RuntimeError("progress heartbeat failed")
        return super().write(value)

    def flush(self) -> None:
        if self._fail_flush.is_set():
            self._fail_flush.clear()
            raise RuntimeError("progress heartbeat failed")
        super().flush()
