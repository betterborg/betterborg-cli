"""Primary-checkout protection for host execution phases."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from betterborg_cli.host_execution.git import _hardened_git_environment, _status_path


class PrimaryCheckoutContaminationError(RuntimeError):
    """Raised when host work starts dirty or changes the primary checkout."""


class PrimaryCheckoutGuard:
    """Snapshot and compare primary-checkout status without repairing it."""

    def __init__(
        self,
        primary_repo: Path,
        *,
        ignored_prefixes: Iterable[str] = (".borg/state/",),
    ) -> None:
        self._repo = Path(primary_repo).resolve()
        self._ignored_prefixes = tuple(ignored_prefixes)
        self._snapshots: dict[tuple[str, str], frozenset[str]] = {}
        self._lock = threading.Lock()

    def assert_clean(self, operation: str = "host execution") -> None:
        dirty = sorted(self._snapshot())
        if dirty:
            raise PrimaryCheckoutContaminationError(
                self._message(operation, "before it started", dirty)
            )

    def before_phase(self, task_ref: str, phase: str) -> None:
        with self._lock:
            self._snapshots[(task_ref, phase)] = self._snapshot()

    def after_phase(self, task_ref: str, phase: str) -> None:
        current = self._snapshot()
        with self._lock:
            before = self._snapshots.pop((task_ref, phase), None)
        if before is None:
            raise RuntimeError(f"primary checkout phase was not started: {phase}")
        added = sorted(current - before)
        if added:
            raise PrimaryCheckoutContaminationError(
                self._message(f"{phase} for {task_ref}", "while it ran", added)
            )

    @contextmanager
    def protect(self, task_ref: str, phase: str) -> Iterator[None]:
        """Raise after a host phase if it added primary-checkout dirt."""
        self.before_phase(task_ref, phase)
        active_error: BaseException | None = None
        try:
            yield
        except BaseException as error:
            active_error = error
            raise
        finally:
            try:
                self.after_phase(task_ref, phase)
            except PrimaryCheckoutContaminationError as error:
                if active_error is None:
                    raise
                active_error.add_note(str(error))

    def _snapshot(self) -> frozenset[str]:
        result = subprocess.run(
            ["git", "status", "--porcelain", "-uall"],
            cwd=str(self._repo),
            check=False,
            capture_output=True,
            text=True,
            env=_hardened_git_environment(),
        )
        if result.returncode != 0:
            raise PrimaryCheckoutContaminationError(
                f"unable to inspect primary checkout {self._repo}: "
                f"{result.stderr.strip()}"
            )
        return frozenset(
            line
            for line in result.stdout.splitlines()
            if line and not self._ignored(_status_path(line))
        )

    def _ignored(self, path: str) -> bool:
        return any(
            path == prefix.rstrip("/") or path.startswith(prefix)
            for prefix in self._ignored_prefixes
        )

    def _message(self, operation: str, timing: str, entries: list[str]) -> str:
        details = "\n".join(f"  {line}" for line in entries[:40])
        return (
            f"primary checkout {self._repo} was dirty {timing} during "
            f"{operation}; task work was preserved and execution is blocked:\n"
            f"{details}"
        )
