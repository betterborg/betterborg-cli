"""Primary-checkout protection for host execution phases."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from betterborg_cli.host_execution.git import _hardened_git_environment, _status_path


class PrimaryCheckoutContaminationError(RuntimeError):
    """Raised when host work starts dirty or changes the primary checkout."""


@dataclass(frozen=True)
class _CheckoutSnapshot:
    status: frozenset[str]
    head: str
    branch: str


class PrimaryCheckoutGuard:
    """Snapshot and compare primary-checkout state without repairing it."""

    def __init__(
        self,
        primary_repo: Path,
        *,
        ignored_prefixes: Iterable[str] = (".borg/state/",),
    ) -> None:
        self._repo = Path(primary_repo).resolve()
        self._ignored_prefixes = tuple(ignored_prefixes)
        self._snapshots: dict[tuple[str, str], _CheckoutSnapshot] = {}
        self._lock = threading.Lock()

    def assert_clean(self, operation: str = "host execution") -> None:
        dirty = sorted(self._snapshot().status)
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
        changes = sorted(current.status - before.status)
        if current.head != before.head:
            changes.insert(0, f"HEAD changed: {before.head} -> {current.head}")
        if current.branch != before.branch:
            changes.insert(
                0, f"branch changed: {before.branch} -> {current.branch}"
            )
        if changes:
            raise PrimaryCheckoutContaminationError(
                self._message(
                    f"{phase} for {task_ref}",
                    "while it ran",
                    changes,
                    condition="changed",
                )
            )

    @contextmanager
    def protect(self, task_ref: str, phase: str) -> Iterator[None]:
        """Raise after a host phase if it changed the primary checkout."""
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

    def _snapshot(self) -> _CheckoutSnapshot:
        status = self._git_output(["status", "--porcelain", "-uall"])
        head = self._git_output(["rev-parse", "--verify", "HEAD"]).strip()
        branch = self._git_output(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
        return _CheckoutSnapshot(
            status=frozenset(
                line
                for line in status.splitlines()
                if line and not self._ignored(_status_path(line))
            ),
            head=head,
            branch=branch,
        )

    def _git_output(self, arguments: list[str]) -> str:
        result = subprocess.run(
            ["git", *arguments],
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
        return result.stdout

    def _ignored(self, path: str) -> bool:
        return any(
            path == prefix.rstrip("/") or path.startswith(prefix)
            for prefix in self._ignored_prefixes
        )

    def _message(
        self,
        operation: str,
        timing: str,
        entries: list[str],
        *,
        condition: str = "was dirty",
    ) -> str:
        details = "\n".join(f"  {line}" for line in entries[:40])
        return (
            f"primary checkout {self._repo} {condition} {timing} during "
            f"{operation}; task work was preserved and execution is blocked:\n"
            f"{details}"
        )
