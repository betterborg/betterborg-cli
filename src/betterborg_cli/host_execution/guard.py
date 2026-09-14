"""Primary-checkout protection for host execution phases."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from betterborg_cli.host_execution.git import SafeGit, UnsafeGitError, _status_entries
from betterborg_cli.repo_paths import MANAGED_IGNORE_RULE

#: Where a guard finding is recorded on the exception a failing phase already
#: raised. A phase that both raised and dirtied the checkout gets its
#: contamination attached rather than raised, and a caller that has to tell a
#: breach from a failure reads it from the same place it reads the other arms.
_ATTACHED_ATTRIBUTE = "_betterborg_checkout_contamination"


class CheckoutCondition(StrEnum):
    """Which of the guard's checks raised.

    A caller that treats a checkout dirty before a phase began differently from
    one the phase changed while it ran reads this rather than the message. Each
    raise site names its member once and the message is rendered from that same
    member, so neither a reword nor a mistyped argument can give one check's
    wording to the other check's decision. ``UNREADABLE`` is the state neither
    check reached, which is why it names no snapshot.
    """

    DIRTY = "was dirty"
    CHANGED = "changed"
    UNREADABLE = "could not be inspected"


class PrimaryCheckoutContaminationError(RuntimeError):
    """Raised when host work starts dirty or changes the primary checkout."""

    def __init__(self, message: str, *, condition: CheckoutCondition) -> None:
        super().__init__(message)
        self.condition = condition


def attached_contamination(
    error: BaseException,
) -> PrimaryCheckoutContaminationError | None:
    """Return the guard finding recorded beside a phase's own failure."""

    found = getattr(error, _ATTACHED_ATTRIBUTE, None)
    return found if isinstance(found, PrimaryCheckoutContaminationError) else None


def checkout_was_changed(error: BaseException) -> bool:
    """Return whether this failure carries a phase changing the checkout.

    The guard raises its finding when the phase itself did not, and attaches it
    to the phase's own exception when it did, so both the error in hand and the
    one hanging off it are read. A checkout that was dirty before the phase
    began is not this: the phase did not write it, and it is not evidence the
    phase wrote anything.
    """

    for candidate in (error, attached_contamination(error)):
        if (
            isinstance(candidate, PrimaryCheckoutContaminationError)
            and candidate.condition is CheckoutCondition.CHANGED
        ):
            return True
    return False


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
        ignored_prefixes: Iterable[str] = (MANAGED_IGNORE_RULE,),
        git: SafeGit | None = None,
    ) -> None:
        self._repo = Path(primary_repo).resolve()
        if git is not None and git.cwd != self._repo:
            raise ValueError("checkout guard Git binding must match primary repo")
        self._git = git or SafeGit(self._repo)
        self._ignored_prefixes = tuple(ignored_prefixes)
        self._snapshots: dict[tuple[str, str], _CheckoutSnapshot] = {}
        self._lock = threading.Lock()

    def assert_clean(self, operation: str = "host execution") -> None:
        dirty = sorted(self._snapshot().status)
        if dirty:
            raise self._contamination(
                operation,
                "before it started",
                dirty,
                condition=CheckoutCondition.DIRTY,
            )

    def before_phase(self, task_ref: str, phase: str) -> None:
        snapshot = self._snapshot()
        dirty = sorted(snapshot.status)
        if dirty:
            raise self._contamination(
                f"{phase} for {task_ref}",
                "before it started",
                dirty,
                condition=CheckoutCondition.DIRTY,
            )
        with self._lock:
            self._snapshots[(task_ref, phase)] = snapshot

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
            raise self._contamination(
                f"{phase} for {task_ref}",
                "while it ran",
                changes,
                condition=CheckoutCondition.CHANGED,
            )

    @contextmanager
    def protect(self, task_ref: str, phase: str) -> Iterator[None]:
        """Require clean entry and raise if a phase changes the checkout."""
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
                # A phase that both raised and changed the checkout has to be
                # tellable from one that only raised, so the finding is
                # recorded on the exception as well as read into its notes.
                setattr(active_error, _ATTACHED_ATTRIBUTE, error)
                active_error.add_note(str(error))

    def _snapshot(self) -> _CheckoutSnapshot:
        status = self._git_output(
            ["status", "--porcelain=v1", "-z", "-uall"]
        )
        head = self._git_output(["rev-parse", "--verify", "HEAD"]).strip()
        branch = self._git_output(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
        return _CheckoutSnapshot(
            # Runtime-owned paths are safe to ignore only while untracked.
            # Once Git tracks a path, every modification is contamination.
            status=frozenset(
                entry
                for entry, paths in _status_entries(status)
                if not (
                    entry.startswith("?? ")
                    and all(self._ignored(path) for path in paths)
                )
            ),
            head=head,
            branch=branch,
        )

    def _git_output(self, arguments: list[str]) -> str:
        try:
            result = self._git.run(arguments, check=False)
        except UnsafeGitError as error:
            raise self._unreadable(str(error)) from error
        if result.returncode != 0:
            raise self._unreadable(result.stderr.strip())
        return result.stdout

    def _ignored(self, path: str) -> bool:
        return any(
            path == prefix.rstrip("/") or path.startswith(prefix)
            for prefix in self._ignored_prefixes
        )

    def _unreadable(self, detail: str) -> PrimaryCheckoutContaminationError:
        """Build the condition neither snapshot check reached.

        Named once here as the other two are named once at their own raise
        sites, so this one cannot be given another check's decision either.
        """
        condition = CheckoutCondition.UNREADABLE
        return PrimaryCheckoutContaminationError(
            f"primary checkout {self._repo} {condition.value}: {detail}",
            condition=condition,
        )

    def _contamination(
        self,
        operation: str,
        timing: str,
        entries: list[str],
        *,
        condition: CheckoutCondition,
    ) -> PrimaryCheckoutContaminationError:
        """Build one condition's wording and its decision from one member.

        A raise site names the condition once, so it cannot give one check's
        message to the other check's decision.
        """
        details = "\n".join(f"  {line}" for line in entries[:40])
        return PrimaryCheckoutContaminationError(
            f"primary checkout {self._repo} {condition.value} {timing} during "
            f"{operation}; task work was preserved and execution is blocked:\n"
            f"{details}",
            condition=condition,
        )
