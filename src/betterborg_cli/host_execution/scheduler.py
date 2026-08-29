"""Lease-backed dependency scheduler for host task execution."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID

from betterborg_cli.agent_runtime import CancellationToken
from betterborg_cli.store import (
    ExecutionOwnershipError,
    ExecutionRunAcquisition,
    ExecutionRunStatus,
    SqliteStore,
    TaskClaim,
    TaskRuntime,
    TaskRuntimeStatus,
)

_TERMINAL_TASK_STATUSES = frozenset(
    {
        TaskRuntimeStatus.DONE,
        TaskRuntimeStatus.BLOCKED,
        TaskRuntimeStatus.FAILED,
    }
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class HostSchedulerConfig:
    """Timing and concurrency limits for one execution scheduler."""

    jobs: int = 1
    lease_duration: timedelta = timedelta(minutes=2)
    heartbeat_interval: timedelta = timedelta(seconds=30)
    poll_interval_seconds: float = 0.02

    def __post_init__(self) -> None:
        if self.jobs < 1:
            raise ValueError("scheduler jobs must be positive")
        if self.lease_duration <= timedelta(0):
            raise ValueError("scheduler lease duration must be positive")
        if not timedelta(0) < self.heartbeat_interval < self.lease_duration:
            raise ValueError(
                "scheduler heartbeat interval must be positive and shorter "
                "than its lease"
            )
        if self.poll_interval_seconds <= 0:
            raise ValueError("scheduler poll interval must be positive")


@dataclass(frozen=True, slots=True)
class ScheduledTaskContext:
    """Owned task claim passed to an injected host task implementation."""

    store: SqliteStore = field(repr=False)
    claim: TaskClaim
    owner_token: str = field(repr=False)
    cancel: CancellationToken
    clock: Callable[[], datetime] = field(repr=False)

    @property
    def runtime(self) -> TaskRuntime:
        """Return the task's latest durable runtime projection."""
        runtime = self.store.get_task_runtime(self.claim.task_id)
        if runtime is None:
            raise KeyError(f"task runtime {self.claim.task_id} not found")
        return runtime

    def transition(
        self,
        expected_status: TaskRuntimeStatus,
        new_status: TaskRuntimeStatus,
        **changes: object,
    ) -> TaskRuntime:
        """Persist one claim-owned phase transition for injected behavior."""
        return self.store.transition_task_runtime(
            self.claim.run_id,
            self.owner_token,
            self.claim.id,
            self.claim.claim_token,
            expected_status=expected_status,
            new_status=new_status,
            now=self.clock(),
            **changes,
        )


class HostTaskBehavior(Protocol):
    """Temporary task-runtime seam used until concrete phases are assembled."""

    def __call__(self, context: ScheduledTaskContext) -> TaskRuntimeStatus | None: ...


@dataclass(frozen=True, slots=True)
class HostSchedulerResult:
    """Durable outcome of starting or observing one execution operation."""

    operation_id: UUID
    acquired: bool
    status: ExecutionRunStatus
    total: int
    done: int
    failed: int
    blocked: int
    pending: int

    @property
    def active_operation_id(self) -> UUID | None:
        """Return the already-active operation ID for a duplicate caller."""
        if not self.acquired and self.status is ExecutionRunStatus.RUNNING:
            return self.operation_id
        return None


class HostTaskScheduler:
    """Claim dependency-ready tasks and run them under one leased operation."""

    def __init__(
        self,
        store: SqliteStore,
        behavior: HostTaskBehavior,
        *,
        config: HostSchedulerConfig | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._store = store
        self._behavior = behavior
        self._config = config or HostSchedulerConfig()
        self._clock = clock

    def run(
        self,
        borg_id: UUID,
        generation_id: UUID,
        *,
        cancel: CancellationToken | None = None,
    ) -> HostSchedulerResult:
        """Run or observe the Borg's current-generation execution operation."""
        started_at = self._clock()
        acquisition = self._store.acquire_execution_run(
            borg_id,
            generation_id,
            lease_duration=self._config.lease_duration,
            now=started_at,
        )
        return self.run_acquired(generation_id, acquisition, cancel=cancel)

    def run_acquired(
        self,
        generation_id: UUID,
        acquisition: ExecutionRunAcquisition,
        *,
        cancel: CancellationToken | None = None,
    ) -> HostSchedulerResult:
        """Schedule a run already acquired by the host integration owner."""
        if not acquisition.acquired:
            return self._result(
                acquisition.operation_id,
                generation_id,
                acquired=False,
            )
        token = cancel or CancellationToken()
        started_at = self._clock()
        owner_token = acquisition.owner_token
        if owner_token is None:  # Defensive narrowing of the acquisition contract.
            raise RuntimeError("acquired execution run has no owner token")

        active: dict[Future[TaskRuntimeStatus | None], TaskClaim] = {}
        next_heartbeat = started_at + self._config.heartbeat_interval
        with ThreadPoolExecutor(
            max_workers=self._config.jobs,
            thread_name_prefix="betterborg-task",
        ) as executor:
            try:
                while True:
                    now = self._clock()
                    if token.is_set():
                        self._drain_cancelled(
                            active,
                            acquisition.run_id,
                            owner_token,
                            next_heartbeat,
                        )
                        self._store.interrupt_execution_run(
                            acquisition.run_id,
                            owner_token,
                            reason="execution cancelled",
                            now=self._clock(),
                        )
                        return self._result(
                            acquisition.run_id, generation_id, acquired=True
                        )

                    if now >= next_heartbeat:
                        self._store.renew_execution_run(
                            acquisition.run_id,
                            owner_token,
                            lease_duration=self._config.lease_duration,
                            now=now,
                        )
                        next_heartbeat = now + self._config.heartbeat_interval

                    self._settle_completed(active, owner_token)

                    while len(active) < self._config.jobs and not token.is_set():
                        claim = self._store.claim_dependency_ready_task(
                            acquisition.run_id,
                            owner_token,
                            lease_duration=self._config.lease_duration,
                            now=self._clock(),
                        )
                        if claim is None:
                            break
                        context = ScheduledTaskContext(
                            store=self._store,
                            claim=claim,
                            owner_token=owner_token,
                            cancel=token,
                            clock=self._clock,
                        )
                        active[executor.submit(self._behavior, context)] = claim

                    if not active:
                        return self._finish(
                            acquisition.run_id,
                            generation_id,
                            owner_token,
                        )

                    wait(
                        tuple(active),
                        timeout=self._config.poll_interval_seconds,
                        return_when=FIRST_COMPLETED,
                    )
            except ExecutionOwnershipError:
                token.cancel()
                self._drain(active)
                raise

    def _settle_completed(
        self,
        active: dict[Future[TaskRuntimeStatus | None], TaskClaim],
        owner_token: str,
    ) -> None:
        for future, claim in list(active.items()):
            if not future.done():
                continue
            del active[future]
            try:
                outcome = future.result()
                if outcome is not None and outcome not in _TERMINAL_TASK_STATUSES:
                    raise ValueError("task behavior returned a nonterminal status")
                runtime = self._store.get_task_runtime(claim.task_id)
                if runtime is None:
                    raise KeyError(f"task runtime {claim.task_id} not found")
                if runtime.status in _TERMINAL_TASK_STATUSES:
                    if outcome is not None and outcome is not runtime.status:
                        raise ValueError(
                            "task behavior returned a status different from its "
                            "durable outcome"
                        )
                    continue
                terminal_status = outcome or TaskRuntimeStatus.FAILED
                reason = (
                    None
                    if outcome is not None
                    else "task behavior returned without a terminal outcome"
                )
            except Exception as error:  # Isolate one injected task implementation.
                runtime = self._store.get_task_runtime(claim.task_id)
                if runtime is None or runtime.status in _TERMINAL_TASK_STATUSES:
                    continue
                terminal_status = TaskRuntimeStatus.FAILED
                reason = f"task behavior failed: {error}"
            try:
                self._store.transition_task_runtime(
                    claim.run_id,
                    owner_token,
                    claim.id,
                    claim.claim_token,
                    expected_status=runtime.status,
                    new_status=terminal_status,
                    state_reason=reason,
                    now=self._clock(),
                )
            except ExecutionOwnershipError:
                # Cancellation or lease loss already reconciled the durable claim.
                pass

    @staticmethod
    def _drain(
        active: dict[Future[TaskRuntimeStatus | None], TaskClaim],
    ) -> None:
        if active:
            wait(tuple(active))

    def _drain_cancelled(
        self,
        active: dict[Future[TaskRuntimeStatus | None], TaskClaim],
        run_id: UUID,
        owner_token: str,
        next_heartbeat: datetime,
    ) -> None:
        """Fence interruption behind active work while retaining ownership."""
        pending = {future for future in active if not future.done()}
        while pending:
            _, pending = wait(
                pending,
                timeout=self._config.poll_interval_seconds,
                return_when=FIRST_COMPLETED,
            )
            now = self._clock()
            if now >= next_heartbeat:
                self._store.renew_execution_run(
                    run_id,
                    owner_token,
                    lease_duration=self._config.lease_duration,
                    now=now,
                )
                next_heartbeat = now + self._config.heartbeat_interval

    def _finish(
        self,
        run_id: UUID,
        generation_id: UUID,
        owner_token: str,
    ) -> HostSchedulerResult:
        runtimes = self._store.list_task_runtimes(generation_id)
        status = (
            ExecutionRunStatus.COMPLETED
            if all(runtime.status is TaskRuntimeStatus.DONE for runtime in runtimes)
            else ExecutionRunStatus.FAILED
        )
        self._store.finish_execution_run(
            run_id,
            owner_token,
            status=status,
            now=self._clock(),
        )
        return self._result(run_id, generation_id, acquired=True)

    def _result(
        self,
        run_id: UUID,
        generation_id: UUID,
        *,
        acquired: bool,
    ) -> HostSchedulerResult:
        run = self._store.get_execution_run(run_id)
        if run is None:
            raise KeyError(f"execution run {run_id} not found")
        runtimes = self._store.list_task_runtimes(generation_id)
        counts = {
            status: sum(runtime.status is status for runtime in runtimes)
            for status in TaskRuntimeStatus
        }
        return HostSchedulerResult(
            operation_id=run_id,
            acquired=acquired,
            status=run.status,
            total=len(runtimes),
            done=counts[TaskRuntimeStatus.DONE],
            failed=counts[TaskRuntimeStatus.FAILED],
            blocked=counts[TaskRuntimeStatus.BLOCKED],
            pending=counts[TaskRuntimeStatus.PENDING],
        )
