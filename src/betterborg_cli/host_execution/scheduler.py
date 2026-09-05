"""Lease-backed dependency scheduler for host task execution."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import partial
from threading import Lock
from typing import Protocol
from uuid import UUID

from betterborg_cli.agent_runtime import CancellationToken
from betterborg_cli.progress import (
    AgentActivity,
    ProgressError,
    RunProgress,
    StageSpec,
    StageState,
)
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


ActivitySink = Callable[[AgentActivity], None]
TaskActivitySink = Callable[[UUID, AgentActivity], None]


@dataclass(frozen=True, slots=True)
class HostSchedulerConfig:
    """Timing and concurrency limits for one execution scheduler."""

    jobs: int = 1
    review_passes: int = 3
    lease_duration: timedelta = timedelta(minutes=2)
    heartbeat_interval: timedelta = timedelta(seconds=30)
    poll_interval_seconds: float = 0.02

    def __post_init__(self) -> None:
        if self.jobs < 1:
            raise ValueError("scheduler jobs must be positive")
        if self.review_passes < 1:
            raise ValueError("scheduler review passes must be positive")
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
    activity: ActivitySink | None = field(default=None, repr=False, compare=False)
    _transitioned: Callable[[TaskRuntime], None] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def stage_key(self) -> str:
        """Return this task's stable key in the shared progress renderer."""
        return str(self.claim.task_id)

    def activity_sink(self, agent_label: str) -> ActivitySink | None:
        """Return a labelled view of the task-bound redacting activity sink."""
        label = agent_label.strip()
        if not label:
            raise ValueError("agent activity label must not be empty")
        if self.activity is None:
            return None

        def emit(activity: AgentActivity) -> None:
            detail = label
            if activity.detail:
                detail = f"{label}: {activity.detail}"
            self.activity(AgentActivity(activity.kind, detail))

        return emit

    @property
    def agent_activity_sink(self) -> ActivitySink | None:
        """Return the legacy unlabelled task-bound provider sink."""
        return self.activity

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
        runtime = self.store.transition_task_runtime(
            self.claim.run_id,
            self.owner_token,
            self.claim.id,
            self.claim.claim_token,
            expected_status=expected_status,
            new_status=new_status,
            now=self.clock(),
            **changes,
        )
        if self._transitioned is not None:
            self._transitioned(runtime)
        return runtime

    def reconcile_progress(self) -> TaskRuntime:
        """Reflect the latest durable row after a lower-level transition."""
        runtime = self.runtime
        if self._transitioned is not None:
            self._transitioned(runtime)
        return runtime


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
        activity_handoff: (
            Callable[[UUID, AgentActivity], AgentActivity] | None
        ) = None,
        progress: RunProgress | None = None,
        interruption_cleanup: Callable[[], None] | None = None,
    ) -> None:
        self._store = store
        self._behavior = behavior
        self._config = config or HostSchedulerConfig()
        self._clock = clock
        self._activity_handoff = activity_handoff
        self._progress = progress
        self._interruption_cleanup = interruption_cleanup
        self._progress_projection_lock = Lock()
        self._render_error: BaseException | None = None

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
        try:
            result = self._run_acquired(generation_id, acquisition, cancel=cancel)
        except BaseException as error:
            self._finalize_progress_projection(generation_id, error)
            raise
        self._finalize_progress_projection(generation_id)
        return result

    def _run_acquired(
        self,
        generation_id: UUID,
        acquisition: ExecutionRunAcquisition,
        *,
        cancel: CancellationToken | None = None,
    ) -> HostSchedulerResult:
        """Run an acquired operation before final presentation reconciliation."""
        token = cancel or CancellationToken()
        started_at = self._clock()
        owner_token = acquisition.owner_token
        if owner_token is None:  # Defensive narrowing of the acquisition contract.
            raise RuntimeError("acquired execution run has no owner token")

        active: dict[Future[TaskRuntimeStatus | None], TaskClaim] = {}
        next_heartbeat = started_at + self._config.heartbeat_interval
        try:
            self._seed_progress(acquisition.run_id, generation_id)
        except KeyboardInterrupt as error:
            self._interrupt_after_keyboard_interrupt(
                error,
                token,
                active,
                acquisition.run_id,
                generation_id,
                owner_token,
                next_heartbeat,
            )
            raise
        except ProgressError:
            if not self._cancellation_requested(token):
                raise
            self._interrupt(
                token,
                active,
                acquisition.run_id,
                generation_id,
                owner_token,
                next_heartbeat,
            )
            return self._result(acquisition.run_id, generation_id, acquired=True)

        with ThreadPoolExecutor(
            max_workers=self._config.jobs,
            thread_name_prefix="betterborg-task",
        ) as executor:
            try:
                while True:
                    now = self._clock()
                    if self._cancellation_requested(token):
                        self._interrupt(
                            token,
                            active,
                            acquisition.run_id,
                            generation_id,
                            owner_token,
                            next_heartbeat,
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

                    while (
                        len(active) < self._config.jobs
                        and not self._cancellation_requested(token)
                    ):
                        claim = self._store.claim_dependency_ready_task(
                            acquisition.run_id,
                            owner_token,
                            lease_duration=self._config.lease_duration,
                            now=self._clock(),
                        )
                        if claim is None:
                            break
                        if self._cancellation_requested(token):
                            break
                        self._start_progress(claim)
                        if self._cancellation_requested(token):
                            break
                        context = ScheduledTaskContext(
                            store=self._store,
                            claim=claim,
                            owner_token=owner_token,
                            cancel=token,
                            clock=self._clock,
                            activity=(
                                partial(self._report_activity, claim.task_id)
                                if self._activity_handoff is not None
                                or self._progress is not None
                                else None
                            ),
                            _transitioned=self._transition_progress,
                        )
                        active[executor.submit(self._behavior, context)] = claim

                    if self._cancellation_requested(token):
                        continue
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
            except KeyboardInterrupt as error:
                self._interrupt_after_keyboard_interrupt(
                    error,
                    token,
                    active,
                    acquisition.run_id,
                    generation_id,
                    owner_token,
                    next_heartbeat,
                )
                raise
            except ProgressError:
                if not self._cancellation_requested(token):
                    raise
                self._interrupt(
                    token,
                    active,
                    acquisition.run_id,
                    generation_id,
                    owner_token,
                    next_heartbeat,
                )
                return self._result(
                    acquisition.run_id, generation_id, acquired=True
                )
            except ExecutionOwnershipError:
                token.cancel()
                try:
                    self._drain(active)
                    if self._interruption_cleanup is not None:
                        self._interruption_cleanup()
                finally:
                    self._reconcile_interrupted_progress(
                        acquisition.run_id, generation_id
                    )
                raise

    def _cancellation_requested(self, token: CancellationToken) -> bool:
        """Return whether either execution cancellation surface has fired."""
        return token.is_set() or (
            self._progress is not None and self._progress.cancelling
        )

    def _seed_progress(self, run_id: UUID, generation_id: UUID) -> None:
        """Declare tasks and seed only authoritative terminal projections."""
        if self._progress is None or self._progress.cancelling:
            return
        records = self._store.list_task_records(generation_id)
        runtimes = {
            runtime.task_id: runtime
            for runtime in self._store.list_task_runtimes(generation_id)
        }
        run = self._store.get_execution_run(run_id)
        if run is None:
            raise KeyError(f"execution run {run_id} not found")
        rows = {
            row.task_id: row
            for row in self._store.list_task_runtime(run.borg_id)
            if row.generation_id == generation_id
        }
        for record in records:
            stage_key = str(record.id)
            if stage_key not in self._progress.stages:
                spec = StageSpec(stage_key, record.title)
                self._project_progress(self._progress.declare, spec)
            stage = self._progress.stages[stage_key]
            if stage.state is not StageState.PENDING:
                continue
            runtime = runtimes[record.id]
            row = rows.get(record.id)
            duration = row.duration_seconds if row is not None else None
            result = self._progress_result(runtime)
            if runtime.status is TaskRuntimeStatus.DONE:
                self._project_progress(
                    self._progress.seed_completed, stage_key, result, duration
                )
            elif runtime.status in {
                TaskRuntimeStatus.FAILED,
                TaskRuntimeStatus.BLOCKED,
            }:
                self._project_progress(
                    self._progress.seed_failed, stage_key, result, duration
                )

    def _start_progress(self, claim: TaskClaim) -> None:
        """Start a task timer only after its durable claim was accepted."""
        if self._progress is None:
            return
        stage = self._progress.stages.get(str(claim.task_id))
        if stage is not None and stage.state is StageState.PENDING:
            self._project_progress(self._progress.start, stage.key)
            runtime = self._store.get_task_runtime(claim.task_id)
            if runtime is None:
                raise KeyError(f"task runtime {claim.task_id} not found")
            detail = self._progress_detail(runtime, phase=claim.resume_phase)
            self._project_progress(self._progress.update, stage.key, detail)

    def _transition_progress(self, runtime: TaskRuntime) -> None:
        """Reflect one already-accepted durable task transition."""
        if self._progress is None:
            return
        stage = self._progress.stages.get(str(runtime.task_id))
        if stage is None or stage.state is not StageState.RUNNING:
            return
        result = self._progress_result(runtime)
        if runtime.status is TaskRuntimeStatus.DONE:
            self._project_progress(self._progress.complete, stage.key, result)
        elif runtime.status in {TaskRuntimeStatus.FAILED, TaskRuntimeStatus.BLOCKED}:
            self._project_progress(self._progress.fail, stage.key, result)
        else:
            detail = self._progress_detail(runtime)
            self._project_progress(self._progress.update, stage.key, detail)

    def _report_activity(self, task_id: UUID, activity: AgentActivity) -> None:
        """Hand off activity before forwarding only its returned projection."""
        reported = activity
        if self._activity_handoff is not None:
            reported = self._activity_handoff(task_id, activity)
        if self._progress is None:
            return
        stage = self._progress.stages.get(str(task_id))
        if stage is not None and stage.state is StageState.RUNNING:
            self._project_progress(self._progress.activity, stage.key, reported)

    def _reconcile_interrupted_progress(
        self, run_id: UUID, generation_id: UUID
    ) -> None:
        """Project post-interruption durable state into declared display rows."""
        if self._progress is None:
            return
        run = self._store.get_execution_run(run_id)
        if run is None:
            raise KeyError(f"execution run {run_id} not found")
        durations = {
            row.task_id: row.duration_seconds
            for row in self._store.list_task_runtime(run.borg_id)
            if row.generation_id == generation_id
        }
        for runtime in self._store.list_task_runtimes(generation_id):
            stage = self._progress.stages.get(str(runtime.task_id))
            if stage is None:
                continue
            if runtime.status in _TERMINAL_TASK_STATUSES:
                if stage.state is StageState.RUNNING:
                    self._transition_progress(runtime)
                elif stage.state is StageState.PENDING:
                    result = self._progress_result(runtime)
                    duration = durations.get(runtime.task_id)
                    if runtime.status is TaskRuntimeStatus.DONE:
                        self._project_progress(
                            self._progress.seed_completed,
                            stage.key,
                            result,
                            duration,
                        )
                    else:
                        self._project_progress(
                            self._progress.seed_failed,
                            stage.key,
                            result,
                            duration,
                        )
            elif stage.state is StageState.RUNNING:
                self._project_progress(
                    self._progress.stop, stage.key, "execution cancelled"
                )

    def _project_progress(
        self, projection: Callable[..., object], *args: object
    ) -> None:
        """Apply one lifecycle projection while retaining renderer failures."""
        progress = self._progress
        if progress is None:
            return
        with self._progress_projection_lock:
            self._capture_autonomous_render_error(progress)
            try:
                projection(*args)
            except ProgressError:
                raise
            except Exception as error:
                self._remember_render_error(error)
            self._capture_autonomous_render_error(progress)

    def _capture_progress_render_error(self) -> None:
        progress = self._progress
        if progress is None:
            return
        with self._progress_projection_lock:
            self._capture_autonomous_render_error(progress)

    def _capture_autonomous_render_error(self, progress: RunProgress) -> None:
        try:
            progress.raise_if_render_failed()
        except ProgressError:
            raise
        except BaseException as error:
            self._remember_render_error(error)

    def _remember_render_error(self, error: BaseException) -> None:
        if self._render_error is None:
            self._render_error = error

    def _finalize_progress_projection(
        self,
        generation_id: UUID,
        authoritative_error: BaseException | None = None,
    ) -> None:
        """Reconcile terminal rows, then surface or annotate rendering failure."""
        reconciliation_error: BaseException | None = None
        try:
            if self._progress is not None:
                self._capture_progress_render_error()
                for runtime in self._store.list_task_runtimes(generation_id):
                    if runtime.status in _TERMINAL_TASK_STATUSES:
                        self._transition_progress(runtime)
                self._capture_progress_render_error()
        except BaseException as error:
            reconciliation_error = error

        with self._progress_projection_lock:
            render_error = self._render_error
            self._render_error = None

        final_error = authoritative_error or reconciliation_error
        if authoritative_error is not None and reconciliation_error is not None:
            authoritative_error.add_note(
                "execution progress reconciliation also failed: "
                f"{reconciliation_error}"
            )
        if render_error is not None:
            if final_error is None:
                raise render_error
            if render_error is not final_error:
                final_error.add_note(
                    f"execution progress rendering also failed: {render_error}"
                )
        if authoritative_error is None and reconciliation_error is not None:
            raise reconciliation_error

    def _progress_detail(
        self,
        runtime: TaskRuntime,
        *,
        phase: str | None = None,
    ) -> str:
        """Project one durable active phase without changing its lifecycle."""
        phase_label = runtime.status.value if phase is None else phase
        if phase_label == TaskRuntimeStatus.REVIEW.value:
            pass_number = runtime.review_round + 1
            if 1 <= pass_number <= self._config.review_passes:
                return (
                    f"review (pass {pass_number}/{self._config.review_passes})"
                )
        elif phase_label == TaskRuntimeStatus.FIX.value:
            pass_number = runtime.review_round
            if 1 <= pass_number < self._config.review_passes:
                return f"fix (pass {pass_number}/{self._config.review_passes})"

        detail = phase_label
        if runtime.state_reason:
            detail += f": {runtime.state_reason}"
        return detail

    @staticmethod
    def _progress_result(runtime: TaskRuntime) -> str:
        if runtime.status is TaskRuntimeStatus.DONE:
            return "merged"
        result = runtime.status.value
        if runtime.state_reason:
            result += f": {runtime.state_reason}"
        return result

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
                    self._transition_progress(runtime)
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
                runtime = self._store.transition_task_runtime(
                    claim.run_id,
                    owner_token,
                    claim.id,
                    claim.claim_token,
                    expected_status=runtime.status,
                    new_status=terminal_status,
                    state_reason=reason,
                    now=self._clock(),
                )
                self._transition_progress(runtime)
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

    def _interrupt(
        self,
        token: CancellationToken,
        active: dict[Future[TaskRuntimeStatus | None], TaskClaim],
        run_id: UUID,
        generation_id: UUID,
        owner_token: str,
        next_heartbeat: datetime,
    ) -> None:
        """Fence active work before durably interrupting and reconciling a run."""
        token.cancel()
        if self._progress is not None:
            self._project_progress(self._progress.begin_cancellation)
        try:
            self._drain_cancelled(active, run_id, owner_token, next_heartbeat)
            self._store.interrupt_execution_run(
                run_id,
                owner_token,
                reason="execution cancelled",
                now=self._clock(),
            )
            if self._interruption_cleanup is not None:
                self._interruption_cleanup()
        finally:
            self._reconcile_interrupted_progress(run_id, generation_id)

    def _interrupt_after_keyboard_interrupt(
        self,
        error: KeyboardInterrupt,
        token: CancellationToken,
        active: dict[Future[TaskRuntimeStatus | None], TaskClaim],
        run_id: UUID,
        generation_id: UUID,
        owner_token: str,
        next_heartbeat: datetime,
    ) -> None:
        """Preserve the user interrupt while recording cancellation failures."""
        try:
            self._interrupt(
                token,
                active,
                run_id,
                generation_id,
                owner_token,
                next_heartbeat,
            )
        except BaseException as interruption_error:
            error.add_note(
                f"execution interruption reconciliation failed: {interruption_error}"
            )

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
        for runtime in runtimes:
            self._transition_progress(runtime)
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
