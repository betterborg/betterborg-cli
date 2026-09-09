"""Concrete trust-gated host execution assembly."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from threading import Event, Lock, Thread
from typing import Any
from uuid import UUID

from betterborg_cli.agent_runtime import CancellationToken
from betterborg_cli.host_execution.coding import HostCodingPhase
from betterborg_cli.host_execution.environment import (
    EnvironmentMaterializationError,
    HostEnvironmentManager,
    declared_secret_mask_values,
    redact_secrets,
)
from betterborg_cli.host_execution.guard import PrimaryCheckoutContaminationError
from betterborg_cli.host_execution.merge import HostMergePhase
from betterborg_cli.host_execution.preflight import (
    AnalyzerPlanLoader,
    HostPreflight,
    HostPreflightBlock,
    HostPreflightPlan,
)
from betterborg_cli.host_execution.review import HostReviewFixPhase
from betterborg_cli.host_execution.sanity import HostSanityPhase
from betterborg_cli.host_execution.scheduler import (
    HostSchedulerConfig,
    HostSchedulerResult,
    HostTaskScheduler,
    ScheduledTaskContext,
    TaskActivitySink,
)
from betterborg_cli.host_execution.worktrees import HostWorktreeManager, WorktreeError
from betterborg_cli.progress import AgentActivity, RunProgress
from betterborg_cli.store import ExecutionRunStatus, SqliteStore, TaskRuntimeStatus


class HostExecutionError(RuntimeError):
    """Raised when run-scoped host setup cannot be completed safely."""


TaskActivityHandoff = Callable[[UUID, AgentActivity], AgentActivity]


@dataclass(frozen=True, slots=True)
class _ExecutionActivityBinding:
    """Mask one acquired run's activity before any reporter can observe it."""

    mask_values: tuple[str, ...] = field(repr=False)
    reporter: TaskActivitySink | None = field(repr=False)

    def emit(self, task_id: UUID, activity: AgentActivity) -> AgentActivity:
        """Return freshly redacted activity after notifying the observer."""
        detail = activity.detail
        if detail is not None:
            detail = redact_secrets(detail, self.mask_values)
        redacted = AgentActivity(activity.kind, detail)
        try:
            if self.reporter is not None:
                self.reporter(task_id, redacted)
        except Exception:
            # Activity reporters are observational and cannot change execution.
            pass
        return redacted


@dataclass(frozen=True, slots=True)
class HostExecutionResult:
    """Outcome of validation, reconciliation, and optional scheduling."""

    preflight: HostPreflightPlan | HostPreflightBlock
    scheduler: HostSchedulerResult | None = None

    @property
    def operation_id(self) -> UUID | None:
        return self.scheduler.operation_id if self.scheduler is not None else None

    @property
    def active_operation_id(self) -> UUID | None:
        return (
            self.scheduler.active_operation_id if self.scheduler is not None else None
        )

    @property
    def acquired(self) -> bool:
        return self.scheduler.acquired if self.scheduler is not None else False

    @property
    def status(self) -> ExecutionRunStatus | None:
        return self.scheduler.status if self.scheduler is not None else None


class _SetupLeaseHeartbeats:
    """Keep run ownership live until the scheduler takes over heartbeats."""

    def __init__(
        self,
        store: SqliteStore,
        run_id: UUID,
        owner_token: str,
        config: HostSchedulerConfig,
        clock,
        *,
        started_at: datetime,
    ) -> None:
        self._store = store
        self._run_id = run_id
        self._owner_token = owner_token
        self._config = config
        self._clock = clock
        self._next = started_at + config.heartbeat_interval
        self._stop = Event()
        self._lock = Lock()
        self._error: BaseException | None = None
        self._thread = Thread(
            target=self._run,
            name="betterborg-setup-heartbeat",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self._renew_through(self._clock())
        if self._error is not None:
            raise self._error

    def checkpoint(self) -> None:
        """Surface heartbeat failures between setup stages."""
        self._renew_through(self._clock())
        if self._error is not None:
            raise self._error

    def _run(self) -> None:
        poll_seconds = min(self._config.heartbeat_interval.total_seconds(), 0.1)
        while not self._stop.wait(poll_seconds):
            try:
                self._renew_through(self._clock())
            except BaseException as error:
                self._error = error
                self._stop.set()
                return

    def _renew_through(self, now: datetime) -> None:
        with self._lock:
            while self._error is None and self._next <= now:
                self._store.renew_execution_run(
                    self._run_id,
                    self._owner_token,
                    lease_duration=self._config.lease_duration,
                    now=self._next,
                )
                self._next += self._config.heartbeat_interval


class HostTaskRuntime:
    """Run every concrete phase for one scheduler-owned task claim."""

    def __init__(
        self,
        plan: HostPreflightPlan,
        *,
        environment_manager: HostEnvironmentManager,
        coding: HostCodingPhase,
        review_fix: HostReviewFixPhase,
        merge: HostMergePhase,
        sanity: HostSanityPhase,
        secret_values: Mapping[str, str] | None = None,
        publication_lock: Any | None = None,
    ) -> None:
        self.plan = plan
        self._environment = environment_manager
        self._coding = coding
        self._review_fix = review_fix
        self._merge = merge
        self._sanity = sanity
        self._secret_values = dict(secret_values or {})
        self._publication_lock = publication_lock or Lock()

    def with_secret_values(
        self,
        secret_values: Mapping[str, str],
    ) -> HostTaskRuntime:
        """Bind one run's validated secret values without shared mutation."""
        return HostTaskRuntime(
            self.plan,
            environment_manager=self._environment,
            coding=self._coding,
            review_fix=self._review_fix,
            merge=self._merge,
            sanity=self._sanity,
            secret_values=secret_values,
            publication_lock=self._publication_lock,
        )

    def __call__(self, context: ScheduledTaskContext) -> TaskRuntimeStatus:
        """Materialize, execute, and publish one durable task."""
        if context.cancel.is_set():
            return self._durable_status(context)
        try:
            materialization = self._environment.materialize_claimed_task(
                context.store,
                self.plan,
                context.claim,
                context.owner_token,
                secret_values=self._secret_values,
                activity=context.activity,
                task_transition=context.transition,
            )
        except EnvironmentMaterializationError:
            context.reconcile_progress()
            return self._durable_status(context)
        if context.cancel.is_set():
            return self._durable_status(context)

        published_status: TaskRuntimeStatus | None = None
        task_environment = dict(materialization.environment)
        try:
            status = self._restore_resume_phase(context)
            if status is TaskRuntimeStatus.CODING:
                status = self._coding.run(
                    context,
                    environment={
                        **task_environment,
                        **self._agent_secrets(),
                    },
                    preparation_note=materialization.preparation_note,
                )
            if context.cancel.is_set():
                return self._durable_status(context)
            if status in {TaskRuntimeStatus.REVIEW, TaskRuntimeStatus.FIX}:
                status = self._review_fix.run(
                    context,
                    environment=task_environment,
                    review_environment=self._agent_secrets(),
                    fix_environment=self._agent_secrets(),
                )
            if context.cancel.is_set():
                return self._durable_status(context)
            if status is TaskRuntimeStatus.MERGING:
                # One project ref is shared by every task.  Keep merge-tip
                # production and sanity/base advancement in one process-local
                # critical section so jobs=2 cannot publish a stale tip.
                with self._publication_lock:
                    if context.cancel.is_set():
                        return self._durable_status(context)
                    merge_result = self._merge.run(
                        context,
                        environment={
                            **task_environment,
                            **self._agent_secrets(),
                        },
                    )
                    if merge_result.tip is not None and not context.cancel.is_set():
                        published_status = self._sanity.run(
                            context,
                            merge_result.tip,
                            secret_values=self._secret_values,
                        ).status
        except EnvironmentMaterializationError as error:
            runtime = context.runtime
            if runtime.status not in {
                TaskRuntimeStatus.DONE,
                TaskRuntimeStatus.BLOCKED,
                TaskRuntimeStatus.FAILED,
            }:
                context.transition(
                    runtime.status,
                    TaskRuntimeStatus.BLOCKED,
                    resume_phase=runtime.resume_phase,
                    state_reason=str(error),
                )
            return self._durable_status(context)

        return published_status or self._durable_status(context)

    @staticmethod
    def _restore_resume_phase(context: ScheduledTaskContext) -> TaskRuntimeStatus:
        """Route reclaimed work to the phase that owns its durable attestation."""
        resume_phase = context.claim.resume_phase
        if resume_phase in {
            TaskRuntimeStatus.ENVIRONMENT.value,
            TaskRuntimeStatus.CODING.value,
        }:
            return TaskRuntimeStatus.CODING

        later_phase = {
            TaskRuntimeStatus.REVIEW.value: TaskRuntimeStatus.REVIEW,
            TaskRuntimeStatus.FIX.value: TaskRuntimeStatus.FIX,
            TaskRuntimeStatus.MERGING.value: TaskRuntimeStatus.MERGING,
        }.get(resume_phase)
        if later_phase is None:
            context.transition(
                TaskRuntimeStatus.CODING,
                TaskRuntimeStatus.BLOCKED,
                resume_phase=resume_phase,
                state_reason=f"unsupported task resume phase: {resume_phase}",
            )
            return TaskRuntimeStatus.BLOCKED

        # Environment rematerialization stages every reclaimed task through
        # CODING. Later phases may legitimately have advanced HEAD beyond the
        # original coding commit, so restore the durable route without replaying
        # coding. Review/fix and merge validate their own exact attestations.
        context.transition(
            TaskRuntimeStatus.CODING,
            TaskRuntimeStatus.REVIEW,
            resume_phase=later_phase.value,
        )
        if later_phase in {TaskRuntimeStatus.FIX, TaskRuntimeStatus.MERGING}:
            context.transition(
                TaskRuntimeStatus.REVIEW,
                later_phase,
                resume_phase=later_phase.value,
            )
        return later_phase

    def _agent_secrets(self) -> dict[str, str]:
        environment: dict[str, str] = {}
        for secret in self.plan.secret_requirements:
            if secret.scope not in {"all", "agent"}:
                continue
            value = self._secret_values.get(secret.name)
            if value is None:
                raise EnvironmentMaterializationError(
                    f"agent-scoped secret value is unavailable: {secret.name}"
                )
            environment[secret.name] = value
        return environment

    @staticmethod
    def _durable_status(context: ScheduledTaskContext) -> TaskRuntimeStatus:
        return context.runtime.status


class HostExecutionService:
    """Sole integration owner for concrete leased host execution."""

    def __init__(
        self,
        store: SqliteStore,
        preflight: HostPreflight,
        runtime: HostTaskRuntime,
        *,
        worktree_manager: HostWorktreeManager,
        scheduler_config: HostSchedulerConfig | None = None,
        clock=None,
        activity: TaskActivitySink | None = None,
        progress: RunProgress | None = None,
    ) -> None:
        self._store = store
        self._preflight = preflight
        self._runtime = runtime
        self._worktrees = worktree_manager
        self._scheduler_config = scheduler_config
        self._clock = clock
        self._activity = activity
        self._progress = progress

    def run(
        self,
        borg_id: UUID,
        generation_id: UUID,
        analyzer_plan: Mapping[str, Any] | AnalyzerPlanLoader,
        *,
        secret_values: Mapping[str, str] | None = None,
        cancel: CancellationToken | None = None,
        validated_preflight: HostPreflightPlan | HostPreflightBlock | None = None,
    ) -> HostExecutionResult:
        """Validate, reconcile, acquire, prepare, and schedule host work."""
        secrets = dict(secret_values or {})
        validated = validated_preflight
        if validated is None:
            validated = self._preflight.validate(
                analyzer_plan,
                available_secret_names=secrets,
            )
        elif self._preflight.validated_result is not validated:
            raise HostExecutionError(
                "prevalidated host plan was not produced by this preflight"
            )
        if isinstance(validated, HostPreflightBlock):
            return HostExecutionResult(validated)
        if validated != self._runtime.plan:
            raise HostExecutionError(
                "concrete task runtime does not match the validated preflight plan"
            )
        activity = _ExecutionActivityBinding(
            declared_secret_mask_values(validated, secrets),
            self._activity,
        )
        activity_handoff: TaskActivityHandoff = activity.emit
        self._sweep_expired_runs()
        config = self._scheduler_config or HostSchedulerConfig()
        acquired_at = self._now()
        acquisition = self._store.acquire_execution_run(
            borg_id,
            generation_id,
            lease_duration=config.lease_duration,
            now=acquired_at,
        )
        behavior = self._runtime
        if acquisition.acquired:
            runtime = self._runtime.with_secret_values(secrets)
            owner_token = acquisition.owner_token
            if owner_token is None:
                raise HostExecutionError("acquired execution run has no owner token")
            borg = self._store.get_borg(borg_id)
            if borg is None:
                raise HostExecutionError(f"Borg {borg_id} not found")
            heartbeats = _SetupLeaseHeartbeats(
                self._store,
                acquisition.run_id,
                owner_token,
                config,
                self._now,
                started_at=acquired_at,
            )
            heartbeats.start()
            try:
                # Acquisition itself atomically expires a prior owner.  That
                # can happen after the pre-acquisition reconciliation above,
                # so repeat the sweep while the new lease is heartbeating and
                # before any new worktree setup or task dispatch begins.
                self._sweep_expired_runs()
                heartbeats.checkpoint()
                self._worktrees.prepare_current_task_worktrees(
                    self._store,
                    run_id=acquisition.run_id,
                    owner_token=owner_token,
                    generation_id=generation_id,
                    project_name=borg.name,
                    now=self._now(),
                )
                heartbeats.checkpoint()
            except BaseException as setup_error:
                try:
                    heartbeats.stop()
                except BaseException as heartbeat_error:
                    setup_error.add_note(
                        f"setup heartbeat shutdown failed: {heartbeat_error}"
                    )
                self._interrupt_failed_setup(
                    acquisition.run_id,
                    owner_token,
                    setup_error,
                )
                raise
            else:
                try:
                    heartbeats.stop()
                except BaseException as setup_error:
                    self._interrupt_failed_setup(
                        acquisition.run_id,
                        owner_token,
                        setup_error,
                    )
                    raise
            behavior = partial(self._run_claimed_task, runtime, borg.name)
        scheduler = HostTaskScheduler(
            self._store,
            behavior,
            config=self._scheduler_config,
            activity_handoff=(
                activity_handoff
                if self._activity is not None or self._progress is not None
                else None
            ),
            progress=self._progress,
            expired_run_sweep=self._sweep_expired_runs,
            **({"clock": self._clock} if self._clock is not None else {}),
        )

        scheduled = scheduler.run_acquired(
            generation_id,
            acquisition,
            cancel=cancel,
        )
        # A cancelled run already swept expired runs inside the scheduler's
        # own interruption fence.
        if scheduled.status is not ExecutionRunStatus.CANCELLED:
            self._sweep_expired_runs()
        return HostExecutionResult(validated, scheduled)

    def _run_claimed_task(
        self,
        runtime: HostTaskRuntime,
        project_name: str,
        context: ScheduledTaskContext,
    ) -> TaskRuntimeStatus:
        """Refresh a never-started claim before entering concrete phases."""
        task_id = context.claim.task_id
        unstarted = not self._store.list_environment_attempts(task_id) and not (
            self._store.list_agent_attempts(task_id)
        )
        if unstarted:
            try:
                self._worktrees.refresh_unstarted_task_worktree(
                    context.runtime,
                    project_name=project_name,
                )
            except (PrimaryCheckoutContaminationError, WorktreeError) as error:
                context.transition(
                    TaskRuntimeStatus.CLAIMED,
                    TaskRuntimeStatus.BLOCKED,
                    state_reason=str(error),
                )
                return TaskRuntimeStatus.BLOCKED
        return runtime(context)

    def _sweep_expired_runs(self) -> None:
        """Interrupt every run whose lease expired, on any Borg.

        Acquisition expires only the Borg being acquired, so this is the one
        place a run left behind by another Borg of this repository is stopped.
        """
        self._store.reconcile_expired_execution_runs(now=self._now())

    def _interrupt_failed_setup(
        self,
        run_id: UUID,
        owner_token: str,
        error: BaseException,
    ) -> None:
        try:
            self._store.interrupt_execution_run(
                run_id,
                owner_token,
                reason="host execution setup failed",
                now=self._now(),
            )
        except BaseException as interrupt_error:
            error.add_note(f"setup run interruption failed: {interrupt_error}")
        try:
            self._sweep_expired_runs()
        except BaseException as sweep_error:
            error.add_note(f"setup expired-run sweep failed: {sweep_error}")

    def _now(self) -> datetime:
        return self._clock() if self._clock is not None else datetime.now(UTC)
