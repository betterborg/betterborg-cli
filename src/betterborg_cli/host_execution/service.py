"""Concrete trust-gated host execution assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from betterborg_cli.agent_runtime import CancellationToken
from betterborg_cli.host_execution.coding import HostCodingPhase
from betterborg_cli.host_execution.compose import (
    ComposeCleanupResult,
    ComposeStackError,
    HostComposeManager,
)
from betterborg_cli.host_execution.environment import (
    EnvironmentMaterializationError,
    HostEnvironmentManager,
)
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
)
from betterborg_cli.host_execution.worktrees import HostWorktreeManager
from betterborg_cli.store import ExecutionRunStatus, SqliteStore, TaskRuntimeStatus


class HostExecutionError(RuntimeError):
    """Raised when run-scoped host setup cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class HostExecutionResult:
    """Outcome of validation, reconciliation, and optional scheduling."""

    preflight: HostPreflightPlan | HostPreflightBlock
    scheduler: HostSchedulerResult | None = None
    cleanup: tuple[ComposeCleanupResult, ...] = ()

    @property
    def operation_id(self) -> UUID | None:
        return self.scheduler.operation_id if self.scheduler is not None else None

    @property
    def active_operation_id(self) -> UUID | None:
        return (
            self.scheduler.active_operation_id
            if self.scheduler is not None
            else None
        )

    @property
    def acquired(self) -> bool:
        return self.scheduler.acquired if self.scheduler is not None else False

    @property
    def status(self) -> ExecutionRunStatus | None:
        return self.scheduler.status if self.scheduler is not None else None


class HostTaskRuntime:
    """Run every concrete phase for one scheduler-owned task claim."""

    def __init__(
        self,
        plan: HostPreflightPlan,
        *,
        environment_manager: HostEnvironmentManager,
        compose_manager: HostComposeManager,
        coding: HostCodingPhase,
        review_fix: HostReviewFixPhase,
        merge: HostMergePhase,
        sanity: HostSanityPhase,
        secret_values: Mapping[str, str] | None = None,
    ) -> None:
        self.plan = plan
        self._environment = environment_manager
        self._compose = compose_manager
        self._coding = coding
        self._review_fix = review_fix
        self._merge = merge
        self._sanity = sanity
        self._secret_values = dict(secret_values or {})

    def with_secret_values(
        self, secret_values: Mapping[str, str]
    ) -> HostTaskRuntime:
        """Bind one run's validated secret values without shared mutation."""
        return HostTaskRuntime(
            self.plan,
            environment_manager=self._environment,
            compose_manager=self._compose,
            coding=self._coding,
            review_fix=self._review_fix,
            merge=self._merge,
            sanity=self._sanity,
            secret_values=secret_values,
        )

    def __call__(self, context: ScheduledTaskContext) -> TaskRuntimeStatus:
        """Materialize, execute, publish, and clean one durable task."""
        try:
            materialization = self._environment.materialize_claimed_task(
                context.store,
                self.plan,
                context.claim,
                context.owner_token,
                secret_values=self._secret_values,
            )
        except EnvironmentMaterializationError:
            return self._durable_status(context)

        stack = None
        try:
            stack = self._compose.start_claimed_stack(
                context.store,
                self.plan,
                context.claim,
                context.owner_token,
            )
            service_environment = dict(materialization.environment)
            if stack is not None:
                service_environment.update(stack.environment)

            status = self._coding.run(
                context,
                environment={
                    **service_environment,
                    **self._agent_secrets("coding"),
                },
            )
            if status is TaskRuntimeStatus.REVIEW:
                status = self._review_fix.run(
                    context,
                    environment=service_environment,
                    review_environment=self._agent_secrets("review"),
                    fix_environment=self._agent_secrets("fix"),
                )
            merge_result = None
            if status is TaskRuntimeStatus.MERGING:
                merge_result = self._merge.run(
                    context,
                    environment={
                        **service_environment,
                        **self._agent_secrets("merge"),
                    },
                )
        except ComposeStackError:
            return self._durable_status(context)
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
        finally:
            if stack is not None:
                self._compose.stop_claimed_stack(
                    context.store,
                    stack,
                    context.claim,
                    context.owner_token,
                )

        if merge_result is None or merge_result.tip is None:
            return self._durable_status(context)
        return self._sanity.run(
            context,
            merge_result.tip,
            secret_values=self._secret_values,
        ).status

    def _agent_secrets(self, phase: str) -> dict[str, str]:
        environment: dict[str, str] = {}
        for secret in self.plan.secret_requirements:
            if secret.scope not in {"all", "agent"}:
                continue
            if phase not in secret.used_by and "agent" not in secret.used_by:
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
        compose_manager: HostComposeManager,
        scheduler_config: HostSchedulerConfig | None = None,
        clock=None,
    ) -> None:
        self._store = store
        self._preflight = preflight
        self._runtime = runtime
        self._worktrees = worktree_manager
        self._compose = compose_manager
        self._scheduler_config = scheduler_config
        self._clock = clock

    def run(
        self,
        borg_id: UUID,
        generation_id: UUID,
        analyzer_plan: Mapping[str, Any] | AnalyzerPlanLoader,
        *,
        secret_values: Mapping[str, str] | None = None,
        external_urls: Mapping[str, str] | None = None,
        cancel: CancellationToken | None = None,
    ) -> HostExecutionResult:
        """Validate, reconcile, acquire, prepare, and schedule host work."""
        secrets = dict(secret_values or {})
        validated = self._preflight.validate(
            analyzer_plan,
            available_secret_names=secrets,
            external_urls=external_urls,
        )
        if isinstance(validated, HostPreflightBlock):
            return HostExecutionResult(validated)
        if validated != self._runtime.plan:
            raise HostExecutionError(
                "concrete task runtime does not match the validated preflight plan"
            )
        runtime = self._runtime.with_secret_values(secrets)

        cleanup = list(self._cleanup_stale())
        acquisition = self._store.acquire_execution_run(
            borg_id,
            generation_id,
            lease_duration=(
                self._scheduler_config or HostSchedulerConfig()
            ).lease_duration,
            now=self._now(),
        )
        scheduler = HostTaskScheduler(
            self._store,
            runtime,
            config=self._scheduler_config,
            **({"clock": self._clock} if self._clock is not None else {}),
        )
        if acquisition.acquired:
            owner_token = acquisition.owner_token
            if owner_token is None:
                raise HostExecutionError("acquired execution run has no owner token")
            borg = self._store.get_borg(borg_id)
            if borg is None:
                raise HostExecutionError(f"Borg {borg_id} not found")
            try:
                self._worktrees.prepare_current_task_worktrees(
                    self._store,
                    run_id=acquisition.run_id,
                    owner_token=owner_token,
                    generation_id=generation_id,
                    project_name=borg.name,
                    now=self._now(),
                )
            except BaseException:
                self._store.interrupt_execution_run(
                    acquisition.run_id,
                    owner_token,
                    reason="host execution setup failed",
                    now=self._now(),
                )
                cleanup.extend(self._cleanup_stale())
                raise

        scheduled = scheduler.run_acquired(
            generation_id,
            acquisition,
            cancel=cancel,
        )
        cleanup.extend(self._cleanup_stale())
        return HostExecutionResult(validated, scheduled, tuple(cleanup))

    def _cleanup_stale(self) -> tuple[ComposeCleanupResult, ...]:
        resources = self._store.reconcile_expired_execution_runs(now=self._now())
        if not resources:
            return ()
        return self._compose.cleanup_stale_projects(self._store, resources)

    def _now(self):  # noqa: ANN202 - store accepts None as its real-time clock.
        return self._clock() if self._clock is not None else None
