"""Shared state-mutating workflow orchestration for CLI and MCP adapters."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import UUID

from betterborg_cli.agent_runtime.base import CancellationToken
from betterborg_cli.execution_estimate import (
    EXECUTION_ESTIMATE_VERSION,
    estimate_generation,
    phase_billing_from_config,
)
from betterborg_cli.host_execution import HostExecutionResult, HostPreflightBlock
from betterborg_cli.planning import (
    SupervisorLoop,
    TaskPublication,
    TaskPublisher,
    approved_plan_digest,
    render_plan_markdown,
    validate_plan,
)
from betterborg_cli.progress import RunProgress, StageSpec, StageState
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import RepositoryConfig
from betterborg_cli.repository_files import (
    publish_repository_text,
    require_git_trackable,
)
from betterborg_cli.store import (
    Borg,
    BorgState,
    ExecutionDecision,
    PlanApproval,
    PlanningAttempt,
    PlanningAttemptStatus,
    PrdSession,
    Repository,
    SqliteStore,
)

PlanningAgentFactory = Callable[[], Any]
_EXECUTION_PREFLIGHT_STAGE_KEY = "preflight"


class HostInvoker(Protocol):
    """Invoke host execution with the command's shared control context."""

    def __call__(
        self,
        paths: RepoPaths,
        store: SqliteStore,
        config: RepositoryConfig,
        repository_id: UUID,
        borg_id: UUID,
        generation_id: UUID,
        *,
        cancel: CancellationToken | None,
        progress: RunProgress | None,
    ) -> HostExecutionResult: ...


@dataclass(frozen=True, slots=True)
class PlanApprovalWorkflowResult:
    """Durable result of approving and decomposing one exact plan."""

    borg: Borg
    approval: PlanApproval
    plan_path: Path
    publication: TaskPublication | None


@dataclass(frozen=True, slots=True)
class ExecutionDecisionRequest:
    """Transport-owned choice made after presenting a fresh estimate."""

    source: str
    decision: Literal["approved", "bypassed"]


@dataclass(frozen=True, slots=True)
class ExecutionWorkflowResult:
    """Shared estimate gate and optional host-execution result."""

    borg: Borg
    approval: PlanApproval | None
    prd_session: PrdSession | None
    publication: TaskPublication
    estimate: dict[str, Any]
    decision: ExecutionDecision | None
    host_result: HostExecutionResult | None
    decision_event: Literal["required", "recorded", "concurrent", "existing"]


def approve_plan_workflow(
    paths: RepoPaths,
    config: RepositoryConfig,
    name: str,
    *,
    planning_agent: PlanningAgentFactory,
    on_bound: Callable[[], None] | None = None,
    cancel: CancellationToken | None = None,
    progress: RunProgress | None = None,
) -> PlanApprovalWorkflowResult:
    """Bind, decompose, reconcile, and validate one plan approval."""
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        repository = _repository(store, config)
        borg = _borg(store, repository, name)
        approval, plan_path = bind_plan_approval(
            paths,
            store,
            borg,
            cancel=cancel,
        )
        if on_bound is not None:
            on_bound()
        borg = store.get_borg(borg.id)
        if borg is None:
            raise RuntimeError(f"Borg {name!r} disappeared during approval")

        publication = None
        if borg.state in {BorgState.PM_WORKING, BorgState.SUPERVISOR_WORKING}:
            agent = planning_agent()
            supervisor = SupervisorLoop(
                repository,
                borg,
                store,
                agent,
                pm_agent=agent,
                approved_plan=approval.manifest["plan"],
                plan_approval=approval,
                cancel=cancel,
                progress=progress,
            ).run()
            borg = supervisor.borg
            publication = supervisor.publication
            if borg.state is BorgState.READY_TO_EXECUTE and publication is None:
                raise RuntimeError(
                    "Supervisor reached ready state without durable task publication"
                )

        if borg.state is BorgState.READY_TO_EXECUTE:
            if publication is None:
                publication = TaskPublisher(
                    repository,
                    store,
                    cancel=cancel,
                ).reconcile(borg.id)
            if publication is None:
                raise RuntimeError(
                    f"Borg {name!r} is ready to execute but has no current tasks"
                )
        elif borg.state is not BorgState.BLOCKED:
            raise RuntimeError(
                f"decomposition stopped in unexpected state {borg.state.value!r}"
            )

    return PlanApprovalWorkflowResult(borg, approval, plan_path, publication)


def execute_workflow(
    paths: RepoPaths,
    config: RepositoryConfig,
    name: str,
    *,
    decide: Callable[[dict[str, Any]], ExecutionDecisionRequest | None],
    invoke_host: HostInvoker,
    cancel: CancellationToken | None = None,
    progress: RunProgress | None = None,
) -> ExecutionWorkflowResult:
    """Verify, estimate, persist the gate, and invoke the sole host service."""
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        repository = _repository(store, config)
        borg = _borg(store, repository, name)
        if borg.state is not BorgState.READY_TO_EXECUTE:
            raise ValueError(f"Borg {name!r} is not ready to execute")

        publication = TaskPublisher(
            repository,
            store,
            cancel=cancel,
        ).inspect_current_task_files(borg.id)
        generation = publication.generation
        approval = next(
            (
                item
                for item in store.list_plan_approvals(borg.id)
                if item.id == generation.plan_approval_id
            ),
            None,
        )
        prd_session = store.get_prd_session_for_borg(borg.id)
        decision = store.get_current_execution_decision(borg.id)
        estimate = (
            decision.snapshot
            if decision is not None
            else estimate_generation(
                generation.id,
                [item.task for item in publication.files],
                store.list_task_completion_samples(),
                phase_billing_from_config(config),
            )
        )

        event: Literal["required", "recorded", "concurrent", "existing"]
        if decision is None:
            requested = decide(estimate)
            if requested is None:
                return ExecutionWorkflowResult(
                    borg,
                    approval,
                    prd_session,
                    publication,
                    estimate,
                    None,
                    None,
                    "required",
                )
            batch = next(
                (
                    item
                    for item in store.list_task_batches(borg.id)
                    if item.id == generation.batch_id
                ),
                None,
            )
            if approval is None or batch is None:
                raise RuntimeError("current task generation has no approval or batch")
            decision = ExecutionDecision(
                borg_id=borg.id,
                generation_id=generation.id,
                approved_plan_digest=approval.plan_digest,
                task_batch_digest=batch.digest,
                estimate_version=EXECUTION_ESTIMATE_VERSION,
                source=requested.source,
                snapshot=estimate,
                decision=requested.decision,
            )
            try:
                store.append_execution_decision(decision)
            except sqlite3.IntegrityError:
                concurrent = store.get_current_execution_decision(borg.id)
                if concurrent is None or concurrent.generation_id != generation.id:
                    raise
                decision = concurrent
                event = "concurrent"
            else:
                event = "recorded"
        else:
            event = "existing"

        if decision.decision not in {"approved", "bypassed"}:
            raise RuntimeError(
                "current generation has an unsupported execution decision"
            )
        if progress is not None:
            progress.declare(
                StageSpec(_EXECUTION_PREFLIGHT_STAGE_KEY, "Preflight")
            )
            progress.start(_EXECUTION_PREFLIGHT_STAGE_KEY)
        try:
            host_result = invoke_host(
                paths,
                store,
                config,
                repository.id,
                borg.id,
                generation.id,
                cancel=cancel,
                progress=progress,
            )
        except BaseException as error:
            if _preflight_is_running(progress):
                assert progress is not None
                detail = str(error).strip() or type(error).__name__
                if isinstance(error, KeyboardInterrupt) or (
                    cancel is not None and cancel.is_set()
                ):
                    progress.stop(_EXECUTION_PREFLIGHT_STAGE_KEY, detail)
                else:
                    progress.fail(_EXECUTION_PREFLIGHT_STAGE_KEY, detail)
            raise
        if _preflight_is_running(progress):
            assert progress is not None
            if isinstance(host_result.preflight, HostPreflightBlock):
                progress.fail(
                    _EXECUTION_PREFLIGHT_STAGE_KEY,
                    host_result.preflight.reason,
                )
            else:
                progress.complete(_EXECUTION_PREFLIGHT_STAGE_KEY, "ready")

    return ExecutionWorkflowResult(
        borg,
        approval,
        prd_session,
        publication,
        estimate,
        decision,
        host_result,
        event,
    )


def _preflight_is_running(progress: RunProgress | None) -> bool:
    return (
        progress is not None
        and progress.stages[_EXECUTION_PREFLIGHT_STAGE_KEY].state
        is StageState.RUNNING
    )


def bind_plan_approval(
    paths: RepoPaths,
    store: SqliteStore,
    borg: Borg,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[PlanApproval, Path]:
    """Bind or recover one approval for the latest exact Architect plan."""
    plan_attempt = validated_current_plan_attempt(paths, store, borg)
    current_plan = plan_attempt.result
    digest = approved_plan_digest(current_plan)
    body = render_plan_markdown(current_plan)
    body_digest = "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()
    relative_path = Path(".betterborg/plans") / f"{borg.name}.md"
    plan_path = paths.root / relative_path

    approvals = store.list_plan_approvals(borg.id)
    if approvals:
        approval = approvals[-1]
        if borg.state is BorgState.PLAN_APPROVAL_PENDING:
            raise ValueError(f"Borg {borg.name!r} already has a plan approval")
        manifest_plan = approval.manifest.get("plan")
        if (
            approval.attempt_id != plan_attempt.id
            or approval.plan_digest != digest
            or manifest_plan != current_plan
            or approval.manifest.get("plan.md") != body_digest
            or approval.manifest.get("plan_path") != relative_path.as_posix()
        ):
            raise ValueError(
                f"Borg {borg.name!r} approval does not match its current plan"
            )
        try:
            existing = plan_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            publish_repository_text(plan_path, body, root=paths.root, overwrite=True)
        else:
            if existing != body:
                raise ValueError(f"approved plan Markdown drifted: {relative_path}")
        require_git_trackable(relative_path, root=paths.root, cancel=cancel)
        return approval, plan_path

    if borg.state is not BorgState.PLAN_APPROVAL_PENDING:
        raise ValueError(
            f"Borg {borg.name!r} cannot approve a plan from state "
            f"{borg.state.value!r}; a plan must be awaiting approval"
        )
    publish_repository_text(plan_path, body, root=paths.root, overwrite=True)
    require_git_trackable(relative_path, root=paths.root, cancel=cancel)
    approval = PlanApproval(
        borg_id=borg.id,
        attempt_id=plan_attempt.id,
        plan_digest=digest,
        manifest={
            "plan": current_plan,
            "plan.md": body_digest,
            "plan_path": relative_path.as_posix(),
        },
        approved_by="operator",
    )
    with store.transaction():
        store.append_plan_approval(approval)
        store.compare_and_set_borg_state(
            borg.id,
            expected_state=borg.state,
            expected_version=borg.state_version,
            new_state=BorgState.PM_WORKING,
        )
    return approval, plan_path


def validated_current_plan_attempt(
    paths: RepoPaths,
    store: SqliteStore,
    borg: Borg,
) -> PlanningAttempt:
    """Return the exact latest complete Architect plan exposed to operators."""
    attempt = next(
        (
            item
            for item in reversed(store.list_planning_attempts(borg.id))
            if item.phase == "architect_plan"
            and item.status is PlanningAttemptStatus.COMPLETED
            and item.result is not None
        ),
        None,
    )
    if attempt is None:
        raise ValueError(
            f"Borg {borg.name!r} does not have a stored plan; "
            f"run 'betterborg plan start {borg.name}' first"
        )
    validate_plan(attempt.result, paths.root, check_repository_state=False)
    return attempt


def _repository(store: SqliteStore, config: RepositoryConfig) -> Repository:
    repository = store.get_repository(config.repository_id)
    if repository is None:
        raise ValueError("repository is not initialized; run 'betterborg init' first")
    return repository


def _borg(store: SqliteStore, repository: Repository, name: str) -> Borg:
    borg = store.get_borg_by_name(repository.id, name)
    if borg is None:
        raise ValueError(f"Borg {name!r} does not exist")
    return borg
