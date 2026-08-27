"""Typed MCP stdio tools backed by BetterBorg's existing workflow services."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Literal

import click
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from betterborg_cli.agent_runtime.api_tools import ApiAgentRole
from betterborg_cli.agent_runtime.selection import select_agent
from betterborg_cli.execution_estimate import (
    EXECUTION_ESTIMATE_VERSION,
    estimate_generation,
    phase_billing_from_config,
)
from betterborg_cli.host_execution import HostPreflightBlock
from betterborg_cli.onboarding import CreateService
from betterborg_cli.planning import SupervisorLoop, TaskPublisher
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import load_repository_config
from betterborg_cli.repository_service import RepositoryService
from betterborg_cli.store import (
    BorgState,
    ExecutionDecision,
    ExecutionRunStatus,
    SqliteStore,
    TaskRuntimeCost,
    TaskRuntimeRow,
)
from betterborg_cli.workspace_trust import require_workspace_trust


class ProtocolModel(BaseModel):
    """Immutable base for values serialized as MCP structured content."""

    model_config = ConfigDict(frozen=True)


class Artifact(ProtocolModel):
    """One repository-relative durable workflow output."""

    kind: str
    path: str


class NextAction(ProtocolModel):
    """One typed follow-on MCP invocation."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class WorkflowResult(ProtocolModel):
    """Common typed result retained across every MCP workflow tool."""

    status: str
    artifacts: tuple[Artifact, ...] = ()
    next_actions: tuple[NextAction, ...] = ()
    data: dict[str, Any] = Field(default_factory=dict)


class RuntimeCost(ProtocolModel):
    """MCP representation of the shared billing-aware task cost."""

    api_spend_usd: float | None
    api_spend_unknown: bool
    subscription_included: bool


class RuntimeTask(ProtocolModel):
    """Exact phase-07 runtime projection for one current task."""

    generation_id: str
    task_id: str
    task_ref: str
    stage: str
    stem: str
    position: int
    title: str
    complexity: str
    status: str
    state_reason: str | None
    review_round: int
    attempt_count: int
    duration_seconds: float | None
    cost: RuntimeCost


class TaskListResult(ProtocolModel):
    """Typed current-generation task listing."""

    status: str
    borg: str
    generation_id: str
    generation_digest: str
    approved_plan_digest: str | None
    tasks: tuple[RuntimeTask, ...]
    artifacts: tuple[Artifact, ...]
    next_actions: tuple[NextAction, ...]


server = FastMCP(
    "BetterBorg",
    instructions=(
        "Operate the BetterBorg workflow in the Git repository where the server "
        "was started. Results preserve durable status, artifacts, and next actions."
    ),
)


def _paths(*, trusted: bool) -> RepoPaths:
    paths = RepoPaths.discover()
    if trusted:
        require_workspace_trust(paths, explicit=False, interactive=False)
    return paths


def _relative(paths: RepoPaths, path: Path) -> str:
    return path.resolve().relative_to(paths.root).as_posix()


def _analysis_artifacts(paths: RepoPaths, result: Any) -> tuple[Artifact, ...]:
    artifacts = [Artifact(kind="score", path=_relative(paths, result.score_path))]
    artifacts.extend(
        Artifact(
            kind=f"{prompt.role}_prompt",
            path=_relative(paths, prompt.path),
        )
        for prompt in result.prompts
    )
    artifacts.extend(
        Artifact(kind="improvement_prd", path=_relative(paths, prd.path))
        for prd in result.improvement_prds
    )
    return tuple(artifacts)


def _theme_actions(paths: RepoPaths, result: Any) -> tuple[NextAction, ...]:
    return tuple(
        NextAction(
            tool="create",
            arguments={
                "name": prd.suggested_borg_name,
                "source": _relative(paths, prd.path),
            },
        )
        for prd in result.improvement_prds
    )


@server.tool(name="init")
def initialize() -> WorkflowResult:
    """Initialize and analyze the server's current Git repository."""
    paths = _paths(trusted=True)
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        result = RepositoryService(
            paths,
            store,
            lambda config: select_agent(
                config, ApiAgentRole.ANALYSIS, paths, interactive=False
            ),
        ).initialize()
    return WorkflowResult(
        status="initialized" if result.initialized else "already_initialized",
        artifacts=_analysis_artifacts(paths, result),
        next_actions=_theme_actions(paths, result),
        data={
            "repository_id": str(result.repository.id),
            "analysis_id": str(result.analysis.id),
            "score": result.analysis.overall_score,
        },
    )


@server.tool(name="analyze")
def analyze() -> WorkflowResult:
    """Re-analyze the initialized current Git repository."""
    paths = _paths(trusted=True)
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        result = RepositoryService(
            paths,
            store,
            lambda config: select_agent(
                config, ApiAgentRole.ANALYSIS, paths, interactive=False
            ),
        ).analyze()
    return WorkflowResult(
        status="completed",
        artifacts=_analysis_artifacts(paths, result),
        next_actions=_theme_actions(paths, result),
        data={
            "repository_id": str(result.repository.id),
            "analysis_id": str(result.analysis.id),
            "score": result.analysis.overall_score,
            "previous_score": result.previous_analysis.overall_score,
            "delta": result.analysis.score_delta,
        },
    )


@server.tool(name="create")
def create(
    name: str,
    source: str | None = None,
    confirmed: bool = False,
) -> WorkflowResult:
    """Create a Borg PRD, optionally confirming the returned draft."""
    paths = _paths(trusted=True)
    config = load_repository_config(paths)
    source_path = Path(source) if source is not None else None
    if source_path is not None and not source_path.is_absolute():
        source_path = paths.root / source_path
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        repository = store.get_repository(config.repository_id)
        if repository is None:
            raise ValueError("repository is not initialized; run 'borg init' first")
        result = CreateService(
            repository,
            store,
            select_agent(
                config, ApiAgentRole.PLANNING, paths, interactive=False
            ),
            interactive=False,
        ).create(name, source_path, confirmed=confirmed)

    if result.confirmed:
        status = "confirmed"
        artifacts = (
            Artifact(kind="prd", path=_relative(paths, result.prd_path)),
        )
        actions = (
            NextAction(tool="plan", arguments={"name": name, "action": "start"}),
        )
    elif result.questions:
        status = "needs_input"
        artifacts = ()
        actions = ()
    else:
        status = "draft"
        artifacts = ()
        actions = ()
    return WorkflowResult(
        status=status,
        artifacts=artifacts,
        next_actions=actions,
        data={
            "borg": name,
            "borg_id": str(result.borg.id),
            "questions": list(result.questions),
            "draft_markdown": result.body_md,
        },
    )


def _planning_io(answers: list[str] | None) -> InteractiveIO:
    supplied = iter(answers or ())

    def prompt(_message: str) -> str | None:
        return next(supplied, None)

    return InteractiveIO(
        prompt=prompt,
        confirm=lambda _message, _default: False,
        write=lambda _message: None,
    )


def _planning_state(paths: RepoPaths, name: str) -> tuple[Any, list[dict[str, Any]]]:
    config = load_repository_config(paths)
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        repository = store.get_repository(config.repository_id)
        if repository is None:
            raise ValueError("repository is not initialized; run 'borg init' first")
        borg = store.get_borg_by_name(repository.id, name)
        if borg is None:
            raise ValueError(f"Borg {name!r} does not exist")
        pending = next(
            (
                item
                for item in reversed(store.list_planning_questions(borg.id))
                if item.answers is None
            ),
            None,
        )
        questions = pending.questions if pending is not None else []
    return borg, questions


def _plan_actions(name: str, state: BorgState) -> tuple[NextAction, ...]:
    if state is BorgState.ARCHITECT_AWAITING_ANSWERS:
        return (
            NextAction(tool="plan", arguments={"name": name, "action": "start"}),
        )
    if state is BorgState.PLAN_APPROVAL_PENDING:
        return (
            NextAction(tool="plan", arguments={"name": name, "action": "show"}),
            NextAction(tool="plan", arguments={"name": name, "action": "approve"}),
        )
    if state is BorgState.READY_TO_EXECUTE:
        return (
            NextAction(tool="task_list", arguments={"name": name}),
            NextAction(tool="execute", arguments={"name": name}),
        )
    return ()


def _approve_plan(paths: RepoPaths, name: str) -> tuple[Any, Any, Path, Any]:
    from betterborg_cli import cli as cli_module

    config = load_repository_config(paths)
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        repository = store.get_repository(config.repository_id)
        if repository is None:
            raise ValueError("repository is not initialized; run 'borg init' first")
        borg = store.get_borg_by_name(repository.id, name)
        if borg is None:
            raise ValueError(f"Borg {name!r} does not exist")
        approval, plan_path = cli_module._bind_plan_approval(paths, store, borg)
        borg = store.get_borg(borg.id)
        if borg is None:
            raise RuntimeError(f"Borg {name!r} disappeared during approval")
        if borg.state in {BorgState.PM_WORKING, BorgState.SUPERVISOR_WORKING}:
            agent = select_agent(
                config, ApiAgentRole.PLANNING, paths, interactive=False
            )
            borg = SupervisorLoop(
                repository,
                borg,
                store,
                agent,
                pm_agent=agent,
                approved_plan=approval.manifest["plan"],
                plan_approval=approval,
            ).run().borg
        publication = None
        if borg.state is BorgState.READY_TO_EXECUTE:
            publication = TaskPublisher(repository, store).reconcile(borg.id)
            if publication is None:
                raise RuntimeError(
                    f"Borg {name!r} is ready to execute but has no current tasks"
                )
        elif borg.state is not BorgState.BLOCKED:
            raise RuntimeError(
                f"decomposition stopped in unexpected state {borg.state.value!r}"
            )
    return borg, approval, plan_path, publication


@server.tool(name="plan")
def plan(
    name: str,
    action: Literal["start", "show", "change", "approve"] = "start",
    note: str | None = None,
    answers: list[str] | None = None,
) -> WorkflowResult:
    """Start, inspect, change, or approve a plan; approval decomposes tasks."""
    from betterborg_cli import cli as cli_module

    paths = _paths(trusted=action != "show")
    if action == "approve":
        borg, approval, plan_path, publication = _approve_plan(paths, name)
        artifacts = [
            Artifact(kind="approved_plan", path=_relative(paths, plan_path))
        ]
        if publication is not None:
            artifacts.extend(
                Artifact(kind="task", path=_relative(paths, item.path))
                for item in publication.files
            )
        return WorkflowResult(
            status=borg.state.value,
            artifacts=tuple(artifacts),
            next_actions=_plan_actions(name, borg.state),
            data={"borg": name, "plan_digest": approval.plan_digest},
        )

    if action == "show":
        config = load_repository_config(paths)
        with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
            repository = store.get_repository(config.repository_id)
            if repository is None:
                raise ValueError("repository is not initialized; run 'borg init' first")
            borg = store.get_borg_by_name(repository.id, name)
            if borg is None:
                raise ValueError(f"Borg {name!r} does not exist")
            attempt = cli_module._validated_current_plan_attempt(paths, store, borg)
        return WorkflowResult(
            status=borg.state.value,
            next_actions=_plan_actions(name, borg.state),
            data={"borg": name, "plan": attempt.result},
        )

    if action == "change" and (note is None or not note.strip()):
        raise ValueError("plan change note must not be empty")
    try:
        borg = cli_module._continue_planning(
            paths.root,
            name,
            change_note=note.strip() if action == "change" and note else None,
            io=_planning_io(answers),
        )
        questions: list[dict[str, Any]] = []
    except click.ClickException:
        borg, questions = _planning_state(paths, name)
        if borg.state is not BorgState.ARCHITECT_AWAITING_ANSWERS:
            raise
    return WorkflowResult(
        status=borg.state.value,
        next_actions=_plan_actions(name, borg.state),
        data={"borg": name, "questions": questions},
    )


def _runtime_cost(cost: TaskRuntimeCost) -> RuntimeCost:
    return RuntimeCost(
        api_spend_usd=cost.api_spend_usd,
        api_spend_unknown=cost.api_spend_unknown,
        subscription_included=cost.subscription_included,
    )


def _runtime_task(row: TaskRuntimeRow) -> RuntimeTask:
    return RuntimeTask(
        generation_id=str(row.generation_id),
        task_id=str(row.task_id),
        task_ref=row.task_ref,
        stage=row.stage,
        stem=row.stem,
        position=row.position,
        title=row.title,
        complexity=row.complexity.value,
        status=row.status.value,
        state_reason=row.state_reason,
        review_round=row.review_round,
        attempt_count=row.attempt_count,
        duration_seconds=row.duration_seconds,
        cost=_runtime_cost(row.cost),
    )


@server.tool(name="task_list")
def task_list(name: str) -> TaskListResult:
    """List exact runtime data for the current, verified task generation."""
    from betterborg_cli import cli as cli_module

    paths, publication = cli_module._current_task_publication(name)
    rows = cli_module._current_task_runtime(paths, name, publication)
    return TaskListResult(
        status="completed",
        borg=name,
        generation_id=str(publication.generation.id),
        generation_digest=publication.generation.digest,
        approved_plan_digest=publication.generation.manifest.get(
            "approved_plan_digest"
        ),
        tasks=tuple(_runtime_task(row) for row in rows),
        artifacts=tuple(
            Artifact(kind="task", path=_relative(paths, item.path))
            for item in publication.files
        ),
        next_actions=(NextAction(tool="execute", arguments={"name": name}),),
    )


def _execute(name: str, auto_execute: bool) -> WorkflowResult:
    from betterborg_cli import cli as cli_module

    paths = _paths(trusted=True)
    config = load_repository_config(paths)
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        repository = store.get_repository(config.repository_id)
        if repository is None:
            raise ValueError("repository is not initialized; run 'borg init' first")
        borg = store.get_borg_by_name(repository.id, name)
        if borg is None:
            raise ValueError(f"Borg {name!r} does not exist")
        if borg.state is not BorgState.READY_TO_EXECUTE:
            raise ValueError(f"Borg {name!r} is not ready to execute")
        publication = TaskPublisher(repository, store).inspect_current_task_files(
            borg.id
        )
        generation = publication.generation
        decision = store.get_current_execution_decision(borg.id)
        estimate = decision.snapshot if decision is not None else estimate_generation(
            generation.id,
            [item.task for item in publication.files],
            store.list_task_completion_samples(),
            phase_billing_from_config(config),
        )
        if decision is None and not auto_execute:
            return WorkflowResult(
                status="estimate_approval_required",
                artifacts=tuple(
                    Artifact(kind="task", path=_relative(paths, item.path))
                    for item in publication.files
                ),
                next_actions=(
                    NextAction(
                        tool="execute",
                        arguments={"name": name, "auto_execute": True},
                    ),
                ),
                data={
                    "borg": name,
                    "generation_id": str(generation.id),
                    "estimate": estimate,
                },
            )
        if decision is None:
            approval = next(
                (
                    item
                    for item in store.list_plan_approvals(borg.id)
                    if item.id == generation.plan_approval_id
                ),
                None,
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
                source="mcp_auto_execute",
                snapshot=estimate,
                decision="bypassed",
            )
            try:
                store.append_execution_decision(decision)
            except sqlite3.IntegrityError:
                decision = store.get_current_execution_decision(borg.id)
                if decision is None or decision.generation_id != generation.id:
                    raise
        if decision.decision not in {"approved", "bypassed"}:
            raise RuntimeError(
                "current generation has an unsupported execution decision"
            )
        result = cli_module._invoke_host_execution(
            paths,
            store,
            config,
            repository.id,
            borg.id,
            generation.id,
        )

    if isinstance(result.preflight, HostPreflightBlock):
        status = "blocked"
        operation_id = None
        active_operation_id = None
        reason = result.preflight.reason
    elif result.active_operation_id is not None:
        status = "active"
        operation_id = None
        active_operation_id = str(result.active_operation_id)
        reason = None
    else:
        if result.operation_id is None or result.status is None:
            raise RuntimeError("host execution returned no operation")
        status = result.status.value
        operation_id = str(result.operation_id)
        active_operation_id = None
        reason = None
    actions = ()
    if result.status not in {ExecutionRunStatus.COMPLETED}:
        actions = (NextAction(tool="execute", arguments={"name": name}),)
    return WorkflowResult(
        status=status,
        artifacts=tuple(
            Artifact(kind="task", path=_relative(paths, item.path))
            for item in publication.files
        ),
        next_actions=actions,
        data={
            "borg": name,
            "generation_id": str(generation.id),
            "operation_id": operation_id,
            "active_operation_id": active_operation_id,
            "reason": reason,
            "estimate": estimate,
        },
    )


@server.tool(name="execute")
def execute(name: str, auto_execute: bool = False) -> WorkflowResult:
    """Estimate or run the assembled host execution service."""
    return _execute(name, auto_execute)


def run_stdio_server() -> None:
    """Run only MCP protocol framing on stdout using the stdio transport."""
    server.run(transport="stdio")
