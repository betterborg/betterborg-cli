"""Typed MCP stdio tools backed by BetterBorg's existing workflow services."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import anyio
import click
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, ConfigDict, Field

from betterborg_cli.agent_runtime.api_tools import ApiAgentRole
from betterborg_cli.agent_runtime.selection import select_agent
from betterborg_cli.host_execution import HostPreflightBlock
from betterborg_cli.onboarding import CreateService, OnboardingDispatcher
from betterborg_cli.planning import ArchitectCancelled
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import load_repository_config
from betterborg_cli.repository_service import RepositoryService
from betterborg_cli.store import (
    BorgState,
    ExecutionRunStatus,
    SqliteStore,
    TaskComplexity,
    TaskRuntimeCost,
    TaskRuntimeRow,
    TaskRuntimeStatus,
)
from betterborg_cli.workflow_service import (
    ExecutionDecisionRequest,
    approve_plan_workflow,
    execute_workflow,
    validated_current_plan_attempt,
)
from betterborg_cli.workspace_trust import require_workspace_trust


class ProtocolModel(BaseModel):
    """Immutable base for values serialized as MCP structured content."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ScoreArtifact(ProtocolModel):
    kind: Literal["score"]
    path: str


class PromptArtifact(ProtocolModel):
    kind: Literal["coding_prompt", "review_prompt", "merge_prompt"]
    path: str


class ImprovementPrdArtifact(ProtocolModel):
    kind: Literal["improvement_prd"]
    path: str


class PrdArtifact(ProtocolModel):
    kind: Literal["prd"]
    path: str


class ApprovedPlanArtifact(ProtocolModel):
    kind: Literal["approved_plan"]
    path: str


class TaskArtifact(ProtocolModel):
    kind: Literal["task"]
    path: str


AnalysisArtifact = ScoreArtifact | PromptArtifact | ImprovementPrdArtifact
PlanArtifact = ApprovedPlanArtifact | TaskArtifact


class CreateActionArguments(ProtocolModel):
    name: str
    source: str


class PlanActionArguments(ProtocolModel):
    name: str
    action: Literal["start", "show", "approve"]


class TaskListActionArguments(ProtocolModel):
    name: str


class ExecuteActionArguments(ProtocolModel):
    name: str


class CreateNextAction(ProtocolModel):
    tool: Literal["create"]
    arguments: CreateActionArguments


class PlanNextAction(ProtocolModel):
    tool: Literal["plan"]
    arguments: PlanActionArguments


class TaskListNextAction(ProtocolModel):
    tool: Literal["task_list"]
    arguments: TaskListActionArguments


class ExecuteNextAction(ProtocolModel):
    tool: Literal["execute"]
    arguments: ExecuteActionArguments


class InitializeData(ProtocolModel):
    repository_id: UUID
    analysis_id: UUID
    score: float = Field(ge=0, le=5)


class AnalyzeData(ProtocolModel):
    repository_id: UUID
    analysis_id: UUID
    score: float = Field(ge=0, le=5)
    previous_score: float = Field(ge=0, le=5)
    delta: float


class CreateData(ProtocolModel):
    borg: str
    borg_id: UUID
    questions: tuple[str, ...]
    draft_markdown: str | None


class PlanningQuestionData(ProtocolModel):
    id: str
    question: str
    why: str | None = None
    hint: str | None = None


class PlanRepository(ProtocolModel):
    id: str
    role: Literal["primary", "secondary"] | None = None


class PlanFile(ProtocolModel):
    path: str
    role: Literal["new", "modified", "deleted", "read"]
    repo: str | None = None
    description: str | None = None


class PlanContract(ProtocolModel):
    kind: Literal[
        "db_migration",
        "api_endpoint",
        "type",
        "function_signature",
        "config",
        "event",
        "other",
    ]
    spec: str
    repo: str | None = None


class PlanPhase(ProtocolModel):
    name: str
    title: str
    goal: str
    technical_approach: str
    repositories: tuple[str, ...] = ()
    files_touched: tuple[PlanFile, ...]
    contracts: tuple[PlanContract, ...] = ()
    test_strategy: str
    acceptance_criteria: tuple[str, ...]
    dependencies_on: tuple[str, ...] = ()
    deliverables: tuple[str, ...]
    constraints: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()


class PlanCodePointer(ProtocolModel):
    path: str
    why: str


class PlanDocument(ProtocolModel):
    title: str
    repositories: tuple[PlanRepository, ...] = ()
    summary: str
    overall_approach: str
    phases: tuple[PlanPhase, ...]
    code_pointers: tuple[PlanCodePointer, ...] = ()
    risks: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()


class PlanProgressData(ProtocolModel):
    borg: str
    questions: tuple[PlanningQuestionData, ...]


class PlanShowData(ProtocolModel):
    borg: str
    plan: PlanDocument


class PlanApprovalData(ProtocolModel):
    borg: str
    plan_digest: str


PlanData = PlanProgressData | PlanShowData | PlanApprovalData


class EstimateRangeData(ProtocolModel):
    p50: float | None
    p80: float | None
    unit: Literal["seconds", "USD"]


class TaskMixData(ProtocolModel):
    small: int = Field(ge=0)
    medium: int = Field(ge=0)
    large: int = Field(ge=0)
    unsized: int = Field(ge=0)


class PhaseSampleData(ProtocolModel):
    coding: int = Field(ge=0)
    review: int = Field(ge=0)
    merge: int = Field(ge=0)


class PhaseSourceData(ProtocolModel):
    coding: Literal["dummy_prior", "dummy_prior+local", "local", "unknown"]
    review: Literal["dummy_prior", "dummy_prior+local", "local", "unknown"]
    merge: Literal["dummy_prior", "dummy_prior+local", "local", "unknown"]


class ComplexityEstimateData(ProtocolModel):
    complexity: TaskComplexity
    task_count: int = Field(ge=0)
    sample_size: int = Field(ge=0)
    token_sample_size: PhaseSampleData
    token_source: PhaseSourceData
    source: Literal["dummy_prior", "dummy_prior+local", "local", "unknown"]
    time: EstimateRangeData


class TimeEstimateData(EstimateRangeData):
    unit: Literal["seconds"]
    kind: Literal["total_agent_work"]
    calendar_time: Literal[False]
    unknown_tasks: int = Field(ge=0)


class ApiModelData(ProtocolModel):
    phase: Literal["coding", "review", "merge"]
    provider: str
    model: str


class ApiBillingData(ProtocolModel):
    estimate: EstimateRangeData | None
    unknown: bool
    models: tuple[ApiModelData, ...]
    pricing_catalog_version: str
    pricing_sources: dict[str, str]


class SubscriptionBillingData(ProtocolModel):
    included: bool
    phases: tuple[Literal["coding", "review", "merge"], ...]
    usd: None


class BillingEstimateData(ProtocolModel):
    api: ApiBillingData
    subscription: SubscriptionBillingData
    unknown_phases: tuple[Literal["coding", "review", "merge"], ...]


class EstimateProvenanceData(ProtocolModel):
    prior_version: str
    prior_label: str
    local_blend_sample_count: int = Field(ge=1)


class ExecutionEstimateData(ProtocolModel):
    generation_id: UUID
    task_mix: TaskMixData
    estimable_tasks: int = Field(ge=0)
    sample_size: int = Field(ge=0)
    per_complexity: tuple[ComplexityEstimateData, ...]
    time: TimeEstimateData
    billing: BillingEstimateData
    provenance: EstimateProvenanceData


class ExecuteData(ProtocolModel):
    borg: str
    generation_id: UUID
    operation_id: UUID | None = None
    active_operation_id: UUID | None = None
    reason: str | None = None
    estimate: ExecutionEstimateData


class SetupRequiredData(ProtocolModel):
    cli_command: str


class InitializeResult(ProtocolModel):
    status: Literal["initialized", "already_initialized", "setup_required"]
    artifacts: tuple[AnalysisArtifact | PrdArtifact, ...] = ()
    next_actions: tuple[CreateNextAction | PlanNextAction, ...] = ()
    data: InitializeData | SetupRequiredData


class AnalyzeResult(ProtocolModel):
    status: Literal["completed", "setup_required"]
    artifacts: tuple[AnalysisArtifact, ...] = ()
    next_actions: tuple[CreateNextAction, ...] = ()
    data: AnalyzeData | SetupRequiredData


class CreateResult(ProtocolModel):
    status: Literal["confirmed", "draft", "setup_required"]
    artifacts: tuple[PrdArtifact, ...] = ()
    next_actions: tuple[PlanNextAction, ...] = ()
    data: CreateData | SetupRequiredData


class PlanResult(ProtocolModel):
    status: Literal[
        "draft",
        "architect_working",
        "tech_review_working",
        "plan_approval_pending",
        "pm_working",
        "supervisor_working",
        "tasks_approval_pending",
        "executing",
        "done",
        "blocked",
        "cancelled",
        "setup_required",
    ]
    artifacts: tuple[PlanArtifact, ...] = ()
    next_actions: tuple[PlanNextAction | TaskListNextAction | ExecuteNextAction, ...]
    data: PlanData | SetupRequiredData


class ExecuteResult(ProtocolModel):
    status: Literal[
        "blocked",
        "active",
        "running",
        "completed",
        "failed",
        "cancelled",
        "setup_required",
    ]
    artifacts: tuple[TaskArtifact, ...] = ()
    next_actions: tuple[ExecuteNextAction, ...] = ()
    data: ExecuteData | SetupRequiredData


class RuntimeCost(ProtocolModel):
    """MCP representation of the shared billing-aware task cost."""

    api_spend_usd: float | None
    api_spend_unknown: bool
    subscription_included: bool


class RuntimeTask(ProtocolModel):
    """Exact phase-07 runtime projection for one current task."""

    generation_id: UUID
    task_id: UUID
    task_ref: str
    stage: str
    stem: str
    position: int
    title: str
    complexity: TaskComplexity
    status: TaskRuntimeStatus
    state_reason: str | None
    review_round: int
    attempt_count: int
    duration_seconds: float | None
    cost: RuntimeCost


class TaskListResult(ProtocolModel):
    """Typed current-generation task listing."""

    status: Literal["completed"]
    borg: str
    generation_id: UUID
    generation_digest: str
    approved_plan_digest: str | None
    tasks: tuple[RuntimeTask, ...]
    artifacts: tuple[TaskArtifact, ...]
    next_actions: tuple[ExecuteNextAction, ...]


server = FastMCP(
    "BetterBorg",
    instructions=(
        "Operate the BetterBorg workflow in the Git repository where the server "
        "was started. Results preserve durable status, artifacts, and next actions."
    ),
)


class ElicitedAnswer(BaseModel):
    """Primitive form returned for one free-text InteractiveIO prompt."""

    answer: str = Field(description="Your answer")


class ElicitedConfirmation(BaseModel):
    """Primitive form returned for one InteractiveIO confirmation."""

    approved: bool = Field(description="Approve this operation")


class McpInteractiveIO(InteractiveIO):
    """Bridge synchronous workflow prompts to same-request MCP elicitation."""

    def __init__(self, context: Context) -> None:
        self._context = context
        self._rendered: list[str] = []
        super().__init__(
            prompt=self._prompt,
            confirm=self._confirm,
            write=self._write,
        )

    @staticmethod
    def supported(context: Context) -> bool:
        """Return whether the connected client supports form elicitation."""
        params = context.session.client_params
        if params is None or params.capabilities.elicitation is None:
            return False
        return params.capabilities.elicitation.form is not None

    def _message(self, message: str) -> str:
        if not self._rendered:
            return message
        rendered = "\n".join(self._rendered)
        self._rendered.clear()
        return f"{rendered}\n\n{message}"

    async def _elicit_answer(self, message: str) -> str | None:
        result = await self._context.elicit(self._message(message), ElicitedAnswer)
        if result.action != "accept":
            return None
        return result.data.answer

    async def _elicit_confirmation(self, message: str, default: bool) -> bool:
        class Confirmation(ElicitedConfirmation):
            approved: bool = Field(
                default=default,
                description="Approve this operation",
            )

        result = await self._context.elicit(self._message(message), Confirmation)
        return result.action == "accept" and result.data.approved

    def _prompt(self, message: str) -> str | None:
        return anyio.from_thread.run(self._elicit_answer, message)

    def _confirm(self, message: str, default: bool) -> bool:
        return anyio.from_thread.run(self._elicit_confirmation, message, default)

    def _write(self, message: str) -> None:
        self._rendered.append(message)


def _paths(*, trusted: bool, io: InteractiveIO | None = None) -> RepoPaths:
    paths = RepoPaths.discover()
    if trusted:
        require_workspace_trust(
            paths,
            explicit=False,
            interactive=io is not None,
            confirm=(
                (lambda prompt: io.confirm(prompt, default=False))
                if io is not None
                else None
            ),
        )
    return paths


def _cli_command(*arguments: str) -> str:
    return shlex.join(("borg", *arguments))


def _relative(paths: RepoPaths, path: Path) -> str:
    return path.resolve().relative_to(paths.root).as_posix()


def _analysis_artifacts(
    paths: RepoPaths, result: Any
) -> tuple[AnalysisArtifact, ...]:
    artifacts: list[AnalysisArtifact] = [
        ScoreArtifact(kind="score", path=_relative(paths, result.score_path))
    ]
    artifacts.extend(
        PromptArtifact(
            kind=f"{prompt.role}_prompt",
            path=_relative(paths, prompt.path),
        )
        for prompt in result.prompts
    )
    artifacts.extend(
        ImprovementPrdArtifact(
            kind="improvement_prd",
            path=_relative(paths, prd.path),
        )
        for prd in result.improvement_prds
    )
    return tuple(artifacts)


def _theme_actions(paths: RepoPaths, result: Any) -> tuple[CreateNextAction, ...]:
    return tuple(
        CreateNextAction(
            tool="create",
            arguments=CreateActionArguments(
                name=prd.suggested_borg_name,
                source=_relative(paths, prd.path),
            )
        )
        for prd in result.improvement_prds
    )


def _initialize(io: McpInteractiveIO) -> InitializeResult:
    paths = _paths(trusted=True, io=io)
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        result = RepositoryService(
            paths,
            store,
            lambda config: select_agent(
                config, ApiAgentRole.ANALYSIS, paths, interactive=False
            ),
        ).initialize()
        onboarding = None
        if result.initialized:
            config = load_repository_config(paths)
            creator = CreateService(
                result.repository,
                store,
                select_agent(
                    config, ApiAgentRole.PLANNING, paths, interactive=False
                ),
                io=io,
                interactive=True,
            )
            onboarding = OnboardingDispatcher(
                result.repository,
                store,
                io,
                creator,
                result.improvement_prds,
            ).run()

    artifacts: list[AnalysisArtifact | PrdArtifact] = list(
        _analysis_artifacts(paths, result)
    )
    actions: list[CreateNextAction | PlanNextAction] = list(
        _theme_actions(paths, result)
    )
    if onboarding is not None and onboarding.confirmed:
        artifacts.append(
            PrdArtifact(kind="prd", path=_relative(paths, onboarding.prd_path))
        )
        actions = [
            PlanNextAction(
                tool="plan",
                arguments=PlanActionArguments(
                    name=onboarding.borg.name,
                    action="start",
                ),
            )
        ]
    return InitializeResult(
        status="initialized" if result.initialized else "already_initialized",
        artifacts=tuple(artifacts),
        next_actions=tuple(actions),
        data=InitializeData(
            repository_id=result.repository.id,
            analysis_id=result.analysis.id,
            score=result.analysis.overall_score,
        ),
    )


@server.tool(name="init")
async def initialize(ctx: Context) -> InitializeResult:
    """Initialize, analyze, and interactively onboard the current repository."""
    if not McpInteractiveIO.supported(ctx):
        return InitializeResult(
            status="setup_required",
            data=SetupRequiredData(cli_command=_cli_command("init")),
        )
    return await anyio.to_thread.run_sync(_initialize, McpInteractiveIO(ctx))


def _analyze(io: McpInteractiveIO) -> AnalyzeResult:
    paths = _paths(trusted=True, io=io)
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        result = RepositoryService(
            paths,
            store,
            lambda config: select_agent(
                config, ApiAgentRole.ANALYSIS, paths, interactive=False
            ),
        ).analyze()
    return AnalyzeResult(
        status="completed",
        artifacts=_analysis_artifacts(paths, result),
        next_actions=_theme_actions(paths, result),
        data=AnalyzeData(
            repository_id=result.repository.id,
            analysis_id=result.analysis.id,
            score=result.analysis.overall_score,
            previous_score=result.previous_analysis.overall_score,
            delta=result.analysis.score_delta,
        ),
    )


@server.tool(name="analyze")
async def analyze(ctx: Context) -> AnalyzeResult:
    """Re-analyze the initialized current Git repository."""
    if not McpInteractiveIO.supported(ctx):
        return AnalyzeResult(
            status="setup_required",
            data=SetupRequiredData(cli_command=_cli_command("analyze")),
        )
    return await anyio.to_thread.run_sync(_analyze, McpInteractiveIO(ctx))


def _create(name: str, source: str | None, io: McpInteractiveIO) -> CreateResult:
    paths = _paths(trusted=True, io=io)
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
            io=io,
            interactive=True,
        ).create(name, source_path)

    if result.confirmed:
        status = "confirmed"
        artifacts = (
            PrdArtifact(kind="prd", path=_relative(paths, result.prd_path)),
        )
        actions = (
            PlanNextAction(
                tool="plan",
                arguments=PlanActionArguments(name=name, action="start")
            ),
        )
    else:
        status = "draft"
        artifacts = ()
        actions = ()
    return CreateResult(
        status=status,
        artifacts=artifacts,
        next_actions=actions,
        data=CreateData(
            borg=name,
            borg_id=result.borg.id,
            questions=result.questions,
            draft_markdown=result.body_md,
        ),
    )


@server.tool(name="create")
async def create(
    name: str,
    ctx: Context,
    source: str | None = None,
) -> CreateResult:
    """Interactively interview, review, and confirm a Borg PRD."""
    arguments = ["create", name]
    if source is not None:
        arguments.extend(("--prd", source))
    if not McpInteractiveIO.supported(ctx):
        return CreateResult(
            status="setup_required",
            data=SetupRequiredData(cli_command=_cli_command(*arguments)),
        )
    return await anyio.to_thread.run_sync(
        _create,
        name,
        source,
        McpInteractiveIO(ctx),
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


def _plan_actions(
    name: str, state: BorgState
) -> tuple[PlanNextAction | TaskListNextAction | ExecuteNextAction, ...]:
    if state is BorgState.PLAN_APPROVAL_PENDING:
        return (
            PlanNextAction(
                tool="plan",
                arguments=PlanActionArguments(name=name, action="show")
            ),
            PlanNextAction(
                tool="plan",
                arguments=PlanActionArguments(name=name, action="approve")
            ),
        )
    if state is BorgState.READY_TO_EXECUTE:
        return (
            TaskListNextAction(
                tool="task_list",
                arguments=TaskListActionArguments(name=name),
            ),
            ExecuteNextAction(
                tool="execute",
                arguments=ExecuteActionArguments(name=name),
            ),
        )
    return ()


def _approve_plan(paths: RepoPaths, name: str) -> tuple[Any, Any, Path, Any]:
    config = load_repository_config(paths)
    result = approve_plan_workflow(
        paths,
        config,
        name,
        planning_agent=lambda: select_agent(
            config, ApiAgentRole.PLANNING, paths, interactive=False
        ),
    )
    return result.borg, result.approval, result.plan_path, result.publication


def _plan(
    name: str,
    action: Literal["start", "show", "change", "approve"],
    note: str | None,
    io: McpInteractiveIO | None,
) -> PlanResult:
    from betterborg_cli import cli as cli_module

    paths = _paths(trusted=action != "show", io=io)
    if action == "approve":
        assert io is not None
        borg, _questions = _planning_state(paths, name)
        with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
            attempt = validated_current_plan_attempt(paths, store, borg)
        plan_document = PlanDocument.model_validate(attempt.result)
        io.write(json.dumps(attempt.result, indent=2, sort_keys=True))
        if not io.confirm(
            f"Approve the current plan for Borg {name!r} and decompose its tasks?",
            default=False,
        ):
            return PlanResult(
                status=borg.state,
                next_actions=_plan_actions(name, borg.state),
                data=PlanShowData(borg=name, plan=plan_document),
            )
        borg, approval, plan_path, publication = _approve_plan(paths, name)
        artifacts = [
            ApprovedPlanArtifact(
                kind="approved_plan",
                path=_relative(paths, plan_path),
            )
        ]
        if publication is not None:
            artifacts.extend(
                TaskArtifact(kind="task", path=_relative(paths, item.path))
                for item in publication.files
            )
        return PlanResult(
            status=borg.state,
            artifacts=tuple(artifacts),
            next_actions=_plan_actions(name, borg.state),
            data=PlanApprovalData(borg=name, plan_digest=approval.plan_digest),
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
            attempt = validated_current_plan_attempt(paths, store, borg)
        return PlanResult(
            status=borg.state,
            next_actions=_plan_actions(name, borg.state),
            data=PlanShowData(
                borg=name,
                plan=PlanDocument.model_validate(attempt.result),
            ),
        )

    if action == "change" and (note is None or not note.strip()):
        raise ValueError("plan change note must not be empty")
    try:
        borg = cli_module._continue_planning(
            paths.root,
            name,
            change_note=note.strip() if action == "change" and note else None,
            io=io,
        )
        questions: list[dict[str, Any]] = []
    except click.ClickException as error:
        cause = error.__cause__
        if not isinstance(cause, ArchitectCancelled) or str(cause) != (
            "Architect questions are awaiting answers"
        ):
            raise
        borg, _questions = _planning_state(paths, name)
        if borg.state is not BorgState.ARCHITECT_AWAITING_ANSWERS:
            raise
        return PlanResult(
            status="cancelled",
            next_actions=(),
            data=PlanProgressData(borg=name, questions=()),
        )
    return PlanResult(
        status=borg.state,
        next_actions=_plan_actions(name, borg.state),
        data=PlanProgressData(
            borg=name,
            questions=tuple(
                PlanningQuestionData.model_validate(question)
                for question in questions
            ),
        ),
    )


@server.tool(name="plan")
async def plan(
    name: str,
    ctx: Context,
    action: Literal["start", "show", "change", "approve"] = "start",
    note: str | None = None,
) -> PlanResult:
    """Start, inspect, change, or approve a plan; approval decomposes tasks."""
    if action == "show":
        return await anyio.to_thread.run_sync(_plan, name, action, note, None)

    arguments = ["plan", action, name]
    if action == "change" and note is not None:
        arguments.extend(("--note", note))
    if not McpInteractiveIO.supported(ctx):
        return PlanResult(
            status="setup_required",
            next_actions=(),
            data=SetupRequiredData(cli_command=_cli_command(*arguments)),
        )
    return await anyio.to_thread.run_sync(
        _plan,
        name,
        action,
        note,
        McpInteractiveIO(ctx),
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
            TaskArtifact(kind="task", path=_relative(paths, item.path))
            for item in publication.files
        ),
        next_actions=(
            ExecuteNextAction(
                tool="execute",
                arguments=ExecuteActionArguments(name=name),
            ),
        ),
    )


def _execute(name: str, io: McpInteractiveIO) -> ExecuteResult:
    from betterborg_cli import cli as cli_module

    paths = _paths(trusted=True, io=io)
    config = load_repository_config(paths)

    def decide(estimate: dict[str, Any]) -> ExecutionDecisionRequest | None:
        io.write(json.dumps(estimate, indent=2, sort_keys=True))
        if not io.confirm(
            f"Approve this estimate for Borg {name!r} and begin host execution?",
            default=False,
        ):
            return None
        return ExecutionDecisionRequest("mcp_elicitation", "approved")

    workflow = execute_workflow(
        paths,
        config,
        name,
        decide=decide,
        invoke_host=cli_module._invoke_host_execution,
    )
    publication = workflow.publication
    generation = publication.generation
    estimate = workflow.estimate
    result = workflow.host_result
    if result is None:
        return ExecuteResult(
            status="cancelled",
            artifacts=tuple(
                TaskArtifact(kind="task", path=_relative(paths, item.path))
                for item in publication.files
            ),
            data=ExecuteData(
                borg=name,
                generation_id=generation.id,
                estimate=ExecutionEstimateData.model_validate(estimate),
            ),
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
        actions = (
            ExecuteNextAction(
                tool="execute",
                arguments=ExecuteActionArguments(name=name),
            ),
        )
    return ExecuteResult(
        status=status,
        artifacts=tuple(
            TaskArtifact(kind="task", path=_relative(paths, item.path))
            for item in publication.files
        ),
        next_actions=actions,
        data=ExecuteData(
            borg=name,
            generation_id=generation.id,
            operation_id=operation_id,
            active_operation_id=active_operation_id,
            reason=reason,
            estimate=ExecutionEstimateData.model_validate(estimate),
        ),
    )


@server.tool(name="execute")
async def execute(name: str, ctx: Context) -> ExecuteResult:
    """Elicit estimate approval and run the assembled host execution service."""
    if not McpInteractiveIO.supported(ctx):
        return ExecuteResult(
            status="setup_required",
            data=SetupRequiredData(cli_command=_cli_command("execute", name)),
        )
    return await anyio.to_thread.run_sync(_execute, name, McpInteractiveIO(ctx))


def run_stdio_server() -> None:
    """Run only MCP protocol framing on stdout using the stdio transport."""
    server.run(transport="stdio")
