"""Command-line entry point for BetterBorg."""

import hashlib
import json
import os
import re
import shlex
import sqlite3
from functools import wraps
from pathlib import Path
from threading import RLock
from uuid import UUID

import click

from betterborg_cli import __version__
from betterborg_cli.agent_runtime import BillingMode
from betterborg_cli.agent_runtime.api_tools import ApiAgentRole
from betterborg_cli.agent_runtime.selection import resolve_agent_model, select_agent
from betterborg_cli.execution_estimate import (
    DUMMY_PRIOR_LABEL,
    EXECUTION_ESTIMATE_VERSION,
    estimate_generation,
    phase_billing_from_config,
)
from betterborg_cli.host_execution import (
    HostCodingConfig,
    HostCodingPhase,
    HostComposeManager,
    HostEnvironmentManager,
    HostExecutionResult,
    HostExecutionService,
    HostMergeConfig,
    HostMergePhase,
    HostPreflight,
    HostPreflightBlock,
    HostReviewFixConfig,
    HostReviewFixPhase,
    HostSanityPhase,
    HostSchedulerConfig,
    HostTaskRuntime,
    HostWorktreeManager,
)
from betterborg_cli.onboarding import (
    CreateService,
    OnboardingDispatcher,
    create_commands,
)
from betterborg_cli.planning import (
    ArchitectCancelled,
    ArchitectLoop,
    SupervisorCancelled,
    SupervisorLoop,
    TaskDigestDriftError,
    TaskPublication,
    TaskPublisher,
    TechLeadCancelled,
    TechLeadLoop,
    approved_plan_digest,
    render_plan_markdown,
    render_task_markdown,
    task_markdown_digest,
    validate_plan,
)
from betterborg_cli.prd_session import InteractiveIO, validate_borg_name
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import RepositoryConfig, load_repository_config
from betterborg_cli.repository_files import (
    publish_repository_text,
    require_git_trackable,
)
from betterborg_cli.repository_service import RepositoryService
from betterborg_cli.store import (
    Borg,
    BorgState,
    ExecutionDecision,
    PlanApproval,
    PlanChangeRequest,
    PlanningAttempt,
    PlanningAttemptStatus,
    SqliteStore,
    TaskGenerationStatus,
    TaskRecord,
    TaskRuntimeCost,
    TaskRuntimeRow,
)
from betterborg_cli.workspace_trust import (
    UntrustedWorkspaceError,
    require_workspace_trust,
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Work with BetterBorg from the command line."""


@cli.command()
def version() -> None:
    """Print the installed BetterBorg CLI version."""
    click.echo(f"borg {__version__}")


def _stdin_is_interactive() -> bool:
    return click.get_text_stream("stdin").isatty()


def _trusted_workspace_callback(function):
    """Gate a callback before it can load repository-controlled context."""

    @wraps(function)
    def guarded(*args, explicit_trust: bool = False, **kwargs):
        try:
            paths = RepoPaths.discover()
            interactive = _stdin_is_interactive() and not kwargs.get(
                "json_output", False
            )
            require_workspace_trust(
                paths,
                explicit=explicit_trust,
                interactive=interactive,
                confirm=lambda prompt: click.confirm(prompt, default=False),
            )
        except (UntrustedWorkspaceError, ValueError, RuntimeError) as error:
            raise click.ClickException(str(error)) from error
        return function(*args, repository_path=paths.root, **kwargs)

    return guarded


@cli.command(name="trust")
@click.option(
    "--yes",
    "explicit_trust",
    is_flag=True,
    help="Trust without prompting after accepting the host-access consequence.",
)
@_trusted_workspace_callback
def trust_workspace(repository_path: Path) -> None:
    """Trust this workspace for host-capable agent operations."""
    click.echo(f"Trusted workspace: {repository_path}")


@cli.command(name="init")
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit machine-readable initialization output without prompting.",
)
@click.option(
    "--yes",
    "explicit_trust",
    is_flag=True,
    help="Trust this workspace without prompting and initialize it.",
)
@_trusted_workspace_callback
def initialize_repository(
    repository_path: Path,
    json_output: bool,
) -> None:
    """Register and analyze the current Git repository."""
    paths = RepoPaths.discover(repository_path)
    database = paths.state_dir / "borg.sqlite3"
    interactive = _stdin_is_interactive() and not json_output
    try:
        if not database.resolve().is_relative_to(paths.root):
            raise ValueError(f"repository state path escapes repository: {database}")
        with SqliteStore.open(database) as store:
            service = RepositoryService(
                paths,
                store,
                lambda config: select_agent(
                    config,
                    ApiAgentRole.ANALYSIS,
                    paths,
                    interactive=interactive,
                ),
            )
            result = service.initialize()

            if result.initialized and interactive:
                _write_initialized(result)
                config = load_repository_config(paths)
                io = _interactive_io()
                creator = CreateService(
                    result.repository,
                    store,
                    select_agent(
                        config,
                        ApiAgentRole.PLANNING,
                        paths,
                        interactive=True,
                    ),
                    io=io,
                    editor=_edit_markdown,
                )
                OnboardingDispatcher(
                    result.repository,
                    store,
                    io,
                    creator,
                    result.improvement_prds,
                ).run()
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    commands = create_commands(paths.root, result.improvement_prds)
    if json_output:
        click.echo(
            json.dumps(
                {
                    "repository_id": str(result.repository.id),
                    "initialized": result.initialized,
                    "score": result.analysis.overall_score,
                    "create_commands": [shlex.join(command) for command in commands],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    elif not result.initialized:
        click.echo(f"Repository already initialized: {result.repository.id}")
    elif not interactive:
        _write_initialized(result)
        for command in commands:
            click.echo(shlex.join(command))


@cli.command(name="analyze")
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit the analysis result as machine-readable JSON.",
)
@click.option(
    "--yes",
    "explicit_trust",
    is_flag=True,
    help="Trust this workspace without prompting before analyzing it.",
)
@_trusted_workspace_callback
def analyze_repository(
    repository_path: Path,
    json_output: bool,
) -> None:
    """Re-analyze an initialized Git repository and refresh its outputs."""
    paths = RepoPaths.discover(repository_path)
    database = paths.state_dir / "borg.sqlite3"
    interactive = _stdin_is_interactive() and not json_output
    try:
        if not database.resolve().is_relative_to(paths.root):
            raise ValueError(f"repository state path escapes repository: {database}")
        with SqliteStore.open(database) as store:
            result = RepositoryService(
                paths,
                store,
                lambda config: select_agent(
                    config,
                    ApiAgentRole.ANALYSIS,
                    paths,
                    interactive=interactive,
                ),
            ).analyze()
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    analysis = result.analysis
    previous_score = result.previous_analysis.overall_score
    score_delta = analysis.overall_score - previous_score
    if json_output:
        click.echo(
            json.dumps(
                {
                    "analysis_id": str(analysis.id),
                    "repository_id": str(result.repository.id),
                    "score": analysis.overall_score,
                    "previous_score": previous_score,
                    "delta": score_delta,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        click.echo(
            f"Analyzed repository {result.repository.id}: "
            f"score {analysis.overall_score:.2f}/5 "
            f"(previous {previous_score:.2f}/5, "
            f"delta {score_delta:+.2f})."
        )


@cli.command(name="create")
@click.argument("name")
@click.option(
    "--prd",
    "source",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    help="Optional local Markdown PRD to improve.",
)
@click.option(
    "--yes",
    "explicit_trust",
    is_flag=True,
    help="Trust this workspace without prompting before creating the Borg.",
)
@_trusted_workspace_callback
def create_borg(
    repository_path: Path,
    name: str,
    source: Path | None,
) -> None:
    """Brainstorm or improve a PRD and create a named Borg."""
    try:
        _validate_create_name(name)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    if not _stdin_is_interactive():
        raise click.ClickException("borg create requires an interactive terminal")
    paths = RepoPaths.discover(repository_path)
    try:
        config = load_repository_config(paths)
        with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
            repository = store.get_repository(config.repository_id)
            if repository is None:
                raise ValueError("repository is not initialized; run 'borg init' first")
            io = _interactive_io()
            result = CreateService(
                repository,
                store,
                select_agent(
                    config,
                    ApiAgentRole.PLANNING,
                    paths,
                    interactive=True,
                ),
                io=io,
                editor=_edit_markdown,
            ).create(name, source)
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    if result.confirmed:
        click.echo(f"Created Borg {result.borg.name!r}: {result.prd_path}")
        click.echo(f"borg plan start {result.borg.name}")
    elif result.questions:
        click.echo("Borg PRD needs more input before it can be created.")
    else:
        click.echo("Borg draft saved without a confirmed PRD.")


@cli.group()
def plan() -> None:
    """Create and review implementation plans for a Borg."""


@plan.command(name="start")
@click.argument("name")
@click.option(
    "--yes",
    "explicit_trust",
    is_flag=True,
    help="Trust this workspace without prompting before planning.",
)
@_trusted_workspace_callback
def start_plan(repository_path: Path, name: str) -> None:
    """Start or resume planning for the named Borg."""
    borg = _continue_planning(repository_path, name)
    _write_planning_gate(name, borg, changed=False)


@plan.command(name="show")
@click.argument("name")
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit the stored plan as validated machine-readable JSON.",
)
def show_plan(name: str, json_output: bool) -> None:
    """Show the latest complete plan for the named Borg."""
    try:
        paths = RepoPaths.discover()
        config = load_repository_config(paths)
        with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
            repository = store.get_repository(config.repository_id)
            if repository is None:
                raise ValueError("repository is not initialized; run 'borg init' first")
            borg = store.get_borg_by_name(repository.id, name)
            if borg is None:
                raise ValueError(
                    f"Borg {name!r} does not exist; run 'borg create {name}' first"
                )
            attempt = _validated_current_plan_attempt(paths, store, borg)
            stored_plan = attempt.result
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    if json_output:
        click.echo(json.dumps(stored_plan, sort_keys=True, separators=(",", ":")))
    else:
        click.echo(render_plan_markdown(stored_plan), nl=False)


@cli.group()
def task() -> None:
    """Inspect the current executable tasks for a Borg."""


@task.command(name="list")
@click.argument("name")
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit current task metadata as machine-readable JSON.",
)
def list_tasks(name: str, json_output: bool) -> None:
    """List the complete SQLite-current task generation for a Borg."""
    try:
        paths, publication = _current_task_publication(name)
        runtime_rows = _current_task_runtime(paths, name, publication)
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    runtime_by_task = {row.task_id: row for row in runtime_rows}
    tasks = []
    for published in publication.files:
        runtime = runtime_by_task.get(published.task.id)
        if runtime is None:
            raise click.ClickException(
                "current task generation changed while it was being inspected"
            )
        tasks.append(
            {
                **_task_listing_item(paths, published.task, published.path),
                **_task_runtime_listing_item(runtime),
            }
        )
    totals = _task_runtime_totals(runtime_rows)
    if json_output:
        click.echo(
            json.dumps(
                {
                    "approved_plan_digest": publication.generation.manifest.get(
                        "approved_plan_digest"
                    ),
                    "borg": name,
                    "generation_digest": publication.generation.digest,
                    "generation_id": str(publication.generation.id),
                    "tasks": tasks,
                    "totals": totals,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return

    click.echo(
        f"Current task generation for Borg {name!r}: {publication.generation.id}"
    )
    for item in tasks:
        click.echo(
            f"{item['stage']}/{item['stem']} "
            f"[{item['complexity']}] {item['title']}"
        )
        click.echo(f"  Status: {item['status']}")
        if item["state_reason"] is not None:
            click.echo(f"  Reason: {item['state_reason']}")
        click.echo(f"  Review rounds: {item['review_round']}")
        click.echo(f"  Attempts: {item['attempt_count']}")
        click.echo(f"  Duration: {_format_duration(item['duration_seconds'])}")
        click.echo(f"  Cost: {_format_runtime_cost(item['cost'])}")
        click.echo(f"  Task ref: {item['task_ref']}")
        click.echo(f"  Markdown: {item['path']}")
    click.echo(
        f"Totals: {totals['attempt_count']} attempt(s), "
        f"{_format_duration(totals['duration_seconds'])}, "
        f"{_format_runtime_cost(totals['cost'])}"
    )


@task.command(name="estimate")
@click.argument("name")
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit the generation estimate as machine-readable JSON.",
)
def estimate_tasks(name: str, json_output: bool) -> None:
    """Estimate total agent work for the current task generation."""
    try:
        paths, publication = _current_task_publication(name)
        config = load_repository_config(paths)
        with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
            generation = store.get_task_generation(publication.generation.id)
            if (
                generation is None
                or generation.status is not TaskGenerationStatus.CURRENT
            ):
                raise RuntimeError(
                    "current task generation changed while it was being estimated"
                )
            estimate = estimate_generation(
                generation.id,
                [item.task for item in publication.files],
                store.list_task_completion_samples(),
                phase_billing_from_config(config),
            )
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    if json_output:
        click.echo(json.dumps(estimate, sort_keys=True, separators=(",", ":")))
        return

    _write_execution_estimate(name, estimate)


def _write_execution_estimate(name: str, estimate: dict[str, object]) -> None:
    """Render the estimate shared by inspection and the execution gate."""
    mix = estimate["task_mix"]
    time = estimate["time"]
    click.echo(DUMMY_PRIOR_LABEL)
    click.echo(f"Execution estimate for Borg {name!r}: {estimate['generation_id']}")
    click.echo(
        "Task mix: "
        f"{mix['small']} small, {mix['medium']} medium, {mix['large']} large, "
        f"{mix['unsized']} unsized"
    )
    click.echo(
        "Total agent work (not calendar time): "
        f"P50 {_format_duration(time['p50'])}, "
        f"P80 {_format_duration(time['p80'])}"
    )
    if time["unknown_tasks"]:
        click.echo(f"Unknown time: {time['unknown_tasks']} task(s)")
    click.echo(f"Local completion sample: {estimate['sample_size']} task(s)")
    for item in estimate["per_complexity"]:
        click.echo(
            f"  {item['complexity']}: {item['task_count']} task(s), "
            f"n={item['sample_size']}, source={item['source']}, "
            f"P50 {_format_duration(item['time']['p50'])}, "
            f"P80 {_format_duration(item['time']['p80'])}"
        )

    billing = estimate["billing"]
    api = billing["api"]
    if api["unknown"]:
        click.echo("API estimate: unknown (billing, usage, or model price is missing)")
    elif api["estimate"] is None:
        click.echo("API estimate: not used")
    else:
        click.echo(
            f"API estimate: P50 ${api['estimate']['p50']:.4f}, "
            f"P80 ${api['estimate']['p80']:.4f} USD"
        )
    subscription = billing["subscription"]
    if subscription["included"]:
        click.echo(
            "Subscription work included for "
            f"{', '.join(subscription['phases'])}; USD: unknown/not applicable"
        )
    if billing["unknown_phases"]:
        click.echo(
            "Billing mode unknown for: " + ", ".join(billing["unknown_phases"])
        )


@cli.command(name="execute")
@click.argument("name")
@click.option(
    "--auto-execute",
    is_flag=True,
    help="Record an estimate bypass and execute without asking for approval.",
)
@_trusted_workspace_callback
def execute_borg(repository_path: Path, name: str, auto_execute: bool) -> None:
    """Run the current, digest-verified task generation for a Borg."""
    paths = RepoPaths.discover(repository_path)
    try:
        config = load_repository_config(paths)
        with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
            repository = store.get_repository(config.repository_id)
            if repository is None:
                raise ValueError("repository is not initialized; run 'borg init' first")
            borg = store.get_borg_by_name(repository.id, name)
            if borg is None:
                raise ValueError(
                    f"Borg {name!r} does not exist; run 'borg create {name}' first"
                )
            if borg.state is not BorgState.READY_TO_EXECUTE:
                raise ValueError(f"Borg {name!r} is not ready to execute")

            # This verifies the complete immutable tree before estimating,
            # selecting an agent, or allowing HostExecutionService to claim.
            publication = TaskPublisher(
                repository, store
            ).inspect_current_task_files(borg.id)
            generation = publication.generation
            decision = store.get_current_execution_decision(borg.id)
            if decision is None:
                estimate = estimate_generation(
                    generation.id,
                    [item.task for item in publication.files],
                    store.list_task_completion_samples(),
                    phase_billing_from_config(config),
                )
                _write_execution_estimate(name, estimate)
                if auto_execute:
                    source = "auto_execute"
                    outcome = "bypassed"
                else:
                    click.confirm(
                        "Approve this estimate and begin host execution?",
                        default=False,
                        abort=True,
                    )
                    source = "interactive"
                    outcome = "approved"
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
                    raise RuntimeError(
                        "current task generation has no approval or batch"
                    )
                decision = ExecutionDecision(
                    borg_id=borg.id,
                    generation_id=generation.id,
                    approved_plan_digest=approval.plan_digest,
                    task_batch_digest=batch.digest,
                    estimate_version=EXECUTION_ESTIMATE_VERSION,
                    source=source,
                    snapshot=estimate,
                    decision=outcome,
                )
                store.append_execution_decision(decision)
                click.echo(
                    f"Recorded execution estimate {outcome} for generation "
                    f"{generation.id}."
                )
            elif decision.decision not in {"approved", "bypassed"}:
                raise RuntimeError(
                    "current generation has an unsupported execution decision"
                )
            else:
                click.echo(
                    f"Using recorded execution decision for generation "
                    f"{generation.id}."
                )

            result = _invoke_host_execution(
                paths,
                store,
                config,
                repository.id,
                borg.id,
                generation.id,
            )
    except (OSError, RuntimeError, ValueError, sqlite3.IntegrityError) as error:
        raise click.ClickException(str(error)) from error

    _write_host_execution_result(result)


def _invoke_host_execution(
    paths: RepoPaths,
    store: SqliteStore,
    config: RepositoryConfig,
    repository_id: UUID,
    borg_id: UUID,
    generation_id: UUID,
) -> HostExecutionResult:
    """Assemble and invoke the sole concrete phase-07 host service."""
    analysis = store.get_prior_ready_analysis(repository_id)
    if analysis is None:
        raise RuntimeError("repository has no completed analysis; run 'borg analyze'")
    analyzer_plan = analysis.analysis_json
    preflight = HostPreflight(paths.root)
    validated = preflight.validate(analyzer_plan)
    if isinstance(validated, HostPreflightBlock):
        return HostExecutionResult(validated)

    coding_agent = select_agent(
        config,
        ApiAgentRole.CODING,
        paths,
        interactive=_stdin_is_interactive(),
    )
    review_agent = select_agent(
        config,
        ApiAgentRole.REVIEW,
        paths,
        interactive=_stdin_is_interactive(),
    )
    merge_agent = select_agent(
        config,
        ApiAgentRole.MERGE,
        paths,
        interactive=_stdin_is_interactive(),
    )
    environment = HostEnvironmentManager(paths.root)
    compose = HostComposeManager(paths.root)
    worktrees = HostWorktreeManager(
        paths.root,
        paths.worktrees_dir,
        source_branch=config.default_branch,
    )
    repository_lock = RLock()

    def locked_repository():
        return repository_lock

    runtime = HostTaskRuntime(
        validated,
        environment_manager=environment,
        compose_manager=compose,
        coding=HostCodingPhase(
            paths.root,
            coding_agent,
            config=HostCodingConfig(
                model=resolve_agent_model(coding_agent, config.agents.coding.model),
                billing_mode=_agent_billing_mode(coding_agent.name),
                effort=config.agents.coding.effort,
            ),
        ),
        review_fix=HostReviewFixPhase(
            paths.root,
            review_agent,
            config=HostReviewFixConfig(
                review_model=resolve_agent_model(
                    review_agent, config.agents.review.model
                ),
                review_passes=config.execution.review_passes,
                review_billing_mode=_agent_billing_mode(review_agent.name),
                fix_billing_mode=_agent_billing_mode(review_agent.name),
                review_effort=config.agents.review.effort,
                fix_effort=config.agents.review.effort,
            ),
        ),
        merge=HostMergePhase(
            paths.root,
            merge_agent,
            config=HostMergeConfig(
                model=resolve_agent_model(merge_agent, config.agents.merge.model),
                billing_mode=_agent_billing_mode(merge_agent.name),
                effort=config.agents.merge.effort,
            ),
            repository_lock=locked_repository,
        ),
        sanity=HostSanityPhase(
            paths.root,
            validated,
            environment_manager=environment,
            compose_manager=compose,
            worktree_manager=worktrees,
            repository_lock=locked_repository,
        ),
    )
    service = HostExecutionService(
        store,
        preflight,
        runtime,
        worktree_manager=worktrees,
        compose_manager=compose,
        scheduler_config=HostSchedulerConfig(jobs=config.execution.jobs),
    )
    secrets = {
        name: os.environ[name]
        for name in validated.required_secret_names
        if name in os.environ
    }
    return service.run(
        borg_id,
        generation_id,
        analyzer_plan,
        secret_values=secrets,
    )


def _agent_billing_mode(adapter_name: str) -> BillingMode:
    if adapter_name in {"claude", "codex"}:
        return BillingMode.SUBSCRIPTION
    return BillingMode.API


def _write_host_execution_result(result: HostExecutionResult) -> None:
    if isinstance(result.preflight, HostPreflightBlock):
        raise click.ClickException(result.preflight.reason)
    if result.active_operation_id is not None:
        click.echo(f"Execution already active: {result.active_operation_id}")
        return
    if result.operation_id is None or result.status is None:
        raise click.ClickException("host execution returned no operation")
    click.echo(
        f"Execution operation {result.operation_id}: {result.status.value}"
    )


@task.command(name="show")
@click.argument("name")
@click.argument("task_ref")
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit the stored task contract as machine-readable JSON.",
)
def show_task(name: str, task_ref: str, json_output: bool) -> None:
    """Show one task from the complete SQLite-current generation."""
    try:
        paths, publication = _current_task_publication(name)
        published = next(
            (
                item
                for item in publication.files
                if task_ref
                in {item.task.task_ref, f"{item.task.stage}/{item.task.stem}"}
            ),
            None,
        )
        if published is None:
            raise ValueError(
                f"current task {task_ref!r} does not exist for Borg {name!r}"
            )
        body = published.path.read_bytes()
        expected = render_task_markdown(published.task.task).encode("utf-8")
        if body != expected or task_markdown_digest(body) != published.task.digest:
            raise TaskDigestDriftError(
                f"task file {published.task.stage}/{published.task.stem}.md "
                "digest drifted"
            )
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    if json_output:
        item = _task_listing_item(paths, published.task, published.path)
        click.echo(
            json.dumps(
                {**item, "task": published.task.task},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        click.echo(body.decode("utf-8"), nl=False)


def _current_task_publication(
    name: str,
) -> tuple[RepoPaths, TaskPublication]:
    """Load and verify the sole current generation without reconciling state."""
    paths = RepoPaths.discover()
    config = load_repository_config(paths)
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        repository = store.get_repository(config.repository_id)
        if repository is None:
            raise ValueError("repository is not initialized; run 'borg init' first")
        borg = store.get_borg_by_name(repository.id, name)
        if borg is None:
            raise ValueError(
                f"Borg {name!r} does not exist; run 'borg create {name}' first"
            )
        publication = TaskPublisher(repository, store).inspect_current_task_files(
            borg.id
        )
    return paths, publication


def _current_task_runtime(
    paths: RepoPaths, name: str, publication: TaskPublication
) -> list[TaskRuntimeRow]:
    """Load runtime rows and guard against a concurrent generation change."""
    config = load_repository_config(paths)
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        repository = store.get_repository(config.repository_id)
        if repository is None:
            raise ValueError("repository is not initialized; run 'borg init' first")
        borg = store.get_borg_by_name(repository.id, name)
        if borg is None:
            raise ValueError(
                f"Borg {name!r} does not exist; run 'borg create {name}' first"
            )
        runtime_rows = store.list_task_runtime(borg.id)
        generation_ids = {row.generation_id for row in runtime_rows}
        if generation_ids != {publication.generation.id}:
            raise RuntimeError(
                "current task generation changed while it was being inspected"
            )
    return runtime_rows


def _task_listing_item(
    paths: RepoPaths, record: TaskRecord, path: Path
) -> dict[str, object]:
    """Return stable public metadata for one verified current task."""
    return {
        "complexity": record.complexity.value,
        "dependencies": record.task.get("dependencies", []),
        "digest": record.digest,
        "path": path.relative_to(paths.root).as_posix(),
        "position": record.position,
        "stage": record.stage,
        "stem": record.stem,
        "task_ref": record.task_ref,
        "title": record.title,
    }


def _task_runtime_listing_item(row: TaskRuntimeRow) -> dict[str, object]:
    """Serialize the shared task-runtime projection for CLI consumers."""
    return {
        "attempt_count": row.attempt_count,
        "cost": _task_runtime_cost_item(row.cost),
        "duration_seconds": row.duration_seconds,
        "review_round": row.review_round,
        "state_reason": row.state_reason,
        "status": row.status.value,
    }


def _task_runtime_cost_item(cost: TaskRuntimeCost) -> dict[str, object]:
    return {
        "api_spend_unknown": cost.api_spend_unknown,
        "api_spend_usd": cost.api_spend_usd,
        "subscription_included": cost.subscription_included,
    }


def _task_runtime_totals(rows: list[TaskRuntimeRow]) -> dict[str, object]:
    """Aggregate display totals without turning missing measurements into zero."""
    attempted = [row for row in rows if row.attempt_count]
    durations = [
        row.duration_seconds
        for row in attempted
        if row.duration_seconds is not None
    ]
    api_spend_unknown = not attempted or any(
        row.cost.api_spend_unknown for row in attempted
    )
    api_costs = [
        row.cost.api_spend_usd
        for row in attempted
        if row.cost.api_spend_usd is not None
    ]
    return {
        "attempt_count": sum(row.attempt_count for row in rows),
        "cost": _task_runtime_cost_item(
            TaskRuntimeCost(
                api_spend_usd=(
                    None
                    if api_spend_unknown or not api_costs
                    else float(sum(api_costs))
                ),
                api_spend_unknown=api_spend_unknown,
                subscription_included=any(
                    row.cost.subscription_included for row in attempted
                ),
            )
        ),
        "duration_seconds": float(sum(durations)) if durations else None,
    }


def _format_duration(value: object) -> str:
    if value is None:
        return "unknown duration"
    seconds = float(value)
    if seconds < 60:
        return f"{seconds:g}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _format_runtime_cost(value: object) -> str:
    if not isinstance(value, dict):
        raise TypeError("runtime cost must be serialized before formatting")
    parts = []
    if value["api_spend_unknown"]:
        parts.append("API spend unknown")
    elif value["api_spend_usd"] is not None:
        parts.append(f"${float(value['api_spend_usd']):.4f} API")
    if value["subscription_included"]:
        parts.append("subscription included")
    return " + ".join(parts) if parts else "unknown cost"


@plan.command(name="approve")
@click.argument("name")
@click.option(
    "--yes",
    "explicit_trust",
    is_flag=True,
    help="Trust this workspace without prompting before decomposition.",
)
@_trusted_workspace_callback
def approve_plan(repository_path: Path, name: str) -> None:
    """Approve the current plan and prepare its executable task generation."""
    paths = RepoPaths.discover(repository_path)
    resumable = False
    try:
        config = load_repository_config(paths)
        with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
            repository = store.get_repository(config.repository_id)
            if repository is None:
                raise ValueError("repository is not initialized; run 'borg init' first")
            borg = store.get_borg_by_name(repository.id, name)
            if borg is None:
                raise ValueError(
                    f"Borg {name!r} does not exist; run 'borg create {name}' first"
                )

            approval, plan_path = _bind_plan_approval(paths, store, borg)
            borg = store.get_borg(borg.id)
            if borg is None:
                raise RuntimeError(f"Borg {name!r} disappeared during approval")
            resumable = True

            if borg.state in {BorgState.PM_WORKING, BorgState.SUPERVISOR_WORKING}:
                agent = select_agent(
                    config,
                    ApiAgentRole.PLANNING,
                    paths,
                    interactive=_stdin_is_interactive(),
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
    except (SupervisorCancelled, KeyboardInterrupt) as error:
        message = str(error).strip()
        detail = f" ({message})" if message else ""
        raise click.ClickException(
            f"Decomposition for Borg {name!r} was interrupted{detail}. "
            f"Run 'borg plan approve {name}' to resume."
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        if resumable:
            message = str(error).strip()
            detail = f" ({message})" if message else ""
            raise click.ClickException(
                f"Decomposition for Borg {name!r} could not continue{detail}. "
                f"Run 'borg plan approve {name}' to resume."
            ) from error
        raise click.ClickException(str(error)) from error

    relative_plan = plan_path.relative_to(paths.root).as_posix()
    click.echo(f"Approved plan: {relative_plan} ({approval.plan_digest})")
    if borg.state is BorgState.READY_TO_EXECUTE:
        click.echo(f"Borg {name!r} is ready to execute.")
        click.echo("Current tasks:")
        for item in publication.files:
            click.echo(f"  {item.path.relative_to(paths.root).as_posix()}")
    else:
        click.echo(f"Task decomposition blocked for Borg {name!r}.")


def _bind_plan_approval(
    paths: RepoPaths,
    store: SqliteStore,
    borg: Borg,
) -> tuple[PlanApproval, Path]:
    """Bind or recover one approval for the latest exact Architect plan."""
    plan_attempt = _validated_current_plan_attempt(paths, store, borg)
    current_plan = plan_attempt.result
    digest = approved_plan_digest(current_plan)
    body = render_plan_markdown(current_plan)
    body_digest = "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()
    relative_path = Path(".borg/plans") / f"{borg.name}.md"
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
            publish_repository_text(
                plan_path, body, root=paths.root, overwrite=True
            )
        else:
            if existing != body:
                raise ValueError(f"approved plan Markdown drifted: {relative_path}")
        require_git_trackable(relative_path, root=paths.root)
        return approval, plan_path

    if borg.state is not BorgState.PLAN_APPROVAL_PENDING:
        raise ValueError(
            f"Borg {borg.name!r} cannot approve a plan from state "
            f"{borg.state.value!r}; a plan must be awaiting approval"
        )
    publish_repository_text(plan_path, body, root=paths.root, overwrite=True)
    require_git_trackable(relative_path, root=paths.root)
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


def _validated_current_plan_attempt(
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
            f"run 'borg plan start {borg.name}' first"
        )
    validate_plan(
        attempt.result,
        paths.root,
        check_repository_state=False,
    )
    return attempt

@plan.command(name="change")
@click.argument("name")
@click.option(
    "--note",
    help="Plain-language changes for the Architect to apply.",
)
@click.option(
    "--yes",
    "explicit_trust",
    is_flag=True,
    help="Trust this workspace without prompting before revising the plan.",
)
@_trusted_workspace_callback
def change_plan(repository_path: Path, name: str, note: str | None) -> None:
    """Request changes to a plan awaiting human approval."""
    if note is None:
        note = _prompt("Change note")
    if note is None or not note.strip():
        raise click.ClickException("plan change note must not be empty")
    note = note.strip()

    borg = _continue_planning(repository_path, name, change_note=note)
    _write_planning_gate(name, borg, changed=True)


def _continue_planning(
    repository_path: Path,
    name: str,
    *,
    change_note: str | None = None,
) -> Borg:
    """Load and drain one initial or change-request planning lifecycle."""
    paths = RepoPaths.discover(repository_path)
    change_requested = change_note is not None
    resumable = False
    try:
        config = load_repository_config(paths)
        with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
            repository = store.get_repository(config.repository_id)
            if repository is None:
                raise ValueError("repository is not initialized; run 'borg init' first")
            borg = store.get_borg_by_name(repository.id, name)
            if borg is None:
                raise ValueError(
                    f"Borg {name!r} does not exist; run 'borg create {name}' first"
                )

            if change_note is not None:
                if borg.state is not BorgState.PLAN_APPROVAL_PENDING:
                    raise ValueError(
                        f"Borg {name!r} cannot change its plan from state "
                        f"{borg.state.value!r}; a plan must be awaiting approval"
                    )
                requests = store.list_plan_change_requests(borg.id)
                request = PlanChangeRequest(
                    borg_id=borg.id,
                    round=max((item.round for item in requests), default=0) + 1,
                    note=change_note,
                )
                with store.transaction():
                    store.append_plan_change_request(request)
                    borg = store.compare_and_set_borg_state(
                        borg.id,
                        expected_state=borg.state,
                        expected_version=borg.state_version,
                        new_state=BorgState.ARCHITECT_WORKING,
                    )

            if borg.state not in {
                BorgState.PLAN_APPROVAL_PENDING,
                BorgState.BLOCKED,
            }:
                resumable = True
                agent = select_agent(
                    config,
                    ApiAgentRole.PLANNING,
                    paths,
                    interactive=_stdin_is_interactive(),
                )
                io = _interactive_io()
                if borg.state is BorgState.DRAFT:
                    borg = ArchitectLoop(
                        repository,
                        borg,
                        store,
                        agent,
                        io=io,
                    ).run().borg
                borg = TechLeadLoop(
                    repository,
                    borg,
                    store,
                    agent,
                    architect_agent=agent,
                    io=io,
                ).run().borg
    except (ArchitectCancelled, TechLeadCancelled, KeyboardInterrupt) as error:
        message = str(error).strip()
        detail = f" ({message})" if message else ""
        action = "Plan change" if change_requested else "Planning"
        raise click.ClickException(
            f"{action} for Borg {name!r} was interrupted{detail}. "
            f"Run 'borg plan start {name}' to resume."
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        if resumable:
            message = str(error).strip()
            detail = f" ({message})" if message else ""
            action = "Plan change" if change_requested else "Planning"
            raise click.ClickException(
                f"{action} for Borg {name!r} could not continue{detail}. "
                f"Run 'borg plan start {name}' to resume."
            ) from error
        raise click.ClickException(str(error)) from error
    return borg


def _write_planning_gate(name: str, borg: Borg, *, changed: bool) -> None:
    """Report the actionable terminal gate reached by a planning lifecycle."""
    if borg.state is BorgState.PLAN_APPROVAL_PENDING:
        suffix = " after applying the change" if changed else ""
        click.echo(f"Plan approval pending for Borg {name!r}{suffix}.")
        click.echo(f"Review it with: borg plan show {name}")
    elif borg.state is BorgState.BLOCKED:
        suffix = " while applying the change" if changed else ""
        click.echo(f"Planning blocked for Borg {name!r}{suffix}.")
        click.echo(f"Review the saved Tech Lead findings with: borg plan show {name}")
    else:
        raise click.ClickException(
            f"Planning stopped in unexpected state {borg.state.value!r}. "
            f"Run 'borg plan start {name}' to resume."
        )


def _write_initialized(result) -> None:
    click.echo(
        f"Initialized repository {result.repository.id} "
        f"with score {result.analysis.overall_score:.2f}/5."
    )


def _validate_create_name(name: str) -> None:
    validate_borg_name(name)
    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None:
        raise ValueError(
            "Borg name must use kebab-case lowercase letters and numbers"
        )


def _interactive_io() -> InteractiveIO:
    return InteractiveIO(
        prompt=_prompt,
        confirm=lambda message, default: click.confirm(message, default=default),
        write=click.echo,
    )


def _prompt(message: str) -> str | None:
    try:
        return click.prompt(message, default="", show_default=False)
    except click.Abort:
        return None


def _edit_markdown(body: str) -> str | None:
    return click.edit(body, extension=".md")
