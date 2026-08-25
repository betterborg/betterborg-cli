"""Command-line entry point for BetterBorg."""

import json
import re
import shlex
from functools import wraps
from pathlib import Path

import click

from betterborg_cli import __version__
from betterborg_cli.agent_runtime.api_tools import ApiAgentRole
from betterborg_cli.agent_runtime.selection import select_agent
from betterborg_cli.onboarding import (
    CreateService,
    OnboardingDispatcher,
    create_commands,
)
from betterborg_cli.planning import (
    ArchitectCancelled,
    ArchitectLoop,
    TechLeadCancelled,
    TechLeadLoop,
    render_plan_markdown,
    validate_plan,
)
from betterborg_cli.prd_session import InteractiveIO, validate_borg_name
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import load_repository_config
from betterborg_cli.repository_service import RepositoryService
from betterborg_cli.store import BorgState, PlanningAttemptStatus, SqliteStore
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
    def guarded(*args, explicit_trust: bool, **kwargs):
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

            if borg.state not in {
                BorgState.PLAN_APPROVAL_PENDING,
                BorgState.BLOCKED,
            }:
                interactive = _stdin_is_interactive()
                agent = select_agent(
                    config,
                    ApiAgentRole.PLANNING,
                    paths,
                    interactive=interactive,
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
        raise click.ClickException(
            f"Planning for Borg {name!r} was interrupted{detail}. "
            f"Run 'borg plan start {name}' to resume."
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    if borg.state is BorgState.PLAN_APPROVAL_PENDING:
        click.echo(f"Plan approval pending for Borg {name!r}.")
        click.echo(f"Review it with: borg plan show {name}")
    elif borg.state is BorgState.BLOCKED:
        click.echo(f"Planning blocked for Borg {name!r}.")
        click.echo(f"Review the saved Tech Lead findings with: borg plan show {name}")
    else:
        raise click.ClickException(
            f"Planning stopped in unexpected state {borg.state.value!r}"
        )


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
                    f"Borg {name!r} does not have a stored plan; "
                    f"run 'borg plan start {name}' first"
                )
            stored_plan = attempt.result
            validate_plan(
                stored_plan,
                paths.root,
                check_repository_state=False,
            )
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    if json_output:
        click.echo(json.dumps(stored_plan, sort_keys=True, separators=(",", ":")))
    else:
        click.echo(render_plan_markdown(stored_plan), nl=False)


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
