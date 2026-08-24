"""Command-line entry point for BetterBorg."""

import json
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
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import load_repository_config
from betterborg_cli.repository_service import RepositoryService
from betterborg_cli.store import SqliteStore
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


@cli.command(name="create")
@click.option(
    "--name",
    required=True,
    help="Name for the Borg and its confirmed PRD.",
)
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
    elif result.questions:
        click.echo("Borg PRD needs more input before it can be created.")
    else:
        click.echo("Borg draft saved without a confirmed PRD.")


def _write_initialized(result) -> None:
    click.echo(
        f"Initialized repository {result.repository.id} "
        f"with score {result.analysis.overall_score:.2f}/5."
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
