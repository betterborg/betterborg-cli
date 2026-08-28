"""Command-line entry point for BetterBorg."""

from functools import wraps
from pathlib import Path

import click

from betterborg_cli import __version__
from betterborg_cli.agent_runtime.api_tools import ApiAgentRole
from betterborg_cli.agent_runtime.selection import select_agent
from betterborg_cli.repo_paths import RepoPaths
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
            require_workspace_trust(
                paths,
                explicit=explicit_trust,
                interactive=_stdin_is_interactive(),
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
    "--yes",
    "explicit_trust",
    is_flag=True,
    help="Trust this workspace without prompting and initialize it.",
)
@_trusted_workspace_callback
def initialize_repository(repository_path: Path) -> None:
    """Register and analyze the current Git repository."""
    paths = RepoPaths.discover(repository_path)
    database = paths.state_dir / "borg.sqlite3"
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
                    interactive=_stdin_is_interactive(),
                ),
            )
            result = service.initialize()
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    if result.initialized:
        click.echo(
            f"Initialized repository {result.repository.id} "
            f"with score {result.analysis.overall_score:.2f}/5."
        )
    else:
        click.echo(f"Repository already initialized: {result.repository.id}")
