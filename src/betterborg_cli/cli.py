"""Command-line entry point for BetterBorg."""

import click

from betterborg_cli import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Work with BetterBorg from the command line."""


@cli.command()
def version() -> None:
    """Print the installed BetterBorg CLI version."""
    click.echo(f"borg {__version__}")
