"""Tests for the public CLI bootstrap."""

from pathlib import Path

from click.testing import CliRunner
from pytest import MonkeyPatch

from betterborg_cli import __version__
from betterborg_cli.cli import cli


def test_help_lists_bootstrap_commands(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "Work with BetterBorg" in result.output
    assert "version" in result.output


def test_version_does_not_initialize_repository(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = cli_runner.invoke(cli, ["version"])

    assert result.exit_code == 0
    assert result.output == f"borg {__version__}\n"
    assert list(tmp_path.iterdir()) == []
