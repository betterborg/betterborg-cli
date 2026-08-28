"""Shared fixtures for BetterBorg CLI tests."""

import sqlite3
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner


@pytest.fixture
def cli_runner() -> CliRunner:
    """Return Click's isolated command-line test runner."""
    return CliRunner()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create an initialized temporary Git repository."""
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "BetterBorg Tests"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "config",
            "user.email",
            "tests@betterborg.dev",
        ],
        check=True,
    )
    return tmp_path


@pytest.fixture
def committed_git_repo(git_repo: Path) -> Path:
    """Create a temporary Git repository with one tracked commit."""
    (git_repo / "README.md").write_text(
        "# Test repository\n\nBuild and test docs.\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(git_repo), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "commit", "--quiet", "-m", "initial"],
        check=True,
    )
    return git_repo


@pytest.fixture
def sqlite_db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a temporary SQLite database and close it after the test."""
    connection = sqlite3.connect(tmp_path / "test.sqlite3")
    try:
        yield connection
    finally:
        connection.close()
