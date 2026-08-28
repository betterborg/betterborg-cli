"""Shared fixtures for BetterBorg CLI tests."""

import sqlite3
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner

from betterborg_cli.repo_analysis import DIMENSIONS
from betterborg_cli.store import (
    Borg,
    PrdSession,
    Repository,
    RepositoryAnalysis,
    RepositoryPackage,
    SqliteStore,
)


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


@pytest.fixture
def write_repository_config():
    """Return the shared repository-config writer for planning tests."""
    return _write_repository_config


@pytest.fixture
def persist_repository_analysis():
    """Return the shared analysis persistence helper for planning tests."""
    return _persist_repository_analysis


@pytest.fixture
def persist_planning_context():
    """Return the shared complete planning-context factory."""
    return _persist_planning_context


def _write_repository_config(root: Path, repository: Repository) -> None:
    (root / ".borg").mkdir()
    (root / ".borg/config.toml").write_text(
        "version = 1\n\n"
        "[repository]\n"
        f'id = "{repository.id}"\n'
        'default_branch = "main"\n',
        encoding="utf-8",
    )


def _persist_repository_analysis(
    store: SqliteStore, repository: Repository
) -> tuple[RepositoryAnalysis, list[RepositoryPackage]]:
    head_sha = subprocess.run(
        ["git", "-C", str(repository.root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    analysis = RepositoryAnalysis(
        repository_id=repository.id,
        head_sha=head_sha,
        summary="A compact test repository.",
        primary_language="Python",
        is_monorepo=False,
        overall_score=4.0,
        analysis_json={
            "packages": [{"path": "."}],
            "themes": [],
            "command_catalog": {"commands": []},
            "environment": {"files": []},
            "required_secrets": [],
            "service_dependencies": [],
        },
    )
    package = RepositoryPackage(
        repository_id=repository.id,
        analysis_id=analysis.id,
        package_path=".",
        package_name="test-repository",
        primary_language="Python",
        rubric={dimension: {"score": 4} for dimension in DIMENSIONS},
        overall_score=4.0,
    )
    packages = [package]
    store.append_analysis(analysis, packages)
    return analysis, packages


def _persist_planning_context(
    root: Path, store: SqliteStore, name: str
) -> tuple[Repository, Borg]:
    repository = Repository(root=root)
    borg = Borg(repository_id=repository.id, name=name)
    _write_repository_config(root, repository)
    prd_path = Path(".borg/prds") / f"{name}.md"
    (root / prd_path).parent.mkdir(parents=True)
    (root / prd_path).write_text(f"# {name}\n", encoding="utf-8")
    store.add_repository(repository)
    store.add_borg(borg)
    store.add_prd_session(
        PrdSession(
            repository_id=repository.id,
            borg_id=borg.id,
            prd_path=prd_path,
        )
    )
    _persist_repository_analysis(store, repository)
    return repository, borg
