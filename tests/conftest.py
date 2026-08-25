"""Shared fixtures for BetterBorg CLI tests."""

import shutil
import sqlite3
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner
from pytest import MonkeyPatch

from betterborg_cli import cli as cli_module
from betterborg_cli.agent_runtime.mock import MockAdapter
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.repo_analysis import DIMENSIONS
from betterborg_cli.repo_paths import RepoPaths
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


@pytest.fixture
def planning_cli_repository(persist_planning_context):
    """Return a factory for a planning repository with CLI-visible state."""

    def create(root: Path, name: str):
        paths = RepoPaths.discover(root)
        fixture_database = root.parent / f"{name}.sqlite3"
        with SqliteStore.open(fixture_database) as store:
            repository, _borg = persist_planning_context(root, store, name)
        paths.state_dir.mkdir(parents=True)
        shutil.copyfile(fixture_database, paths.state_dir / "borg.sqlite3")
        return repository, paths

    return create


@pytest.fixture
def planning_plan_response():
    """Build a valid plan response shared by planning lifecycle tests."""
    return _planning_plan_response


@pytest.fixture
def tech_lead_approval_response():
    """Build an approving Tech Lead response."""
    return _tech_lead_approval_response


@pytest.fixture
def tech_lead_change_request_response():
    """Build a Tech Lead response that requests a plan revision."""
    return _tech_lead_change_request_response


@pytest.fixture
def configure_interactive_cli(monkeypatch: MonkeyPatch):
    """Return the shared interactive CLI dependency configurator."""

    def configure(
        root: Path,
        adapter: MockAdapter,
        io: InteractiveIO,
        *,
        state_home: Path,
    ) -> None:
        monkeypatch.chdir(root)
        monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
        monkeypatch.setattr(cli_module, "_stdin_is_interactive", lambda: True)
        monkeypatch.setattr(
            cli_module, "select_agent", lambda *_args, **_kwargs: adapter
        )
        monkeypatch.setattr(cli_module, "_interactive_io", lambda: io)

    return configure


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


def _planning_plan_response(
    *, summary: str = "Add a small, tested release workflow."
) -> dict:
    return {
        "title": "Release workflow",
        "summary": summary,
        "overall_approach": (
            "Extend the existing repository conventions and verify public behavior."
        ),
        "phases": [
            {
                "name": "01-release-workflow",
                "title": "Add release workflow",
                "goal": "Document and test the release path.",
                "technical_approach": "Update the tracked README convention.",
                "files_touched": [
                    {
                        "path": "README.md",
                        "role": "modified",
                        "description": "Document the release workflow.",
                    }
                ],
                "test_strategy": "Assert the documented public workflow.",
                "acceptance_criteria": ["The release path is documented."],
                "deliverables": ["Release workflow documentation."],
                "dependencies_on": [],
            }
        ],
        "code_pointers": [
            {"path": "README.md", "why": "It owns repository guidance."}
        ],
        "risks": [],
        "open_questions": [],
    }


def _tech_lead_approval_response() -> dict:
    return {
        "decision": "approve",
        "summary": "The plan is ready for human approval.",
        "findings": [],
    }


def _tech_lead_change_request_response(message: str) -> dict:
    return {
        "decision": "request_changes",
        "summary": message,
        "findings": [
            {
                "severity": "major",
                "message": message,
                "suggestion": "Clarify the plan and its verification.",
            }
        ],
    }
