"""Idempotent repository initialization through the public CLI."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from click.testing import CliRunner
from pytest import MonkeyPatch

from betterborg_cli import cli as cli_module
from betterborg_cli.agent_runtime.api_tools import ApiAgentRole
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.agent_runtime.selection import SelectedAgent
from betterborg_cli.cli import cli
from betterborg_cli.repo_analysis import DIMENSIONS, PROMPT_ROLES
from betterborg_cli.repo_paths import MANAGED_IGNORE_BEGIN, RepoPaths
from betterborg_cli.repository_config import load_repository_config
from betterborg_cli.store import SqliteStore


def _analysis_payload() -> dict[str, object]:
    return {
        "summary": "A small Python command-line application.",
        "primary_language": "python",
        "is_monorepo": False,
        "packages": [
            {
                "path": ".",
                "name": "root",
                "primary_language": "python",
                "rubric": {
                    dimension: {
                        "score": 3,
                        "evidence": f"README.md describes {dimension}",
                    }
                    for dimension in DIMENSIONS
                },
            }
        ],
        "recommendations": [
            {
                "id": "rec-ci",
                "title": "Document the CI checks",
                "package_path": ".",
                "dimension": "ci",
                "manifest_evidence": ["README.md"],
                "estimated_delta": 1,
                "effort": "S",
                "overlap_group": None,
            }
        ],
        "themes": [
            {
                "id": "theme-ci",
                "title": "Make validation visible",
                "recommendation_ids": ["rec-ci"],
                "effort": "S",
                "effort_rationale": "One documentation edit.",
            }
        ],
    }


def _adapter(repository: Path) -> tuple[MockAdapter, SelectedAgent]:
    adapter = MockAdapter(name="openai")
    adapter.queue(MockResponse(payload=_analysis_payload()))

    def prompt_response(spec):
        role = next(
            role for role in PROMPT_ROLES if f".{role}." in spec.result_path.name
        )
        return {
            "body_md": (
                f"# {role.title()} agent\n\n"
                f"Complete repository-specific {role} instructions."
            )
        }

    for _role in PROMPT_ROLES:
        adapter.queue(MockResponse(dynamic=prompt_response))
    return adapter, SelectedAgent(
        role=ApiAgentRole.ANALYSIS,
        adapter=adapter,
        paths=RepoPaths.discover(repository),
    )


def test_init_creates_outputs_once_and_preserves_repository_identity(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    git_repo = committed_git_repo
    paths = RepoPaths.discover(git_repo)
    adapter, selected = _adapter(git_repo)
    selections = 0

    def select_mock(*_args, **_kwargs):
        nonlocal selections
        selections += 1
        return selected

    monkeypatch.chdir(git_repo)
    monkeypatch.setenv("XDG_STATE_HOME", str(git_repo.parent / "machine-state"))
    monkeypatch.setattr(cli_module, "select_agent", select_mock)

    first = cli_runner.invoke(cli, ["init", "--yes"])

    assert first.exit_code == 0, first.output
    config = load_repository_config(paths)
    first_ignore = paths.gitignore.read_text(encoding="utf-8")
    assert "Initialized repository" in first.output
    assert str(config.repository_id) in first.output
    assert "Score: 3.00/5 (estimated)" in paths.score_report.read_text(
        encoding="utf-8"
    )
    for role in PROMPT_ROLES:
        prompt = paths.prompts_dir / f"{role}.system.md"
        assert prompt.is_file()
        assert f"# {role.title()} agent" in prompt.read_text(encoding="utf-8")
    improvement = paths.improvement_prds_dir / "theme-ci.md"
    assert improvement.is_file()
    assert "Theme Ci" in improvement.read_text(encoding="utf-8")
    assert len(adapter.calls) == 4
    assert selections == 1

    second = cli_runner.invoke(cli, ["init", "--yes"])

    assert second.exit_code == 0, second.output
    assert second.output == f"Repository already initialized: {config.repository_id}\n"
    assert paths.gitignore.read_text(encoding="utf-8") == first_ignore
    assert first_ignore.count(MANAGED_IGNORE_BEGIN) == 1
    assert len(adapter.calls) == 4
    assert selections == 1

    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        analyses = store.list_analyses(config.repository_id)
        operations = store.list_operations(config.repository_id)
        assert len(analyses) == 1
        assert [operation.kind for operation in operations] == [
            "repository.initialized"
        ]
        assert store.get_repository(config.repository_id).root == git_repo

    with sqlite3.connect(paths.state_dir / "borg.sqlite3") as connection:
        repository_count = connection.execute(
            "SELECT COUNT(*) FROM repositories"
        ).fetchone()[0]
        assert repository_count == 1
