"""Idempotent repository initialization through the public CLI."""

from __future__ import annotations

import json
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
from betterborg_cli.store import Borg, PrdSession, SqliteStore


def _analysis_payload(
    *,
    score: int = 3,
    recommendation_title: str = "Document the CI checks",
    theme_title: str = "Make validation visible",
) -> dict[str, object]:
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
                        "score": score,
                        "evidence": f"README.md describes {dimension}",
                    }
                    for dimension in DIMENSIONS
                },
            }
        ],
        "recommendations": [
            {
                "id": "rec-ci",
                "title": recommendation_title,
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
                "title": theme_title,
                "recommendation_ids": ["rec-ci"],
                "effort": "S",
                "effort_rationale": "One documentation edit.",
            }
        ],
    }


def _queue_analysis(
    adapter: MockAdapter,
    payload: dict[str, object],
    *,
    prompt_prefix: str = "",
) -> None:
    adapter.queue(MockResponse(payload=payload))

    def prompt_response(spec):
        role = next(
            role for role in PROMPT_ROLES if f".{role}." in spec.result_path.name
        )
        return {
            "body_md": (
                f"# {prompt_prefix}{role.title()} agent\n\n"
                f"Complete repository-specific {role} instructions."
            )
        }

    for _role in PROMPT_ROLES:
        adapter.queue(MockResponse(dynamic=prompt_response))


def _adapter(repository: Path) -> tuple[MockAdapter, SelectedAgent]:
    adapter = MockAdapter(name="openai")
    _queue_analysis(adapter, _analysis_payload())
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
    assert (
        "borg create --name 'Theme Ci' --prd "
        ".borg/prds/improvements/theme-ci.md\n"
    ) in first.output

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
        assert store.get_borg_by_name(config.repository_id, "Theme Ci") is None

    with sqlite3.connect(paths.state_dir / "borg.sqlite3") as connection:
        repository_count = connection.execute(
            "SELECT COUNT(*) FROM repositories"
        ).fetchone()[0]
        assert repository_count == 1


def test_json_init_never_prompts_and_emits_exact_create_commands(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    git_repo = committed_git_repo
    paths = RepoPaths.discover(git_repo)
    adapter, selected = _adapter(git_repo)
    selected_interactivity: list[bool] = []

    def select_mock(*_args, **kwargs):
        selected_interactivity.append(kwargs["interactive"])
        return selected

    monkeypatch.chdir(git_repo)
    monkeypatch.setenv("XDG_STATE_HOME", str(git_repo.parent / "machine-state"))
    monkeypatch.setattr(cli_module, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr(cli_module, "select_agent", select_mock)

    result = cli_runner.invoke(cli, ["init", "--yes", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "create_commands": [
            "borg create --name 'Theme Ci' --prd "
            ".borg/prds/improvements/theme-ci.md"
        ],
        "initialized": True,
        "repository_id": str(load_repository_config(paths).repository_id),
        "score": 3.0,
    }
    assert selected_interactivity == [False]
    assert paths.improvement_prds_dir.joinpath("theme-ci.md").is_file()
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        assert (
            store.get_borg_by_name(
                load_repository_config(paths).repository_id, "Theme Ci"
            )
            is None
        )


def test_first_interactive_init_presents_doors_and_creates_selected_theme(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    git_repo = committed_git_repo
    paths = RepoPaths.discover(git_repo)
    adapter, selected = _adapter(git_repo)
    adapter.queue(
        MockResponse(
            payload={
                "questions": [],
                "prd_markdown": "# CI Borg\n\nMake repository validation visible.",
            }
        )
    )
    selections: list[ApiAgentRole] = []

    def select_mock(_config, role, _paths, **_kwargs):
        selections.append(role)
        return selected

    monkeypatch.chdir(git_repo)
    monkeypatch.setenv("XDG_STATE_HOME", str(git_repo.parent / "machine-state"))
    monkeypatch.setattr(cli_module, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr(cli_module, "select_agent", select_mock)

    result = cli_runner.invoke(
        cli,
        ["init", "--yes"],
        input="1\n\n\nn\ny\n",
    )

    assert result.exit_code == 0, result.output
    assert "Initialized repository" in result.output
    assert "1. Fix the repo" in result.output
    assert "2. Improve an existing PRD" in result.output
    assert "3. Brainstorm a new PRD" in result.output
    assert "Make validation visible — predicted impact +0.125; effort S" in (
        result.output
    )
    generated = paths.improvement_prds_dir / "theme-ci.md"
    confirmed = paths.tracked_dir / "prds" / "Theme Ci.md"
    assert generated.is_file()
    assert confirmed.read_text(encoding="utf-8") == (
        "# CI Borg\n\nMake repository validation visible.\n"
    )
    assert generated.read_text(encoding="utf-8").startswith(
        "# Make validation visible\n"
    )
    assert selections == [ApiAgentRole.ANALYSIS, ApiAgentRole.PLANNING]
    assert len(adapter.calls) == 5


def test_analyze_appends_history_and_refreshes_generated_outputs(
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

    initialized = cli_runner.invoke(cli, ["init", "--yes"])

    assert initialized.exit_code == 0, initialized.output
    config = load_repository_config(paths)
    confirmed_path = paths.tracked_dir / "prds" / "Confirmed.md"
    confirmed_body = "# Confirmed\n\nKeep this approved product requirement.\n"
    confirmed_path.write_text(confirmed_body, encoding="utf-8")
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = Borg(repository_id=config.repository_id, name="Confirmed")
        session = PrdSession(
            repository_id=config.repository_id,
            borg_id=borg.id,
            prd_path=Path(".borg/prds/Confirmed.md"),
        )
        store.add_borg(borg)
        store.add_prd_session(session)

    _queue_analysis(
        adapter,
        _analysis_payload(
            score=4,
            recommendation_title="Automate the CI checks",
            theme_title="Automate validation",
        ),
        prompt_prefix="Refreshed ",
    )
    result = cli_runner.invoke(cli, ["analyze"])

    assert result.exit_code == 0, result.output
    assert result.output == (
        f"Analyzed repository {config.repository_id}: score 4.00/5 "
        "(previous 3.00/5, delta +1.00).\n"
    )
    assert load_repository_config(paths).repository_id == config.repository_id
    assert confirmed_path.read_text(encoding="utf-8") == confirmed_body
    assert paths.improvement_prds_dir.joinpath("theme-ci.md").read_text(
        encoding="utf-8"
    ).startswith("# Automate validation\n")
    score_report = paths.score_report.read_text(encoding="utf-8")
    assert "Score: 4.00/5 (estimated)" in score_report
    assert "Previous: 3.00" in score_report
    assert "Delta: +1.00" in score_report
    assert selections == 2
    assert len(adapter.calls) == 8

    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        analyses = store.list_analyses(config.repository_id)
        assert len(analyses) == 2
        assert analyses[1].prior_analysis_id == analyses[0].id
        assert analyses[1].score_delta == 1
        latest_prompts = store.get_latest_generated_prompts(config.repository_id)
        assert set(latest_prompts) == set(PROMPT_ROLES)
        assert all(prompt.version == 2 for prompt in latest_prompts.values())
        assert all(
            prompt.analysis_id == analyses[1].id
            for prompt in latest_prompts.values()
        )
        assert all(
            prompt.body_md.startswith("# Refreshed ")
            for prompt in latest_prompts.values()
        )
        assert store.get_repository(config.repository_id).id == config.repository_id
        assert store.get_borg(borg.id) == borg
        assert store.get_prd_session(session.id) == session

    with sqlite3.connect(paths.state_dir / "borg.sqlite3") as connection:
        repository_count = connection.execute(
            "SELECT COUNT(*) FROM repositories"
        ).fetchone()[0]
        assert repository_count == 1


def test_json_analyze_emits_current_previous_and_delta_without_prompting(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    git_repo = committed_git_repo
    paths = RepoPaths.discover(git_repo)
    adapter, selected = _adapter(git_repo)
    selected_interactivity: list[bool] = []

    def select_mock(*_args, **kwargs):
        selected_interactivity.append(kwargs["interactive"])
        return selected

    monkeypatch.chdir(git_repo)
    monkeypatch.setenv("XDG_STATE_HOME", str(git_repo.parent / "machine-state"))
    monkeypatch.setattr(cli_module, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr(cli_module, "select_agent", select_mock)

    initialized = cli_runner.invoke(cli, ["init", "--yes", "--json"])
    assert initialized.exit_code == 0, initialized.output

    _queue_analysis(adapter, _analysis_payload(score=2), prompt_prefix="New ")
    result = cli_runner.invoke(cli, ["analyze", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
        "analysis_id": payload["analysis_id"],
        "delta": -1.0,
        "previous_score": 3.0,
        "repository_id": str(load_repository_config(paths).repository_id),
        "score": 2.0,
    }
    assert selected_interactivity == [False, False]


def test_analyze_rejects_an_uninitialized_repository(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    git_repo = committed_git_repo
    paths = RepoPaths.discover(git_repo)
    monkeypatch.chdir(git_repo)
    monkeypatch.setenv("XDG_STATE_HOME", str(git_repo.parent / "machine-state"))

    result = cli_runner.invoke(cli, ["analyze", "--yes"])

    assert result.exit_code == 1
    assert "repository is not initialized; run 'borg init' first" in result.output
    assert not paths.tracked_dir.joinpath("config.toml").exists()
