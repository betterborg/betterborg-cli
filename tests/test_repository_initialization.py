"""Idempotent repository initialization through the public CLI."""

from __future__ import annotations

import json
import os
import re
import signal
import sqlite3
import subprocess
import threading
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from pytest import MonkeyPatch

from betterborg_cli import cli as cli_module
from betterborg_cli import repository_files as repository_files_module
from betterborg_cli import repository_service as repository_service_module
from betterborg_cli.agent_runtime import CancellationToken, run_captured
from betterborg_cli.agent_runtime.api_tools import ApiAgentRole
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.agent_runtime.selection import SelectedAgent
from betterborg_cli.cli import cli
from betterborg_cli.progress import RunProgress, StageState
from betterborg_cli.repo_analysis import DIMENSIONS, PROMPT_ROLES
from betterborg_cli.repo_analysis import analyzer as analyzer_module
from betterborg_cli.repo_analysis import improvement_prds as improvement_prds_module
from betterborg_cli.repo_analysis import prompts_manager as prompts_manager_module
from betterborg_cli.repo_paths import MANAGED_IGNORE_BEGIN, RepoPaths
from betterborg_cli.repository_config import (
    CONFIG_FILENAME,
    AgentChoice,
    AgentStage,
    load_repository_config,
)
from betterborg_cli.repository_service import (
    RepositoryInitializationError,
    RepositoryService,
    _default_branch,
)
from betterborg_cli.run_control import RunControl
from betterborg_cli.store import (
    Borg,
    PrdSession,
    Repository,
    RepositoryAnalysis,
    RepositoryPackage,
    SqliteStore,
)


def _analysis_payload(
    *,
    score: int = 3,
    recommendation_title: str = "Document the CI checks",
    theme_id: str = "theme-ci",
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
                "id": theme_id,
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


def _selected_agent(
    repository: Path,
    *,
    adapter: str,
    model: str,
    effort: str = "high",
) -> SelectedAgent:
    return SelectedAgent(
        role=ApiAgentRole.ANALYSIS,
        adapter=MockAdapter(name=adapter),
        paths=RepoPaths.discover(repository),
        model=model,
        effort=effort,
    )


def test_initial_config_pins_and_reuses_the_bootstrap_selection(
    committed_git_repo: Path,
) -> None:
    paths = RepoPaths.discover(committed_git_repo)
    selected = _selected_agent(
        committed_git_repo,
        adapter="codex",
        model="gpt-5.6-sol",
    )
    factory_configs = []

    def agent_factory(config):
        factory_configs.append(config)
        return selected

    with SqliteStore.open(paths.state_dir / "pinned-config.sqlite3") as store:
        service = RepositoryService(paths, store, agent_factory)
        repository, config, bootstrap_agent = service._ensure_repository()

        assert bootstrap_agent is selected
        assert repository.id == config.repository_id
        assert len(factory_configs) == 1
        assert factory_configs[0].agents.resolve(AgentStage.ANALYSIS).adapter is None
        for stage in AgentStage:
            assert getattr(config.agents, stage.value) == AgentChoice()
            assert config.agents.resolve(stage) == config.agents.defaults

        body = paths.tracked_dir.joinpath(CONFIG_FILENAME).read_text(
            encoding="utf-8"
        )
        assert [line for line in body.splitlines() if line.startswith("[")] == [
            "[repository]",
            "[agents.defaults]",
            *(f"[agents.{stage.value}]" for stage in AgentStage),
        ]
        assert config.agents.defaults.adapter == "codex"
        assert config.agents.defaults.model == "gpt-5.6-sol"
        assert config.agents.defaults.effort == "high"
        assert "credential" not in body.casefold()
        assert "token" not in body.casefold()
        assert "secret" not in body.casefold()

        same_repository, same_config, second_bootstrap = service._ensure_repository()

        assert same_repository == repository
        assert same_config == config
        assert second_bootstrap is None
        assert len(factory_configs) == 1
        assert paths.tracked_dir.joinpath(CONFIG_FILENAME).read_text(
            encoding="utf-8"
        ) == body


def test_init_creates_outputs_once_and_preserves_repository_identity(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    git_repo = committed_git_repo
    paths = RepoPaths.discover(git_repo)
    adapter, selected = _adapter(git_repo)
    selections: list[AgentStage] = []

    def select_mock(_config, stage, _paths, **_kwargs):
        selections.append(stage)
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
    assert "theme-ci" in improvement.read_text(encoding="utf-8")
    assert len(adapter.calls) == 4
    assert selections == [AgentStage.ANALYSIS]
    assert (
        "betterborg create theme-ci --prd "
        ".betterborg/prds/improvements/theme-ci.md\n"
    ) in first.output
    prompts_complete = "✔ Generate role prompts"
    drafts_started = "⠋ Draft improvement PRDs"
    drafts_complete = "✔ Draft improvement PRDs"
    assert first.output.count(drafts_complete) == 1
    assert first.output.index(prompts_complete) < first.output.index(drafts_started)
    assert first.output.index(drafts_started) < first.output.index(drafts_complete)

    second = cli_runner.invoke(cli, ["init", "--yes"])

    assert second.exit_code == 0, second.output
    assert "✔ Discover evidence" in second.output
    assert "✔ Analyze repository" in second.output
    assert second.output.endswith(
        f"Repository already initialized: {config.repository_id}\n"
    )
    assert paths.gitignore.read_text(encoding="utf-8") == first_ignore
    assert first_ignore.count(MANAGED_IGNORE_BEGIN) == 1
    assert len(adapter.calls) == 4
    assert selections == [AgentStage.ANALYSIS]

    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        analyses = store.list_analyses(config.repository_id)
        operations = store.list_operations(config.repository_id)
        assert len(analyses) == 1
        assert [operation.kind for operation in operations] == [
            "repository.initialized"
        ]
        assert store.get_repository(config.repository_id).root == git_repo
        assert store.get_borg_by_name(config.repository_id, "theme-ci") is None

    with sqlite3.connect(paths.state_dir / "betterborg.sqlite3") as connection:
        repository_count = connection.execute(
            "SELECT COUNT(*) FROM repositories"
        ).fetchone()[0]
        assert repository_count == 1


@pytest.mark.parametrize("interrupt_after_claim", [False, True])
def test_initial_config_interruption_leaves_no_file_or_complete_parseable_config(
    committed_git_repo: Path,
    monkeypatch: MonkeyPatch,
    interrupt_after_claim: bool,
) -> None:
    paths = RepoPaths.discover(committed_git_repo)
    repository = Repository(root=paths.root)
    config_path = paths.tracked_dir / CONFIG_FILENAME
    original_link = os.link

    def interrupt(source: Path, destination: Path) -> None:
        if interrupt_after_claim:
            original_link(source, destination)
        raise KeyboardInterrupt("config publication interrupted")

    monkeypatch.setattr(repository_files_module.os, "link", interrupt)

    with SqliteStore.open(paths.state_dir / "atomic-config.sqlite3") as store:
        service = RepositoryService(
            paths,
            store,
            lambda _config: MockAdapter(name="openai"),
        )
        with pytest.raises(KeyboardInterrupt, match="config publication interrupted"):
            service._write_initial_config(repository)

    if interrupt_after_claim:
        config = load_repository_config(paths)
        assert config.repository_id == repository.id
        assert config.agents.defaults.adapter == "openai"
        assert config.agents.defaults.model == "gpt-5.6-sol"
        assert config.agents.defaults.effort == "high"
        assert all(
            config.agents.resolve(stage) == config.agents.defaults
            for stage in AgentStage
        )
        assert config_path.read_text(encoding="utf-8").endswith("\n")
    else:
        assert not config_path.exists()
    assert list(paths.tracked_dir.glob(".config.toml.*.tmp")) == []


def test_initial_config_create_race_preserves_the_winning_repository_identity(
    committed_git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    paths = RepoPaths.discover(committed_git_repo)
    winning_repository = Repository(root=paths.root)
    config_path = paths.tracked_dir / CONFIG_FILENAME
    winning_body = (
        "version = 1\n\n"
        "[repository]\n"
        f'id = "{winning_repository.id}"\n'
        'default_branch = "winning-branch"\n\n'
        "[agents.defaults]\n"
        'adapter = "codex"\n'
        'model = "gpt-5.6-sol"\n'
        'effort = "high"\n\n'
        + "\n\n".join(f"[agents.{stage.value}]" for stage in AgentStage)
        + "\n"
    )
    unrelated_temporary = paths.tracked_dir / ".config.toml.concurrent.tmp"
    original_link = os.link

    def publish_winner_then_lose(source: Path, destination: Path) -> None:
        unrelated_temporary.write_text(
            "owned by another initializer\n",
            encoding="utf-8",
        )
        winner_temporary = paths.tracked_dir / ".config.toml.winner.tmp"
        winner_temporary.write_text(winning_body, encoding="utf-8")
        original_link(winner_temporary, destination)
        winner_temporary.unlink()
        original_link(source, destination)

    monkeypatch.setattr(
        repository_files_module.os,
        "link",
        publish_winner_then_lose,
    )

    losing_selection = _selected_agent(
        committed_git_repo,
        adapter="openai",
        model="gpt-5.6-sol",
    )
    winning_selection = _selected_agent(
        committed_git_repo,
        adapter="codex",
        model="gpt-5.6-sol",
    )
    factory_configs = []

    def agent_factory(config):
        factory_configs.append(config)
        if config.agents.resolve(AgentStage.ANALYSIS).adapter == "codex":
            return winning_selection
        return losing_selection

    with SqliteStore.open(paths.state_dir / "config-race.sqlite3") as store:
        service = RepositoryService(paths, store, agent_factory)
        first_repository, first_config, bootstrap_agent = (
            service._ensure_repository()
        )
        second_repository, second_config, second_bootstrap = (
            service._ensure_repository()
        )

        assert first_repository.id == winning_repository.id
        assert second_repository.id == winning_repository.id
        assert first_config.repository_id == winning_repository.id
        assert second_config.repository_id == winning_repository.id
        assert store.get_repository(winning_repository.id) == first_repository
        assert bootstrap_agent is winning_selection
        assert second_bootstrap is None
        assert len(factory_configs) == 2
        assert factory_configs[0].repository_id != winning_repository.id
        assert factory_configs[0].agents.defaults.adapter is None
        assert factory_configs[1] == first_config

    assert config_path.read_text(encoding="utf-8") == winning_body
    assert unrelated_temporary.read_text(encoding="utf-8") == (
        "owned by another initializer\n"
    )
    assert list(paths.tracked_dir.glob(".config.toml.*.tmp")) == [
        unrelated_temporary
    ]


def test_initial_config_rejects_a_tracked_directory_outside_the_repository(
    committed_git_repo: Path,
) -> None:
    paths = RepoPaths.discover(committed_git_repo)
    outside = committed_git_repo.parent / f"{committed_git_repo.name}-outside"
    outside.mkdir()
    paths.tracked_dir.symlink_to(outside, target_is_directory=True)

    with SqliteStore.open(outside / "containment.sqlite3") as store:
        service = RepositoryService(
            paths,
            store,
            lambda _config: MockAdapter(name="openai"),
        )
        with pytest.raises(RepositoryInitializationError, match="escapes repository"):
            service._write_initial_config(Repository(root=paths.root))

    assert not outside.joinpath(CONFIG_FILENAME).exists()


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
    expected = {
        "create_commands": [
            "betterborg create theme-ci --prd "
            ".betterborg/prds/improvements/theme-ci.md"
        ],
        "initialized": True,
        "repository_id": str(load_repository_config(paths).repository_id),
        "score": 3.0,
    }
    assert json.loads(result.output) == expected
    assert result.output == json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ) + "\n"
    assert selected_interactivity == [False]
    assert paths.improvement_prds_dir.joinpath("theme-ci.md").is_file()
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        assert (
            store.get_borg_by_name(
                load_repository_config(paths).repository_id, "theme-ci"
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
    selections: list[AgentStage] = []

    def select_mock(_config, stage, _paths, **_kwargs):
        selections.append(stage)
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
    confirmed = paths.tracked_dir / "prds" / "theme-ci.md"
    assert generated.is_file()
    assert confirmed.read_text(encoding="utf-8") == (
        "# CI Borg\n\nMake repository validation visible.\n"
    )
    assert generated.read_text(encoding="utf-8").startswith(
        "# Make validation visible\n"
    )
    assert selections == [AgentStage.ANALYSIS, AgentStage.REQUIREMENTS]
    assert len(adapter.calls) == 5


def test_analyze_appends_history_and_refreshes_generated_outputs(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    git_repo = committed_git_repo
    paths = RepoPaths.discover(git_repo)
    adapter, selected = _adapter(git_repo)
    selections: list[AgentStage] = []

    def select_mock(_config, stage, _paths, **_kwargs):
        selections.append(stage)
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
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        borg = Borg(repository_id=config.repository_id, name="Confirmed")
        session = PrdSession(
            repository_id=config.repository_id,
            borg_id=borg.id,
            prd_path=Path(".betterborg/prds/Confirmed.md"),
        )
        store.add_borg(borg)
        store.add_prd_session(session)

    _queue_analysis(
        adapter,
        _analysis_payload(
            score=4,
            recommendation_title="Automate the CI checks",
            theme_id="theme-automation",
            theme_title="Automate validation",
        ),
        prompt_prefix="Refreshed ",
    )
    result = cli_runner.invoke(cli, ["analyze"])

    assert result.exit_code == 0, result.output
    assert result.output.endswith(
        f"Analyzed repository {config.repository_id}: score 4.00/5 "
        "(previous 3.00/5, delta +1.00).\n"
    )
    assert "✔ Discover evidence" in result.output
    assert "1 evidence files" in result.output
    assert "✔ Analyze repository" in result.output
    assert "score 4.00/5" in result.output
    assert load_repository_config(paths).repository_id == config.repository_id
    assert confirmed_path.read_text(encoding="utf-8") == confirmed_body
    assert not paths.improvement_prds_dir.joinpath("theme-ci.md").exists()
    assert paths.improvement_prds_dir.joinpath("theme-automation.md").read_text(
        encoding="utf-8"
    ).startswith("# Automate validation\n")
    score_report = paths.score_report.read_text(encoding="utf-8")
    assert "Score: 4.00/5 (estimated)" in score_report
    assert "Previous: 3.00" in score_report
    assert "Delta: +1.00" in score_report
    assert selections == [AgentStage.ANALYSIS, AgentStage.ANALYSIS]
    assert len(adapter.calls) == 8

    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
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

    with sqlite3.connect(paths.state_dir / "betterborg.sqlite3") as connection:
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
    expected = {
        "analysis_id": payload["analysis_id"],
        "delta": -1.0,
        "previous_score": 3.0,
        "repository_id": str(load_repository_config(paths).repository_id),
        "score": 2.0,
    }
    assert payload == expected
    assert result.output == json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ) + "\n"
    assert selected_interactivity == [False, False]


def test_analyze_reports_the_predecessor_linked_during_persistence(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    git_repo = committed_git_repo
    paths = RepoPaths.discover(git_repo)
    adapter, selected = _adapter(git_repo)
    monkeypatch.chdir(git_repo)
    monkeypatch.setenv("XDG_STATE_HOME", str(git_repo.parent / "machine-state"))
    monkeypatch.setattr(cli_module, "select_agent", lambda *_args, **_kwargs: selected)

    initialized = cli_runner.invoke(cli, ["init", "--yes", "--json"])
    assert initialized.exit_code == 0, initialized.output
    config = load_repository_config(paths)

    def persist_intervening_analysis(_spec) -> dict[str, object]:
        payload = _analysis_payload(score=4)
        package_payload = payload["packages"][0]
        assert isinstance(package_payload, dict)
        with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as other_store:
            prior = other_store.get_prior_ready_analysis(config.repository_id)
            assert prior is not None
            analysis = RepositoryAnalysis(
                repository_id=config.repository_id,
                head_sha=prior.head_sha,
                summary="An intervening concurrent analysis.",
                primary_language="python",
                is_monorepo=False,
                overall_score=4,
                analysis_json=payload,
                prior_analysis_id=prior.id,
                score_delta=1,
            )
            package = RepositoryPackage(
                repository_id=config.repository_id,
                analysis_id=analysis.id,
                package_path=".",
                package_name="root",
                primary_language="python",
                rubric=package_payload["rubric"],
                overall_score=4,
            )
            other_store.append_analysis(analysis, [package])
        return _analysis_payload(score=2)

    adapter.queue(MockResponse(dynamic=persist_intervening_analysis))

    def prompt_response(spec):
        role = next(
            role for role in PROMPT_ROLES if f".{role}." in spec.result_path.name
        )
        return {
            "body_md": (
                f"# Concurrent {role.title()} agent\n\n"
                f"Complete repository-specific {role} instructions."
            )
        }

    for _role in PROMPT_ROLES:
        adapter.queue(MockResponse(dynamic=prompt_response))

    result = cli_runner.invoke(cli, ["analyze", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["score"] == 2
    assert payload["previous_score"] == 4
    assert payload["delta"] == -2
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        current = next(
            analysis
            for analysis in store.list_analyses(config.repository_id)
            if str(analysis.id) == payload["analysis_id"]
        )
        previous = store.get_prior_ready_analysis(
            config.repository_id,
            before_analysis_id=current.id,
        )
        assert previous is not None
        assert previous.overall_score == payload["previous_score"]
        assert current.score_delta == payload["delta"]


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
    assert "repository is not initialized; run 'betterborg init' first" in result.output
    assert not paths.tracked_dir.joinpath("config.toml").exists()


def test_default_branch_cancellation_reaps_process_tree_and_starts_no_later_work(
    committed_git_repo: Path,
    real_process_harness: Any,
    monkeypatch: MonkeyPatch,
) -> None:
    paths = RepoPaths.discover(committed_git_repo)
    cancel = CancellationToken(grace_seconds=0.05)
    factory_calls = 0
    errors: list[BaseException] = []
    forced_exits: list[int] = []

    def command_runner(
        _command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return run_captured(
            real_process_harness.resistant_argv("default-branch"),
            cancel=kwargs["cancel"],
            check=bool(kwargs["check"]),
        )

    def blocking_default_branch(
        repository_root: Path, *, cancel: CancellationToken | None = None
    ) -> str:
        return _default_branch(
            repository_root,
            cancel=cancel,
            command_runner=command_runner,
        )

    def agent_factory(_config):
        nonlocal factory_calls
        factory_calls += 1
        return MockAdapter()

    monkeypatch.setattr(
        repository_service_module,
        "_default_branch",
        blocking_default_branch,
    )
    control = RunControl(cancel, exit_function=forced_exits.append).install()
    try:
        with SqliteStore.open(paths.state_dir / "cancel-default.sqlite3") as store:
            service = RepositoryService(paths, store, agent_factory, cancel=cancel)

            def initialize() -> None:
                try:
                    service.initialize()
                except BaseException as error:
                    errors.append(error)

            worker = threading.Thread(target=initialize)
            worker.start()
            real_process_harness.wait_for_marker("default-branch.parent.pid")
            real_process_harness.wait_for_marker("default-branch.child.pid")
            with pytest.raises(KeyboardInterrupt):
                os.kill(os.getpid(), signal.SIGINT)
            assert control.wait_for_cancellation(timeout=1)
            worker.join(timeout=2)

            assert not worker.is_alive()
            assert len(errors) == 1
            assert isinstance(errors[0], KeyboardInterrupt)
    finally:
        control.close()
    real_process_harness.assert_tree_absent("default-branch")
    assert forced_exits == []
    assert not (paths.tracked_dir / CONFIG_FILENAME).exists()
    assert factory_calls == 0


def test_init_ctrl_c_during_git_head_reports_stopped_and_exits_interrupted(
    committed_git_repo: Path,
    real_process_harness: Any,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    paths = RepoPaths.discover(committed_git_repo)
    adapter, selected = _adapter(committed_git_repo)
    cancel = CancellationToken(grace_seconds=0.05)
    progress_stream = StringIO()
    progress = RunProgress(stream=progress_stream)
    sender_errors: list[BaseException] = []
    git_head_tokens: list[CancellationToken | None] = []
    original_git_head = analyzer_module._git_head

    def blocking_git_head(
        repository_root: Path,
        *,
        cancel: CancellationToken | None = None,
        command_runner=run_captured,
    ) -> str:
        def blocking_runner(
            _command: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            git_head_tokens.append(kwargs["cancel"])
            assert progress.stages["discover"].state is StageState.RUNNING
            return run_captured(
                real_process_harness.resistant_argv("cli-git-head"),
                cancel=kwargs["cancel"],
                check=bool(kwargs["check"]),
            )

        return original_git_head(
            repository_root,
            cancel=cancel,
            command_runner=blocking_runner,
        )

    def interrupt_after_probe_starts() -> None:
        try:
            real_process_harness.wait_for_marker("cli-git-head.parent.pid")
            real_process_harness.wait_for_marker("cli-git-head.child.pid")
            os.kill(os.getpid(), signal.SIGINT)
        except BaseException as error:
            sender_errors.append(error)

    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setenv("XDG_STATE_HOME", str(committed_git_repo.parent / "state"))
    monkeypatch.setattr(cli_module, "CancellationToken", lambda: cancel)
    monkeypatch.setattr(cli_module, "RunProgress", lambda **_kwargs: progress)
    monkeypatch.setattr(cli_module, "_stdin_is_interactive", lambda: False)
    monkeypatch.setattr(cli_module, "select_agent", lambda *_args, **_kwargs: selected)
    monkeypatch.setattr(analyzer_module, "_git_head", blocking_git_head)
    monkeypatch.setattr(
        analyzer_module,
        "build_discovery_workspace",
        lambda *_args, **_kwargs: pytest.fail("discovery must not start"),
    )

    sender = threading.Thread(target=interrupt_after_probe_starts)
    sender.start()
    try:
        exit_code = cli_module.main(["init", "--yes"], prog_name="betterborg")
    finally:
        sender.join(timeout=2)

    assert not sender.is_alive()
    assert sender_errors == []
    assert exit_code == 130
    assert git_head_tokens == [cancel]
    assert progress.stages["discover"].state is StageState.STOPPED
    assert progress.stages["analyze"].state is StageState.PENDING
    assert progress.closed
    output = progress_stream.getvalue()
    assert "stopping…" in output
    assert "■ Discover evidence" in output
    assert "interrupted" in output
    assert "✖ Discover evidence" not in output
    assert re.fullmatch(
        r"0 of 1 stage finished in \d+:\d{2}; 0 failed and 1 stopped\.",
        output.splitlines()[-1],
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == ""
    real_process_harness.assert_tree_absent("cli-git-head")
    assert adapter.calls == []

    config = load_repository_config(paths)
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        assert store.list_analyses(config.repository_id) == []
        assert store.list_operations(config.repository_id) == []


def test_default_branch_keeps_detached_head_classification(
    committed_git_repo: Path,
) -> None:
    def rejected(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "detached")

    with pytest.raises(
        RepositoryInitializationError,
        match="Git HEAD is detached",
    ):
        _default_branch(committed_git_repo, command_runner=rejected)


def test_prompt_cancellation_after_durable_fanout_starts_no_later_init_work(
    committed_git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    paths = RepoPaths.discover(committed_git_repo)
    adapter, selected = _adapter(committed_git_repo)
    cancel = CancellationToken()
    durable_roles: set[str] = set()
    durable_roles_lock = threading.Lock()
    reconcile = prompts_manager_module.get_durable_role_prompt

    def reconcile_then_cancel(*args: object, **kwargs: object):
        retained = reconcile(*args, **kwargs)
        analysis_id = kwargs.get("analysis_id")
        role = kwargs.get("role")
        if retained is not None and analysis_id is not None and isinstance(role, str):
            with durable_roles_lock:
                durable_roles.add(role)
                if durable_roles == set(PROMPT_ROLES):
                    cancel.cancel()
        return retained

    monkeypatch.setattr(
        prompts_manager_module,
        "get_durable_role_prompt",
        reconcile_then_cancel,
    )

    with SqliteStore.open(paths.state_dir / "cancel-prompts.sqlite3") as store:
        service = RepositoryService(
            paths,
            store,
            lambda _config: selected,
            cancel=cancel,
        )

        with pytest.raises(KeyboardInterrupt):
            service.initialize()

        repository = store.get_repository(load_repository_config(paths).repository_id)
        assert repository is not None
        assert set(store.get_latest_generated_prompts(repository.id)) == set(
            PROMPT_ROLES
        )
        assert store.list_operations(repository.id) == []

    assert len(adapter.calls) == 4
    assert list(paths.improvement_prds_dir.glob("*.md")) == []


def test_init_cancellation_after_atomic_improvement_publication_stops_stage(
    committed_git_repo: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    paths = RepoPaths.discover(committed_git_repo)
    adapter, selected = _adapter(committed_git_repo)
    cancel = CancellationToken()
    progress_stream = StringIO()
    progress = RunProgress(stream=progress_stream)
    published: list[Path] = []
    publish = improvement_prds_module.publish_repository_text

    def publish_then_cancel(
        path: Path,
        body: str,
        *,
        root: Path,
        overwrite: bool,
    ) -> None:
        publish(path, body, root=root, overwrite=overwrite)
        published.append(path)
        cancel.cancel()

    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setenv("XDG_STATE_HOME", str(committed_git_repo.parent / "state"))
    monkeypatch.setattr(cli_module, "CancellationToken", lambda: cancel)
    monkeypatch.setattr(cli_module, "RunProgress", lambda **_kwargs: progress)
    monkeypatch.setattr(cli_module, "_stdin_is_interactive", lambda: False)
    monkeypatch.setattr(cli_module, "select_agent", lambda *_args, **_kwargs: selected)
    monkeypatch.setattr(
        improvement_prds_module,
        "publish_repository_text",
        publish_then_cancel,
    )

    exit_code = cli_module.main(["init", "--yes"], prog_name="betterborg")

    assert exit_code == 130
    assert published == [paths.improvement_prds_dir / "theme-ci.md"]
    assert paths.improvement_prds_dir.joinpath("theme-ci.md").read_text(
        encoding="utf-8"
    ).startswith("# Make validation visible\n")
    assert progress.stages["prompts"].state is StageState.COMPLETED
    assert progress.stages["improvement-prds"].state is StageState.STOPPED
    assert progress.stages["improvement-prds"].result == "interrupted"
    assert progress.closed
    output = progress_stream.getvalue()
    assert output.index("✔ Generate role prompts") < output.index(
        "⠋ Draft improvement PRDs"
    )
    assert "■ Draft improvement PRDs" in output
    assert "interrupted" in output
    assert "✖ Draft improvement PRDs" not in output
    assert "✔ Draft improvement PRDs" not in output
    assert len(adapter.calls) == 4
    config = load_repository_config(paths)
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        assert store.list_operations(config.repository_id) == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == ""


def test_incomplete_initialization_seeds_retained_analysis_without_restart(
    committed_git_repo: Path,
) -> None:
    paths = RepoPaths.discover(committed_git_repo)
    initial_adapter = MockAdapter(name="openai").queue(
        MockResponse(payload=_analysis_payload())
    )

    def partial_prompt_response(spec):
        role = next(
            role for role in PROMPT_ROLES if f".{role}." in spec.result_path.name
        )
        if role == "review":
            raise RuntimeError("review generator unavailable")
        return {
            "body_md": f"# {role.title()} agent\n\nRetained role instructions."
        }

    for _role in PROMPT_ROLES:
        initial_adapter.queue(MockResponse(dynamic=partial_prompt_response))

    with SqliteStore.open(paths.state_dir / "retained.sqlite3") as store:
        setup = RepositoryService(paths, store, lambda _config: initial_adapter)
        with pytest.raises(
            RepositoryInitializationError,
            match="review: adapter crashed: review generator unavailable",
        ):
            setup.initialize()

        repository = store.get_repository(load_repository_config(paths).repository_id)
        assert repository is not None
        analysis = store.get_prior_ready_analysis(repository.id)
        assert analysis is not None
        assert set(store.get_latest_generated_prompts(repository.id)) == {
            "coding",
            "merge",
        }
        call_names = [call.result_path.name for call in initial_adapter.calls]
        assert call_names[0] == f"{analysis.id}.json"
        assert set(call_names[1:]) == {
            f"{analysis.id}.coding.json",
            f"{analysis.id}.review.json",
            f"{analysis.id}.merge.json",
        }

        retry_adapter = MockAdapter(name="openai").queue(
            MockResponse(
                payload={
                    "body_md": (
                        "# Review agent\n\nComplete repository-specific review "
                        "instructions."
                    )
                }
            )
        )
        progress = RunProgress(stream=StringIO())
        result = RepositoryService(
            paths,
            store,
            lambda _config: retry_adapter,
            progress=progress,
        ).initialize()

        assert result.analysis == analysis
        assert [call.result_path.name for call in retry_adapter.calls] == [
            f"{analysis.id}.review.json"
        ]

    for key in ("discover", "analyze"):
        record = progress.stages[key]
        assert record.state is StageState.COMPLETED
        assert record.retained is True
        assert record.started_at is None
        assert record.finished_at is None
        assert record.duration_seconds is None

    prompt_stage = progress.stages["prompts"]
    assert prompt_stage.state is StageState.COMPLETED
    assert prompt_stage.retained is False
    assert prompt_stage.started_at is not None
    assert prompt_stage.result == "3 prompts"
    for role in ("coding", "merge"):
        child = prompt_stage.children[role]
        assert child.state is StageState.COMPLETED
        assert child.retained is True
        assert child.started_at is None
        assert child.finished_at is None
        assert child.duration_seconds is None
    review = prompt_stage.children["review"]
    assert review.state is StageState.COMPLETED
    assert review.retained is False
    assert review.started_at is not None

    all_retained_progress = RunProgress(stream=StringIO())

    def fail_factory(_config):
        pytest.fail("an all-retained initialization must not select an agent")

    with SqliteStore.open(paths.state_dir / "retained.sqlite3") as store:
        retained_result = RepositoryService(
            paths,
            store,
            fail_factory,
            progress=all_retained_progress,
        ).initialize()

    assert retained_result.initialized is False
    retained_parent = all_retained_progress.stages["prompts"]
    assert retained_parent.state is StageState.COMPLETED
    assert retained_parent.retained is True
    assert retained_parent.started_at is None
    assert retained_parent.finished_at is None
    assert retained_parent.duration_seconds is None
    assert all(
        child.state is StageState.COMPLETED
        and child.retained
        and child.started_at is None
        for child in retained_parent.children.values()
    )
