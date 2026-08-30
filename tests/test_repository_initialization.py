"""Idempotent repository initialization through the public CLI."""

from __future__ import annotations

import json
import os
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
from betterborg_cli.repo_analysis import DIMENSIONS, PROMPT_ROLES, run_analyzer
from betterborg_cli.repo_paths import MANAGED_IGNORE_BEGIN, RepoPaths
from betterborg_cli.repository_config import CONFIG_FILENAME, load_repository_config
from betterborg_cli.repository_service import (
    RepositoryInitializationError,
    RepositoryService,
    _default_branch,
)
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
    assert "theme-ci" in improvement.read_text(encoding="utf-8")
    assert len(adapter.calls) == 4
    assert selections == 1
    assert (
        "borg create theme-ci --prd "
        ".borg/prds/improvements/theme-ci.md\n"
    ) in first.output

    second = cli_runner.invoke(cli, ["init", "--yes"])

    assert second.exit_code == 0, second.output
    assert "completed Discover evidence" in second.output
    assert "completed Analyze repository" in second.output
    assert second.output.endswith(
        f"Repository already initialized: {config.repository_id}\n"
    )
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
        assert store.get_borg_by_name(config.repository_id, "theme-ci") is None

    with sqlite3.connect(paths.state_dir / "borg.sqlite3") as connection:
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
        service = RepositoryService(paths, store, lambda _config: MockAdapter())
        with pytest.raises(KeyboardInterrupt, match="config publication interrupted"):
            service._write_initial_config(repository)

    if interrupt_after_claim:
        config = load_repository_config(paths)
        assert config.repository_id == repository.id
        assert config_path.read_text(encoding="utf-8").endswith("\n")
    else:
        assert not config_path.exists()
    assert list(paths.tracked_dir.glob(".config.toml.*.tmp")) == []


def test_initial_config_create_race_preserves_the_winning_repository_identity(
    committed_git_repo: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    paths = RepoPaths.discover(committed_git_repo)
    losing_repository = Repository(root=paths.root)
    winning_repository = Repository(root=paths.root)
    config_path = paths.tracked_dir / CONFIG_FILENAME
    winning_body = (
        "version = 1\n\n"
        "[repository]\n"
        f'id = "{winning_repository.id}"\n'
        'default_branch = "winning-branch"\n'
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

    with SqliteStore.open(paths.state_dir / "config-race.sqlite3") as store:
        service = RepositoryService(paths, store, lambda _config: MockAdapter())
        service._write_initial_config(losing_repository)
        first_repository, first_config = service._ensure_repository()
        second_repository, second_config = service._ensure_repository()

        assert first_repository.id == winning_repository.id
        assert second_repository.id == winning_repository.id
        assert first_config.repository_id == winning_repository.id
        assert second_config.repository_id == winning_repository.id
        assert store.get_repository(winning_repository.id) == first_repository

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
        service = RepositoryService(paths, store, lambda _config: MockAdapter())
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
    assert json.loads(result.output) == {
        "create_commands": [
            "borg create theme-ci --prd "
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
    confirmed = paths.tracked_dir / "prds" / "theme-ci.md"
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
    assert "completed Discover evidence — 1 evidence files" in result.output
    assert "completed Analyze repository — score 4.00/5" in result.output
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
        with SqliteStore.open(paths.state_dir / "borg.sqlite3") as other_store:
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
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
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
    assert "repository is not initialized; run 'borg init' first" in result.output
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
        cancel.cancel()
        worker.join(timeout=2)

        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], KeyboardInterrupt)
    real_process_harness.assert_tree_absent("default-branch")
    assert not (paths.tracked_dir / CONFIG_FILENAME).exists()
    assert factory_calls == 0


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


def test_incomplete_initialization_seeds_retained_analysis_without_restart(
    committed_git_repo: Path,
) -> None:
    paths = RepoPaths.discover(committed_git_repo)
    initial_adapter = MockAdapter(name="openai").queue(
        MockResponse(payload=_analysis_payload())
    )

    with SqliteStore.open(paths.state_dir / "retained.sqlite3") as store:
        setup = RepositoryService(paths, store, lambda _config: initial_adapter)
        repository, _config = setup._ensure_repository()
        analysis = run_analyzer(
            repository,
            store,
            initial_adapter,
            artifact_dir=paths.artifacts_dir / "analysis",
        )
        paths.prompts_dir.mkdir(parents=True, exist_ok=True)
        for role in ("coding", "merge"):
            body = f"# {role.title()} agent\n\nRetained role instructions."
            (paths.prompts_dir / f"{role}.system.md").write_text(
                body,
                encoding="utf-8",
            )
            store.append_generated_prompt(
                repository_id=repository.id,
                analysis_id=analysis.id,
                role=role,
                body_md=body,
            )

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
