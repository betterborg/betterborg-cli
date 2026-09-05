"""Bounded analyzer persistence and migration-002 contracts."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import threading
from collections.abc import Callable, Sequence
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from betterborg_cli.agent_runtime import CancellationToken, run_captured
from betterborg_cli.agent_runtime.api_tools import ApiAgentRole
from betterborg_cli.agent_runtime.base import AgentCapabilities
from betterborg_cli.agent_runtime.codex import CodexAdapter
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.agent_runtime.selection import SelectedAgent, select_agent
from betterborg_cli.progress import (
    AgentActivity,
    AgentActivityKind,
    RunProgress,
    StageSpec,
    StageState,
)
from betterborg_cli.repo_analysis import (
    DIMENSIONS,
    AnalyzerError,
    run_analyzer,
)
from betterborg_cli.repo_analysis import analyzer as analyzer_module
from betterborg_cli.repo_analysis.discovery import DiscoveryFile, DiscoveryManifest
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import (
    AgentChoice,
    AgentChoices,
    AgentStage,
    RepositoryConfig,
)
from betterborg_cli.run_control import RunControl
from betterborg_cli.store import Repository, SqliteStore


def _payload(score: float = 3) -> dict[str, object]:
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
                "title": "Document the CI checks",
                "package_path": ".",
                "dimension": "ci",
                "manifest_evidence": ["README.md"],
                "estimated_delta": 1,
                "effort": "L",
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


def _commit_repository(repo: Path) -> str:
    (repo / "README.md").write_text("# Example\n\nBuild and test docs.\n")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "--quiet", "-m", "initial"],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_dynamic_selected_agent_persists_append_only_history_and_prompts(
    git_repo: Path,
) -> None:
    repo_root = git_repo
    head_sha = _commit_repository(repo_root)
    repository = Repository(root=repo_root)
    database = git_repo / "state.sqlite3"
    observed_workspaces: list[Path] = []

    first_payload = _payload(3)
    second_payload = _payload(4)

    def response(payload: dict[str, object]):
        def dynamic(spec):
            observed_workspaces.append(spec.cwd)
            assert spec.cwd != repo_root
            assert json.loads((spec.cwd / "analysis_input.json").read_text())[
                "files"
            ][0]["path"] == "README.md"
            assert sorted(
                path.relative_to(spec.cwd).as_posix()
                for path in spec.cwd.rglob("*")
                if path.is_file()
            ) == ["analysis_input.json", "files/README.md"]
            return deepcopy(payload)

        return dynamic

    adapter = MockAdapter(name="openai").queue(
        MockResponse(dynamic=response(first_payload))
    )
    adapter.queue(MockResponse(dynamic=response(second_payload)))
    selected = SelectedAgent(
        role=ApiAgentRole.ANALYSIS,
        adapter=adapter,
        paths=RepoPaths.discover(repo_root),
    )

    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        first = run_analyzer(
            repository,
            store,
            selected,
            artifact_dir=git_repo / "artifacts",
        )
        second = run_analyzer(
            repository,
            store,
            selected,
            artifact_dir=git_repo / "artifacts",
        )

        assert store.applied_migrations() == tuple(range(1, 12))
        assert len(observed_workspaces) == 2
        assert all(call.model == "gpt-5.6-sol" for call in adapter.calls)
        assert all(
            call.allowed_tools == ("list_files", "read_file", "search_text")
            for call in adapter.calls
        )
        assert first.head_sha == second.head_sha == head_sha
        assert first.prior_analysis_id is None
        assert first.score_delta is None
        assert second.prior_analysis_id == first.id
        assert second.score_delta == 1
        assert second.analysis_json["themes"][0]["effort"] == "S"
        assert second.analysis_json["themes"][0][
            "normalized_impact"
        ] == pytest.approx(1 / 8)
        assert second.analysis_json["themes"][0]["recommendations"] == [
            {"id": "rec-ci", "effective_delta": 1.0, "delta_clamped": False}
        ]
        assert store.get_prior_ready_analysis(
            repository.id, before_analysis_id=second.id
        ) == first
        assert [row.id for row in store.list_analyses(repository.id)] == [
            first.id,
            second.id,
        ]
        [package] = store.list_packages(second.id)
        assert package.package_path == "."
        assert package.overall_score == 4

        coding_v1 = store.append_generated_prompt(
            repository_id=repository.id,
            analysis_id=first.id,
            role="coding",
            body_md="First coding prompt",
        )
        coding_v2 = store.append_generated_prompt(
            repository_id=repository.id,
            analysis_id=second.id,
            role="coding",
            body_md="Second coding prompt",
        )
        review_v1 = store.append_generated_prompt(
            repository_id=repository.id,
            analysis_id=second.id,
            role="review",
            body_md="Review prompt",
        )
        assert (coding_v1.version, coding_v2.version, review_v1.version) == (1, 2, 1)
        assert store.get_latest_generated_prompts(repository.id) == {
            "coding": coding_v2,
            "review": review_v1,
        }

        with store.locked_connection() as connection:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(
                    "UPDATE repository_analyses SET summary = 'changed' WHERE id = ?",
                    (str(first.id),),
                )
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(
                    "DELETE FROM generated_prompts WHERE id = ?",
                    (str(coding_v1.id),),
                )

    with SqliteStore.open(database) as reopened:
        assert reopened.list_analyses(repository.id) == [first, second]
        assert reopened.list_packages(second.id) == [package]
        assert (
            reopened.get_latest_generated_prompts(repository.id)["coding"]
            == coding_v2
        )


def test_invalid_analyzer_output_is_not_persisted(git_repo: Path) -> None:
    repo_root = git_repo
    _commit_repository(repo_root)
    repository = Repository(root=repo_root)
    invalid = _payload()
    del invalid["themes"][0]["effort"]
    adapter = MockAdapter(name="openai").queue(MockResponse(payload=invalid))

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)

        with pytest.raises(AnalyzerError, match="effort"):
            run_analyzer(
                repository,
                store,
                adapter,
                artifact_dir=git_repo / "artifacts",
            )

        assert store.list_analyses(repository.id) == []


def test_git_head_cancellation_stops_discovery_before_workspace_or_agent(
    git_repo: Path,
    real_process_harness: Any,
) -> None:
    _commit_repository(git_repo)
    repository = Repository(root=git_repo)
    adapter = MockAdapter(name="openai").queue(MockResponse(payload=_payload()))
    cancel = CancellationToken(grace_seconds=0.05)
    progress = RunProgress(
        [
            StageSpec("discover", "Discover evidence"),
            StageSpec("analyze", "Analyze repository"),
        ],
        stream=StringIO(),
    )
    workspace = git_repo.parent / "analysis-workspace"
    errors: list[BaseException] = []
    forced_exits: list[int] = []

    def command_runner(
        _command: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return run_captured(
            real_process_harness.resistant_argv("git-head"),
            cancel=kwargs["cancel"],
            check=bool(kwargs["check"]),
        )

    control = RunControl(
        cancel,
        progress=progress,
        exit_function=forced_exits.append,
    ).install()
    try:
        with SqliteStore.open(git_repo / "state.sqlite3") as store:
            store.add_repository(repository)

            def analyze() -> None:
                try:
                    run_analyzer(
                        repository,
                        store,
                        adapter,
                        artifact_dir=git_repo / "artifacts",
                        workspace_dir=workspace,
                        cancel=cancel,
                        progress=progress,
                        command_runner=command_runner,
                    )
                except BaseException as error:
                    errors.append(error)

            worker = threading.Thread(target=analyze)
            worker.start()
            real_process_harness.wait_for_marker("git-head.parent.pid")
            real_process_harness.wait_for_marker("git-head.child.pid")
            with pytest.raises(KeyboardInterrupt):
                os.kill(os.getpid(), signal.SIGINT)
            assert control.wait_for_cancellation(timeout=1)
            worker.join(timeout=2)

            assert not worker.is_alive()
            assert len(errors) == 1
            assert isinstance(errors[0], KeyboardInterrupt)
            assert progress.stages["discover"].state is StageState.STOPPED
            assert progress.stages["analyze"].state is StageState.PENDING
            assert store.list_analyses(repository.id) == []
    finally:
        control.close()

    real_process_harness.assert_tree_absent("git-head")
    assert forced_exits == []
    assert not workspace.exists()
    assert adapter.calls == []


def test_git_head_keeps_ordinary_unreadable_head_classification(
    git_repo: Path,
) -> None:
    _commit_repository(git_repo)
    repository = Repository(root=git_repo)
    command: list[str] = []

    def rejected(
        argv: Sequence[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        command.extend(argv)
        return subprocess.CompletedProcess(argv, 128, "", "unreadable")

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        with pytest.raises(AnalyzerError, match="readable Git HEAD"):
            run_analyzer(
                repository,
                store,
                MockAdapter(),
                artifact_dir=git_repo / "artifacts",
                command_runner=rejected,
            )

    assert command[-2:] == ["rev-parse", "HEAD"]


def test_analyzer_forwards_agent_activity_and_completes_after_persistence(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _commit_repository(git_repo)
    repository = Repository(root=git_repo)
    activity = AgentActivity(AgentActivityKind.READING, "README.md")
    adapter = MockAdapter(name="claude").queue(
        MockResponse(payload=_payload(), activities=(activity,))
    )
    selected = SelectedAgent(
        role=ApiAgentRole.ANALYSIS,
        adapter=adapter,
        paths=RepoPaths.discover(git_repo),
        model="opus-4-8",
    )
    manifest = DiscoveryManifest(
        repo_name=git_repo.name,
        files=[
            DiscoveryFile(
                path="README.md" if number == 0 else f"src/file-{number}.py",
                workspace_path=(
                    "files/README.md"
                    if number == 0
                    else f"files/src/file-{number}.py"
                ),
                category="source",
                size_bytes=1,
                copied_bytes=1,
            )
            for number in range(142)
        ],
        total_copied_bytes=1_800_000,
    )
    monkeypatch.setattr(
        analyzer_module,
        "build_discovery_workspace",
        lambda *_args, **_kwargs: manifest,
    )
    progress = RunProgress(
        [
            StageSpec("discover", "Discover evidence"),
            StageSpec("analyze", "Analyze repository"),
        ],
        stream=StringIO(),
    )

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        analysis = run_analyzer(
            repository,
            store,
            selected,
            artifact_dir=git_repo / "artifacts",
            progress=progress,
        )

        assert store.list_analyses(repository.id) == [analysis]

    assert progress.stages["discover"].state is StageState.COMPLETED
    assert progress.stages["discover"].result == "142 files · 1.8 MB"
    analyze = progress.stages["analyze"]
    assert analyze.state is StageState.COMPLETED
    assert analyze.activity == activity
    assert analyze.result == "claude · opus-4-8"


def test_analyzer_resolves_the_anthropic_default_model(git_repo: Path) -> None:
    _commit_repository(git_repo)
    repository = Repository(root=git_repo)
    adapter = MockAdapter(name="anthropic").queue(MockResponse(payload=_payload()))
    selected = SelectedAgent(
        role=ApiAgentRole.ANALYSIS,
        adapter=adapter,
        paths=RepoPaths.discover(git_repo),
    )

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        run_analyzer(
            repository,
            store,
            selected,
            artifact_dir=git_repo / "artifacts",
        )

    assert adapter.calls[0].model == "claude-opus-5"


@pytest.mark.parametrize("effort", ["high", "low"])
def test_analyzer_passes_stage_effort_to_selected_anthropic(
    git_repo: Path, effort: str
) -> None:
    _commit_repository(git_repo)
    repository = Repository(root=git_repo)
    selected = select_agent(
        RepositoryConfig(
            version=1,
            repository_id=repository.id,
            default_branch="main",
            agents=AgentChoices(
                analysis=AgentChoice(adapter="anthropic", effort=effort)
            ),
        ),
        AgentStage.ANALYSIS,
        RepoPaths.discover(git_repo),
        interactive=False,
        credentials={"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o"},
        executable_lookup=lambda _binary: None,
    )
    adapter = MockAdapter(name="anthropic").queue(MockResponse(payload=_payload()))
    selected.adapter = adapter

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        analysis = run_analyzer(
            repository,
            store,
            selected,
            artifact_dir=git_repo / "artifacts",
        )

        assert store.list_analyses(repository.id) == [analysis]

    assert len(adapter.calls) == 1
    assert adapter.calls[0].effort == effort


def test_native_analyzer_runs_read_only_in_the_bounded_workspace(
    git_repo: Path,
) -> None:
    _commit_repository(git_repo)
    repository = Repository(root=git_repo)
    observed: dict[str, object] = {}

    def runner(
        command: Sequence[str],
        cwd: Path,
        _stdin_text: str,
        log_path: Path,
        _cancel: object,
        _env: object,
        _on_line: Callable[[str], None] | None,
    ) -> int:
        observed.update(
            command=list(command),
            cwd=cwd,
            visible=sorted(
                path.relative_to(cwd).as_posix()
                for path in cwd.rglob("*")
                if path.is_file()
            ),
        )
        log_path.write_text("", encoding="utf-8")
        Path(command[command.index("-o") + 1]).write_text(
            json.dumps(_payload()), encoding="utf-8"
        )
        return 0

    trusted: list[Path] = []
    selected = SelectedAgent(
        role=ApiAgentRole.ANALYSIS,
        adapter=CodexAdapter(ApiAgentRole.ANALYSIS, proc_runner=runner),
        paths=RepoPaths.discover(git_repo),
        trust_requirement=lambda paths, **_kwargs: trusted.append(paths.root),
    )
    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        analysis = run_analyzer(
            repository,
            store,
            selected,
            artifact_dir=git_repo / "artifacts",
        )

        assert store.list_analyses(repository.id) == [analysis]

    assert trusted == [git_repo.resolve()]

    command = observed["command"]
    assert command[command.index("-s") + 1] == "read-only"
    assert command[command.index("-C") + 1] == str(observed["cwd"])
    assert observed["cwd"] != git_repo
    assert observed["visible"] == ["analysis_input.json", "files/README.md"]


def test_analyzer_rejects_an_unwrapped_native_adapter(git_repo: Path) -> None:
    _commit_repository(git_repo)
    repository = Repository(root=git_repo)
    spawned = False

    def runner(*_args: object, **_kwargs: object) -> int:
        nonlocal spawned
        spawned = True
        return 0

    adapter = CodexAdapter(ApiAgentRole.ANALYSIS, proc_runner=runner)
    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)

        with pytest.raises(AnalyzerError, match="must be wrapped by SelectedAgent"):
            run_analyzer(
                repository,
                store,
                adapter,
                artifact_dir=git_repo / "artifacts",
            )

        assert store.list_analyses(repository.id) == []
    assert not spawned


def test_analyzer_rejects_an_adapter_without_a_read_only_boundary(
    git_repo: Path,
) -> None:
    _commit_repository(git_repo)
    repository = Repository(root=git_repo)
    adapter = MockAdapter(capabilities=AgentCapabilities())

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)

        with pytest.raises(AnalyzerError, match="read-only execution boundary"):
            run_analyzer(
                repository,
                store,
                adapter,
                artifact_dir=git_repo / "artifacts",
            )
