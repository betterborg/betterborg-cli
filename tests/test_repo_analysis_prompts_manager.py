"""Stable role-prompt generation and partial-failure contracts."""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from threading import Barrier, Lock, Thread
from typing import Any

import pytest

from betterborg_cli.agent_runtime import CancellationToken, run_captured
from betterborg_cli.agent_runtime.api_tools import ApiAgentRole
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.agent_runtime.selection import SelectedAgent
from betterborg_cli.progress import (
    AgentActivity,
    AgentActivityKind,
    ChildSpec,
    RunProgress,
    StageSpec,
    StageState,
)
from betterborg_cli.repo_analysis import (
    PROMPT_ROLES,
    AnalyzerError,
    PromptManagerConfig,
    generate_role_prompts,
    prompts_manager,
)
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.run_control import RunControl
from betterborg_cli.store import (
    Repository,
    RepositoryAnalysis,
    RepositoryPackage,
    SqliteStore,
)


def _append_analysis(
    store: SqliteStore,
    repository: Repository,
    *,
    score: float,
    prior: RepositoryAnalysis | None = None,
) -> RepositoryAnalysis:
    analysis = RepositoryAnalysis(
        repository_id=repository.id,
        head_sha=f"head-{score}",
        summary=f"Python CLI analysis at score {score}.",
        primary_language="python",
        is_monorepo=False,
        overall_score=score,
        analysis_json={
            "summary": "A Python CLI with pytest and Ruff checks.",
            "primary_language": "python",
            "overall_score": score,
            "packages": [
                {
                    "path": ".",
                    "name": "root",
                    "primary_language": "python",
                    "overall_score": score,
                }
            ],
            "recommendations": [],
            "themes": [],
        },
        prior_analysis_id=prior.id if prior is not None else None,
        score_delta=score - prior.overall_score if prior is not None else None,
    )
    package = RepositoryPackage(
        repository_id=repository.id,
        analysis_id=analysis.id,
        package_path=".",
        package_name="root",
        primary_language="python",
        rubric={},
        overall_score=score,
    )
    store.append_analysis(analysis, [package])
    return analysis


def _role_keyed_responses(
    adapter: MockAdapter,
    bodies: dict[str, str | Exception],
) -> None:
    def respond(spec):
        for role, response in bodies.items():
            if f".{role}." not in spec.result_path.name:
                continue
            if isinstance(response, Exception):
                raise response
            return {
                "body_md": response,
                "notes": f"generated {role}",
            }
        raise AssertionError(f"unrecognized prompt result path: {spec.result_path}")

    for _role in bodies:
        adapter.queue(MockResponse(dynamic=respond))


def _selected_adapter(git_repo: Path, bodies: dict[str, str | Exception]):
    adapter = MockAdapter(name="openai")
    _role_keyed_responses(adapter, bodies)
    selected = SelectedAgent(
        role=ApiAgentRole.ANALYSIS,
        adapter=adapter,
        paths=RepoPaths.discover(git_repo),
        model="prompt-model",
    )
    return adapter, selected


def test_generates_all_role_metadata_at_stable_paths(git_repo: Path) -> None:
    repository = Repository(root=git_repo)
    bodies = {
        role: f"# {role.title()} agent\n\nComplete repository-specific {role} guidance."
        for role in PROMPT_ROLES
    }
    adapter, selected = _selected_adapter(git_repo, bodies)

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        analysis = _append_analysis(store, repository, score=3)

        runs = generate_role_prompts(
            repository,
            analysis,
            store,
            selected,
            artifact_dir=git_repo / "artifacts",
        )

        assert [run.role for run in runs] == list(PROMPT_ROLES)
        assert all(run.ok and run.version == 1 for run in runs)
        latest = store.get_latest_generated_prompts(repository.id)
        assert set(latest) == set(PROMPT_ROLES)
        for run in runs:
            assert run.path == git_repo / ".borg" / "prompts" / (
                f"{run.role}.system.md"
            )
            assert run.path.read_text(encoding="utf-8") == bodies[run.role]
            assert latest[run.role].analysis_id == analysis.id
            assert latest[run.role].body_md == bodies[run.role]

    calls = {spec.result_path.stem.rsplit(".", 1)[-1]: spec for spec in adapter.calls}
    assert set(calls) == set(PROMPT_ROLES)
    for role, spec in calls.items():
        assert spec.cwd == git_repo
        assert spec.model == "prompt-model"
        assert role in spec.system_prompt
        assert '"overall_score": 3' in spec.user_prompt
        assert "Prior" not in spec.user_prompt
        assert spec.allowed_tools == ("list_files", "read_file", "search_text")


def test_role_children_run_concurrently_and_reconcile_durable_activity(
    git_repo: Path,
) -> None:
    repository = Repository(root=git_repo)
    barrier = Barrier(len(PROMPT_ROLES))
    adapter = MockAdapter(name="openai")

    def respond(spec):
        role = next(
            role for role in PROMPT_ROLES if f".{role}." in spec.result_path.name
        )
        barrier.wait(timeout=2)
        return MockResponse(
            payload={
                "body_md": (
                    f"# {role.title()} agent\n\n"
                    f"Complete repository-specific {role} guidance."
                )
            },
            activities=(
                AgentActivity(AgentActivityKind.READING, f"{role}.toml"),
            ),
        )

    for _role in PROMPT_ROLES:
        adapter.queue(MockResponse(dynamic=respond))
    selected = SelectedAgent(
        role=ApiAgentRole.ANALYSIS,
        adapter=adapter,
        paths=RepoPaths.discover(git_repo),
        model="prompt-model",
    )
    progress = RunProgress(
        [
            StageSpec(
                "prompts",
                "Generate role prompts",
                tuple(
                    ChildSpec(role, f"{role.title()} prompt")
                    for role in PROMPT_ROLES
                ),
            )
        ],
        stream=StringIO(),
    )

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        analysis = _append_analysis(store, repository, score=3)

        runs = generate_role_prompts(
            repository,
            analysis,
            store,
            selected,
            artifact_dir=git_repo / "artifacts",
            progress=progress,
        )

        assert all(run.ok for run in runs)
        latest = store.get_latest_generated_prompts(repository.id)
        assert all(
            (git_repo / f".borg/prompts/{role}.system.md").read_text()
            == latest[role].body_md
            for role in PROMPT_ROLES
        )

    parent = progress.stages["prompts"]
    assert parent.state is StageState.COMPLETED
    assert parent.result == "3 prompts"
    for role in PROMPT_ROLES:
        child = parent.children[role]
        assert child.state is StageState.COMPLETED
        assert child.retained is False
        assert child.activity == AgentActivity(
            AgentActivityKind.READING,
            f"{role}.toml",
        )
        assert child.result == "prompt v1"


def test_cancellation_during_prompt_root_discovery_starts_no_prompt_work(
    git_repo: Path,
    real_process_harness: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Repository(root=git_repo)
    adapter, selected = _selected_adapter(
        git_repo,
        {
            role: f"# {role.title()} agent\n\nComplete role guidance."
            for role in PROMPT_ROLES
        },
    )
    cancel = CancellationToken(grace_seconds=0.05)
    progress = RunProgress(
        [
            StageSpec(
                "prompts",
                "Generate role prompts",
                tuple(
                    ChildSpec(role, f"{role.title()} prompt")
                    for role in PROMPT_ROLES
                ),
            )
        ],
        stream=StringIO(),
    )
    original_discover = RepoPaths.discover.__func__

    def blocking_discover(
        cls,
        start: Path | None = None,
        *,
        cancel: CancellationToken | None = None,
        command_runner=run_captured,
    ) -> RepoPaths:
        def blocking_runner(
            _command: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return run_captured(
                real_process_harness.resistant_argv("prompt-discovery"),
                cancel=kwargs["cancel"],
                check=bool(kwargs["check"]),
            )

        return original_discover(
            cls,
            start,
            cancel=cancel,
            command_runner=blocking_runner,
        )

    monkeypatch.setattr(RepoPaths, "discover", classmethod(blocking_discover))
    errors: list[BaseException] = []
    forced_exits: list[int] = []

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        analysis = _append_analysis(store, repository, score=3)

        def generate() -> None:
            try:
                generate_role_prompts(
                    repository,
                    analysis,
                    store,
                    selected,
                    artifact_dir=git_repo / "artifacts",
                    cancel=cancel,
                    progress=progress,
                )
            except BaseException as error:
                errors.append(error)

        control = RunControl(
            cancel,
            progress=progress,
            exit_function=forced_exits.append,
        ).install()
        try:
            worker = Thread(target=generate)
            worker.start()
            real_process_harness.wait_for_marker("prompt-discovery.parent.pid")
            real_process_harness.wait_for_marker("prompt-discovery.child.pid")
            with pytest.raises(KeyboardInterrupt):
                os.kill(os.getpid(), signal.SIGINT)
            assert control.wait_for_cancellation(timeout=1)
            worker.join(timeout=2)
            assert not worker.is_alive()
        finally:
            control.close()

        assert len(errors) == 1
        assert isinstance(errors[0], KeyboardInterrupt)
        assert store.get_latest_generated_prompts(repository.id) == {}

    real_process_harness.assert_tree_absent("prompt-discovery")
    assert forced_exits == []
    assert adapter.calls == []
    parent = progress.stages["prompts"]
    assert parent.state is StageState.PENDING
    assert all(
        child.state is StageState.PENDING for child in parent.children.values()
    )


def test_cancellation_after_durable_prompt_publication_completes_the_child(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Repository(root=git_repo)
    adapter, selected = _selected_adapter(
        git_repo,
        {"coding": "# Coding agent\n\nComplete repository-specific guidance."},
    )
    cancel = CancellationToken()
    progress = RunProgress(
        [
            StageSpec(
                "prompts",
                "Generate role prompts",
                (ChildSpec("coding", "Coding prompt"),),
            )
        ],
        stream=StringIO(),
    )
    reconcile = prompts_manager.get_durable_role_prompt

    def reconcile_after_cancellation(*args, **kwargs):
        retained = reconcile(*args, **kwargs)
        cancel.cancel()
        progress.begin_cancellation()
        return retained

    monkeypatch.setattr(
        prompts_manager,
        "get_durable_role_prompt",
        reconcile_after_cancellation,
    )

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        analysis = _append_analysis(store, repository, score=3)

        runs = generate_role_prompts(
            repository,
            analysis,
            store,
            selected,
            artifact_dir=git_repo / "artifacts",
            roles=("coding",),
            cancel=cancel,
            progress=progress,
        )

        assert runs[0].ok
        assert store.get_latest_generated_prompts(repository.id)["coding"] == (
            runs[0].prompt
        )

    parent = progress.stages["prompts"]
    assert parent.state is StageState.COMPLETED
    assert parent.children["coding"].state is StageState.COMPLETED
    assert parent.children["coding"].result == "prompt v1"


def test_cancellation_before_prompt_publication_stops_without_retaining(
    git_repo: Path,
) -> None:
    repository = Repository(root=git_repo)
    cancel = CancellationToken()
    progress = RunProgress(
        [
            StageSpec(
                "prompts",
                "Generate role prompts",
                (ChildSpec("coding", "Coding prompt"),),
            )
        ],
        stream=StringIO(),
    )
    adapter = MockAdapter(name="openai")

    def interrupt(_spec):
        cancel.cancel()
        progress.begin_cancellation()
        raise KeyboardInterrupt

    adapter.queue(MockResponse(dynamic=interrupt))
    selected = SelectedAgent(
        role=ApiAgentRole.ANALYSIS,
        adapter=adapter,
        paths=RepoPaths.discover(git_repo),
    )

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        analysis = _append_analysis(store, repository, score=3)

        with pytest.raises(KeyboardInterrupt):
            generate_role_prompts(
                repository,
                analysis,
                store,
                selected,
                artifact_dir=git_repo / "artifacts",
                roles=("coding",),
                cancel=cancel,
                progress=progress,
            )

        assert store.get_latest_generated_prompts(repository.id) == {}

    parent = progress.stages["prompts"]
    assert parent.state is StageState.STOPPED
    assert parent.children["coding"].state is StageState.STOPPED
    assert not (git_repo / ".borg/prompts/coding.system.md").exists()


def test_cancellation_between_prompt_parent_and_child_start_stops_parent(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Repository(root=git_repo)
    cancel = CancellationToken()
    progress = RunProgress(
        [
            StageSpec(
                "prompts",
                "Generate role prompts",
                tuple(
                    ChildSpec(role, f"{role.title()} prompt")
                    for role in PROMPT_ROLES
                ),
            )
        ],
        stream=StringIO(),
    )
    adapter, selected = _selected_adapter(
        git_repo,
        {
            role: f"# {role.title()} agent\n\nComplete role guidance."
            for role in PROMPT_ROLES
        },
    )
    original_start_child = progress.start_child
    cancellation_lock = Lock()
    cancellation_started = False

    def cancel_before_child_start(stage_key: str, child_key: str):
        nonlocal cancellation_started
        with cancellation_lock:
            if not cancellation_started:
                cancellation_started = True
                cancel.cancel()
                progress.begin_cancellation()
        return original_start_child(stage_key, child_key)

    monkeypatch.setattr(progress, "start_child", cancel_before_child_start)

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        analysis = _append_analysis(store, repository, score=3)

        with pytest.raises(KeyboardInterrupt):
            generate_role_prompts(
                repository,
                analysis,
                store,
                selected,
                artifact_dir=git_repo / "artifacts",
                cancel=cancel,
                progress=progress,
            )

        assert store.get_latest_generated_prompts(repository.id) == {}

    assert adapter.calls == []
    parent = progress.stages["prompts"]
    assert parent.state is StageState.STOPPED
    assert all(
        child.state is StageState.PENDING for child in parent.children.values()
    )
    progress.close()
    assert progress.closed


def test_main_thread_interrupt_while_waiting_for_prompt_child_stops_parent(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Repository(root=git_repo)
    cancel = CancellationToken()
    progress = RunProgress(
        [
            StageSpec(
                "prompts",
                "Generate role prompts",
                tuple(
                    ChildSpec(role, f"{role.title()} prompt")
                    for role in PROMPT_ROLES
                ),
            )
        ],
        stream=StringIO(),
    )
    adapter = MockAdapter(name="openai")
    for role in PROMPT_ROLES:
        adapter.queue(
            MockResponse(
                payload={
                    "body_md": f"# {role.title()} agent\n\nComplete role guidance."
                },
                delay_seconds=10,
            )
        )
    selected = SelectedAgent(
        role=ApiAgentRole.ANALYSIS,
        adapter=adapter,
        paths=RepoPaths.discover(git_repo),
    )

    class InterruptingFuture:
        def __init__(self, future: Any, *, interrupt: bool) -> None:
            self.future = future
            self.interrupt = interrupt

        def result(self) -> Any:
            if self.interrupt:
                assert adapter.wait_for_response_consumption(timeout=2)
                cancel.cancel()
                progress.begin_cancellation()
                raise KeyboardInterrupt
            return self.future.result()

    class InterruptingExecutor:
        def __init__(self, max_workers: int) -> None:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            self.submissions = 0

        def __enter__(self) -> InterruptingExecutor:
            self.executor.__enter__()
            return self

        def __exit__(self, *exc_info: object) -> bool | None:
            return self.executor.__exit__(*exc_info)

        def submit(
            self,
            function: Callable[..., Any],
            *args: Any,
        ) -> InterruptingFuture:
            future = self.executor.submit(function, *args)
            self.submissions += 1
            return InterruptingFuture(future, interrupt=self.submissions == 1)

    monkeypatch.setattr(prompts_manager, "ThreadPoolExecutor", InterruptingExecutor)

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        analysis = _append_analysis(store, repository, score=3)

        with pytest.raises(KeyboardInterrupt):
            generate_role_prompts(
                repository,
                analysis,
                store,
                selected,
                artifact_dir=git_repo / "artifacts",
                cancel=cancel,
                progress=progress,
            )

        assert store.get_latest_generated_prompts(repository.id) == {}

    parent = progress.stages["prompts"]
    assert parent.state is StageState.STOPPED
    assert all(
        child.state in {StageState.PENDING, StageState.STOPPED}
        for child in parent.children.values()
    )
    progress.close()
    assert progress.closed


@pytest.mark.parametrize("symlink_path", [Path(".borg"), Path(".borg/prompts")])
def test_stable_prompt_directory_cannot_escape_repository_through_symlink(
    git_repo: Path,
    symlink_path: Path,
) -> None:
    repository = Repository(root=git_repo)
    outside = git_repo.parent / f"{git_repo.name}-outside"
    outside.mkdir()
    link = git_repo / symlink_path
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    adapter, selected = _selected_adapter(
        git_repo,
        {"coding": "# Coding agent\n\nComplete repository-specific coding guidance."},
    )

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        analysis = _append_analysis(store, repository, score=3)

        with pytest.raises(ValueError, match="prompt directory escapes repository"):
            generate_role_prompts(
                repository,
                analysis,
                store,
                selected,
                artifact_dir=git_repo / "artifacts",
                roles=("coding",),
            )

        assert adapter.calls == []
        assert store.get_latest_generated_prompts(repository.id) == {}
        assert list(outside.iterdir()) == []


def test_prompt_manager_rejects_effort_for_anthropic_before_adapter_call(
    git_repo: Path,
) -> None:
    repository = Repository(root=git_repo)
    adapter = MockAdapter(name="anthropic")
    _role_keyed_responses(
        adapter,
        {"coding": "# Coding agent\n\nComplete repository-specific coding guidance."},
    )
    selected = SelectedAgent(
        role=ApiAgentRole.ANALYSIS,
        adapter=adapter,
        paths=RepoPaths.discover(git_repo),
    )

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        analysis = _append_analysis(store, repository, score=3)

        with pytest.raises(
            AnalyzerError, match="Anthropic does not support an effort override"
        ):
            generate_role_prompts(
                repository,
                analysis,
                store,
                selected,
                artifact_dir=git_repo / "artifacts",
                config=PromptManagerConfig(effort="high"),
                roles=("coding",),
            )

        assert adapter.calls == []
        assert store.get_latest_generated_prompts(repository.id) == {}


def test_partial_failure_preserves_score_then_reanalysis_refreshes_prompts(
    git_repo: Path,
) -> None:
    repository = Repository(root=git_repo)
    first_bodies: dict[str, str | Exception] = {
        "coding": "# Coding v1\n\nFirst complete coding prompt body.",
        "review": RuntimeError("review generator unavailable"),
        "merge": "# Merge v1\n\nFirst complete merge prompt body.",
    }
    _first_adapter, first_selected = _selected_adapter(git_repo, first_bodies)

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        first_analysis = _append_analysis(store, repository, score=2)
        first_runs = generate_role_prompts(
            repository,
            first_analysis,
            store,
            first_selected,
            artifact_dir=git_repo / "artifacts",
        )

        first_by_role = {run.role: run for run in first_runs}
        assert first_by_role["coding"].ok
        assert not first_by_role["review"].ok
        assert first_by_role["review"].error == (
            "adapter crashed: review generator unavailable"
        )
        assert first_by_role["merge"].ok
        assert store.get_analysis(first_analysis.id) == first_analysis
        assert store.get_analysis(first_analysis.id).overall_score == 2
        assert set(store.get_latest_generated_prompts(repository.id)) == {
            "coding",
            "merge",
        }
        assert not (git_repo / ".borg/prompts/review.system.md").exists()

        second_analysis = _append_analysis(
            store,
            repository,
            score=4,
            prior=first_analysis,
        )
        second_bodies: dict[str, str | Exception] = {
            role: f"# {role.title()} v2\n\nRefreshed complete {role} prompt body."
            for role in PROMPT_ROLES
        }
        second_adapter, second_selected = _selected_adapter(git_repo, second_bodies)
        second_runs = generate_role_prompts(
            repository,
            second_analysis,
            store,
            second_selected,
            artifact_dir=git_repo / "artifacts",
        )

        assert all(run.ok for run in second_runs)
        assert [row.id for row in store.list_analyses(repository.id)] == [
            first_analysis.id,
            second_analysis.id,
        ]
        latest = store.get_latest_generated_prompts(repository.id)
        assert latest["coding"].version == 2
        assert latest["merge"].version == 2
        assert latest["review"].version == 1
        assert all(
            prompt.analysis_id == second_analysis.id for prompt in latest.values()
        )
        for role, body in second_bodies.items():
            assert (git_repo / f".borg/prompts/{role}.system.md").read_text() == body

    second_calls = {
        spec.result_path.stem.rsplit(".", 1)[-1]: spec
        for spec in second_adapter.calls
    }
    assert "# Prior coding prompt" in second_calls["coding"].user_prompt
    assert "# Prior merge prompt" in second_calls["merge"].user_prompt
    assert "# Prior review prompt" not in second_calls["review"].user_prompt


def test_stable_publish_failure_does_not_commit_prompt_metadata(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Repository(root=git_repo)
    first_body = "# Coding v1\n\nFirst complete coding prompt body."
    _first_adapter, first_selected = _selected_adapter(
        git_repo,
        {"coding": first_body},
    )

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        first_analysis = _append_analysis(store, repository, score=2)
        first_run = generate_role_prompts(
            repository,
            first_analysis,
            store,
            first_selected,
            artifact_dir=git_repo / "artifacts",
            roles=("coding",),
        )[0]
        assert first_run.ok

        second_analysis = _append_analysis(
            store,
            repository,
            score=4,
            prior=first_analysis,
        )
        failed_body = (
            "# Coding unpublished\n\nThis body must never become the prior prompt."
        )
        _failed_adapter, failed_selected = _selected_adapter(
            git_repo,
            {"coding": failed_body},
        )

        def fail_publish(_source: Path, _destination: Path) -> None:
            raise OSError("stable publish unavailable")

        with monkeypatch.context() as publish_failure:
            publish_failure.setattr(prompts_manager.os, "replace", fail_publish)
            failed_run = generate_role_prompts(
                repository,
                second_analysis,
                store,
                failed_selected,
                artifact_dir=git_repo / "artifacts",
                roles=("coding",),
            )[0]

        stable_path = git_repo / ".borg/prompts/coding.system.md"
        assert not failed_run.ok
        assert failed_run.error == (
            "prompt could not be recorded: stable publish unavailable"
        )
        assert stable_path.read_text(encoding="utf-8") == first_body
        latest_after_failure = store.get_latest_generated_prompts(repository.id)
        assert latest_after_failure["coding"].version == 1
        assert latest_after_failure["coding"].body_md == first_body
        assert list(stable_path.parent.glob(".coding.system.md.*.tmp")) == []

        retry_body = (
            "# Coding v2\n\nSuccessfully published refreshed coding prompt body."
        )
        retry_adapter, retry_selected = _selected_adapter(
            git_repo,
            {"coding": retry_body},
        )
        retry_run = generate_role_prompts(
            repository,
            second_analysis,
            store,
            retry_selected,
            artifact_dir=git_repo / "artifacts",
            roles=("coding",),
        )[0]

        assert retry_run.ok
        assert retry_run.version == 2
        assert stable_path.read_text(encoding="utf-8") == retry_body
        latest_after_retry = store.get_latest_generated_prompts(repository.id)
        assert latest_after_retry["coding"].body_md == retry_body

    assert first_body in retry_adapter.calls[0].user_prompt
    assert failed_body not in retry_adapter.calls[0].user_prompt


def test_metadata_commit_failure_restores_prior_stable_prompt(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Repository(root=git_repo)
    first_body = "# Coding v1\n\nFirst complete coding prompt body."
    _first_adapter, first_selected = _selected_adapter(
        git_repo,
        {"coding": first_body},
    )

    with SqliteStore.open(git_repo / "state.sqlite3") as store:
        store.add_repository(repository)
        first_analysis = _append_analysis(store, repository, score=2)
        first_run = generate_role_prompts(
            repository,
            first_analysis,
            store,
            first_selected,
            artifact_dir=git_repo / "artifacts",
            roles=("coding",),
        )[0]
        assert first_run.ok

        second_analysis = _append_analysis(
            store,
            repository,
            score=4,
            prior=first_analysis,
        )
        failed_body = (
            "# Coding unpublished\n\nCommit failure must not publish this body."
        )
        _failed_adapter, failed_selected = _selected_adapter(
            git_repo,
            {"coding": failed_body},
        )
        original_transaction = store.transaction
        transaction_calls = 0

        @contextmanager
        def fail_outer_commit():
            nonlocal transaction_calls
            transaction_calls += 1
            is_outer_transaction = transaction_calls == 1
            with original_transaction() as connection:
                yield connection
                if is_outer_transaction:
                    raise OSError("metadata commit unavailable")

        with monkeypatch.context() as commit_failure:
            commit_failure.setattr(store, "transaction", fail_outer_commit)
            failed_run = generate_role_prompts(
                repository,
                second_analysis,
                store,
                failed_selected,
                artifact_dir=git_repo / "artifacts",
                roles=("coding",),
            )[0]

        stable_path = git_repo / ".borg/prompts/coding.system.md"
        assert not failed_run.ok
        assert failed_run.error == (
            "prompt could not be recorded: metadata commit unavailable"
        )
        assert stable_path.read_text(encoding="utf-8") == first_body
        latest_after_failure = store.get_latest_generated_prompts(repository.id)
        assert latest_after_failure["coding"].version == 1
        assert latest_after_failure["coding"].body_md == first_body
        assert list(stable_path.parent.glob(".coding.system.md.*.bak")) == []

        retry_body = (
            "# Coding v2\n\nSuccessfully committed refreshed coding prompt body."
        )
        _retry_adapter, retry_selected = _selected_adapter(
            git_repo,
            {"coding": retry_body},
        )
        retry_run = generate_role_prompts(
            repository,
            second_analysis,
            store,
            retry_selected,
            artifact_dir=git_repo / "artifacts",
            roles=("coding",),
        )[0]

        assert retry_run.ok
        assert retry_run.version == 2
        assert stable_path.read_text(encoding="utf-8") == retry_body
