"""CLI contracts for the generation-bound host execution gate."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import click
import pytest
from click.testing import CliRunner
from progress_test_support import (
    FailingStringIO,
    FakeClock,
    WaitableStringIO,
    terminal_screen,
    terminal_text,
)

from betterborg_cli import cli as cli_module
from betterborg_cli.agent_runtime import (
    ApiAgentRole,
    CancellationToken,
    MockAdapter,
    SelectedAgent,
    run_captured,
)
from betterborg_cli.cli import CliRunContext, cli
from betterborg_cli.host_execution import (
    HostCommand,
    HostExecutionResult,
    HostPreflightPlan,
    HostSchedulerConfig,
    ScheduledTaskContext,
)
from betterborg_cli.planning import TaskPublisher
from betterborg_cli.progress import (
    AgentActivity,
    AgentActivityKind,
    ChildSpec,
    RunProgress,
    StageSpec,
    StageState,
)
from betterborg_cli.repository_config import AgentStage
from betterborg_cli.store import (
    BorgState,
    ExecutionRunStatus,
    PlanApproval,
    SqliteStore,
    TaskRuntimeStatus,
)


def _task_body(round_number: int) -> dict[str, object]:
    return {
        "stage": "08-estimate-publish",
        "stem": f"{round_number:02d}-execute-gate",
        "title": f"Execute generation {round_number}",
        "why": "The approved generation needs a host execution gate.",
        "scope": ["Exercise the generation-bound gate."],
        "implementation_notes": [],
        "acceptance_criteria": ["The generation is gated."],
        "tests": ["Verify the execution decision."],
        "dependencies": [],
        "out_of_scope": [],
        "plan_refs": ["P1.deliverable.1"],
        "estimate_complexity": "small",
    }


def _seed_executable_generation(
    root: Path,
    planning_cli_repository,
    approved_task_generation,
    *,
    name: str = "execute-gate",
    round_number: int = 1,
    task_count: int = 1,
    task_titles: tuple[str, ...] | None = None,
    approval: PlanApproval | None = None,
):
    if approval is None:
        repository, paths = planning_cli_repository(root, name)
    else:
        paths = cli_module.RepoPaths.discover(root)
        config = cli_module.load_repository_config(paths)
        with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
            repository = store.get_repository(config.repository_id)
        assert repository is not None

    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, name)
        assert borg is not None
        if borg.state is not BorgState.READY_TO_EXECUTE:
            borg = store.compare_and_set_borg_state(
                borg.id,
                expected_state=borg.state,
                expected_version=borg.state_version,
                new_state=BorgState.READY_TO_EXECUTE,
            )
        if approval is None:
            approval = PlanApproval(
                borg_id=borg.id,
                plan_digest="sha256:approved-plan",
                manifest={
                    "plan": {
                        "title": f"Deliver {name}",
                        "summary": "Ship the completed Betterborg project.",
                        "phases": [
                            {
                                "name": "01-delivery",
                                "title": "Deliver the project",
                                "goal": "Publish the completed local work.",
                            }
                        ],
                    }
                },
            )
            store.append_plan_approval(approval)
        bodies = []
        if task_titles is not None and len(task_titles) != task_count:
            raise ValueError("task titles must match the requested task count")
        for position in range(1, task_count + 1):
            body = _task_body(round_number)
            body["stem"] = f"{round_number:02d}-execute-gate-{position:02d}"
            body["title"] = (
                task_titles[position - 1]
                if task_titles is not None
                else f"Execute task {position}"
            )
            bodies.append(body)
        fixture = approved_task_generation(
            store,
            borg,
            approval,
            body=bodies,
            round_number=round_number,
            task_ref=f"T-{round_number}",
        )
        publication = TaskPublisher(repository, store).publish(
            fixture.generation.id
        )
    return repository, paths, borg, approval, fixture, publication


def _execution_result(
    status: ExecutionRunStatus = ExecutionRunStatus.COMPLETED,
):
    return SimpleNamespace(
        preflight=HostPreflightPlan(
            repository_root=Path.cwd(),
            commands=(),
            prepare_commands=(),
            materialize_commands=(),
            environment_files=(),
            executables=(),
            required_secret_names=(),
            compose_files=(),
            services=(),
        ),
        active_operation_id=None,
        operation_id=uuid4(),
        status=status,
    )


def _trust(
    cli_runner: CliRunner,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(root)
    monkeypatch.setenv("XDG_STATE_HOME", str(root.parent / "machine-state"))
    trusted = cli_runner.invoke(cli, ["trust", "--yes"])
    assert trusted.exit_code == 0, trusted.output


def _create_project_branch(root: Path, name: str) -> str:
    branch = f"project/{name}"
    subprocess.run(
        ["git", "-C", str(root), "branch", branch, "HEAD"],
        check=True,
    )
    return _project_branch_sha(root, name)


def _project_branch_sha(root: Path, name: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", f"project/{name}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _add_bare_origin(root: Path, name: str) -> Path:
    remote = root.parent / f"{name}-origin.git"
    subprocess.run(["git", "init", "--quiet", "--bare", str(remote)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "remote", "add", "origin", str(remote)],
        check=True,
    )
    return remote


def _remote_project_sha(remote: Path, name: str) -> str | None:
    result = subprocess.run(
        [
            "git",
            "--git-dir",
            str(remote),
            "rev-parse",
            "--verify",
            f"refs/heads/project/{name}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _configure_github_origin(root: Path, remote: Path, repository: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "remote",
            "set-url",
            "origin",
            f"https://github.com/{repository}.git",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "remote",
            "set-url",
            "--add",
            "--push",
            "origin",
            str(remote),
        ],
        check=True,
    )


def _install_fake_gh(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    binary_dir = root.parent / f"{root.name}-fake-gh-bin"
    binary_dir.mkdir()
    args_path = root.parent / f"{root.name}-fake-gh-args"
    body_path = root.parent / f"{root.name}-fake-gh-body"
    executable = binary_dir / "gh"
    executable.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = auth ]; then\n"
        "  if [ \"${FAKE_GH_AUTH_FAIL:-}\" = 1 ]; then\n"
        "    echo 'not logged into github.com' >&2\n"
        "    exit 1\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = repo ]; then\n"
        "  echo main\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n' \"$@\" > \"$FAKE_GH_ARGS\"\n"
        "cat > \"$FAKE_GH_BODY\"\n"
        "if [ \"${FAKE_GH_PR_FAIL:-}\" = 1 ]; then\n"
        "  echo 'pull request creation rejected' >&2\n"
        "  exit 1\n"
        "fi\n"
        "echo 'https://github.com/acme/widgets/pull/42'\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("FAKE_GH_ARGS", str(args_path))
    monkeypatch.setenv("FAKE_GH_BODY", str(body_path))
    monkeypatch.setenv("PATH", f"{binary_dir}:{os.environ['PATH']}")
    return args_path, body_path


def test_execute_requires_trust_then_approves_resumes_and_regates_generation(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, paths, borg, approval, first, _publication = (
        _seed_executable_generation(
            committed_git_repo,
            planning_cli_repository,
            approved_task_generation,
        )
    )
    monkeypatch.chdir(committed_git_repo)
    monkeypatch.setenv(
        "XDG_STATE_HOME", str(committed_git_repo.parent / "machine-state")
    )
    calls: list[object] = []
    config_calls: list[Path] = []
    load_config = cli_module.load_repository_config

    def observed_load_config(observed_paths):
        config_calls.append(observed_paths.root)
        return load_config(observed_paths)

    def invoke_host(
        _paths,
        store,
        _config,
        _repository_id,
        borg_id,
        generation_id,
        *,
        cancel,
        progress,
    ):
        # Host execution owns run acquisition, so observing the immutable row
        # here proves the gate commits before any claim can occur.
        decision = store.get_current_execution_decision(borg_id)
        assert decision is not None
        assert decision.generation_id == generation_id
        assert cancel is not None
        assert progress is not None
        calls.append(decision)
        return _execution_result()

    monkeypatch.setattr(cli_module, "_invoke_host_execution", invoke_host)
    monkeypatch.setattr(cli_module, "load_repository_config", observed_load_config)

    untrusted = cli_runner.invoke(cli, ["execute", "execute-gate", "--auto-execute"])
    assert untrusted.exit_code == 1
    assert "workspace is not trusted" in untrusted.output
    assert calls == []
    assert config_calls == []

    _trust(cli_runner, committed_git_repo, monkeypatch)
    approved = cli_runner.invoke(
        cli,
        ["execute", "execute-gate"],
        input="y\n",
    )
    assert approved.exit_code == 0, approved.output
    assert approved.output.startswith("⠋ Estimate and decision")
    assert "DUMMY DATA" in approved.output
    assert "✔ Estimate and decision" in approved.output
    assert "approved" in approved.output
    assert "Recorded execution estimate approved" in approved.output
    assert approved.output.count("none failed or stopped.") == 1
    assert approved.output.index(" finished in ") < approved.output.index(
        "Execution operation"
    )
    assert len(calls) == 1
    assert config_calls == [committed_git_repo]
    assert calls[0].decision == "approved"
    assert calls[0].source == "interactive"

    resumed = cli_runner.invoke(cli, ["execute", "execute-gate"])
    assert resumed.exit_code == 0, resumed.output
    assert "Using recorded execution decision" in resumed.output
    assert "Approve this estimate" not in resumed.output
    assert len(calls) == 2
    assert calls[1].id == calls[0].id

    _repository, _paths, _borg, _approval, second, _publication = (
        _seed_executable_generation(
            committed_git_repo,
            planning_cli_repository,
            approved_task_generation,
            round_number=2,
            approval=approval,
        )
    )
    regated = cli_runner.invoke(
        cli,
        ["execute", "execute-gate"],
        input="y\n",
    )
    assert regated.exit_code == 0, regated.output
    assert "Approve this estimate and begin host execution?" in regated.output
    assert len(calls) == 3
    assert calls[2].generation_id == second.generation.id
    assert calls[2].id != calls[0].id

    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        assert store.get_execution_decision(borg.id, first.generation.id) == calls[0]
        assert store.get_current_execution_decision(borg.id) == calls[2]


def test_auto_execute_records_bypass_without_skipping_workspace_trust(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repository, paths, borg, _approval, fixture, _publication = (
        _seed_executable_generation(
            committed_git_repo,
            planning_cli_repository,
            approved_task_generation,
            name="auto-execute",
        )
    )
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: _execution_result(),
    )

    result = cli_runner.invoke(
        cli, ["execute", "auto-execute", "--auto-execute"]
    )

    assert result.exit_code == 0, result.output
    assert "Approve this estimate" not in result.output
    assert "Recorded execution estimate bypassed" in result.output
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        decision = store.get_current_execution_decision(borg.id)
    assert decision is not None
    assert decision.generation_id == fixture.generation.id
    assert decision.decision == "bypassed"
    assert decision.source == "auto_execute"
    assert decision.snapshot["generation_id"] == str(fixture.generation.id)


def test_execute_declines_under_suspended_progress_without_invoking_host(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repository, paths, borg, _approval, _fixture, _publication = (
        _seed_executable_generation(
            committed_git_repo,
            planning_cli_repository,
            approved_task_generation,
            name="declined-execution",
        )
    )
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: pytest.fail(
            "declined execution must not invoke the host"
        ),
    )

    result = cli_runner.invoke(
        cli,
        ["execute", "declined-execution"],
        input="n\n",
    )

    assert result.exit_code == 1
    assert "✔ Estimate and decision" in result.output
    assert "declined" in result.output
    assert result.output.index("[y/N]: n") < result.output.index(
        "✔ Estimate and decision"
    )
    assert result.output.index("✔ Estimate and decision") < (
        result.output.index(" finished in ")
    )
    assert "Aborted!" in result.output
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        assert store.get_current_execution_decision(borg.id) is None


def test_execute_threads_one_control_context_and_suspends_trust_and_confirmation(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "execution-control"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    _trust(cli_runner, committed_git_repo, monkeypatch)

    class TrackingProgress(RunProgress):
        suspended = False
        suspension_count = 0

        @contextmanager
        def suspend(self):
            self.suspension_count += 1
            self.suspended = True
            try:
                with super().suspend() as progress:
                    yield progress
            finally:
                self.suspended = False

    token = CancellationToken()
    stream = StringIO()
    progress = TrackingProgress(stream=stream, clock=FakeClock())
    run = CliRunContext(token, progress)
    discovered_tokens: list[CancellationToken | None] = []
    host_contexts: list[tuple[CancellationToken | None, RunProgress | None]] = []
    discover = cli_module.RepoPaths.discover

    def observed_discover(*args, **kwargs):
        discovered_tokens.append(kwargs.get("cancel"))
        return discover(*args, **kwargs)

    def confirm(*_args, **_kwargs):
        assert progress.suspended
        return True

    def invoke_host(*_args, cancel=None, progress=None):
        host_contexts.append((cancel, progress))
        return _execution_result()

    monkeypatch.setattr(cli_module.RepoPaths, "discover", observed_discover)
    monkeypatch.setattr(cli_module.click, "confirm", confirm)
    monkeypatch.setattr(cli_module, "_invoke_host_execution", invoke_host)

    result = cli_runner.invoke(cli, ["execute", name], obj=run)

    assert result.exit_code == 0, result.output
    assert discovered_tokens
    assert all(observed is token for observed in discovered_tokens)
    assert host_contexts == [(token, progress)]
    assert progress.suspension_count == 3
    assert progress.closed
    assert stream.getvalue().splitlines()[0] == (
        "⠋ Estimate and decision  0:00  thinking"
    )
    estimate = progress.stages["estimate-decision"]
    assert estimate.state is StageState.COMPLETED
    assert estimate.result == "approved"
    preflight = progress.stages["preflight"]
    assert preflight.state is StageState.COMPLETED
    assert preflight.result == "ready"


@pytest.mark.parametrize("blocked_seam", ["configuration", "publication"])
def test_execute_projects_the_first_live_frame_before_repository_setup_returns(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
    blocked_seam: str,
) -> None:
    name = f"execution-preview-{blocked_seam}"
    _repository, _paths, _borg, _approval, fixture, _publication = (
        _seed_executable_generation(
            committed_git_repo,
            planning_cli_repository,
            approved_task_generation,
            name=name,
        )
    )
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    stream = WaitableStringIO(interactive=True)
    reporters: list[RunProgress] = []
    frames: list[tuple[RunProgress, tuple[str, ...]]] = []
    real_progress = RunProgress
    real_refresh = RunProgress._refresh_transient

    def progress_factory(**kwargs) -> RunProgress:
        progress = real_progress(stream=stream, width=100, **kwargs)
        reporters.append(progress)
        return progress

    def observed_refresh(progress, *, started=None, parent_label=None) -> None:
        frame = tuple(line.plain for line in progress._live_lines())
        if frame and not progress._suspension_depth:
            frames.append((progress, frame))
        real_refresh(progress, started=started, parent_label=parent_label)

    entered = threading.Event()
    release = threading.Event()
    actual_load_config = cli_module.load_repository_config

    def blocked_load_config(paths):
        entered.set()
        assert release.wait(timeout=2)
        return actual_load_config(paths)

    actual_inspect = TaskPublisher.inspect_current_task_files

    def blocked_inspect(publisher, borg_id):
        entered.set()
        assert release.wait(timeout=2)
        return actual_inspect(publisher, borg_id)

    monkeypatch.setattr(cli_module, "RunProgress", progress_factory)
    monkeypatch.setattr(RunProgress, "_refresh_transient", observed_refresh)
    if blocked_seam == "configuration":
        monkeypatch.setattr(cli_module, "load_repository_config", blocked_load_config)
    else:
        monkeypatch.setattr(
            TaskPublisher, "inspect_current_task_files", blocked_inspect
        )
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: _execution_result(),
    )
    monkeypatch.setattr(
        cli_module,
        "_push_project_base",
        lambda _git, _name: "pushed",
    )
    monkeypatch.setattr(
        cli_module,
        "_open_rollup_pull_request",
        lambda *_args, **_kwargs: "opened",
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        invocation = executor.submit(
            cli_runner.invoke,
            cli,
            ["execute", name, "--auto-execute", "--push", "--pr"],
        )
        assert entered.wait(timeout=2)
        try:
            progress = next(
                reporter for reporter in reporters if reporter._enabled
            )
            rendered = stream.wait_for(
                lambda value: all(
                    label in terminal_text(value)
                    for label in (
                        "Estimate and decision",
                        "Preflight",
                        "Push project branch",
                        "Open rollup pull request",
                    )
                )
            )
            assert "  ◦ Preflight" in terminal_text(rendered)
            assert "  ◦ Push project branch" in terminal_text(rendered)
            assert "  ◦ Open rollup pull request" in terminal_text(rendered)
            assert tuple(progress.stages) == ("estimate-decision",)
            assert tuple(
                spec.label for spec in progress._projection_snapshot().previews
            ) == (
                "Preflight",
                "Push project branch",
                "Open rollup pull request",
            )
        finally:
            release.set()
        result = invocation.result(timeout=5)

    assert result.exit_code == 0, result.output
    first_frame = next(frame for reporter, frame in frames if reporter is progress)
    estimate_line = next(
        line for line in first_frame if "Estimate and decision" in line
    )
    assert estimate_line[:3] in {
        f"  {frame}" for frame in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    }
    assert "  ◦ Preflight" in first_frame
    assert "  ◦ Push project branch" in first_frame
    assert "  ◦ Open rollup pull request" in first_frame
    task_key = str(fixture.task.id)
    assert progress.stages["preflight"].state is StageState.COMPLETED
    assert progress.stages["push-project"].state is StageState.COMPLETED
    assert progress.stages["rollup-pr"].state is StageState.COMPLETED
    assert task_key not in progress.stages


def test_execute_adopts_requested_follow_up_previews_in_action_order(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "follow-up-preview-adoption"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    stream = WaitableStringIO(interactive=True)
    progress = RunProgress(stream=stream, width=100, heartbeat_interval=0.01)
    push_entered = threading.Event()
    release_push = threading.Event()
    pr_entered = threading.Event()
    release_pr = threading.Event()
    actions: list[str] = []

    def blocked_push(_git, _name) -> str:
        actions.append("push")
        push_entered.set()
        assert release_push.wait(timeout=2)
        return "pushed"

    def blocked_pr(*_args, **_kwargs) -> str:
        actions.append("pr")
        pr_entered.set()
        assert release_pr.wait(timeout=2)
        return "opened"

    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: _execution_result(),
    )
    monkeypatch.setattr(cli_module, "_push_project_base", blocked_push)
    monkeypatch.setattr(cli_module, "_open_rollup_pull_request", blocked_pr)

    with ThreadPoolExecutor(max_workers=1) as executor:
        invocation = executor.submit(
            cli_runner.invoke,
            cli,
            ["execute", name, "--auto-execute", "--push", "--pr"],
            obj=CliRunContext(CancellationToken(), progress),
        )
        assert push_entered.wait(timeout=2)
        try:
            push_frame = stream.wait_for(
                lambda value: "  ◦ Open rollup pull request"
                in terminal_screen(value)
            )
            assert "  ◦ Open rollup pull request" in terminal_screen(push_frame)
            assert progress.stages["push-project"].state is StageState.RUNNING
            assert "rollup-pr" not in progress.stages
            preview_keys = tuple(
                preview.key
                for preview in progress._projection_snapshot().previews
            )
            assert preview_keys.count("rollup-pr") == 1
            assert "push-project" not in preview_keys

            release_push.set()
            assert pr_entered.wait(timeout=2)
            pr_frame = stream.wait_for(
                lambda value: any(
                    "Open rollup pull request" in line
                    for line in terminal_screen(value).splitlines()
                )
            )
            assert progress.stages["push-project"].state is StageState.COMPLETED
            assert progress.stages["rollup-pr"].state is StageState.RUNNING
            assert all(
                preview.key != "rollup-pr"
                for preview in progress._projection_snapshot().previews
            )
            assert sum(
                "Open rollup pull request" in line
                for line in terminal_screen(pr_frame).splitlines()
            ) == 1
        finally:
            release_push.set()
            release_pr.set()
        result = invocation.result(timeout=5)

    assert result.exit_code == 0, result.output
    assert actions == ["push", "pr"]
    assert progress.stages["push-project"].result == "pushed"
    assert progress.stages["rollup-pr"].result == "opened"


def test_failed_push_discards_undeclared_pr_preview(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "failed-push-skips-pr"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    _trust(cli_runner, committed_git_repo, monkeypatch)
    stream = StringIO()
    progress = RunProgress(stream=stream, heartbeat_interval=0.01)
    pr_called = False

    def rejected_push(_git, _name) -> str:
        raise click.ClickException("push rejected")

    def unexpected_pr(*_args, **_kwargs) -> str:
        nonlocal pr_called
        pr_called = True
        return "opened"

    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: _execution_result(),
    )
    monkeypatch.setattr(cli_module, "_push_project_base", rejected_push)
    monkeypatch.setattr(cli_module, "_open_rollup_pull_request", unexpected_pr)

    result = cli_runner.invoke(
        cli,
        ["execute", name, "--auto-execute", "--push", "--pr"],
        obj=CliRunContext(CancellationToken(), progress),
    )

    assert result.exit_code == 1
    assert result.output.encode().endswith(b"Error: push rejected\n")
    assert not pr_called
    assert "rollup-pr" not in progress.stages
    assert progress._projection_snapshot().previews == ()
    assert not any(
        token in stream.getvalue()
        for token in (
            "✔ Open rollup pull request",
            "✖ Open rollup pull request",
            "■ Open rollup pull request",
        )
    )
    assert (
        "2 of 3 stages finished in 0:00; 1 failed and 0 stopped."
        in stream.getvalue()
    )


@pytest.mark.parametrize(
    (
        "task_count",
        "completed_count",
        "follow_up_arguments",
        "expected_live_account",
        "expected_summary_count",
    ),
    [
        (1, 0, ("--push", "--pr"), None, 5),
        (14, 3, (), "3 done · 2 running · 9 pending", 16),
    ],
)
def test_execute_projection_survives_concrete_setup_and_scheduler_adoption(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
    task_count: int,
    completed_count: int,
    follow_up_arguments: tuple[str, ...],
    expected_live_account: str | None,
    expected_summary_count: int,
) -> None:
    name = f"execution-projection-{task_count}"
    long_execution_titles = (
        "auth-refactor",
        "rate-limiter",
        "config-loader",
        "webhook-retry",
        "db-migration",
        *(f"pending-task-{number}" for number in range(6, 15)),
    )
    _repository, paths, borg, _approval, _fixture, publication = (
        _seed_executable_generation(
            committed_git_repo,
            planning_cli_repository,
            approved_task_generation,
            name=name,
            task_count=task_count,
            task_titles=long_execution_titles if task_count == 14 else None,
        )
    )
    config_path = paths.tracked_dir / "config.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + "\n[execution]\njobs = 2\nreview_passes = 3\n",
        encoding="utf-8",
    )
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)

    task_records = tuple(item.task for item in publication.files)
    task_keys = tuple(str(task.id) for task in task_records)
    task_labels = tuple(task.title for task in task_records)
    stream = WaitableStringIO(interactive=True)
    progress_clock = FakeClock()
    progress_ref: dict[str, RunProgress] = {}
    real_progress = RunProgress

    def progress_factory(**kwargs) -> RunProgress:
        progress = real_progress(
            stream=stream if kwargs.get("enabled", True) else StringIO(),
            clock=progress_clock,
            width=100,
            **kwargs,
        )
        if kwargs.get("enabled", True):
            progress_ref["progress"] = progress
        return progress

    follow_up_labels = (
        ("Push project branch", "Open rollup pull request")
        if follow_up_arguments
        else ()
    )
    setup_checkpoints: list[str] = []

    def assert_setup_projection(checkpoint: str) -> None:
        progress = progress_ref["progress"]
        snapshot = progress._projection_snapshot()
        stages = {stage.record.key: stage.record for stage in snapshot.stages}
        assert stages["estimate-decision"].state is StageState.COMPLETED
        assert stages["preflight"].state is StageState.COMPLETED
        assert not set(task_keys).intersection(stages)
        assert tuple(spec.label for spec in snapshot.previews) == (
            *task_labels,
            *follow_up_labels,
        )
        assert snapshot.cohort_keys == frozenset(
            (*task_keys, *(spec.key for spec in snapshot.previews[-2:]))
            if follow_up_arguments
            else task_keys
        )
        setup_checkpoints.append(checkpoint)

    actual_write_estimate = cli_module._write_execution_estimate
    projected_follow_ups: tuple[StageSpec, ...] = ()

    def observed_write_estimate(project_name, estimate) -> None:
        nonlocal projected_follow_ups
        progress = progress_ref["progress"]
        snapshot = progress._projection_snapshot()
        assert tuple(spec.label for spec in snapshot.previews) == (
            "Preflight",
            *task_labels,
            *follow_up_labels,
        )
        assert tuple(progress.stages) == ("estimate-decision",)
        projected_follow_ups = tuple(
            spec for spec in snapshot.previews if spec.label in follow_up_labels
        )
        setup_checkpoints.append("decision")
        actual_write_estimate(project_name, estimate)

    plan = HostPreflightPlan(
        repository_root=committed_git_repo,
        commands=(),
        prepare_commands=(HostCommand("prepare", ("prepare",), "."),),
        materialize_commands=(),
        environment_files=(),
        executables=(),
        required_secret_names=(),
        compose_files=(),
        services=(),
    )

    class ObservedPreflight:
        def __init__(self, *_args, **_kwargs) -> None:
            self.validated_result = None

        def validate(self, *_args, **_kwargs) -> HostPreflightPlan:
            progress = progress_ref["progress"]
            snapshot = progress._projection_snapshot()
            assert snapshot.stages[-1].record.key == "preflight"
            assert snapshot.stages[-1].record.state is StageState.RUNNING
            assert tuple(spec.label for spec in snapshot.previews) == (
                *task_labels,
                *follow_up_labels,
            )
            setup_checkpoints.append("preflight-validation")
            self.validated_result = plan
            return plan

    selected_stages: list[AgentStage] = []

    def select_observed_agent(_config, stage, *_args, **_kwargs):
        assert_setup_projection(f"agent-{stage.value}")
        selected_stages.append(stage)
        return SimpleNamespace(name="mock", model="test-model", effort=None)

    class ObservedWorktrees:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def prepare_current_task_worktrees(self, *_args, **_kwargs):
            assert_setup_projection("worktree-preparation")
            return [
                SimpleNamespace(task_id=task.id, path=committed_git_repo)
                for task in task_records
            ]

        def refresh_unstarted_task_worktree(self, *_args, **_kwargs) -> bool:
            return False

    class ObservedCompose:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def cleanup_stale_projects(self, *_args, **_kwargs) -> tuple[object, ...]:
            return ()

    active_lock = threading.Lock()
    active_started = 0
    target_active = min(2, task_count - completed_count)
    scheduler_blocked = threading.Event()
    release_scheduler = threading.Event()

    class BlockingRuntime:
        def __init__(self, runtime_plan, **_kwargs) -> None:
            self.plan = runtime_plan

        def with_secret_values(self, _secret_values):
            return self

        def prepare_reusable_caches(self, *_args, **_kwargs) -> tuple[str, ...]:
            assert_setup_projection("cache-preparation")
            return ("prepared",)

        def __call__(self, context: ScheduledTaskContext) -> TaskRuntimeStatus:
            nonlocal active_started
            task_index = next(
                index
                for index, task in enumerate(task_records)
                if task.id == context.claim.task_id
            )
            if task_index < completed_count:
                durations = (134.0, 182.0, 108.0)
                progress = progress_ref["progress"]
                with active_lock:
                    progress_clock.now = durations[task_index]
                    progress.stages[context.stage_key].started_at = 0.0
                    context.transition(
                        TaskRuntimeStatus.CLAIMED,
                        TaskRuntimeStatus.DONE,
                    )
                return TaskRuntimeStatus.DONE

            if task_count == 14 and task_index == 3:
                context.transition(
                    TaskRuntimeStatus.CLAIMED,
                    TaskRuntimeStatus.REVIEW,
                    resume_phase="review",
                    review_round=1,
                )
                activity_sink = context.activity_sink("review")
                assert activity_sink is not None
                activity_sink(
                    AgentActivity(
                        AgentActivityKind.READING,
                        "src/webhook/retry.go",
                    )
                )
            elif task_count == 14 and task_index == 4:
                context.transition(
                    TaskRuntimeStatus.CLAIMED,
                    TaskRuntimeStatus.CODING,
                    resume_phase="coding",
                )
                activity_sink = context.activity_sink("coding")
                assert activity_sink is not None
                activity_sink(AgentActivity(AgentActivityKind.THINKING))
            with active_lock:
                active_started += 1
                if active_started == target_active:
                    scheduler_blocked.set()
            assert release_scheduler.wait(timeout=3)
            context.transition(
                context.runtime.status,
                TaskRuntimeStatus.DONE,
            )
            return TaskRuntimeStatus.DONE

    actual_reconcile = SqliteStore.reconcile_expired_execution_runs
    stale_checks = 0

    def observed_reconcile(store, *args, **kwargs):
        nonlocal stale_checks
        progress = progress_ref.get("progress")
        if progress is not None and not set(task_keys).intersection(progress.stages):
            stale_checks += 1
            assert_setup_projection(f"stale-cleanup-{stale_checks}")
        return actual_reconcile(store, *args, **kwargs)

    actual_acquire = SqliteStore.acquire_execution_run

    def observed_acquire(store, *args, **kwargs):
        acquisition = actual_acquire(store, *args, **kwargs)
        assert acquisition.acquired
        assert_setup_projection("run-acquisition")
        return acquisition

    actual_follow_up = cli_module._run_execution_follow_up
    adopted_follow_ups: list[StageSpec] = []

    def observed_follow_up(progress, spec, action, **kwargs) -> None:
        adopted_follow_ups.append(spec)
        actual_follow_up(progress, spec, action, **kwargs)

    actual_scheduler_config = HostSchedulerConfig
    monkeypatch.setattr(cli_module, "RunProgress", progress_factory)
    monkeypatch.setattr(
        cli_module, "_write_execution_estimate", observed_write_estimate
    )
    monkeypatch.setattr(cli_module, "HostPreflight", ObservedPreflight)
    monkeypatch.setattr(cli_module, "select_agent", select_observed_agent)
    monkeypatch.setattr(
        cli_module, "HostEnvironmentManager", lambda *_a, **_k: object()
    )
    monkeypatch.setattr(cli_module, "HostComposeManager", ObservedCompose)
    monkeypatch.setattr(cli_module, "HostWorktreeManager", ObservedWorktrees)
    monkeypatch.setattr(cli_module, "HostCodingPhase", lambda *_a, **_k: object())
    monkeypatch.setattr(cli_module, "HostReviewFixPhase", lambda *_a, **_k: object())
    monkeypatch.setattr(cli_module, "HostMergePhase", lambda *_a, **_k: object())
    monkeypatch.setattr(cli_module, "HostSanityPhase", lambda *_a, **_k: object())
    monkeypatch.setattr(cli_module, "HostTaskRuntime", BlockingRuntime)
    monkeypatch.setattr(
        cli_module,
        "HostSchedulerConfig",
        lambda *, jobs, review_passes: actual_scheduler_config(
            jobs=jobs,
            review_passes=review_passes,
            poll_interval_seconds=0.005,
        ),
    )
    monkeypatch.setattr(
        SqliteStore, "reconcile_expired_execution_runs", observed_reconcile
    )
    monkeypatch.setattr(SqliteStore, "acquire_execution_run", observed_acquire)
    monkeypatch.setattr(
        cli_module, "_run_execution_follow_up", observed_follow_up
    )
    monkeypatch.setattr(cli_module, "_push_project_base", lambda *_args: "pushed")
    monkeypatch.setattr(
        cli_module,
        "_open_rollup_pull_request",
        lambda *_args, **_kwargs: "opened",
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        invocation = executor.submit(
            cli_runner.invoke,
            cli,
            ["execute", name, "--auto-execute", *follow_up_arguments],
        )
        assert scheduler_blocked.wait(timeout=3)
        try:
            progress = progress_ref["progress"]
            if task_count == 14:
                with progress._lock:
                    progress_clock.now = 72.0
                    progress.stages[task_keys[3]].started_at = 31.0
                    progress.stages[task_keys[4]].started_at = 0.0
                    monkeypatch.setattr(
                        progress,
                        "_current_spinner_frame",
                        lambda _record=None: "⠋",
                    )
                progress.refresh()
            snapshot = progress._projection_snapshot()
            task_stages = {
                stage.record.key: stage.record
                for stage in snapshot.stages
                if stage.record.key in task_keys
            }
            assert len(task_stages) == task_count
            assert sum(
                stage.state is StageState.COMPLETED
                for stage in task_stages.values()
            ) == completed_count
            assert sum(
                stage.state is StageState.RUNNING
                for stage in task_stages.values()
            ) == target_active
            retained = tuple(
                progress.stages[key]
                for key, stage in task_stages.items()
                if stage.state is StageState.COMPLETED
            )
            pending = tuple(
                progress.stages[key]
                for key, stage in task_stages.items()
                if stage.state is StageState.PENDING
            )
            assert all(
                not stage.retained
                and stage.started_at is not None
                and stage.finished_at is not None
                for stage in retained
            )
            assert all(
                not stage.retained
                and stage.started_at is None
                and stage.finished_at is None
                for stage in pending
            )
            assert tuple(spec.label for spec in snapshot.previews) == follow_up_labels
            assert len(snapshot.stages) == task_count + 2
            if expected_live_account is not None:
                expected_literal_account = (
                    "✔ auth-refactor    2:14  merged\n"
                    "✔ rate-limiter     3:02  merged\n"
                    "✔ config-loader    1:48  merged\n"
                    "\n"
                    "  3 done · 2 running · 9 pending\n"
                    "  ⠋ webhook-retry   0:41  review (pass 2/3)\n"
                    "      └ reading src/webhook/retry.go\n"
                    "  ⠋ db-migration    1:12  coding\n"
                    "      └ thinking\n"
                    "\n"
                    "  ctrl-c to stop"
                )
                output = stream.wait_for(
                    lambda value: expected_literal_account
                    in terminal_screen(value)
                )
                rendered = terminal_text(output)
                visible = terminal_screen(output)
                assert expected_literal_account in visible
                assert "✔ Estimate and decision" in visible
                assert "✔ Preflight" in visible
                assert expected_live_account in visible
                assert "5 done · 2 running · 9 pending" not in rendered
        finally:
            release_scheduler.set()
        result = invocation.result(timeout=10)

    assert result.exit_code == 0, result.output
    assert selected_stages == [AgentStage.CODING, AgentStage.REVIEW, AgentStage.MERGE]
    assert setup_checkpoints == [
        "decision",
        "preflight-validation",
        "agent-coding",
        "agent-review",
        "agent-merge",
        "stale-cleanup-1",
        "run-acquisition",
        "stale-cleanup-2",
        "worktree-preparation",
        "cache-preparation",
    ]
    assert stale_checks == 2
    assert len(progress.stages) == expected_summary_count
    assert all(
        stage.state is StageState.COMPLETED
        for stage in progress.stages.values()
    )
    assert (
        f"{expected_summary_count} of {expected_summary_count} stages finished "
        f"in {'1:12' if task_count == 14 else '0:00'}; "
        "none failed or stopped."
    ) in terminal_text(stream.getvalue())
    assert len(adopted_follow_ups) == len(projected_follow_ups)
    assert all(
        adopted is projected
        for adopted, projected in zip(
            adopted_follow_ups, projected_follow_ups, strict=True
        )
    )


@pytest.mark.parametrize("interactive", [False, True], ids=["plain", "interactive"])
def test_execute_reporter_finished_rows_match_in_plain_and_interactive_modes(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
    interactive: bool,
) -> None:
    real_progress = RunProgress
    actual_scheduler_config = HostSchedulerConfig
    name = "reporter-parity-run"
    _repository, paths, _borg, _approval, _fixture, publication = (
        _seed_executable_generation(
            committed_git_repo,
            planning_cli_repository,
            approved_task_generation,
            name=name,
            task_titles=("reporter-parity",),
        )
    )
    config_path = paths.tracked_dir / "config.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + "\n[execution]\njobs = 1\nreview_passes = 3\n",
        encoding="utf-8",
    )
    stream = WaitableStringIO(interactive=interactive)
    progress_clock = FakeClock()
    progress_ref: dict[str, RunProgress] = {}
    task = publication.files[0].task
    plan = HostPreflightPlan(
        repository_root=committed_git_repo,
        commands=(),
        prepare_commands=(HostCommand("prepare", ("prepare",), "."),),
        materialize_commands=(),
        environment_files=(),
        executables=(),
        required_secret_names=(),
        compose_files=(),
        services=(),
    )

    def progress_factory(**kwargs) -> RunProgress:
        progress = real_progress(
            stream=stream if kwargs.get("enabled", True) else StringIO(),
            clock=progress_clock,
            width=100,
            **kwargs,
        )
        if kwargs.get("enabled", True):
            progress_ref["progress"] = progress
        return progress

    class ReadyPreflight:
        def __init__(self, *_args, **_kwargs) -> None:
            self.validated_result = None

        def validate(self, *_args, **_kwargs) -> HostPreflightPlan:
            self.validated_result = plan
            return plan

    class PreparedWorktrees:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def prepare_current_task_worktrees(self, *_args, **_kwargs):
            return [SimpleNamespace(task_id=task.id, path=committed_git_repo)]

        def refresh_unstarted_task_worktree(self, *_args, **_kwargs) -> bool:
            return False

    class CleanCompose:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def cleanup_stale_projects(self, *_args, **_kwargs) -> tuple[object, ...]:
            return ()

    class DeterministicRuntime:
        def __init__(self, runtime_plan, **_kwargs) -> None:
            self.plan = runtime_plan

        def with_secret_values(self, _secret_values):
            return self

        def prepare_reusable_caches(self, *_args, **_kwargs) -> tuple[str, ...]:
            return ("prepared",)

        def __call__(self, context: ScheduledTaskContext) -> TaskRuntimeStatus:
            progress = progress_ref["progress"]
            progress.declare_child(
                context.stage_key,
                ChildSpec("checks", "Checks"),
            )
            progress.start_child(context.stage_key, "checks")
            progress_clock.now = 4.0
            progress.complete_child(
                context.stage_key,
                "checks",
                "142 files",
            )
            progress_clock.now = 65.0
            progress.stages[context.stage_key].started_at = 0.0
            context.transition(
                TaskRuntimeStatus.CLAIMED,
                TaskRuntimeStatus.DONE,
            )
            return TaskRuntimeStatus.DONE

    with monkeypatch.context() as run_patch:
        run_patch.delenv("CI", raising=False)
        run_patch.delenv("NO_COLOR", raising=False)
        run_patch.delenv("TERM", raising=False)
        run_patch.setattr(cli_module, "RunProgress", progress_factory)
        run_patch.setattr(cli_module, "HostPreflight", ReadyPreflight)
        run_patch.setattr(
            cli_module,
            "select_agent",
            lambda *_args, **_kwargs: SimpleNamespace(
                name="mock", model="test-model", effort=None
            ),
        )
        run_patch.setattr(
            cli_module, "HostEnvironmentManager", lambda *_a, **_k: object()
        )
        run_patch.setattr(cli_module, "HostComposeManager", CleanCompose)
        run_patch.setattr(cli_module, "HostWorktreeManager", PreparedWorktrees)
        run_patch.setattr(cli_module, "HostCodingPhase", lambda *_a, **_k: object())
        run_patch.setattr(
            cli_module, "HostReviewFixPhase", lambda *_a, **_k: object()
        )
        run_patch.setattr(cli_module, "HostMergePhase", lambda *_a, **_k: object())
        run_patch.setattr(cli_module, "HostSanityPhase", lambda *_a, **_k: object())
        run_patch.setattr(cli_module, "HostTaskRuntime", DeterministicRuntime)
        run_patch.setattr(
            cli_module,
            "HostSchedulerConfig",
            lambda *, jobs, review_passes: actual_scheduler_config(
                jobs=jobs,
                review_passes=review_passes,
                poll_interval_seconds=0.005,
            ),
        )
        _trust(cli_runner, committed_git_repo, run_patch)
        result = cli_runner.invoke(
            cli,
            ["execute", name, "--auto-execute"],
        )

    assert result.exit_code == 0, result.output
    finished_rows = tuple(
        line
        for line in terminal_screen(stream.getvalue()).splitlines()
        if line.startswith(("✔ ", "├ ✔ ", "└ ✔ "))
    )
    assert finished_rows == (
        "✔ Estimate and decision  0:00  bypassed",
        "✔ Preflight              0:00  ready",
        "✔ reporter-parity        1:05  merged",
        "└ ✔ Checks                 0:04  142 files",
    )


def test_execute_without_push_succeeds_without_a_remote(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "local-only"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    local_sha = _create_project_branch(committed_git_repo, name)
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: _execution_result(),
    )

    result = cli_runner.invoke(cli, ["execute", name, "--auto-execute"])

    assert result.exit_code == 0, result.output
    assert ": completed" in result.output
    assert "Pushed" not in result.output
    assert _project_branch_sha(committed_git_repo, name) == local_sha


def test_failure_after_preflight_does_not_reclassify_preflight(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "post-preflight-failure"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    _trust(cli_runner, committed_git_repo, monkeypatch)
    progress = RunProgress(stream=StringIO())

    def fail_after_preflight(*_args, progress=None, **_kwargs):
        assert progress is not None
        progress.complete("preflight", "ready")
        raise RuntimeError("task setup failed")

    monkeypatch.setattr(
        cli_module, "_invoke_host_execution", fail_after_preflight
    )

    result = cli_runner.invoke(
        cli,
        ["execute", name, "--auto-execute"],
        obj=CliRunContext(CancellationToken(), progress),
    )

    assert result.exit_code == 1
    assert "task setup failed" in result.output
    assert progress.stages["preflight"].state is StageState.COMPLETED
    assert progress.stages["preflight"].result == "ready"


def test_push_option_publishes_completed_project_branch_non_force(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "push-success"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    local_sha = _create_project_branch(committed_git_repo, name)
    remote = _add_bare_origin(committed_git_repo, name)
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: _execution_result(),
    )

    result = cli_runner.invoke(cli, ["execute", name, "--push"], input="y\n")

    assert result.exit_code == 0, result.output
    assert ": completed" in result.output
    assert "✔ Push project branch" in result.output
    assert "Pushed project/push-success" in result.output
    assert f"Pushed project/{name} to origin." in result.output
    assert result.output.index(" finished in ") < result.output.index(
        "Execution operation"
    )
    assert _remote_project_sha(remote, name) == local_sha


def test_push_missing_remote_reports_delivery_failure_and_preserves_local_branch(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "missing-remote"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    local_sha = _create_project_branch(committed_git_repo, name)
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: _execution_result(),
    )

    result = cli_runner.invoke(
        cli, ["execute", name, "--auto-execute", "--push"]
    )

    assert result.exit_code == 1
    assert ": completed" in result.output
    assert "✖ Push project branch" in result.output
    assert "Local execution completed, but push" in result.output
    assert "origin" in result.output
    assert result.output.index(" finished in ") < result.output.index(
        "Execution operation"
    )
    assert result.output.index("Execution operation") < result.output.rindex(
        "Local execution completed, but push"
    )
    assert _project_branch_sha(committed_git_repo, name) == local_sha


def test_push_timeout_keeps_existing_error_and_local_branch(
    committed_git_repo: Path,
) -> None:
    name = "push-timeout"
    local_sha = _create_project_branch(committed_git_repo, name)
    observed_timeouts: list[float | None] = []
    observed_environments: list[dict[str, str]] = []

    def runner(command, **kwargs):
        if tuple(command[1:2]) == ("push",):
            observed_timeouts.append(kwargs["timeout"])
            observed_environments.append(dict(kwargs["env"]))
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return run_captured(command, **kwargs)

    git = cli_module.SafeGit(committed_git_repo, command_runner=runner)

    with pytest.raises(click.ClickException, match="timed out after 60 seconds"):
        cli_module._push_project_base(git, name)

    assert observed_timeouts == [60]
    assert observed_environments[0]["GIT_TERMINAL_PROMPT"] == "0"
    assert observed_environments[0]["GCM_INTERACTIVE"] == "never"
    assert _project_branch_sha(committed_git_repo, name) == local_sha


def test_follow_up_autonomous_render_failure_fails_successful_action_stage() -> None:
    stream = FailingStringIO()
    progress = RunProgress(
        stream=stream,
        heartbeat_interval=0.01,
    )
    action_entered = threading.Event()
    release_action = threading.Event()

    def action() -> str:
        action_entered.set()
        assert release_action.wait(timeout=2)
        return "published"

    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(
            cli_module._run_execution_follow_up,
            progress,
            StageSpec("push-project", "Push project branch"),
            action,
        )
        assert action_entered.wait(timeout=2)
        worker = progress._cadence_worker
        assert worker is not None
        stream.fail_next_write()
        worker.join(timeout=2)
        assert not worker.is_alive()
        release_action.set()
        with pytest.raises(
            RuntimeError, match="progress heartbeat failed"
        ) as caught:
            running.result(timeout=2)

    stage = progress.stages["push-project"]
    assert str(caught.value) == "progress heartbeat failed"
    assert stage.state is StageState.FAILED
    assert stage.result == "progress heartbeat failed"
    assert progress._cadence_worker is None
    progress.raise_if_render_failed()


def test_follow_up_action_failure_wins_autonomous_render_failure_race() -> None:
    stream = FailingStringIO()
    progress = RunProgress(stream=stream, heartbeat_interval=0.01)
    action_entered = threading.Event()
    release_action = threading.Event()
    action_error = click.ClickException("push rejected")

    def action() -> str:
        action_entered.set()
        assert release_action.wait(timeout=2)
        raise action_error

    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(
            cli_module._run_execution_follow_up,
            progress,
            StageSpec("push-project", "Push project branch"),
            action,
        )
        assert action_entered.wait(timeout=2)
        worker = progress._cadence_worker
        assert worker is not None
        stream.fail_next_write()
        worker.join(timeout=2)
        assert not worker.is_alive()
        release_action.set()
        with pytest.raises(click.ClickException) as caught:
            running.result(timeout=2)

    stage = progress.stages["push-project"]
    assert caught.value is action_error
    assert stage.state is StageState.FAILED
    assert stage.result == "push rejected"
    assert caught.value.__notes__ == [
        "execution follow-up progress rendering also failed: "
        "progress heartbeat failed"
    ]
    expected = StringIO()
    click.ClickException("push rejected").show(file=expected)
    actual = StringIO()
    caught.value.show(file=actual)
    assert actual.getvalue().encode() == expected.getvalue().encode()
    assert progress._cadence_worker is None
    progress.raise_if_render_failed()


def test_follow_up_completion_output_failure_preserves_completed_stage() -> None:
    stream = FailingStringIO()
    progress = RunProgress(stream=stream)

    def action() -> str:
        stream.fail_next_write()
        return "published"

    with pytest.raises(RuntimeError, match="progress heartbeat failed"):
        cli_module._run_execution_follow_up(
            progress,
            StageSpec("push-project", "Push project branch"),
            action,
        )

    stage = progress.stages["push-project"]
    assert stage.state is StageState.COMPLETED
    assert stage.result == "published"


def test_push_interrupt_reaps_tree_stops_stage_and_preserves_core_report(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    real_process_harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "push-interrupt"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    local_sha = _create_project_branch(committed_git_repo, name)
    _trust(cli_runner, committed_git_repo, monkeypatch)
    operation_id = uuid4()
    resistant = real_process_harness.resistant_argv("execute-push")
    source = r'''
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from betterborg_cli import cli as cli_module
from betterborg_cli.agent_runtime import run_captured
from betterborg_cli.host_execution import HostPreflightPlan
from betterborg_cli.progress import RunProgress, StageState
from betterborg_cli.store import ExecutionRunStatus

repository = Path(sys.argv[1])
marker_root = Path(sys.argv[2])
resistant = tuple(json.loads(sys.argv[3]))
operation_id = UUID(sys.argv[4])
name = sys.argv[5]
actual_safe_git = cli_module.SafeGit


class FastProgress(RunProgress):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, heartbeat_interval=0.01, **kwargs)

    def _render_cadence_frame(self, stopped):
        keep_running = super()._render_cadence_frame(stopped)
        stage = self.stages.get("push-project")
        if (
            stage is not None
            and stage.state is StageState.RUNNING
            and stage.activity is not None
        ):
            (marker_root / "push-heartbeat").write_text(
                "refreshed", encoding="utf-8"
            )
        return keep_running


def runner(command, **kwargs):
    if tuple(command[1:2]) == ("push",):
        (marker_root / "push-token").write_text(
            "bound" if kwargs.get("cancel") is not None else "missing",
            encoding="utf-8",
        )
        return run_captured(resistant, **kwargs)
    return run_captured(command, **kwargs)


def safe_git(cwd, **kwargs):
    (marker_root / "push-activity").write_text(
        "bound" if kwargs.get("activity") is not None else "missing",
        encoding="utf-8",
    )
    return actual_safe_git(cwd, command_runner=runner, **kwargs)


preflight = HostPreflightPlan(
    repository_root=repository,
    commands=(),
    prepare_commands=(),
    materialize_commands=(),
    environment_files=(),
    executables=(),
    required_secret_names=(),
    compose_files=(),
    services=(),
)
cli_module.RunProgress = FastProgress
cli_module.SafeGit = safe_git
cli_module._invoke_host_execution = lambda *_args, **_kwargs: SimpleNamespace(
    preflight=preflight,
    active_operation_id=None,
    operation_id=operation_id,
    status=ExecutionRunStatus.COMPLETED,
)
raise SystemExit(
    cli_module.main(["execute", name, "--auto-execute", "--push"])
)
'''
    process = real_process_harness.launch_python(
        source,
        str(committed_git_repo),
        str(real_process_harness.root),
        json.dumps(resistant),
        str(operation_id),
        name,
        name="execute-push",
    )
    real_process_harness.wait_for_marker("execute-push.child.pid")
    real_process_harness.wait_for_marker("push-heartbeat")
    real_process_harness.signal(process, signal.SIGINT)

    assert real_process_harness.wait_for_exit(process) == 130
    stdout, stderr = process.communicate()
    output = stdout + stderr
    real_process_harness.assert_tree_absent("execute-push")
    assert real_process_harness.wait_for_marker("push-token") == "bound"
    assert real_process_harness.wait_for_marker("push-activity") == "bound"
    assert output.count("⠋ Push project branch") >= 2
    assert "running git push origin refs/heads/project/push-interrupt" in output
    assert "■ Push project branch" in output
    assert "✖ Push project branch" not in output
    assert " finished in " in output
    report = f"Execution operation {operation_id}: completed"
    assert report in output
    assert _project_branch_sha(committed_git_repo, name) == local_sha


def test_push_denies_remote_rewind_instead_of_forcing_project_branch(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "non-force"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    local_sha = _create_project_branch(committed_git_repo, name)
    remote = _add_bare_origin(committed_git_repo, name)
    reference = f"refs/heads/project/{name}"
    subprocess.run(
        ["git", "-C", str(committed_git_repo), "push", "origin", reference],
        check=True,
        capture_output=True,
    )
    tree = subprocess.run(
        ["git", "-C", str(committed_git_repo), "rev-parse", f"{reference}^{{tree}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote_sha = subprocess.run(
        [
            "git",
            "-C",
            str(committed_git_repo),
            "commit-tree",
            tree,
            "-p",
            local_sha,
        ],
        check=True,
        capture_output=True,
        input="remote-only commit\n",
        text=True,
    ).stdout.strip()
    subprocess.run(
        [
            "git",
            "-C",
            str(committed_git_repo),
            "push",
            "origin",
            f"{remote_sha}:{reference}",
        ],
        check=True,
        capture_output=True,
    )
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: _execution_result(),
    )

    result = cli_runner.invoke(
        cli, ["execute", name, "--auto-execute", "--push"]
    )

    assert result.exit_code == 1
    assert ": completed" in result.output
    assert "non-fast-forward" in result.output or "fetch first" in result.output
    assert _project_branch_sha(committed_git_repo, name) == local_sha
    assert _remote_project_sha(remote, name) == remote_sha


def test_push_credential_failure_preserves_local_branch(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "credential-failure"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    local_sha = _create_project_branch(committed_git_repo, name)
    remote = _add_bare_origin(committed_git_repo, name)
    credential_denial = committed_git_repo.parent / "deny-ssh-credential"
    credential_denial.write_text(
        "#!/bin/sh\n"
        "echo 'git@localhost: Permission denied (publickey).' >&2\n"
        "exit 255\n",
        encoding="utf-8",
    )
    credential_denial.chmod(0o755)
    subprocess.run(
        [
            "git",
            "-C",
            str(committed_git_repo),
            "remote",
            "set-url",
            "origin",
            f"ssh://git@localhost{remote}",
        ],
        check=True,
    )
    monkeypatch.setenv("GIT_SSH_COMMAND", str(credential_denial))
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: _execution_result(),
    )

    result = cli_runner.invoke(
        cli, ["execute", name, "--auto-execute", "--push"]
    )

    assert result.exit_code == 1
    assert ": completed" in result.output
    assert "Permission denied (publickey)" in result.output
    assert _project_branch_sha(committed_git_repo, name) == local_sha
    assert _remote_project_sha(remote, name) is None


def test_push_and_pr_wait_for_local_execution_to_complete(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "active-execution"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    _create_project_branch(committed_git_repo, name)
    _trust(cli_runner, committed_git_repo, monkeypatch)
    active = _execution_result(ExecutionRunStatus.RUNNING)
    active.active_operation_id = uuid4()
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: active,
    )

    result = cli_runner.invoke(
        cli, ["execute", name, "--auto-execute", "--push", "--pr"]
    )

    assert result.exit_code == 0, result.output
    assert f"Execution already active: {active.active_operation_id}" in result.output
    assert "push" not in result.output.casefold()
    assert "pull request" not in result.output.casefold()


def test_pr_option_opens_rollup_with_prd_and_rendered_plan(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "pr-success"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    local_sha = _create_project_branch(committed_git_repo, name)
    remote = _add_bare_origin(committed_git_repo, name)
    reference = f"refs/heads/project/{name}"
    subprocess.run(
        ["git", "-C", str(committed_git_repo), "push", str(remote), reference],
        check=True,
        capture_output=True,
    )
    _configure_github_origin(committed_git_repo, remote, "acme/widgets")
    args_path, body_path = _install_fake_gh(committed_git_repo, monkeypatch)
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: _execution_result(),
    )

    result = cli_runner.invoke(cli, ["execute", name, "--auto-execute", "--pr"])

    assert result.exit_code == 0, result.output
    assert ": completed" in result.output
    assert "Pushed" not in result.output
    assert "✔ Open rollup pull request" in result.output
    assert "Opened rollup pull request" in result.output
    assert "Opened rollup pull request" in result.output
    assert args_path.read_text(encoding="utf-8").splitlines() == [
        "pr",
        "create",
        "--repo",
        "acme/widgets",
        "--head",
        f"project/{name}",
        "--base",
        "main",
        "--title",
        f"Deliver {name}",
        "--body-file",
        "-",
    ]
    body = body_path.read_text(encoding="utf-8")
    assert body.startswith(f"# {name}\n\n---\n\n# Deliver {name}")
    assert "### 01-delivery — Deliver the project" in body
    assert _project_branch_sha(committed_git_repo, name) == local_sha
    assert _remote_project_sha(remote, name) == local_sha


def test_rollup_pr_commands_keep_runner_contract_and_report_activity(
    committed_git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel = CancellationToken()
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    activities = []
    gh = "/test/bin/gh"

    def runner(command, **kwargs):
        argv = tuple(command)
        calls.append((argv, kwargs))
        if argv[0] == "git":
            stdout = "https://github.com/acme/widgets.git\n"
        elif argv[1:3] == ("repo", "view"):
            stdout = "main\n"
        elif argv[1:3] == ("pr", "create"):
            stdout = "https://github.com/acme/widgets/pull/42\n"
        else:
            stdout = ""
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(cli_module.shutil, "which", lambda _name: gh)

    result = cli_module._open_rollup_pull_request(
        committed_git_repo,
        "runner-contract",
        {"title": "Runner contract"},
        None,
        cancel=cancel,
        command_runner=runner,
        activity=activities.append,
    )

    assert result.endswith(": https://github.com/acme/widgets/pull/42")
    assert [call[0] for call in calls] == [
        (
            "git",
            "-C",
            str(committed_git_repo),
            "remote",
            "get-url",
            "origin",
        ),
        (gh, "auth", "status", "--active", "--hostname", "github.com"),
        (
            gh,
            "repo",
            "view",
            "acme/widgets",
            "--json",
            "defaultBranchRef",
            "--jq",
            ".defaultBranchRef.name",
        ),
        (
            gh,
            "pr",
            "create",
            "--repo",
            "acme/widgets",
            "--head",
            "project/runner-contract",
            "--base",
            "main",
            "--title",
            "Runner contract",
            "--body-file",
            "-",
        ),
    ]
    assert [call[1]["timeout"] for call in calls] == [10, 30, 30, 60]
    assert all(call[1]["cancel"] is cancel for call in calls)
    assert all(call[1]["check"] is False for call in calls)
    assert "cwd" not in calls[0][1]
    for _command, kwargs in calls[1:]:
        assert kwargs["cwd"] == committed_git_repo
        assert kwargs["env"]["GH_PROMPT_DISABLED"] == "1"
    assert "input" not in calls[2][1]
    assert calls[3][1]["input"].startswith("# Runner contract")
    assert [activity.detail for activity in activities] == [
        shlex.join(call[0]) for call in calls
    ]


def test_combined_push_and_pr_publishes_branch_before_opening_rollup(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "push-and-pr"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    local_sha = _create_project_branch(committed_git_repo, name)
    remote = _add_bare_origin(committed_git_repo, name)
    _configure_github_origin(committed_git_repo, remote, "acme/widgets")
    args_path, _body_path = _install_fake_gh(committed_git_repo, monkeypatch)
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: _execution_result(),
    )

    result = cli_runner.invoke(
        cli,
        ["execute", name, "--auto-execute", "--push", "--pr"],
    )

    assert result.exit_code == 0, result.output
    assert "✔ Push project branch" in result.output
    assert "✔ Open rollup pull request" in result.output
    assert result.output.index("Pushed project/") < result.output.index(
        "Opened rollup pull request"
    )
    assert result.output.index(" finished in ") < result.output.index(
        "Execution operation"
    )
    assert args_path.exists()
    assert _remote_project_sha(remote, name) == local_sha


@pytest.mark.parametrize(
    ("blocked_command", "command_text"),
    [
        ("remote", "git -C"),
        ("auth", "auth status --active --hostname github.com"),
        (
            "repo",
            "repo view acme/widgets --json defaultBranchRef --jq",
        ),
        ("create", "pr create --repo acme/widgets"),
    ],
)
def test_pr_interrupt_reaps_each_command_stops_stage_and_preserves_core_report(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    real_process_harness,
    monkeypatch: pytest.MonkeyPatch,
    blocked_command: str,
    command_text: str,
) -> None:
    name = f"pr-interrupt-{blocked_command}"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    local_sha = _create_project_branch(committed_git_repo, name)
    remote = _add_bare_origin(committed_git_repo, name)
    _configure_github_origin(committed_git_repo, remote, "acme/widgets")
    _install_fake_gh(committed_git_repo, monkeypatch)
    _trust(cli_runner, committed_git_repo, monkeypatch)
    operation_id = uuid4()
    process_name = f"execute-pr-{blocked_command}"
    resistant = real_process_harness.resistant_argv(process_name)
    source = r'''
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from betterborg_cli import cli as cli_module
from betterborg_cli.host_execution import HostPreflightPlan
from betterborg_cli.progress import RunProgress, StageState
from betterborg_cli.store import ExecutionRunStatus

repository = Path(sys.argv[1])
marker_root = Path(sys.argv[2])
resistant = tuple(json.loads(sys.argv[3]))
operation_id = UUID(sys.argv[4])
name = sys.argv[5]
blocked_command = sys.argv[6]
actual_run_captured = cli_module.run_captured


class FastProgress(RunProgress):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, heartbeat_interval=0.01, **kwargs)

    def _render_cadence_frame(self, stopped):
        keep_running = super()._render_cadence_frame(stopped)
        stage = self.stages.get("rollup-pr")
        if (
            stage is not None
            and stage.state is StageState.RUNNING
            and stage.activity is not None
        ):
            (marker_root / "pr-heartbeat").write_text(
                "refreshed", encoding="utf-8"
            )
        return keep_running


def command_kind(command):
    if command[0] == "git":
        return "remote"
    if tuple(command[1:3]) == ("auth", "status"):
        return "auth"
    if tuple(command[1:3]) == ("repo", "view"):
        return "repo"
    if tuple(command[1:3]) == ("pr", "create"):
        return "create"
    return "other"


def runner(command, **kwargs):
    if command_kind(command) == blocked_command:
        (marker_root / "pr-token").write_text(
            "bound" if kwargs.get("cancel") is not None else "missing",
            encoding="utf-8",
        )
        (marker_root / "pr-command").write_text(
            shlex.join(command), encoding="utf-8"
        )
        return actual_run_captured(resistant, **kwargs)
    return actual_run_captured(command, **kwargs)


preflight = HostPreflightPlan(
    repository_root=repository,
    commands=(),
    prepare_commands=(),
    materialize_commands=(),
    environment_files=(),
    executables=(),
    required_secret_names=(),
    compose_files=(),
    services=(),
)
cli_module.RunProgress = FastProgress
cli_module.run_captured = runner
cli_module._invoke_host_execution = lambda *_args, **_kwargs: SimpleNamespace(
    preflight=preflight,
    active_operation_id=None,
    operation_id=operation_id,
    status=ExecutionRunStatus.COMPLETED,
)
raise SystemExit(
    cli_module.main(["execute", name, "--auto-execute", "--pr"])
)
'''
    process = real_process_harness.launch_python(
        source,
        str(committed_git_repo),
        str(real_process_harness.root),
        json.dumps(resistant),
        str(operation_id),
        name,
        blocked_command,
        name=process_name,
    )
    real_process_harness.wait_for_marker(f"{process_name}.child.pid")
    real_process_harness.wait_for_marker("pr-heartbeat")
    real_process_harness.signal(process, signal.SIGINT)

    assert real_process_harness.wait_for_exit(process) == 130
    stdout, stderr = process.communicate()
    output = stdout + stderr
    real_process_harness.assert_tree_absent(process_name)
    assert real_process_harness.wait_for_marker("pr-token") == "bound"
    observed_command = real_process_harness.wait_for_marker("pr-command")
    assert command_text in observed_command
    assert output.count("⠋ Open rollup pull request") >= 2
    assert f"running {observed_command}" in output
    assert "■ Open rollup pull request" in output
    assert "✖ Open rollup pull request" not in output
    assert " finished in " in output
    report = f"Execution operation {operation_id}: completed"
    assert report in output
    assert _project_branch_sha(committed_git_repo, name) == local_sha


def test_pr_missing_remote_preserves_completed_local_branch(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "pr-missing-remote"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    local_sha = _create_project_branch(committed_git_repo, name)
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: _execution_result(),
    )

    result = cli_runner.invoke(cli, ["execute", name, "--auto-execute", "--pr"])

    assert result.exit_code == 1
    assert ": completed" in result.output
    assert "origin remote is missing" in result.output
    assert _project_branch_sha(committed_git_repo, name) == local_sha


def test_pr_rejects_unsupported_remote_without_invoking_gh(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "pr-unsupported-remote"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    local_sha = _create_project_branch(committed_git_repo, name)
    _add_bare_origin(committed_git_repo, name)
    args_path, _body_path = _install_fake_gh(committed_git_repo, monkeypatch)
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: _execution_result(),
    )

    result = cli_runner.invoke(cli, ["execute", name, "--auto-execute", "--pr"])

    assert result.exit_code == 1
    assert ": completed" in result.output
    assert "not a supported github.com remote" in result.output
    assert not args_path.exists()
    assert _project_branch_sha(committed_git_repo, name) == local_sha


def test_pr_missing_gh_preserves_completed_local_branch(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "pr-missing-gh"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    local_sha = _create_project_branch(committed_git_repo, name)
    remote = _add_bare_origin(committed_git_repo, name)
    _configure_github_origin(committed_git_repo, remote, "acme/widgets")
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.setattr(cli_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: _execution_result(),
    )

    result = cli_runner.invoke(cli, ["execute", name, "--auto-execute", "--pr"])

    assert result.exit_code == 1
    assert ": completed" in result.output
    assert "gh executable was not found" in result.output
    assert _project_branch_sha(committed_git_repo, name) == local_sha


def test_pr_authentication_failure_preserves_completed_local_branch(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "pr-auth-failure"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    local_sha = _create_project_branch(committed_git_repo, name)
    remote = _add_bare_origin(committed_git_repo, name)
    _configure_github_origin(committed_git_repo, remote, "acme/widgets")
    args_path, _body_path = _install_fake_gh(committed_git_repo, monkeypatch)
    monkeypatch.setenv("FAKE_GH_AUTH_FAIL", "1")
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: _execution_result(),
    )

    result = cli_runner.invoke(cli, ["execute", name, "--auto-execute", "--pr"])

    assert result.exit_code == 1
    assert ": completed" in result.output
    assert "GitHub CLI authentication failed" in result.output
    assert "not logged into github.com" in result.output
    assert not args_path.exists()
    assert _project_branch_sha(committed_git_repo, name) == local_sha


def test_pr_creation_failure_preserves_completed_local_branch(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "pr-create-failure"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    local_sha = _create_project_branch(committed_git_repo, name)
    remote = _add_bare_origin(committed_git_repo, name)
    _configure_github_origin(committed_git_repo, remote, "acme/widgets")
    args_path, body_path = _install_fake_gh(committed_git_repo, monkeypatch)
    monkeypatch.setenv("FAKE_GH_PR_FAIL", "1")
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: _execution_result(),
    )

    result = cli_runner.invoke(cli, ["execute", name, "--auto-execute", "--pr"])

    assert result.exit_code == 1
    assert ": completed" in result.output
    assert "pull request creation rejected" in result.output
    assert args_path.exists()
    assert body_path.exists()
    assert _project_branch_sha(committed_git_repo, name) == local_sha


def test_pr_rejects_external_prd_symlink_without_uploading_host_file(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "pr-external-prd"
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name=name,
    )
    local_sha = _create_project_branch(committed_git_repo, name)
    remote = _add_bare_origin(committed_git_repo, name)
    _configure_github_origin(committed_git_repo, remote, "acme/widgets")
    args_path, body_path = _install_fake_gh(committed_git_repo, monkeypatch)
    outside = committed_git_repo.parent / f"{name}-host-secret.md"
    outside.write_text("host secret must not be uploaded\n", encoding="utf-8")
    prd_path = committed_git_repo / ".betterborg/prds" / f"{name}.md"
    prd_path.unlink()
    prd_path.symlink_to(outside)
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: _execution_result(),
    )

    result = cli_runner.invoke(cli, ["execute", name, "--auto-execute", "--pr"])

    assert result.exit_code == 1
    assert ": completed" in result.output
    assert "repository path is not a regular file" in result.output
    assert not args_path.exists()
    assert not body_path.exists()
    assert _project_branch_sha(committed_git_repo, name) == local_sha


def test_concurrent_decision_insert_reaches_active_host_execution(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repository, paths, borg, _approval, fixture, _publication = (
        _seed_executable_generation(
            committed_git_repo,
            planning_cli_repository,
            approved_task_generation,
            name="concurrent-decision",
        )
    )
    _trust(cli_runner, committed_git_repo, monkeypatch)
    original_append = SqliteStore.append_execution_decision
    winner = []

    def lose_decision_insert(store, decision):
        concurrent_decision = replace(
            decision,
            id=uuid4(),
            source="concurrent_invocation",
        )
        with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as contender:
            original_append(contender, concurrent_decision)
        winner.append(concurrent_decision)
        original_append(store, decision)

    active_operation_id = uuid4()
    host_calls = []

    def invoke_host(
        _paths,
        store,
        _config,
        _repository_id,
        borg_id,
        generation_id,
        **_kwargs,
    ):
        host_calls.append((borg_id, generation_id))
        assert store.get_current_execution_decision(borg_id) == winner[0]
        result = _execution_result(ExecutionRunStatus.RUNNING)
        result.active_operation_id = active_operation_id
        return result

    monkeypatch.setattr(
        cli_module.SqliteStore,
        "append_execution_decision",
        lose_decision_insert,
    )
    monkeypatch.setattr(cli_module, "_invoke_host_execution", invoke_host)

    result = cli_runner.invoke(
        cli,
        ["execute", "concurrent-decision", "--auto-execute"],
    )

    assert result.exit_code == 0, result.output
    assert "decision recorded by a concurrent invocation" in result.output
    assert f"Execution already active: {active_operation_id}" in result.output
    assert host_calls == [(borg.id, fixture.generation.id)]
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        assert store.get_current_execution_decision(borg.id) == winner[0]


@pytest.mark.parametrize(
    "status",
    [ExecutionRunStatus.FAILED, ExecutionRunStatus.CANCELLED],
)
def test_execute_exits_nonzero_for_unsuccessful_host_execution(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
    status: ExecutionRunStatus,
) -> None:
    _seed_executable_generation(
        committed_git_repo,
        planning_cli_repository,
        approved_task_generation,
        name="unsuccessful-execution",
    )
    _trust(cli_runner, committed_git_repo, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: _execution_result(status),
    )

    result = cli_runner.invoke(
        cli,
        ["execute", "unsuccessful-execution", "--auto-execute"],
    )

    assert result.exit_code == 1
    assert "Execution operation " in result.output
    assert f": {status.value}" in result.output


def test_task_digest_drift_blocks_before_decision_or_host_execution(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repository, paths, borg, _approval, _fixture, publication = (
        _seed_executable_generation(
            committed_git_repo,
            planning_cli_repository,
            approved_task_generation,
            name="digest-drift",
        )
    )
    _trust(cli_runner, committed_git_repo, monkeypatch)
    publication.files[0].path.write_text("drifted\n", encoding="utf-8")
    monkeypatch.setattr(
        cli_module,
        "_invoke_host_execution",
        lambda *_args, **_kwargs: pytest.fail(
            "host execution must not run after digest drift"
        ),
    )

    result = cli_runner.invoke(
        cli, ["execute", "digest-drift", "--auto-execute"]
    )

    assert result.exit_code == 1
    assert "digest drifted" in result.output
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        assert store.get_current_execution_decision(borg.id) is None


def test_execute_assembly_invokes_the_concrete_host_execution_service(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, paths, borg, _approval, fixture, _publication = (
        _seed_executable_generation(
            committed_git_repo,
            planning_cli_repository,
            approved_task_generation,
            name="concrete-service",
        )
    )
    _trust(cli_runner, committed_git_repo, monkeypatch)
    config = cli_module.load_repository_config(paths)
    monkeypatch.setenv("EXECUTE_TOKEN", "owner-secret")
    selected_agents: list[SelectedAgent] = []
    selected_stages: list[AgentStage] = []
    execution_trust: list[object] = []
    selected_settings = {
        AgentStage.CODING: ("selected-coding-model", "coding-effort"),
        AgentStage.REVIEW: ("selected-review-model", "review-effort"),
        AgentStage.MERGE: ("selected-merge-model", "merge-effort"),
    }

    def select(_config, stage, selected_paths, **kwargs):
        selected_stages.append(stage)
        execution_trust.append(kwargs["trust_requirement"])
        model, effort = selected_settings[stage]
        selected = SelectedAgent(
            role=ApiAgentRole(stage.value),
            adapter=MockAdapter(name="openai"),
            paths=selected_paths,
            model=model,
            effort=effort,
        )
        selected_agents.append(selected)
        return selected

    calls: list[tuple[object, ...]] = []

    def run(service, borg_id, generation_id, analyzer_plan, **kwargs):
        assert progress.stages["preflight"].state is StageState.COMPLETED
        assert type(service) is cli_module.HostExecutionService
        assert type(service._runtime) is cli_module.HostTaskRuntime
        assert type(service._runtime._coding) is cli_module.HostCodingPhase
        assert type(service._runtime._review_fix) is cli_module.HostReviewFixPhase
        assert type(service._runtime._merge) is cli_module.HostMergePhase
        assert type(service._runtime._sanity) is cli_module.HostSanityPhase
        coding_config = service._runtime._coding._config
        assert coding_config.model == "selected-coding-model"
        assert coding_config.effort == "coding-effort"
        review_config = service._runtime._review_fix._config
        assert review_config.review_model == "selected-review-model"
        assert review_config.review_effort == "review-effort"
        assert review_config.fix_effort == "review-effort"
        assert review_config.review_passes == config.execution.review_passes
        assert service._scheduler_config.review_passes == (
            config.execution.review_passes
        )
        merge_config = service._runtime._merge._config
        assert merge_config.model == "selected-merge-model"
        assert merge_config.effort == "merge-effort"
        calls.append((borg_id, generation_id, analyzer_plan, kwargs))
        return HostExecutionResult(service._runtime.plan)

    monkeypatch.setattr(cli_module, "select_agent", select)
    monkeypatch.setattr(cli_module.HostExecutionService, "run", run)
    cancel = CancellationToken()
    progress = RunProgress(enabled=False)
    progress.declare(StageSpec("preflight", "Preflight"))
    progress.start("preflight")
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        analysis = store.get_prior_ready_analysis(repository.id)
        assert analysis is not None
        analyzer_plan = {
            **analysis.analysis_json,
            "required_secrets": [
                {
                    "name": "EXECUTE_TOKEN",
                    "used_by": ["coding"],
                    "scope": "agent",
                    "source": "test",
                }
            ],
        }
        monkeypatch.setattr(
            cli_module.SqliteStore,
            "get_prior_ready_analysis",
            lambda _store, _repository_id: SimpleNamespace(
                analysis_json=analyzer_plan
            ),
        )
        result = cli_module._invoke_host_execution(
            paths,
            store,
            config,
                repository.id,
                borg.id,
                fixture.generation.id,
                cancel=cancel,
                progress=progress,
            )

    assert isinstance(result, HostExecutionResult)
    assert selected_stages == [
        AgentStage.CODING,
        AgentStage.REVIEW,
        AgentStage.MERGE,
    ]
    assert len(selected_agents) == 3
    observed_trust_paths: list[Path] = []
    monkeypatch.setattr(
        cli_module,
        "require_workspace_trust",
        lambda observed, **_kwargs: observed_trust_paths.append(observed.root),
    )
    managed_worktree_paths = SimpleNamespace(root=paths.worktrees_dir / "task")
    for requirement in execution_trust:
        requirement(managed_worktree_paths)
    assert observed_trust_paths == [paths.root] * 3
    assert len(calls) == 1
    observed_borg, observed_generation, observed_plan, observed_kwargs = calls[0]
    assert observed_borg == borg.id
    assert observed_generation == fixture.generation.id
    assert observed_plan == analyzer_plan
    assert observed_kwargs["secret_values"] == {
        "EXECUTE_TOKEN": "owner-secret"
    }
    assert observed_kwargs["cancel"] is cancel
    assert isinstance(
        observed_kwargs["validated_preflight"], HostPreflightPlan
    )
