"""CLI contracts for the generation-bound host execution gate."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from io import StringIO
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

import click
import pytest
from click.testing import CliRunner

from betterborg_cli import cli as cli_module
from betterborg_cli.agent_runtime import CancellationToken, MockAdapter, run_captured
from betterborg_cli.cli import CliRunContext, cli
from betterborg_cli.host_execution import HostExecutionResult, HostPreflightPlan
from betterborg_cli.planning import TaskPublisher
from betterborg_cli.progress import RunProgress, StageSpec, StageState
from betterborg_cli.store import (
    BorgState,
    ExecutionRunStatus,
    PlanApproval,
    SqliteStore,
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
    approval: PlanApproval | None = None,
):
    if approval is None:
        repository, paths = planning_cli_repository(root, name)
    else:
        paths = cli_module.RepoPaths.discover(root)
        config = cli_module.load_repository_config(paths)
        with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
            repository = store.get_repository(config.repository_id)
        assert repository is not None

    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
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
                        "summary": "Ship the completed BetterBorg project.",
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
        fixture = approved_task_generation(
            store,
            borg,
            approval,
            body=_task_body(round_number),
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
    assert approved.output.startswith("running Estimate and decision")
    assert "DUMMY DATA" in approved.output
    assert "completed Estimate and decision — approved" in approved.output
    assert "Recorded execution estimate approved" in approved.output
    assert approved.output.index("summary:") < approved.output.index(
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

    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
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
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
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
    assert "completed Estimate and decision — declined" in result.output
    assert result.output.index("[y/N]: n") < result.output.index(
        "completed Estimate and decision — declined"
    )
    assert result.output.index("completed Estimate and decision — declined") < (
        result.output.index("summary:")
    )
    assert "Aborted!" in result.output
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        assert store.get_current_execution_decision(borg.id) is None


def test_execute_threads_one_control_context_and_suspends_confirmation(
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
    progress = TrackingProgress(stream=StringIO())
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
    assert progress.suspension_count == 1
    assert progress.closed
    estimate = progress.stages["estimate-decision"]
    assert estimate.state is StageState.COMPLETED
    assert estimate.result == "approved"
    preflight = progress.stages["preflight"]
    assert preflight.state is StageState.COMPLETED
    assert preflight.result == "ready"


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
    assert "completed Push project branch — Pushed project/push-success" in (
        result.output
    )
    assert f"Pushed project/{name} to origin." in result.output
    assert result.output.index("summary:") < result.output.index(
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
    assert "failed Push project branch" in result.output
    assert "Local execution completed, but push" in result.output
    assert "origin" in result.output
    assert result.output.index("summary:") < result.output.index(
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


def test_follow_up_heartbeat_failure_fails_stage_and_propagates() -> None:
    refresh_attempted = Event()

    class FailingProgress(RunProgress):
        def refresh(self) -> None:
            refresh_attempted.set()
            raise RuntimeError("progress heartbeat failed")

    progress = FailingProgress(stream=StringIO())

    def action() -> str:
        assert refresh_attempted.wait(1)
        return "published"

    with pytest.raises(RuntimeError, match="progress heartbeat failed"):
        cli_module._run_execution_follow_up(
            progress,
            "push-project",
            "Push project branch",
            action,
        )

    stage = progress.stages["push-project"]
    assert stage.state is StageState.FAILED
    assert stage.result == "progress heartbeat failed"


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

    def refresh(self):
        super().refresh()
        stage = self.stages.get("push-project")
        if (
            stage is not None
            and stage.state is StageState.RUNNING
            and stage.activity is not None
        ):
            (marker_root / "push-heartbeat").write_text(
                "refreshed", encoding="utf-8"
            )


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
    assert output.count("running Push project branch") >= 2
    assert "command: git push origin refs/heads/project/push-interrupt" in output
    assert "stopped Push project branch" in output
    assert "failed Push project branch" not in output
    assert "summary:" in output
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
    assert "completed Open rollup pull request — Opened rollup pull request" in (
        result.output
    )
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
    assert "completed Push project branch" in result.output
    assert "completed Open rollup pull request" in result.output
    assert result.output.index("Pushed project/") < result.output.index(
        "Opened rollup pull request"
    )
    assert result.output.index("summary:") < result.output.index(
        "Execution operation"
    )
    assert args_path.exists()
    assert _remote_project_sha(remote, name) == local_sha


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
    prd_path = committed_git_repo / ".borg/prds" / f"{name}.md"
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
        with SqliteStore.open(paths.state_dir / "borg.sqlite3") as contender:
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
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
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
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
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
    adapters: list[MockAdapter] = []
    execution_trust: list[object] = []

    def select(*_args, **kwargs):
        adapter = MockAdapter(name="openai")
        adapters.append(adapter)
        execution_trust.append(kwargs["trust_requirement"])
        return adapter

    calls: list[tuple[object, ...]] = []

    def run(service, borg_id, generation_id, analyzer_plan, **kwargs):
        assert progress.stages["preflight"].state is StageState.COMPLETED
        assert type(service) is cli_module.HostExecutionService
        assert type(service._runtime) is cli_module.HostTaskRuntime
        assert type(service._runtime._coding) is cli_module.HostCodingPhase
        assert type(service._runtime._review_fix) is cli_module.HostReviewFixPhase
        assert type(service._runtime._merge) is cli_module.HostMergePhase
        assert type(service._runtime._sanity) is cli_module.HostSanityPhase
        calls.append((borg_id, generation_id, analyzer_plan, kwargs))
        return HostExecutionResult(service._runtime.plan)

    monkeypatch.setattr(cli_module, "select_agent", select)
    monkeypatch.setattr(cli_module.HostExecutionService, "run", run)
    cancel = CancellationToken()
    progress = RunProgress(enabled=False)
    progress.declare(StageSpec("preflight", "Preflight"))
    progress.start("preflight")
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
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
    assert len(adapters) == 3
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
