"""CLI contracts for the generation-bound host execution gate."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import click
import pytest
from click.testing import CliRunner
from progress_test_support import FailingStringIO

from betterborg_cli import cli as cli_module
from betterborg_cli.agent_runtime import (
    ApiAgentRole,
    CancellationToken,
    MockAdapter,
    SelectedAgent,
    run_captured,
)
from betterborg_cli.cli import CliRunContext, cli
from betterborg_cli.host_execution import HostExecutionResult, HostPreflightPlan
from betterborg_cli.planning import TaskPublisher
from betterborg_cli.progress import RunProgress, StageSpec, StageState
from betterborg_cli.repository_config import AgentStage
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
    assert approved.output.startswith("⠋ Estimate and decision")
    assert "DUMMY DATA" in approved.output
    assert "✔ Estimate and decision" in approved.output
    assert "approved" in approved.output
    assert "Recorded execution estimate approved" in approved.output
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
    assert progress.suspension_count == 2
    assert progress.closed
    estimate = progress.stages["estimate-decision"]
    assert estimate.state is StageState.COMPLETED
    assert estimate.result == "approved"
    preflight = progress.stages["preflight"]
    assert preflight.state is StageState.COMPLETED
    assert preflight.result == "ready"


def test_execute_projects_launch_publication_and_reuses_follow_up_specs(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "execution-preview"
    _repository, _paths, _borg, _approval, fixture, _publication = (
        _seed_executable_generation(
            committed_git_repo,
            planning_cli_repository,
            approved_task_generation,
            name=name,
        )
    )
    _trust(cli_runner, committed_git_repo, monkeypatch)

    class TrackingProgress(RunProgress):
        def __init__(self) -> None:
            super().__init__(stream=StringIO())
            self.previews: list[
                tuple[tuple[StageSpec, ...], tuple[str, ...] | None]
            ] = []
            self.declarations: list[StageSpec] = []

        def preview_pending(
            self,
            specs: tuple[StageSpec, ...],
            *,
            cohort_keys: tuple[str, ...] | None = None,
        ) -> None:
            self.previews.append((specs, cohort_keys))
            super().preview_pending(specs, cohort_keys=cohort_keys)

        def declare(self, spec: StageSpec):
            self.declarations.append(spec)
            return super().declare(spec)

    progress = TrackingProgress()
    run = CliRunContext(CancellationToken(), progress)

    def assert_launch_projection() -> None:
        frame = [line.plain for line in progress._live_lines()]
        assert any("Estimate and decision" in line for line in frame)
        assert "  ◦ Preflight" in frame
        assert "  ◦ Push project branch" in frame
        assert "  ◦ Open rollup pull request" in frame
        assert tuple(progress.stages) == ("estimate-decision",)

    actual_load_config = cli_module.load_repository_config

    def observed_load_config(paths):
        assert_launch_projection()
        return actual_load_config(paths)

    actual_inspect = TaskPublisher.inspect_current_task_files

    def observed_inspect(publisher, borg_id):
        assert_launch_projection()
        return actual_inspect(publisher, borg_id)

    def observed_estimate(_name, _estimate):
        frame = [line.plain for line in progress._live_lines()]
        assert f"  ◦ {fixture.task.title}" in frame
        assert "  ◦ Preflight" in frame
        assert "  ◦ Push project branch" in frame
        assert "  ◦ Open rollup pull request" in frame
        assert tuple(progress.stages) == ("estimate-decision",)

    def invoke_host(*_args, progress=None, **_kwargs):
        assert progress is run.progress
        frame = [line.plain for line in progress._live_lines()]
        assert f"  ◦ {fixture.task.title}" in frame
        assert any("Preflight" in line for line in frame)
        assert "  ◦ Push project branch" in frame
        assert "  ◦ Open rollup pull request" in frame
        return _execution_result()

    monkeypatch.setattr(cli_module, "load_repository_config", observed_load_config)
    monkeypatch.setattr(TaskPublisher, "inspect_current_task_files", observed_inspect)
    monkeypatch.setattr(cli_module, "_write_execution_estimate", observed_estimate)
    monkeypatch.setattr(cli_module, "_invoke_host_execution", invoke_host)
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

    result = cli_runner.invoke(
        cli,
        ["execute", name, "--auto-execute", "--push", "--pr"],
        obj=run,
    )

    assert result.exit_code == 0, result.output
    assert len(progress.previews) == 2
    launch_specs, launch_cohort = progress.previews[0]
    publication_specs, publication_cohort = progress.previews[1]
    assert launch_specs[0] is cli_module.EXECUTION_PREFLIGHT_STAGE
    assert tuple(spec.key for spec in launch_specs[1:]) == (
        "push-project",
        "rollup-pr",
    )
    assert launch_cohort == ("push-project", "rollup-pr")
    assert publication_specs[0] is cli_module.EXECUTION_PREFLIGHT_STAGE
    assert publication_specs[-2] is launch_specs[1]
    assert publication_specs[-1] is launch_specs[2]
    task_key = str(fixture.task.id)
    assert publication_specs[1] == StageSpec(task_key, fixture.task.title)
    assert publication_cohort == (task_key, "push-project", "rollup-pr")
    assert progress.declarations[1] is cli_module.EXECUTION_PREFLIGHT_STAGE
    assert progress.declarations[-2] is launch_specs[1]
    assert progress.declarations[-1] is launch_specs[2]
    assert progress.stages["preflight"].state is StageState.COMPLETED
    assert progress.stages["push-project"].state is StageState.COMPLETED
    assert progress.stages["rollup-pr"].state is StageState.COMPLETED
    assert task_key not in progress.stages


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


def test_follow_up_worker_failure_between_check_and_completion_propagates() -> None:
    stream = FailingStringIO()

    class CompletionRaceProgress(RunProgress):
        def raise_if_render_failed(self) -> None:
            super().raise_if_render_failed()
            stream.fail_next_write()
            worker = self._cadence_worker
            assert worker is not None
            worker.join(timeout=2)
            assert not worker.is_alive()

    progress = CompletionRaceProgress(
        stream=stream,
        heartbeat_interval=0.01,
    )

    def action() -> str:
        return "published"

    with pytest.raises(RuntimeError, match="progress heartbeat failed"):
        cli_module._run_execution_follow_up(
            progress,
            StageSpec("push-project", "Push project branch"),
            action,
        )

    stage = progress.stages["push-project"]
    assert stage.state is StageState.FAILED
    assert stage.result == "progress heartbeat failed"
    RunProgress.raise_if_render_failed(progress)


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
