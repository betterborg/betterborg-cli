"""End-to-end contracts for locked sanity and project-base advancement."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import quote

import pytest
from test_host_merge import (
    RecordingLock,
    _advance_project_base,
    _approved_merge_fixture,
    _project_branch,
)
from test_host_merge import (
    _phase as merge_phase,
)

from betterborg_cli.agent_runtime import MockAdapter
from betterborg_cli.host_execution import (
    HostCommand,
    HostComposeManager,
    HostEnvironmentManager,
    HostExecutable,
    HostPreflightPlan,
    HostSanityPhase,
    HostSecret,
    HostService,
    HostWorktreeManager,
    WorktreeError,
)
from betterborg_cli.store import SqliteStore, TaskRuntimeStatus


@dataclass
class _FakeStack:
    environment: dict[str, str]


class _RecordingCompose:
    def __init__(self, repository_lock: RecordingLock, *, with_stack: bool) -> None:
        self.repository_lock = repository_lock
        self.with_stack = with_stack
        self.stack = (
            _FakeStack({"HEALTHY_URL": "http://127.0.0.1:39123"})
            if with_stack
            else None
        )
        self.started: list[object] = []
        self.stopped: list[object] = []

    def start_claimed_stack(self, store, plan, claim, owner_token):  # noqa: ANN001
        assert self.repository_lock.locked()
        self.started.append(claim)
        return self.stack

    def start_claimed_sanity_stack(  # noqa: ANN001
        self, store, plan, claim, owner_token
    ):
        return self.start_claimed_stack(store, plan, claim, owner_token)

    def stop_claimed_stack(  # noqa: ANN001
        self, store, stack, claim, owner_token
    ) -> None:
        assert self.repository_lock.locked()
        assert stack is not None
        self.stopped.append(stack)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _merged_fixture(tmp_path: Path):
    fixture = _approved_merge_fixture(tmp_path)
    subdir = fixture.repository / "package"
    subdir.mkdir()
    (subdir / ".keep").write_text("package\n", encoding="utf-8")
    _git(fixture.repository, "add", "package/.keep")
    base_commit = _advance_project_base(
        fixture, "README.md", "# Fixture\n\nbase descriptor changed\n"
    )
    repository_lock = RecordingLock()
    with SqliteStore.open(fixture.database) as store:
        merged = merge_phase(fixture, MockAdapter(), repository_lock).run(
            fixture.context(store)
        )
    assert merged.tip is not None
    assert merged.tip.base_commit == base_commit
    return fixture, merged.tip, repository_lock


def _plan(fixture) -> HostPreflightPlan:  # noqa: ANN001
    return HostPreflightPlan(
        repository_root=fixture.repository,
        commands=(
            HostCommand("install", ("catalog-install",), "."),
            HostCommand("test", ("catalog-test",), "package"),
        ),
        prepare_commands=(),
        materialize_commands=(),
        environment_files=(fixture.repository / "README.md",),
        executables=(),
        required_secret_names=("BUILD_TOKEN", "AGENT_TOKEN"),
        compose_files=(),
        services=(
            HostService(
                name="registry",
                kind="external",
                evidence="fixture",
                url_env="REGISTRY_URL",
                url="https://registry.example.test",
            ),
        ),
        package_managers=("cargo", "go", "pnpm"),
        secret_requirements=(
            HostSecret("BUILD_TOKEN", "build", ("install",), "fixture"),
            HostSecret("AGENT_TOKEN", "agent", ("install", "test"), "fixture"),
        ),
    )


def _sanity_phase(
    fixture,  # noqa: ANN001
    plan: HostPreflightPlan,
    repository_lock: RecordingLock,
    compose,  # noqa: ANN001
    runner,  # noqa: ANN001
) -> HostSanityPhase:
    return HostSanityPhase(
        fixture.repository,
        plan,
        environment_manager=HostEnvironmentManager(
            fixture.repository,
            environment={
                "PATH": os.environ["PATH"],
                "UNDECLARED_HOST": "no",
            },
        ),
        compose_manager=compose,
        worktree_manager=HostWorktreeManager(
            fixture.repository,
            fixture.repository.parent / "worktrees",
            source_branch="main",
        ),
        repository_lock=repository_lock,
        command_runner=runner,
    )


def test_sanity_rematerializes_runs_catalog_and_advances_before_cleanup(
    tmp_path: Path,
) -> None:
    fixture, tip, repository_lock = _merged_fixture(tmp_path)
    secret = 'token"with/slash space?x=1&y=2'
    original_plan = _plan(fixture)
    plan = replace(
        original_plan,
        commands=(
            replace(
                original_plan.commands[0],
                argv=("catalog-install", secret),
            ),
            original_plan.commands[1],
        ),
    )
    compose = _RecordingCompose(repository_lock, with_stack=False)
    calls: list[tuple[tuple[str, ...], Path, dict[str, str]]] = []

    def runner(argv, *, cwd, env, **kwargs):  # noqa: ANN001, ANN003
        assert repository_lock.locked()
        calls.append((tuple(argv), Path(cwd), dict(env)))
        leaked = "\n".join((secret, json.dumps(secret)[1:-1], quote(secret, safe="")))
        return subprocess.CompletedProcess(argv, 0, stdout=leaked, stderr="")

    with SqliteStore.open(fixture.database) as store:
        before_attempts = store.list_environment_attempts(fixture.task.id)
        result = _sanity_phase(fixture, plan, repository_lock, compose, runner).run(
            fixture.context(store),
            tip,
            secret_values={
                "BUILD_TOKEN": secret,
                "AGENT_TOKEN": "agent-only-secret",
                "UNDECLARED_TOKEN": "never-injected",
            },
        )
        runtime = store.get_task_runtime(fixture.task.id)
        attempts = store.list_environment_attempts(fixture.task.id)
        sanity_events = store.list_task_execution_events(
            fixture.task.id, kind="sanity.completed"
        )

    assert result.status is TaskRuntimeStatus.DONE
    assert result.commit_sha == tip.commit_sha
    assert runtime is not None and runtime.status is TaskRuntimeStatus.DONE
    assert not Path(runtime.worktree_path).exists()
    assert _git(fixture.repository, "rev-parse", tip.task_branch) == tip.commit_sha
    assert _git(fixture.repository, "rev-parse", _project_branch(fixture)) == (
        tip.commit_sha
    )
    assert [call[0] for call in calls] == [
        ("catalog-install", secret),
        ("catalog-test",),
    ]
    assert [
        call[1].relative_to(Path(runtime.worktree_path)).as_posix() for call in calls
    ] == [
        ".",
        "package",
    ]
    install_env, test_env = calls[0][2], calls[1][2]
    assert install_env["BUILD_TOKEN"] == secret
    assert "BUILD_TOKEN" not in test_env
    assert "AGENT_TOKEN" not in install_env | test_env
    assert "UNDECLARED_TOKEN" not in install_env | test_env
    assert "UNDECLARED_HOST" not in install_env | test_env
    assert install_env["REGISTRY_URL"] == "https://registry.example.test"
    assert install_env["HOME"] == test_env["HOME"]
    assert install_env["XDG_CACHE_HOME"] == test_env["XDG_CACHE_HOME"]
    assert install_env["CARGO_HOME"] == test_env["CARGO_HOME"]
    assert install_env["GOCACHE"] == test_env["GOCACHE"]
    assert install_env["PNPM_STORE_DIR"] == test_env["PNPM_STORE_DIR"]
    assert len(attempts) == len(before_attempts) + 1
    assert attempts[-1].kind == "materialize"
    cache_path = Path(attempts[-1].result["cache_path"])
    assert all(
        Path(value).is_relative_to(cache_path)
        for value in (
            install_env["HOME"],
            install_env["XDG_CACHE_HOME"],
            install_env["CARGO_HOME"],
            install_env["GOCACHE"],
            install_env["PNPM_STORE_DIR"],
        )
    )
    assert result.commands[0].command.argv == ("catalog-install", "[REDACTED]")
    assert secret not in repr(result)
    persisted = json.dumps(sanity_events[-1].payload)
    assert secret not in persisted
    assert json.dumps(secret)[1:-1] not in persisted
    assert quote(secret, safe="") not in persisted
    assert persisted.count("[REDACTED]") == 7
    assert len(compose.started) == 1
    assert compose.stopped == []


def test_sanity_failure_stops_exact_stack_and_never_advances(
    tmp_path: Path,
) -> None:
    fixture, tip, repository_lock = _merged_fixture(tmp_path)
    plan = _plan(fixture)
    compose = _RecordingCompose(repository_lock, with_stack=True)
    calls: list[tuple[str, ...]] = []
    secret = "sanity-build-secret"

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        calls.append(tuple(argv))
        if argv == ["catalog-install"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout="installed", stderr=""
            )
        return subprocess.CompletedProcess(
            argv, 7, stdout=f"failed with {secret}", stderr=""
        )

    with SqliteStore.open(fixture.database) as store:
        result = _sanity_phase(fixture, plan, repository_lock, compose, runner).run(
            fixture.context(store),
            tip,
            secret_values={"BUILD_TOKEN": secret, "AGENT_TOKEN": "agent"},
        )
        runtime = store.get_task_runtime(fixture.task.id)

    assert result.status is TaskRuntimeStatus.BLOCKED
    assert runtime is not None and runtime.status is TaskRuntimeStatus.BLOCKED
    assert secret not in result.reason
    assert "[REDACTED]" in result.reason
    assert calls == [("catalog-install",), ("catalog-test",)]
    assert len(result.commands) == 2
    successful_command, failed_command = result.commands
    assert successful_command.command.argv == ("catalog-install",)
    assert successful_command.returncode == 0
    assert successful_command.stdout == "installed"
    assert failed_command.command.argv == ("catalog-test",)
    assert failed_command.returncode == 7
    assert failed_command.stdout == "failed with [REDACTED]"
    assert failed_command.stderr == ""
    assert len(compose.started) == 1
    assert compose.stopped == [compose.stack]
    assert _git(fixture.repository, "rev-parse", _project_branch(fixture)) == (
        tip.base_commit
    )
    assert Path(runtime.worktree_path).is_dir()


def test_compose_file_drift_durably_blocks_before_base_advancement(
    tmp_path: Path,
) -> None:
    fixture, tip, repository_lock = _merged_fixture(tmp_path)
    missing_compose_file = fixture.repository / "compose.yml"
    plan = replace(
        _plan(fixture),
        executables=(HostExecutable("docker", Path("/validated/docker"), "fixture"),),
        compose_files=(missing_compose_file,),
        services=(
            HostService(
                name="database",
                kind="compose",
                evidence="fixture",
                compose_service="database",
            ),
        ),
    )
    compose = HostComposeManager(
        fixture.repository,
        environment={"PATH": os.environ["PATH"]},
        command_runner=lambda *args, **kwargs: pytest.fail(
            "Compose must not run after validated file drift"
        ),
    )

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    with SqliteStore.open(fixture.database) as store:
        result = _sanity_phase(
            fixture, plan, repository_lock, compose, runner
        ).run(
            fixture.context(store),
            tip,
            secret_values={
                "BUILD_TOKEN": "build-secret",
                "AGENT_TOKEN": "agent-secret",
            },
        )
        runtime = store.get_task_runtime(fixture.task.id)

    assert result.status is TaskRuntimeStatus.BLOCKED
    assert "validated Compose file is missing" in result.reason
    assert runtime is not None and runtime.status is TaskRuntimeStatus.BLOCKED
    assert Path(runtime.worktree_path).is_dir()
    assert _git(fixture.repository, "rev-parse", _project_branch(fixture)) == (
        tip.base_commit
    )


def test_cleanup_failure_blocks_before_completion_while_locked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, tip, repository_lock = _merged_fixture(tmp_path)
    plan = _plan(fixture)
    compose = _RecordingCompose(repository_lock, with_stack=False)

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    with SqliteStore.open(fixture.database) as store:
        phase = _sanity_phase(fixture, plan, repository_lock, compose, runner)

        def fail_cleanup(runtime):  # noqa: ANN001
            assert repository_lock.locked()
            assert runtime.status is TaskRuntimeStatus.MERGING
            raise WorktreeError("injected cleanup failure")

        monkeypatch.setattr(
            phase._worktree_manager,  # noqa: SLF001
            "cleanup_published_task_worktree",
            fail_cleanup,
        )
        result = phase.run(
            fixture.context(store),
            tip,
            secret_values={
                "BUILD_TOKEN": "build-secret",
                "AGENT_TOKEN": "agent-secret",
            },
        )
        runtime = store.get_task_runtime(fixture.task.id)

    assert result.status is TaskRuntimeStatus.BLOCKED
    assert "injected cleanup failure" in result.reason
    assert runtime is not None and runtime.status is TaskRuntimeStatus.BLOCKED
    assert Path(runtime.worktree_path).is_dir()
    assert (
        _git(fixture.repository, "rev-parse", _project_branch(fixture))
        == tip.commit_sha
    )


def test_resume_after_fast_forward_uses_durable_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, tip, repository_lock = _merged_fixture(tmp_path)
    plan = _plan(fixture)
    compose = _RecordingCompose(repository_lock, with_stack=False)
    calls: list[tuple[str, ...]] = []

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    with SqliteStore.open(fixture.database) as store:
        phase = _sanity_phase(fixture, plan, repository_lock, compose, runner)
        original_transition = store.transition_task_runtime

        def interrupt_transition(*args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("interrupted after fast-forward")

        monkeypatch.setattr(store, "transition_task_runtime", interrupt_transition)
        with pytest.raises(RuntimeError, match="interrupted after fast-forward"):
            phase.run(
                fixture.context(store),
                tip,
                secret_values={
                    "BUILD_TOKEN": "build-secret",
                    "AGENT_TOKEN": "agent-secret",
                },
            )
        interrupted = store.get_task_runtime(fixture.task.id)
        assert interrupted is not None
        assert interrupted.status is TaskRuntimeStatus.MERGING
        assert not Path(interrupted.worktree_path).exists()
        assert (
            _git(fixture.repository, "rev-parse", _project_branch(fixture))
            == tip.commit_sha
        )

        monkeypatch.setattr(store, "transition_task_runtime", original_transition)
        resumed = phase.run(fixture.context(store), tip)
        runtime = store.get_task_runtime(fixture.task.id)

    assert resumed.status is TaskRuntimeStatus.DONE
    assert runtime is not None and runtime.status is TaskRuntimeStatus.DONE
    assert not Path(runtime.worktree_path).exists()
    assert calls == [("catalog-install",), ("catalog-test",)]
