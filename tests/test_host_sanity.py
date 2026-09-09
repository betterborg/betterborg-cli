"""End-to-end contracts for locked sanity and project-base advancement."""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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

from betterborg_cli.agent_runtime import CancellationToken, MockAdapter, run_captured
from betterborg_cli.host_execution import (
    HostCommand,
    HostDroppedCommand,
    HostEnvironmentManager,
    HostPreflightPlan,
    HostSanityPhase,
    HostSecret,
    HostWorktreeManager,
    SafeGit,
    WorktreeError,
)
from betterborg_cli.store import SqliteStore, TaskRuntimeStatus


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
    runner,  # noqa: ANN001
    *,
    cancel: CancellationToken | None = None,
    git: SafeGit | None = None,
) -> HostSanityPhase:
    return HostSanityPhase(
        fixture.repository,
        plan,
        environment_manager=HostEnvironmentManager(
            fixture.repository,
            environment={
                "PATH": os.environ["PATH"],
                "HOME": str(fixture.repository.parent),
                "XDG_CACHE_HOME": str(fixture.repository.parent / "cache"),
                "UNDECLARED_HOST": "no",
                "REGISTRY_URL": "https://registry.example.test",
                "BUILD_TOKEN": "operator-shell-value",
                "AGENT_TOKEN": "operator-shell-value",
            },
            cancel=cancel,
            git=git,
        ),
        worktree_manager=HostWorktreeManager(
            fixture.repository,
            fixture.repository.parent / "worktrees",
            source_branch="main",
            cancel=cancel,
            git=git,
        ),
        repository_lock=repository_lock,
        command_runner=runner,
        cancel=cancel,
        git=git,
    )


def test_sanity_attestation_reuses_bound_git_and_reaps_cancelled_probe(
    tmp_path: Path,
    real_process_harness,
) -> None:
    fixture, tip, repository_lock = _merged_fixture(tmp_path)
    plan = _plan(fixture)
    cancel = CancellationToken()
    resistant = real_process_harness.resistant_argv("sanity-attestation-git")
    observed_tokens: list[CancellationToken | None] = []
    activities = []
    with SqliteStore.open(fixture.database) as store:
        runtime = store.get_task_runtime(fixture.task.id)
    assert runtime is not None and runtime.worktree_path is not None
    worktree = Path(runtime.worktree_path).resolve()

    def git_runner(command, **kwargs):  # noqa: ANN001, ANN003
        arguments = tuple(command)
        if (
            Path(kwargs["cwd"]).resolve() == worktree
            and arguments[-3:] == ("rev-parse", "--abbrev-ref", "HEAD")
        ):
            observed_tokens.append(kwargs.get("cancel"))
            return run_captured(resistant, **kwargs)
        return run_captured(command, **kwargs)

    git = SafeGit(
        fixture.repository,
        cancel=cancel,
        command_runner=git_runner,
        activity=activities.append,
    )

    def command_runner(argv, **kwargs):  # noqa: ANN001, ANN003
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    with SqliteStore.open(fixture.database) as store:
        phase = _sanity_phase(
            fixture,
            plan,
            repository_lock,
            command_runner,
            cancel=cancel,
            git=git,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                phase.run,
                fixture.context(store, cancel=cancel),
                tip,
            )
            real_process_harness.wait_for_marker(
                "sanity-attestation-git.child.pid"
            )
            cancel.cancel()
            with pytest.raises(KeyboardInterrupt):
                result.result(timeout=5)

    real_process_harness.assert_tree_absent("sanity-attestation-git")
    assert observed_tokens == [cancel]
    assert any("rev-parse --abbrev-ref HEAD" in item.detail for item in activities)


def test_sanity_catalog_command_reports_redacted_activity_and_reaps_cancellation(
    tmp_path: Path,
    real_process_harness,
) -> None:
    fixture, tip, repository_lock = _merged_fixture(tmp_path)
    secret = 'sanity"secret/with space?x=1&y=2'
    escaped_secret = json.dumps(secret)[1:-1]
    encoded_secret = quote(secret, safe="")
    original_plan = _plan(fixture)
    plan = replace(
        original_plan,
        commands=(
            replace(
                original_plan.commands[0],
                argv=(
                    "catalog-install",
                    secret,
                    escaped_secret,
                    encoded_secret,
                ),
            ),
            original_plan.commands[1],
        ),
    )
    cancel = CancellationToken()
    resistant = real_process_harness.resistant_argv("sanity-catalog-command")
    invocations: list[tuple[tuple[str, ...], dict[str, object]]] = []
    activities = []

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        invocations.append((tuple(argv), dict(kwargs)))
        return run_captured(resistant, **kwargs)

    with SqliteStore.open(fixture.database) as store:
        phase = _sanity_phase(
            fixture,
            plan,
            repository_lock,
            runner,
            cancel=cancel,
        )
        context = fixture.context(store, cancel=cancel, activity=activities.append)
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                phase.run,
                context,
                tip,
                secret_values={"BUILD_TOKEN": secret, "AGENT_TOKEN": "agent"},
            )
            real_process_harness.wait_for_marker(
                "sanity-catalog-command.child.pid"
            )
            cancel.cancel()
            with pytest.raises(KeyboardInterrupt):
                result.result(timeout=5)
        runtime = store.get_task_runtime(fixture.task.id)

    real_process_harness.assert_tree_absent("sanity-catalog-command")
    assert len(invocations) == 1
    command, kwargs = invocations[0]
    assert command == plan.commands[0].argv
    assert kwargs["cancel"] is cancel
    assert kwargs["timeout"] == 600
    assert runtime is not None and runtime.status is TaskRuntimeStatus.MERGING
    assert len(activities) == 1
    detail = activities[0].detail
    assert detail.startswith("sanity: catalog-install")
    assert detail.count("[REDACTED]") == 3
    assert secret not in detail
    assert escaped_secret not in detail
    assert encoded_secret not in detail


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
    calls: list[tuple[tuple[str, ...], Path, dict[str, str]]] = []

    def runner(argv, *, cwd, env, **kwargs):  # noqa: ANN001, ANN003
        assert repository_lock.locked()
        calls.append((tuple(argv), Path(cwd), dict(env)))
        leaked = "\n".join((secret, json.dumps(secret)[1:-1], quote(secret, safe="")))
        return subprocess.CompletedProcess(argv, 0, stdout=leaked, stderr="")

    with SqliteStore.open(fixture.database) as store:
        before_attempts = store.list_environment_attempts(fixture.task.id)
        result = _sanity_phase(fixture, plan, repository_lock, runner).run(
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
    assert install_env["UNDECLARED_HOST"] == "no"
    assert test_env["UNDECLARED_HOST"] == "no"
    assert install_env["REGISTRY_URL"] == "https://registry.example.test"
    assert install_env["HOME"] == str(fixture.repository.parent)
    assert test_env["HOME"] == str(fixture.repository.parent)
    assert install_env["XDG_CACHE_HOME"] == str(fixture.repository.parent / "cache")
    assert test_env["XDG_CACHE_HOME"] == str(fixture.repository.parent / "cache")
    assert len(attempts) == len(before_attempts) + 1
    assert attempts[-1].kind == "materialize"
    assert sanity_events[-1].payload["preparation_key"] == attempts[-1].fingerprint
    assert result.commands[0].command.argv == ("catalog-install", "[REDACTED]")
    assert secret not in repr(result)
    persisted = json.dumps(sanity_events[-1].payload)
    assert secret not in persisted
    assert json.dumps(secret)[1:-1] not in persisted
    assert quote(secret, safe="") not in persisted
    assert persisted.count("[REDACTED]") == 7


def test_sanity_judges_the_catalog_on_the_merged_dependencies(
    tmp_path: Path,
) -> None:
    """The gate prepares the merged tip whatever the reuse rule would say.

    The worktree was prepared before the merge, and a digest of the declared
    commands cannot tell a merged tree from the tree it replaced, so the
    catalog would otherwise judge the dependencies of the tree the merge
    already replaced.
    """
    fixture = _approved_merge_fixture(tmp_path)
    subdir = fixture.repository / "package"
    subdir.mkdir()
    (subdir / ".keep").write_text("package\n", encoding="utf-8")
    _git(fixture.repository, "add", "package/.keep")
    _git(fixture.repository, "commit", "--quiet", "-m", "add package directory")

    installer = tmp_path / "install-dependencies"
    installer.write_text(
        '#!/bin/sh\nset -eu\ncat README.md > "$1"\n', encoding="utf-8"
    )
    installer.chmod(0o755)
    installed = tmp_path / "installed-dependencies"
    original_plan = _plan(fixture)
    plan = replace(
        original_plan,
        materialize_commands=(
            HostCommand("environment", (str(installer), str(installed)), "."),
        ),
    )

    with SqliteStore.open(fixture.database) as store:
        HostEnvironmentManager(
            fixture.repository,
            environment={"PATH": os.environ["PATH"]},
        ).materialize_claimed_task(
            store, plan, fixture.claim, fixture.owner_token
        )
    before_merge = installed.read_text(encoding="utf-8")

    merged_readme = "# Fixture\n\nbase descriptor changed\n"
    _advance_project_base(fixture, "README.md", merged_readme)
    repository_lock = RecordingLock()
    with SqliteStore.open(fixture.database) as store:
        merged = merge_phase(fixture, MockAdapter(), repository_lock).run(
            fixture.context(store)
        )
    assert merged.tip is not None

    judged: list[str] = []

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        judged.append(installed.read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    with SqliteStore.open(fixture.database) as store:
        result = _sanity_phase(fixture, plan, repository_lock, runner).run(
            fixture.context(store),
            merged.tip,
            secret_values={"BUILD_TOKEN": "build", "AGENT_TOKEN": "agent"},
        )

    assert result.status is TaskRuntimeStatus.DONE
    assert before_merge != merged_readme
    assert judged == [merged_readme, merged_readme]


def test_sanity_failure_blocks_and_never_advances(
    tmp_path: Path,
) -> None:
    fixture, tip, repository_lock = _merged_fixture(tmp_path)
    plan = _plan(fixture)
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
        result = _sanity_phase(fixture, plan, repository_lock, runner).run(
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
    assert _git(fixture.repository, "rev-parse", _project_branch(fixture)) == (
        tip.base_commit
    )
    assert Path(runtime.worktree_path).is_dir()


def test_cleanup_failure_blocks_before_completion_while_locked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, tip, repository_lock = _merged_fixture(tmp_path)
    plan = _plan(fixture)

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    with SqliteStore.open(fixture.database) as store:
        phase = _sanity_phase(fixture, plan, repository_lock, runner)

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
    calls: list[tuple[str, ...]] = []

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    with SqliteStore.open(fixture.database) as store:
        phase = _sanity_phase(fixture, plan, repository_lock, runner)
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


def _dropped_catalog_plan(fixture) -> HostPreflightPlan:  # noqa: ANN001
    """Return the plan a host without ``catalog-test``'s program produces."""
    original = _plan(fixture)
    return replace(
        original,
        commands=(original.commands[0],),
        dropped_commands=(
            HostDroppedCommand(
                original.commands[1],
                "host executable is not available: catalog-test "
                "(evidence: analyzer command catalog)",
            ),
        ),
    )


_DROPPED_SUMMARY = (
    "1 sanity command dropped: catalog-test: host executable is not "
    "available: catalog-test (evidence: analyzer command catalog)"
)


def test_published_task_names_the_command_the_host_could_not_run(
    tmp_path: Path,
) -> None:
    fixture, tip, repository_lock = _merged_fixture(tmp_path)
    plan = _dropped_catalog_plan(fixture)
    calls: list[tuple[str, ...]] = []

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    with SqliteStore.open(fixture.database) as store:
        result = _sanity_phase(fixture, plan, repository_lock, runner).run(
            fixture.context(store),
            tip,
            secret_values={"BUILD_TOKEN": "build", "AGENT_TOKEN": "agent"},
        )
        runtime = store.get_task_runtime(fixture.task.id)

    assert result.status is TaskRuntimeStatus.DONE
    assert calls == [("catalog-install",)]
    assert _DROPPED_SUMMARY in result.reason
    assert runtime is not None
    assert _DROPPED_SUMMARY in runtime.state_reason


def test_surviving_command_still_fails_the_task_and_names_the_drop(
    tmp_path: Path,
) -> None:
    fixture, tip, repository_lock = _merged_fixture(tmp_path)
    plan = _dropped_catalog_plan(fixture)

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        return subprocess.CompletedProcess(argv, 3, stdout="broken", stderr="")

    with SqliteStore.open(fixture.database) as store:
        result = _sanity_phase(fixture, plan, repository_lock, runner).run(
            fixture.context(store),
            tip,
            secret_values={"BUILD_TOKEN": "build", "AGENT_TOKEN": "agent"},
        )
        runtime = store.get_task_runtime(fixture.task.id)

    assert result.status is TaskRuntimeStatus.BLOCKED
    assert "sanity command failed with exit code 3" in result.reason
    assert _DROPPED_SUMMARY in result.reason
    assert runtime is not None and runtime.status is TaskRuntimeStatus.BLOCKED
    assert _git(fixture.repository, "rev-parse", _project_branch(fixture)) == (
        tip.base_commit
    )


def test_the_dropped_summary_is_masked_like_every_other_quotation(
    tmp_path: Path,
) -> None:
    """This reason is durable, and the summary quotes the analysis verbatim.

    It is stored as the task's state reason, listed back, and carried into the
    pull request body that is pushed. Everything else this phase quotes from
    the analysis is masked, and a catalogued argv is no safer than the argv of
    a command that ran.
    """
    fixture, tip, repository_lock = _merged_fixture(tmp_path)
    original = _plan(fixture)
    leaking = replace(
        original.commands[1],
        argv=(*original.commands[1].argv, "--token", "build"),
    )
    plan = replace(
        original,
        commands=(original.commands[0],),
        dropped_commands=(
            HostDroppedCommand(
                leaking, "host executable is not available: catalog-test"
            ),
        ),
    )

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    with SqliteStore.open(fixture.database) as store:
        result = _sanity_phase(fixture, plan, repository_lock, runner).run(
            fixture.context(store),
            tip,
            secret_values={"BUILD_TOKEN": "build", "AGENT_TOKEN": "agent"},
        )
        runtime = store.get_task_runtime(fixture.task.id)

    assert result.status is TaskRuntimeStatus.DONE
    assert "--token build" not in result.reason
    assert "catalog-test" in result.reason
    assert runtime is not None
    assert "--token build" not in runtime.state_reason


def test_a_task_with_no_check_to_run_blocks_rather_than_publishing(
    tmp_path: Path,
) -> None:
    """The gate's last backstop, behind preflight's refusal.

    Preflight refuses a run holding no check, so reaching here means something
    upstream let one through. Publishing the task anyway would advance the
    project base on a change nothing verified, and say nothing about it.
    """
    fixture, tip, repository_lock = _merged_fixture(tmp_path)
    plan = replace(_plan(fixture), commands=())
    calls: list[tuple[str, ...]] = []

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    with SqliteStore.open(fixture.database) as store:
        result = _sanity_phase(fixture, plan, repository_lock, runner).run(
            fixture.context(store),
            tip,
            secret_values={"BUILD_TOKEN": "build", "AGENT_TOKEN": "agent"},
        )
        runtime = store.get_task_runtime(fixture.task.id)

    assert result.status is TaskRuntimeStatus.BLOCKED
    assert "sanity command catalog is empty" in result.reason
    assert calls == []
    assert runtime is not None and runtime.status is TaskRuntimeStatus.BLOCKED
    assert _git(fixture.repository, "rev-parse", _project_branch(fixture)) == (
        tip.base_commit
    )
