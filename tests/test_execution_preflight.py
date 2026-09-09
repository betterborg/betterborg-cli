"""Shared materialization scaffold for host execution preflight."""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote
from uuid import UUID, uuid4

import pytest

from betterborg_cli.agent_runtime import CancellationToken, run_captured
from betterborg_cli.host_execution import (
    EnvironmentMaterializationError,
    HostCommand,
    HostEnvironmentManager,
    HostPreflightPlan,
    HostSecret,
    HostWorktreeManager,
)
from betterborg_cli.planning import render_task_markdown, task_markdown_digest
from betterborg_cli.progress import AgentActivityKind
from betterborg_cli.repo_paths import RepoPaths, ensure_managed_gitignore
from betterborg_cli.store import (
    Borg,
    ExecutionAttemptStatus,
    PlanApproval,
    Repository,
    SqliteStore,
    TaskBatch,
    TaskClaim,
    TaskComplexity,
    TaskGeneration,
    TaskRecord,
    TaskRuntimeStatus,
)


@dataclass(frozen=True)
class ExecutionPreflightFixture:
    repository: Path
    database: Path
    worktree_paths: tuple[Path, ...]
    task_ids: tuple[UUID, ...]
    run_id: UUID
    owner_token: str
    commands: list[list[str]] = field(default_factory=list)

    def claim(self, store: SqliteStore) -> TaskClaim:
        claim = store.claim_dependency_ready_task(
            self.run_id,
            self.owner_token,
            lease_duration=timedelta(minutes=30),
        )
        assert claim is not None
        return claim

    def manager(self) -> HostEnvironmentManager:
        def runner(argv, **kwargs):  # noqa: ANN001, ANN003
            self.commands.append(list(argv))
            return run_captured(argv, **kwargs)

        return HostEnvironmentManager(
            self.repository,
            environment={"PATH": os.environ["PATH"]},
            command_runner=runner,
        )


@pytest.fixture
def execution_preflight_fixture(tmp_path: Path):
    """Create claimed-worktree inputs with a fake package manager."""

    def create(*, task_count: int = 1) -> ExecutionPreflightFixture:
        repository = tmp_path / f"repository-{uuid4().hex}"
        repository.mkdir()
        _git(repository, "init", "--quiet", "--initial-branch=main")
        _git(repository, "config", "user.name", "Betterborg Tests")
        _git(repository, "config", "user.email", "tests@betterborg.dev")
        (repository / "README.md").write_text("# Fixture\n", encoding="utf-8")
        (repository / "package.lock").write_text("lock-v1\n", encoding="utf-8")
        (repository / ".gitignore").write_text(
            ".dependencies/\nuntracked.lock\n", encoding="utf-8"
        )
        _write_fake_package_manager(repository / "fake-package-manager")
        nested = repository / "packages"
        nested.mkdir()
        _write_fake_package_manager(nested / "fake-package-manager")
        (nested / "package.lock").write_text("lock-v1\n", encoding="utf-8")
        ensure_managed_gitignore(RepoPaths.discover(repository))

        database = tmp_path / f"state-{uuid4().hex}.sqlite3"
        repository_record = Repository(root=repository)
        borg = Borg(repository_id=repository_record.id, name="EnvironmentFixture")
        approval = PlanApproval(
            borg_id=borg.id,
            plan_digest="sha256:plan",
            manifest={"plan.md": "sha256:plan"},
        )
        batch = TaskBatch(
            borg_id=borg.id,
            plan_approval_id=approval.id,
            round=1,
            digest="sha256:batch",
            manifest={},
        )
        generation = TaskGeneration(
            borg_id=borg.id,
            plan_approval_id=approval.id,
            batch_id=batch.id,
            digest="sha256:generation",
            manifest={},
        )
        tasks = tuple(
            _task_record(generation, borg, position)
            for position in range(1, task_count + 1)
        )
        durable_root = (
            repository
            / ".betterborg/tasks"
            / borg.name
            / str(generation.id)
        )

        with SqliteStore.open(database) as store:
            store.add_repository(repository_record)
            store.add_borg(borg)
            store.append_plan_approval(approval)
            store.append_task_batch(batch)
            store.add_task_generation(generation, tasks)
            for task in tasks:
                path = durable_root / task.stage / f"{task.stem}.md"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(render_task_markdown(task.task), encoding="utf-8")
            store._promote_published_task_generation(
                generation.id,
                durable_root=durable_root,
                tasks_root=repository / ".betterborg/tasks",
                owned_root=repository,
            )

        _git(repository, "add", ".")
        _git(repository, "commit", "--quiet", "-m", "fixture")

        with SqliteStore.open(database) as store:
            acquisition = store.acquire_execution_run(
                borg.id,
                generation.id,
                lease_duration=timedelta(hours=1),
            )
            assert acquisition.owner_token is not None
            worktree_root = tmp_path / f"worktrees-{uuid4().hex}"
            specs = HostWorktreeManager(
                repository,
                worktree_root,
                source_branch="main",
            ).prepare_current_task_worktrees(
                store,
                run_id=acquisition.run_id,
                owner_token=acquisition.owner_token,
                generation_id=generation.id,
                project_name="fixture",
            )

        return ExecutionPreflightFixture(
            repository=repository,
            database=database,
            worktree_paths=tuple(spec.path for spec in specs),
            task_ids=tuple(task.id for task in tasks),
            run_id=acquisition.run_id,
            owner_token=acquisition.owner_token,
        )

    return create


def test_the_unselected_command_list_never_runs(
    execution_preflight_fixture,
) -> None:
    """Exactly one declared list prepares a worktree, in that worktree.

    The assertion has to be that the prepare command never ran: its output
    was always discarded with the disposable worktree it ran in, so a
    worktree-shaped assertion would hold either way.
    """
    fixture = execution_preflight_fixture(task_count=2)
    plan = _plan(fixture.repository)

    with SqliteStore.open(fixture.database) as store:
        for _ in fixture.worktree_paths:
            claim = fixture.claim(store)
            fixture.manager().materialize_claimed_task(
                store, plan, claim, fixture.owner_token
            )
        kinds = {
            attempt.kind
            for task_id in fixture.task_ids
            for attempt in store.list_environment_attempts(task_id)
        }

    assert kinds == {"materialize"}
    assert _command_count(fixture, "prepare") == 0
    assert _command_count(fixture, "materialize") == 2
    for worktree in fixture.worktree_paths:
        assert (worktree / ".dependencies/materialized").is_file()
        assert not (worktree / ".dependencies/prepared").exists()
    assert _git(fixture.repository, "status", "--porcelain") == ""


def test_editing_the_unselected_command_list_does_not_reinstall(
    execution_preflight_fixture,
) -> None:
    """The key carries the list that runs, so the other one cannot invalidate.

    A repository declaring both lists never runs its prepare command, and an
    edit to a command that will never run must not re-install every worktree
    that already ran the one that does.
    """
    fixture = execution_preflight_fixture()
    plan = _plan(fixture.repository)
    edited = replace(
        plan,
        prepare_commands=(
            replace(
                plan.prepare_commands[0],
                argv=("./fake-package-manager", "prepare", "edited"),
            ),
        ),
    )

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        first = fixture.manager().materialize_claimed_task(
            store, plan, claim, fixture.owner_token
        )
        second = fixture.manager().materialize_claimed_task(
            store, edited, claim, fixture.owner_token
        )

    assert second.preparation_key == first.preparation_key
    assert second.materialization_reused is True
    assert _command_count(fixture, "materialize") == 1
    assert _command_count(fixture, "prepare") == 0


def test_the_key_separates_one_command_run_in_two_directories(
    execution_preflight_fixture,
) -> None:
    """A declared command is its working directory as much as its argv.

    Moving a declared install into a subdirectory installs somewhere else,
    so a worktree prepared by the one must not be called prepared for the
    other while the argv they share stays identical.
    """
    fixture = execution_preflight_fixture()
    at_root = _plan(fixture.repository, prepare_action=None)
    in_package = _plan(fixture.repository, prepare_action=None, cwd="packages")
    assert at_root.materialize_commands[0].argv == (
        in_package.materialize_commands[0].argv
    )

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        first = fixture.manager().materialize_claimed_task(
            store, at_root, claim, fixture.owner_token
        )
        moved = fixture.manager().materialize_claimed_task(
            store, in_package, claim, fixture.owner_token
        )

    assert moved.preparation_key != first.preparation_key
    assert moved.materialization_reused is False
    assert _command_count(fixture, "materialize") == 2
    worktree = fixture.worktree_paths[0]
    assert (worktree / "packages/.dependencies/materialized").is_file()


def test_no_cache_directory_and_no_cache_marker_is_created(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    plan = _plan(fixture.repository)
    state = fixture.repository / ".betterborg/state"

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        materialization = fixture.manager().materialize_claimed_task(
            store, plan, claim, fixture.owner_token
        )
        attempts = store.list_environment_attempts(claim.task_id)

    assert [set(attempt.result) for attempt in attempts] == [{"commands"}]
    assert not (state / "environment-cache").exists()
    assert not (fixture.repository.parent / ".betterborg-environments").exists()
    assert list(state.rglob(".betterborg-prepared")) == []
    marker = (
        fixture.worktree_paths[0]
        / ".betterborg/state/environment-materialization"
    )
    assert marker.read_text(encoding="utf-8").strip() == (
        materialization.preparation_key
    )


def test_a_repository_declaring_no_preparation_command_reaches_coding(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    plan = _plan(
        fixture.repository, prepare_action=None, materialize_action=None
    )

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        fixture.manager().materialize_claimed_task(
            store, plan, claim, fixture.owner_token
        )
        runtime = store.get_task_runtime(claim.task_id)
        attempts = store.list_environment_attempts(claim.task_id)

    assert runtime is not None and runtime.status is TaskRuntimeStatus.CODING
    assert [attempt.commands for attempt in attempts] == [[]]
    assert fixture.commands == []


def test_a_gitignored_lockfile_does_not_block_any_task(
    execution_preflight_fixture,
) -> None:
    """A declared file the repository ignores is in no task worktree.

    Preflight sees it in the primary checkout, and a task worktree holds
    tracked files only, so requiring it there blocked every task a
    repository with a normal ignore rule had.
    """
    fixture = execution_preflight_fixture(task_count=2)
    lockfile = fixture.repository / "untracked.lock"
    lockfile.write_text("lock-v1\n", encoding="utf-8")
    plan = replace(
        _plan(fixture.repository, prepare_action=None),
        environment_files=(lockfile,),
    )

    with SqliteStore.open(fixture.database) as store:
        for _ in fixture.worktree_paths:
            claim = fixture.claim(store)
            fixture.manager().materialize_claimed_task(
                store, plan, claim, fixture.owner_token
            )
        statuses = [
            store.get_task_runtime(task_id).status
            for task_id in fixture.task_ids
        ]

    assert not any(
        (worktree / "untracked.lock").exists()
        for worktree in fixture.worktree_paths
    )
    assert statuses == [TaskRuntimeStatus.CODING, TaskRuntimeStatus.CODING]


def test_environment_command_runs_in_the_operator_environment(
    execution_preflight_fixture, tmp_path: Path
) -> None:
    """A repository builds on this machine, so build it the way it builds.

    A synthesized environment hides a toolchain that lives under the real
    home and cold-starts caches that are already warm, and the failure it
    produces is one the operator cannot reproduce by hand.
    """
    fixture = execution_preflight_fixture()
    plan = _plan(fixture.repository, materialize_action=None)
    operator = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path / "operator-home"),
        "XDG_CACHE_HOME": str(tmp_path / "operator-cache"),
        "OPERATOR_TOOLCHAIN": str(tmp_path / "operator-toolchain"),
    }
    environments: list[dict[str, str]] = []

    def runner(argv, *, env, **kwargs):  # noqa: ANN001, ANN003
        environments.append(dict(env))
        return run_captured(argv, env=env, **kwargs)

    manager = HostEnvironmentManager(
        fixture.repository,
        environment=operator,
        command_runner=runner,
    )
    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        materialization = manager.materialize_claimed_task(
            store, plan, claim, fixture.owner_token
        )

    assert environments
    for observed in environments:
        assert observed["HOME"] == operator["HOME"]
        assert observed["XDG_CACHE_HOME"] == operator["XDG_CACHE_HOME"]
        assert observed["OPERATOR_TOOLCHAIN"] == operator["OPERATOR_TOOLCHAIN"]
        assert set(observed) == set(operator) | {"GIT_TERMINAL_PROMPT"}
    assert "BETTERBORG_ENVIRONMENT_ROOT" not in materialization.environment


def test_environment_command_cannot_block_on_a_credential_prompt(
    execution_preflight_fixture,
) -> None:
    """Preparation runs with no timeout and inherits a stdin.

    A command that reaches a private dependency would otherwise wait on a
    credential prompt with nothing left to end it.
    """
    fixture = execution_preflight_fixture()
    plan = _plan(fixture.repository, prepare_action=None)
    environments: list[dict[str, str]] = []

    def runner(argv, *, env, **kwargs):  # noqa: ANN001, ANN003
        environments.append(dict(env))
        return run_captured(argv, env=env, **kwargs)

    manager = HostEnvironmentManager(
        fixture.repository,
        environment={
            "PATH": os.environ["PATH"],
            "GIT_TERMINAL_PROMPT": "1",
        },
        command_runner=runner,
    )
    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        manager.materialize_claimed_task(
            store, plan, claim, fixture.owner_token
        )

    assert [observed["GIT_TERMINAL_PROMPT"] for observed in environments] == ["0"]


def test_a_repository_declaring_only_a_prepare_list_is_prepared_by_it(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    plan = _plan(fixture.repository, materialize_action=None)

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        fixture.manager().materialize_claimed_task(
            store, plan, claim, fixture.owner_token
        )

    assert (fixture.worktree_paths[0] / ".dependencies/prepared").is_file()
    assert _command_count(fixture, "prepare") == 1


def test_restart_reuses_matching_successful_materialization(
    execution_preflight_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = execution_preflight_fixture()
    plan = _plan(fixture.repository)

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        transition = store.transition_task_runtime
        interrupted = False

        def interrupt_after_materialization(*args, **kwargs):
            nonlocal interrupted
            if kwargs.get("new_status") is TaskRuntimeStatus.CODING and not interrupted:
                interrupted = True
                raise RuntimeError("simulated restart")
            return transition(*args, **kwargs)

        monkeypatch.setattr(
            store, "transition_task_runtime", interrupt_after_materialization
        )
        with pytest.raises(RuntimeError, match="simulated restart"):
            fixture.manager().materialize_claimed_task(
                store, plan, claim, fixture.owner_token
            )
        monkeypatch.setattr(store, "transition_task_runtime", transition)

        resumed = fixture.manager().materialize_claimed_task(
            store, plan, claim, fixture.owner_token
        )

    assert resumed.materialization_reused is True
    assert _command_count(fixture, "materialize") == 1


def test_only_a_changed_declared_command_prepares_a_worktree_again(
    execution_preflight_fixture,
) -> None:
    """The key carries the selected command list and nothing else.

    An edit to a declared file no longer reinstalls a worktree, which is
    what keeps an interrupted task from paying for its agent's dependency
    twice; an edit to the command that installs it still does.
    """
    fixture = execution_preflight_fixture()
    plan = _plan(fixture.repository, prepare_action=None)
    worktree = fixture.worktree_paths[0]

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        first = fixture.manager().materialize_claimed_task(
            store, plan, claim, fixture.owner_token
        )

        (worktree / "package.lock").write_text("lock-v2\n", encoding="utf-8")
        unchanged_command = fixture.manager().materialize_claimed_task(
            store, plan, claim, fixture.owner_token
        )

        edited = replace(
            plan,
            materialize_commands=(
                replace(
                    plan.materialize_commands[0],
                    argv=("./fake-package-manager", "materialize", "--offline"),
                ),
            ),
        )
        changed_command = fixture.manager().materialize_claimed_task(
            store, edited, claim, fixture.owner_token
        )
        runtime = store.get_task_runtime(claim.task_id)

    assert unchanged_command.preparation_key == first.preparation_key
    assert unchanged_command.materialization_reused is True
    assert changed_command.preparation_key != first.preparation_key
    assert changed_command.materialization_reused is False
    assert _command_count(fixture, "materialize") == 2
    assert (worktree / "package.lock").read_text() == "lock-v2\n"
    assert runtime is not None and runtime.status is TaskRuntimeStatus.CODING


def test_a_disagreeing_marker_prepares_again_and_an_agreeing_one_does_not(
    execution_preflight_fixture,
) -> None:
    """The stored attempt and the checkout's marker are one condition.

    A completed attempt outlives the dependencies it installed, so a
    checkout that lost them must not be treated as prepared however
    confidently the store remembers the install.
    """
    fixture = execution_preflight_fixture()
    plan = _plan(fixture.repository, prepare_action=None)
    marker = (
        fixture.worktree_paths[0]
        / ".betterborg/state/environment-materialization"
    )

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        first = fixture.manager().materialize_claimed_task(
            store, plan, claim, fixture.owner_token
        )
        agreeing = fixture.manager().materialize_claimed_task(
            store, plan, claim, fixture.owner_token
        )

        marker.write_text("sha256:another-checkout\n", encoding="utf-8")
        disagreeing = fixture.manager().materialize_claimed_task(
            store, plan, claim, fixture.owner_token
        )
        completed = store.find_completed_environment_attempt(
            first.preparation_key, kind="materialize", task_id=claim.task_id
        )

    assert agreeing.materialization_reused is True
    assert disagreeing.materialization_reused is False
    assert completed is not None
    assert disagreeing.preparation_key == first.preparation_key
    assert marker.read_text(encoding="utf-8").strip() == first.preparation_key
    assert _command_count(fixture, "materialize") == 2


def test_a_marker_without_a_completed_attempt_prepares_again(
    execution_preflight_fixture,
) -> None:
    """The marker is written before the attempt recording it completes.

    A process that dies in that window leaves the key on a checkout with no
    completed attempt behind it. Only running the commands again can make
    the pair agree, so the stored attempt has to be consulted even when the
    marker already holds the key.
    """
    fixture = execution_preflight_fixture(task_count=2)
    plan = _plan(fixture.repository, prepare_action=None)

    with SqliteStore.open(fixture.database) as store:
        prepared = fixture.manager().materialize_claimed_task(
            store, plan, fixture.claim(store), fixture.owner_token
        )
        claim = fixture.claim(store)
        runtime = store.get_task_runtime(claim.task_id)
        assert runtime is not None and runtime.worktree_path is not None
        marker = (
            Path(runtime.worktree_path)
            / ".betterborg/state/environment-materialization"
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{prepared.preparation_key}\n", encoding="utf-8")

        materialization = fixture.manager().materialize_claimed_task(
            store, plan, claim, fixture.owner_token
        )
        attempts = store.list_environment_attempts(claim.task_id)

    assert materialization.preparation_key == prepared.preparation_key
    assert materialization.materialization_reused is False
    assert [attempt.kind for attempt in attempts] == ["materialize"]
    assert _command_count(fixture, "materialize") == 2


def test_an_interrupted_preparation_between_two_identical_ones_reinstalls(
    execution_preflight_fixture,
) -> None:
    """An interrupted preparation may already have replaced what it installed.

    Returning to the command that succeeded before finds its own completed
    attempt still in the store, so only invalidating the marker up front
    stops the checkout being called prepared when the interruption left it
    half written.
    """
    fixture = execution_preflight_fixture()
    prepared = _plan(fixture.repository, prepare_action=None)
    interrupted = _plan(
        fixture.repository,
        prepare_action=None,
        materialize_action="fail-dirty",
    )
    dependencies = fixture.worktree_paths[0] / ".dependencies/materialized"
    cancel = CancellationToken()

    def interrupt_once_dirty(argv, **kwargs):  # noqa: ANN001, ANN003
        result = run_captured(argv, **kwargs)
        cancel.cancel()
        return result

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        fixture.manager().materialize_claimed_task(
            store, prepared, claim, fixture.owner_token
        )
        assert dependencies.read_text(encoding="utf-8") == "lock-v1\n"

        with pytest.raises(KeyboardInterrupt):
            HostEnvironmentManager(
                fixture.repository,
                environment={"PATH": os.environ["PATH"]},
                command_runner=interrupt_once_dirty,
                cancel=cancel,
            ).materialize_claimed_task(
                store, interrupted, claim, fixture.owner_token
            )
        assert dependencies.read_text(encoding="utf-8") == "half-installed\n"

        restored = fixture.manager().materialize_claimed_task(
            store, prepared, claim, fixture.owner_token
        )

    assert restored.materialization_reused is False
    assert dependencies.read_text(encoding="utf-8") == "lock-v1\n"
    assert _command_count(fixture, "materialize") == 2


def test_environment_command_contaminating_primary_checkout_blocks_task(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    plan = _plan(
        fixture.repository,
        prepare_action=None,
        materialize_action="materialize",
    )

    def contaminate_primary(*args, **kwargs):
        (fixture.repository / "README.md").write_text(
            "contaminated\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(args[0], 0, "", "")

    manager = HostEnvironmentManager(
        fixture.repository,
        environment={"PATH": os.environ["PATH"]},
        command_runner=contaminate_primary,
    )
    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        with pytest.raises(
            EnvironmentMaterializationError,
            match="primary checkout.*changed",
        ):
            manager.materialize_claimed_task(
                store, plan, claim, fixture.owner_token
            )
        runtime = store.get_task_runtime(claim.task_id)
        attempts = store.list_environment_attempts(claim.task_id)

    assert runtime is not None and runtime.status is TaskRuntimeStatus.BLOCKED
    assert attempts[-1].status is ExecutionAttemptStatus.FAILED
    assert (fixture.repository / "README.md").read_text() == "contaminated\n"


def test_build_secret_is_scoped_and_redacted_from_durable_failure(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    token = "scoped-package-token"
    plan = _plan(
        fixture.repository,
        prepare_action=None,
        materialize_action="fail",
        secrets=(
            HostSecret(
                name="PACKAGE_TOKEN",
                scope="build",
                used_by=("environment",),
                evidence="fixture",
            ),
        ),
    )

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        with pytest.raises(EnvironmentMaterializationError) as caught:
            fixture.manager().materialize_claimed_task(
                store,
                plan,
                claim,
                fixture.owner_token,
                secret_values={"PACKAGE_TOKEN": token},
            )
        attempts = store.list_environment_attempts(claim.task_id)
        runtime = store.get_task_runtime(claim.task_id)

    assert token not in str(caught.value)
    assert attempts[-1].status is ExecutionAttemptStatus.FAILED
    assert attempts[-1].error is not None
    assert token not in attempts[-1].error
    assert "[REDACTED]" in attempts[-1].error
    assert runtime is not None and runtime.status is TaskRuntimeStatus.BLOCKED


def test_encoded_build_secret_is_redacted_from_materialization_result(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    token = 'token"with/slash space?x=1&y=2'
    plan = _plan(
        fixture.repository,
        prepare_action=None,
        materialize_action="materialize",
        secrets=(
            HostSecret(
                name="PACKAGE_TOKEN",
                scope="build",
                used_by=("environment",),
                evidence="fixture",
            ),
        ),
    )

    def emit_encoded_secret(argv, *, env, **kwargs):  # noqa: ANN001, ANN003
        assert env["PACKAGE_TOKEN"] == token
        output = "\n".join(
            (token, json.dumps(token)[1:-1], quote(token, safe=""))
        )
        return subprocess.CompletedProcess(argv, 0, output, output)

    manager = HostEnvironmentManager(
        fixture.repository,
        environment={"PATH": os.environ["PATH"]},
        command_runner=emit_encoded_secret,
    )
    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        manager.materialize_claimed_task(
            store,
            plan,
            claim,
            fixture.owner_token,
            secret_values={"PACKAGE_TOKEN": token},
        )
        attempt = store.list_environment_attempts(claim.task_id)[-1]

    persisted = json.dumps({"error": attempt.error, "result": attempt.result})
    assert token not in persisted
    assert json.dumps(token)[1:-1] not in persisted
    assert quote(token, safe="") not in persisted
    assert persisted.count("[REDACTED]") == 6


def test_build_secret_is_not_exposed_outside_used_by_stage(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    token = "test-only-package-token"
    plan = _plan(
        fixture.repository,
        prepare_action=None,
        materialize_action="capture-secret",
        secrets=(
            HostSecret(
                name="PACKAGE_TOKEN",
                scope="build",
                used_by=("test",),
                evidence="fixture",
            ),
        ),
    )

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        fixture.manager().materialize_claimed_task(
            store,
            plan,
            claim,
            fixture.owner_token,
            secret_values={"PACKAGE_TOKEN": token},
        )

    captured = fixture.worktree_paths[0] / ".dependencies/secret"
    assert captured.read_text(encoding="utf-8") == "unset\n"


def test_declared_secret_is_subtracted_from_the_operator_environment(
    execution_preflight_fixture,
) -> None:
    """An inherited environment has to be subtracted from, not just added to.

    The operator exports the credential a build stage needs, so a stage that
    did not declare it would otherwise inherit it from the shell rather than
    from the declaration that names the stages allowed to see it.
    """
    fixture = execution_preflight_fixture()
    token = "operator-shell-package-token"
    plan = _plan(
        fixture.repository,
        prepare_action=None,
        materialize_action="capture-secret",
        secrets=(
            HostSecret(
                name="PACKAGE_TOKEN",
                scope="build",
                used_by=("test",),
                evidence="fixture",
            ),
        ),
    )

    manager = HostEnvironmentManager(
        fixture.repository,
        environment={"PATH": os.environ["PATH"], "PACKAGE_TOKEN": token},
    )
    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        manager.materialize_claimed_task(
            store,
            plan,
            claim,
            fixture.owner_token,
            secret_values={"PACKAGE_TOKEN": token},
        )

    captured = fixture.worktree_paths[0] / ".dependencies/secret"
    assert captured.read_text(encoding="utf-8") == "unset\n"


def test_build_secret_is_redacted_outside_used_by_stage(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    token = 'token"with/slash space?x=1&y=2'
    plan = _plan(
        fixture.repository,
        prepare_action=None,
        materialize_action="materialize",
        secrets=(
            HostSecret(
                name="PACKAGE_TOKEN",
                scope="build",
                used_by=("test",),
                evidence="fixture",
            ),
        ),
    )
    plan = replace(
        plan,
        materialize_commands=(
            replace(plan.materialize_commands[0], argv=("materialize", token)),
        ),
    )

    def emit_secret(argv, *, env, **kwargs):  # noqa: ANN001, ANN003
        assert "PACKAGE_TOKEN" not in env
        output = "\n".join(
            (token, json.dumps(token)[1:-1], quote(token, safe=""))
        )
        return subprocess.CompletedProcess(argv, 0, output, output)

    manager = HostEnvironmentManager(
        fixture.repository,
        environment={"PATH": os.environ["PATH"]},
        command_runner=emit_secret,
    )
    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        manager.materialize_claimed_task(
            store,
            plan,
            claim,
            fixture.owner_token,
            secret_values={"PACKAGE_TOKEN": token},
        )
        attempt = store.list_environment_attempts(claim.task_id)[-1]

    persisted = json.dumps(
        {
            "commands": attempt.commands,
            "error": attempt.error,
            "result": attempt.result,
        }
    )
    assert token not in persisted
    assert json.dumps(token)[1:-1] not in persisted
    assert quote(token, safe="") not in persisted
    assert "[REDACTED]" in persisted


def test_environment_command_reports_redacted_activity_and_reaps_on_cancel(
    execution_preflight_fixture,
    real_process_harness,
) -> None:
    fixture = execution_preflight_fixture()
    cancel = CancellationToken()
    token = 'token"with/slash space?x=1&y=2'
    command = real_process_harness.resistant_argv("environment-command")
    plan = _plan(
        fixture.repository,
        prepare_action=None,
        materialize_action="unused",
        secrets=(
            HostSecret(
                name="PACKAGE_TOKEN",
                scope="build",
                used_by=("environment",),
                evidence="fixture",
            ),
        ),
    )
    plan = replace(
        plan,
        materialize_commands=(
            HostCommand(
                stage="environment",
                argv=(*command, token),
                cwd=".",
                evidence="fixture",
            ),
        ),
    )
    activities = []
    observed_tokens: list[CancellationToken | None] = []

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        observed_tokens.append(kwargs.get("cancel"))
        return run_captured(argv, **kwargs)

    manager = HostEnvironmentManager(
        fixture.repository,
        environment={"PATH": os.environ["PATH"]},
        command_runner=runner,
        activity=activities.append,
        cancel=cancel,
    )
    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                manager.materialize_claimed_task,
                store,
                plan,
                claim,
                fixture.owner_token,
                secret_values={"PACKAGE_TOKEN": token},
            )
            real_process_harness.wait_for_marker(
                "environment-command.child.pid"
            )
            cancel.cancel()
            with pytest.raises(KeyboardInterrupt):
                result.result(timeout=5)

        runtime = store.get_task_runtime(claim.task_id)

    real_process_harness.assert_tree_absent("environment-command")
    assert observed_tokens == [cancel]
    assert len(activities) == 1
    assert activities[0].kind is AgentActivityKind.COMMAND
    assert activities[0].detail is not None
    assert token not in activities[0].detail
    assert json.dumps(token)[1:-1] not in activities[0].detail
    assert quote(token, safe="") not in activities[0].detail
    assert "[REDACTED]" in activities[0].detail
    assert runtime is not None and runtime.status is TaskRuntimeStatus.ENVIRONMENT


def test_tracked_changes_are_rejected_and_task_work_is_preserved(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    plan = _plan(
        fixture.repository,
        prepare_action=None,
        materialize_action="tracked",
    )

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        with pytest.raises(
            EnvironmentMaterializationError, match="unexpected tracked changes"
        ):
            fixture.manager().materialize_claimed_task(
                store, plan, claim, fixture.owner_token
            )
        runtime = store.get_task_runtime(claim.task_id)

    assert (fixture.worktree_paths[0] / "README.md").read_text() == "changed\n"
    assert runtime is not None and runtime.status is TaskRuntimeStatus.BLOCKED


def test_command_failure_blocks_before_coding(execution_preflight_fixture) -> None:
    fixture = execution_preflight_fixture()
    plan = _plan(
        fixture.repository,
        prepare_action=None,
        materialize_action="fail",
    )

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        with pytest.raises(EnvironmentMaterializationError, match="exit code 7"):
            fixture.manager().materialize_claimed_task(
                store, plan, claim, fixture.owner_token
            )
        runtime = store.get_task_runtime(claim.task_id)

    assert runtime is not None and runtime.status is TaskRuntimeStatus.BLOCKED


def _task_record(
    generation: TaskGeneration, borg: Borg, position: int
) -> TaskRecord:
    body = {
        "stage": "07-host-execution",
        "stem": f"{position:02d}-environment",
        "title": f"Materialize environment {position}",
        "why": "The test needs a claimed task.",
        "scope": ["Materialize dependencies."],
        "implementation_notes": [],
        "acceptance_criteria": ["Dependencies are local."],
        "tests": ["Run the fake package manager."],
        "dependencies": [],
        "out_of_scope": [],
        "plan_refs": ["P1.deliverable.1"],
        "estimate_complexity": "small",
    }
    digest = task_markdown_digest(render_task_markdown(body))
    return TaskRecord(
        generation_id=generation.id,
        borg_id=borg.id,
        task_ref=f"environment-{position}",
        stage=body["stage"],
        stem=body["stem"],
        position=position,
        title=body["title"],
        complexity=TaskComplexity.SMALL,
        digest=digest,
        task=body,
        manifest={"task.md": digest},
    )


def _plan(
    repository: Path,
    *,
    prepare_action: str | None = "prepare",
    materialize_action: str | None = "materialize",
    secrets: tuple[HostSecret, ...] = (),
    cwd: str = ".",
) -> HostPreflightPlan:
    def commands(action: str | None) -> tuple[HostCommand, ...]:
        if action is None:
            return ()
        return (
            HostCommand(
                stage="environment",
                argv=("./fake-package-manager", action),
                cwd=cwd,
                evidence="fixture",
            ),
        )

    return HostPreflightPlan(
        repository_root=repository,
        commands=(),
        prepare_commands=commands(prepare_action),
        materialize_commands=commands(materialize_action),
        environment_files=(repository / "package.lock",),
        executables=(),
        required_secret_names=tuple(secret.name for secret in secrets),
        package_managers=("pip",),
        secret_requirements=secrets,
    )



def _write_fake_package_manager(path: Path) -> None:
    path.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "action=$1\n"
        "case \"$action\" in\n"
        "  prepare)\n"
        "    cat package.lock\n"
        "    mkdir -p .dependencies\n"
        "    printf 'local\\n' > .dependencies/prepared\n"
        "    ;;\n"
        "  materialize)\n"
        "    mkdir -p .dependencies\n"
        "    cp package.lock .dependencies/materialized\n"
        "    ;;\n"
        "  tracked)\n"
        "    printf 'changed\\n' > README.md\n"
        "    ;;\n"
        "  capture-secret)\n"
        "    mkdir -p .dependencies\n"
        "    printf '%s\\n' \"${PACKAGE_TOKEN:-unset}\" > .dependencies/secret\n"
        "    ;;\n"
        "  fail-dirty)\n"
        "    mkdir -p .dependencies\n"
        "    printf 'half-installed\\n' > .dependencies/materialized\n"
        "    exit 7\n"
        "    ;;\n"
        "  fail)\n"
        "    printf '%s\\n' \"${PACKAGE_TOKEN:-package failed}\" >&2\n"
        "    exit 7\n"
        "    ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _command_count(fixture: ExecutionPreflightFixture, action: str) -> int:
    """Count the environment commands the fixture's manager actually ran."""
    return sum(1 for argv in fixture.commands if action in argv)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
