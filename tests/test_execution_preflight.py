"""Shared materialization and service scaffold for host execution preflight."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote
from uuid import UUID, uuid4

import pytest

from betterborg_cli.agent_runtime import CancellationToken, run_captured
from betterborg_cli.host_execution import (
    ComposeStackError,
    EnvironmentMaterializationError,
    HostCommand,
    HostComposeManager,
    HostEnvironmentManager,
    HostExecutable,
    HostPreflightPlan,
    HostSecret,
    HostService,
    HostWorktreeManager,
    compose_project_name,
    service_url_environment,
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

    def compose_manager(self, runner=None) -> HostComposeManager:
        return HostComposeManager(
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
        (repository / "ComposeFixture.Dockerfile").write_text(
            "FROM scratch\n"
            "COPY .dependencies/compose-fixture /compose-fixture\n"
            'ENTRYPOINT ["/compose-fixture"]\n',
            encoding="utf-8",
        )
        (repository / "compose.yml").write_text(
            "services:\n"
            "  healthy:\n"
            "    image: betterborg/shared-compose-fixture:dev\n"
            "    build:\n"
            "      context: .\n"
            "      dockerfile: ComposeFixture.Dockerfile\n"
            "    container_name: betterborg-fixed-collision\n"
            "    depends_on:\n"
            "      unused:\n"
            "        condition: service_started\n"
            "    healthcheck:\n"
            "      test: [CMD, /compose-fixture, --health]\n"
            "      interval: 200ms\n"
            "      timeout: 1s\n"
            "      retries: 20\n"
            "    ports:\n"
            '      - "127.0.0.1:39091:8080"\n'
            "    networks: [fixture]\n"
            "    volumes: [fixture-data:/data]\n"
            "  unused:\n"
            "    build:\n"
            "      context: .\n"
            "      dockerfile: ComposeFixture.Dockerfile\n"
            "networks:\n"
            "  fixture:\n"
            "    name: betterborg-fixed-network\n"
            "volumes:\n"
            "  fixture-data:\n"
            "    name: betterborg-fixed-volume\n",
            encoding="utf-8",
        )
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


def test_jobs_two_compose_stacks_are_healthy_distinct_and_isolated(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture(task_count=2)
    _prepare_compose_fixture(fixture)
    plan = _compose_plan(fixture.repository)
    with SqliteStore.open(fixture.database) as store:
        claims = (fixture.claim(store), fixture.claim(store))

    compose_results: list[subprocess.CompletedProcess[str]] = []
    results_lock = threading.Lock()

    def run_compose(*args, **kwargs):
        result = _run_real_compose(*args, **kwargs)
        with results_lock:
            compose_results.append(result)
        return result

    def start(claim: TaskClaim):
        with SqliteStore.open(fixture.database) as store:
            return fixture.compose_manager(run_compose).start_claimed_stack(
                store, plan, claim, fixture.owner_token
            )

    stacks = []
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(start, claim) for claim in claims]
            errors = []
            for future in futures:
                try:
                    stacks.append(future.result())
                except BaseException as error:
                    stacks.append(None)
                    errors.append(error)
            if errors:
                loopback_failures = [
                    result
                    for result in compose_results
                    if result.returncode != 0
                    and "Unable to enable LOOPBACK FILTERING" in result.stderr
                    and "iptables" in result.stderr
                ]
                if len(loopback_failures) == len(errors):
                    # Docker Engine protects loopback-only publications with a
                    # raw-table firewall rule.  This constrained test kernel
                    # lacks that table, so the only safe runtime behavior here
                    # is to reject startup without widening the binding.
                    overrides = sorted(
                        (fixture.repository / ".betterborg/state/compose").glob(
                            "*/compose.override.yml"
                        )
                    )
                    assert len(overrides) == 2
                    for override in overrides:
                        text = override.read_text(encoding="utf-8")
                        assert 'host_ip: "127.0.0.1"' in text
                        assert 'host_ip: "0.0.0.0"' not in text
                        assert _compose_container_services(
                            override.parent.name
                        ) == set()
                    with SqliteStore.open(fixture.database) as store:
                        assert all(
                            store.get_task_runtime(claim.task_id).status
                            is TaskRuntimeStatus.BLOCKED
                            for claim in claims
                        )
                    return
                raise errors[0]

        first, second = stacks
        assert first is not None and second is not None
        assert first.project_name != second.project_name
        assert first.network_name != second.network_name
        assert set(first.image_names).isdisjoint(second.image_names)
        assert len(first.image_names) == len(second.image_names) == 1
        assert first.environment["SERVICE_URL"] != second.environment["SERVICE_URL"]
        assert _http_body(first.environment["SERVICE_URL"]) == "healthy\n"
        assert _http_body(second.environment["SERVICE_URL"]) == "healthy\n"
        assert first.environment["SEARCH_URL"] == (
            "https://search.example.test/api"
        )
        assert second.environment["SEARCH_URL"] == (
            "https://search.example.test/api"
        )
        for name in (*first.network_names, *second.network_names):
            assert _docker_resource_names("network", name) == {name}
        assert set(first.network_names).isdisjoint(second.network_names)
        first_volumes = _project_resource_names("volume", first.project_name)
        second_volumes = _project_resource_names(
            "volume", second.project_name
        )
        assert first_volumes and second_volumes
        for stack in (first, second):
            assert _compose_container_services(stack.project_name) == {"healthy"}
            assert _compose_container_images(stack.project_name) == set(
                stack.image_names
            )
            assert all(
                _docker_resource_exists("image", image)
                for image in stack.image_names
            )
            assert _compose_published_host_ips(stack.project_name) == {
                "127.0.0.1"
            }

        with SqliteStore.open(fixture.database) as store:
            resources = [
                store.list_compose_resources(claim.task_id) for claim in claims
            ]
            assert [
                {resource.resource_type for resource in owned}
                for owned in resources
            ] == [
                {"project", "network", "image"},
                {"project", "network", "image"},
            ]
            assert [
                {
                    resource.resource_name
                    for resource in owned
                    if resource.resource_type == "network"
                }
                for owned in resources
            ] == [set(first.network_names), set(second.network_names)]
            fixture.compose_manager(None).stop_claimed_stack(
                store, first, claims[0], fixture.owner_token
            )
            first_events = {
                event.kind
                for event in store.list_execution_events(fixture.run_id)
                if event.task_id == claims[0].task_id
            }

        assert _compose_container_services(first.project_name) == set()
        assert not any(
            _docker_resource_exists("image", image) for image in first.image_names
        )
        assert all(
            _docker_resource_exists("image", image) for image in second.image_names
        )
        assert _http_body(second.environment["SERVICE_URL"]) == "healthy\n"
        assert _compose_container_services(second.project_name) == {"healthy"}
        # A successful teardown releases the stack's network and volume as well
        # as its containers and images, and leaves the other claim's untouched.
        assert not any(
            _docker_resource_exists("network", name) for name in first.network_names
        )
        assert all(
            _docker_resource_exists("network", name) for name in second.network_names
        )
        assert _project_resource_names("volume", first.project_name) == set()
        assert (
            _project_resource_names("volume", second.project_name) == second_volumes
        )
        assert {
            "compose.starting",
            "compose.ready",
            "compose.stopping",
            "compose.stopped",
        } <= first_events
        starting = next(
            event
            for event in _execution_events(fixture)
            if event.task_id == claims[1].task_id
            and event.kind == "compose.starting"
        )
        assert "--no-deps" in starting.payload["command"]
        wait_timeout = starting.payload["command"].index("--wait-timeout")
        assert int(starting.payload["command"][wait_timeout + 1]) > 0
    finally:
        for stack, claim in zip(stacks, claims, strict=True):
            if stack is None:
                continue
            # A stack torn down above still reaches the release, and a failure
            # tearing one down must not cost the others theirs.
            try:
                with contextlib.suppress(Exception):
                    if _compose_container_services(stack.project_name):
                        with SqliteStore.open(fixture.database) as store:
                            fixture.compose_manager(None).stop_claimed_stack(
                                store, stack, claim, fixture.owner_token
                            )
            finally:
                with contextlib.suppress(Exception):
                    _release_project_resources(stack.project_name)


def test_failed_compose_startup_releases_every_created_resource(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture(task_count=2)
    _prepare_compose_fixture(fixture)
    plan = _compose_plan(fixture.repository)

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        if "ps" in tuple(argv):
            return subprocess.CompletedProcess(argv, 1, "", "forced health failure")
        return _run_real_compose(argv, **kwargs)

    projects: list[str] = []
    try:
        with SqliteStore.open(fixture.database) as store:
            claims = (fixture.claim(store), fixture.claim(store))
            for claim in claims:
                projects.append(compose_project_name(claim))
                with pytest.raises(ComposeStackError, match="must report healthy"):
                    fixture.compose_manager(runner).start_claimed_stack(
                        store, plan, claim, fixture.owner_token
                    )

        for project_name in projects:
            _assert_project_released(project_name)
    finally:
        for project_name in projects:
            with contextlib.suppress(Exception):
                _release_project_resources(project_name)


def test_startup_failure_releases_resources_when_blocking_the_task_fails(
    execution_preflight_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = execution_preflight_fixture()
    _prepare_compose_fixture(fixture)
    plan = _compose_plan(fixture.repository)

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        if "ps" in tuple(argv):
            return subprocess.CompletedProcess(argv, 1, "", "forced health failure")
        return _run_real_compose(argv, **kwargs)

    project_name = ""
    try:
        with SqliteStore.open(fixture.database) as store:
            claim = fixture.claim(store)
            project_name = compose_project_name(claim)

            def refuse(*_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
                raise RuntimeError("task phase changed before transition")

            monkeypatch.setattr(store, "transition_task_runtime", refuse)

            # Blocking the task runs inside the handler that also tears the
            # project down. A failure there must not cost the teardown, and
            # matching both halves keeps the test honest if blocking ever
            # stops being reached for a freshly claimed task.
            with pytest.raises(
                ComposeStackError,
                match=r"must report healthy(?s:.)*blocking the task also failed",
            ):
                fixture.compose_manager(runner).start_claimed_stack(
                    store, plan, claim, fixture.owner_token
                )

        _assert_project_released(project_name)
    finally:
        if project_name:
            _release_project_resources(project_name)


def test_interrupted_compose_startup_releases_every_created_resource(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    _prepare_compose_fixture(fixture)
    plan = _compose_plan(fixture.repository)
    cancel = CancellationToken()

    created: set[str] = set()

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        result = _run_real_compose(argv, **kwargs)
        if "up" in tuple(argv):
            created.update(_project_resource_names("network", project_name))
            cancel.cancel()
        return result

    project_name = ""
    try:
        with SqliteStore.open(fixture.database) as store:
            claim = fixture.claim(store)
            project_name = compose_project_name(claim)
            with pytest.raises(KeyboardInterrupt):
                fixture.compose_manager(runner).start_claimed_stack(
                    store, plan, claim, fixture.owner_token, cancel=cancel
                )

        # A cancelled "up" that never created anything would satisfy the
        # raises clause and leave nothing to release, passing vacuously.
        assert created
        _assert_project_released(project_name)
    finally:
        if project_name:
            _release_project_resources(project_name)


def test_unexpected_startup_failure_releases_every_created_resource(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    _prepare_compose_fixture(fixture)
    plan = _compose_plan(fixture.repository)

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        if "port" in tuple(argv):
            raise RuntimeError("startup failed outside Compose")
        return _run_real_compose(argv, **kwargs)

    project_name = ""
    try:
        with SqliteStore.open(fixture.database) as store:
            claim = fixture.claim(store)
            project_name = compose_project_name(claim)
            with pytest.raises(RuntimeError, match="outside Compose"):
                fixture.compose_manager(runner).start_claimed_stack(
                    store, plan, claim, fixture.owner_token
                )

        _assert_project_released(project_name)
    finally:
        if project_name:
            _release_project_resources(project_name)


def test_startup_failure_before_the_project_is_recorded_keeps_its_own_error(
    execution_preflight_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = execution_preflight_fixture()
    _prepare_compose_fixture(fixture)
    plan = _compose_plan(fixture.repository)

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)

        def refuse(*_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
            raise RuntimeError("database is locked")

        monkeypatch.setattr(store, "add_compose_resource", refuse)

        # Nothing was recorded, so there is no project to tear down. Tearing one
        # down anyway raises over the failure the caller actually needs to see.
        with pytest.raises(RuntimeError, match="database is locked"):
            fixture.compose_manager(FakeComposeRunner()).start_claimed_stack(
                store, plan, claim, fixture.owner_token
            )


def test_expired_compose_cleanup_timeout_blocks_reclaim_until_retry(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    runner = FakeComposeRunner()
    validated_docker = Path("/opt/validated/bin/docker")
    plan = replace(
        _compose_plan(fixture.repository),
        executables=(HostExecutable(name="docker", path=validated_docker),),
    )
    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        stack = fixture.compose_manager(runner).start_claimed_stack(
            store, plan, claim, fixture.owner_token
        )
        assert stack is not None
        artifact = stack.worktree / "completed-work.txt"
        artifact.write_text("preserved\n", encoding="utf-8")
        source_compose = stack.worktree / "compose.yml"
        source_compose.unlink()
        assert all(
            path.parent == stack.runtime_directory and path.is_file()
            for path in stack.compose_files
        )
        expired_run = store.get_execution_run(fixture.run_id)
        assert expired_run is not None
        expired_at = expired_run.lease_expires_at + timedelta(seconds=1)
        stale = store.reconcile_expired_execution_runs(now=expired_at)
        replacement = store.acquire_execution_run(
            expired_run.borg_id,
            expired_run.generation_id,
            lease_duration=timedelta(hours=1),
            now=expired_at,
        )
        assert replacement.owner_token is not None
        assert store.claim_dependency_ready_task(
            replacement.run_id,
            replacement.owner_token,
            lease_duration=timedelta(minutes=30),
            now=expired_at,
        ) is None

        runner.timeout_down.add(stack.project_name)
        failed = fixture.compose_manager(runner).cleanup_stale_projects(store, stale)
        runtime = store.get_task_runtime(claim.task_id)
        persisted_claim = store.list_task_claims(fixture.run_id)[0]

        assert len(failed) == 1 and failed[0].stopped is False
        assert failed[0].error is not None and "timed out after" in failed[0].error
        assert failed[0].project_name == stack.project_name
        assert failed[0].command == runner.down_commands[-1]
        assert runtime is not None
        assert runtime.status is TaskRuntimeStatus.BLOCKED
        assert runtime.state_reason is not None
        assert stack.project_name in runtime.state_reason
        assert "docker compose" in runtime.state_reason
        assert persisted_claim.released_at is None
        assert store.list_stale_compose_resources(fixture.run_id) == stale

        runner.timeout_down.clear()
        succeeded = fixture.compose_manager(runner).cleanup_stale_projects(
            store, store.list_stale_compose_resources(fixture.run_id)
        )
        reclaimed = store.claim_dependency_ready_task(
            replacement.run_id,
            replacement.owner_token,
            lease_duration=timedelta(minutes=30),
            now=expired_at + timedelta(seconds=1),
        )

    assert len(succeeded) == 1 and succeeded[0].stopped is True
    assert succeeded[0].command == failed[0].command
    assert succeeded[0].command[0] == str(validated_docker)
    assert all(command[0] == str(validated_docker) for command in runner.commands)
    assert runner.timeouts and all(timeout > 0 for timeout in runner.timeouts)
    assert str(source_compose) not in succeeded[0].command
    assert all(str(path) in succeeded[0].command for path in stack.compose_files)
    assert reclaimed is not None and reclaimed.task_id == claim.task_id
    assert artifact.read_text(encoding="utf-8") == "preserved\n"


def test_replayed_cleanup_failure_does_not_block_reclaimed_task(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    runner = FakeComposeRunner()
    plan = _compose_plan(fixture.repository)
    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        stack = fixture.compose_manager(runner).start_claimed_stack(
            store, plan, claim, fixture.owner_token
        )
        assert stack is not None
        expired_run = store.get_execution_run(fixture.run_id)
        assert expired_run is not None
        expired_at = expired_run.lease_expires_at + timedelta(seconds=1)
        stale = store.reconcile_expired_execution_runs(now=expired_at)
        replacement = store.acquire_execution_run(
            expired_run.borg_id,
            expired_run.generation_id,
            lease_duration=timedelta(hours=1),
            now=expired_at,
        )
        assert replacement.owner_token is not None

        cleaned = fixture.compose_manager(runner).cleanup_stale_projects(store, stale)
        reclaimed = store.claim_dependency_ready_task(
            replacement.run_id,
            replacement.owner_token,
            lease_duration=timedelta(minutes=30),
            now=expired_at + timedelta(seconds=1),
        )
        assert reclaimed is not None

        runner.fail_down.add(stack.project_name)
        replayed = fixture.compose_manager(runner).cleanup_stale_projects(store, stale)
        runtime = store.get_task_runtime(claim.task_id)
        failure_events = [
            event
            for event in store.list_execution_events(fixture.run_id)
            if event.kind == "compose.cleanup_failed"
        ]

    assert len(cleaned) == 1 and cleaned[0].stopped is True
    assert len(replayed) == 1 and replayed[0].stopped is True
    assert runtime is not None and runtime.status is TaskRuntimeStatus.CLAIMED
    assert failure_events == []


def test_expiry_during_compose_startup_serializes_cleanup_and_fences_ready(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    runner = FakeComposeRunner()
    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        run = store.get_execution_run(fixture.run_id)
        assert run is not None
    project = (
        f"betterborg-{claim.run_id.hex[:6]}-{claim.task_id.hex[:6]}-{claim.id.hex}"
    )
    runner.pause_up.add(project)

    def start_stack():
        with SqliteStore.open(fixture.database) as store:
            return fixture.compose_manager(runner).start_claimed_stack(
                store,
                _compose_plan(fixture.repository),
                claim,
                fixture.owner_token,
            )

    def cleanup(resources):
        with SqliteStore.open(fixture.database) as store:
            return fixture.compose_manager(runner).cleanup_stale_projects(
                store, resources
            )

    expired_at = run.lease_expires_at + timedelta(seconds=1)
    with ThreadPoolExecutor(max_workers=2) as executor:
        start_future = executor.submit(start_stack)
        assert runner.up_entered.wait(timeout=5)
        with SqliteStore.open(fixture.database) as store:
            stale = store.reconcile_expired_execution_runs(now=expired_at)
            replacement = store.acquire_execution_run(
                run.borg_id,
                run.generation_id,
                lease_duration=timedelta(hours=1),
                now=expired_at,
            )
            assert replacement.owner_token is not None

        cleanup_future = executor.submit(cleanup, stale)
        assert not runner.down_entered.wait(timeout=0.2)
        runner.release_up.set()
        with pytest.raises(
            ComposeStackError, match="ownership expired during startup"
        ):
            start_future.result(timeout=5)
        outcomes = cleanup_future.result(timeout=5)

    with SqliteStore.open(fixture.database) as store:
        reclaimed = store.claim_dependency_ready_task(
            replacement.run_id,
            replacement.owner_token,
            lease_duration=timedelta(minutes=30),
            now=expired_at + timedelta(seconds=1),
        )
        ready_events = [
            event
            for event in store.list_execution_events(fixture.run_id)
            if event.kind == "compose.ready"
        ]

    assert outcomes[0].stopped is True
    assert runner.active == set()
    assert len(runner.down_commands) >= 1
    assert ready_events == []
    assert reclaimed is not None and reclaimed.task_id == claim.task_id


def test_compose_subprocess_environment_excludes_host_credentials(
    execution_preflight_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = execution_preflight_fixture()
    runner = FakeComposeRunner()
    monkeypatch.setenv("BETTERBORG_UNRELATED_TOKEN", "host-secret")
    manager = HostComposeManager(fixture.repository, command_runner=runner)

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        stack = manager.start_claimed_stack(
            store, _compose_plan(fixture.repository), claim, fixture.owner_token
        )
        assert stack is not None
        manager.stop_claimed_stack(store, stack, claim, fixture.owner_token)

    assert runner.environments
    assert all("PATH" in environment for environment in runner.environments)
    assert all(
        "BETTERBORG_UNRELATED_TOKEN" not in environment
        for environment in runner.environments
    )


def test_compose_startup_cancellation_reaps_tree_and_confirms_cleanup(
    execution_preflight_fixture,
    real_process_harness,
) -> None:
    fixture = execution_preflight_fixture()
    cancel = CancellationToken()
    fake = FakeComposeRunner()
    activities = []
    invocations: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        command = tuple(argv)
        invocations.append((command, dict(kwargs)))
        if "up" in command:
            return run_captured(
                real_process_harness.resistant_argv("compose-startup"),
                **kwargs,
            )
        return fake(argv, **kwargs)

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        manager = fixture.compose_manager(runner)
        with ThreadPoolExecutor(max_workers=1) as executor:
            startup = executor.submit(
                manager.start_claimed_stack,
                store,
                _compose_plan(fixture.repository),
                claim,
                fixture.owner_token,
                cancel=cancel,
                activity=activities.append,
            )
            real_process_harness.wait_for_marker("compose-startup.child.pid")
            cancel.cancel()
            with pytest.raises(KeyboardInterrupt):
                startup.result(timeout=5)

        resources = store.list_compose_resources(claim.task_id)
        runtime = store.get_task_runtime(claim.task_id)
        events = [
            event.kind
            for event in store.list_execution_events(fixture.run_id)
            if event.kind.startswith("compose.")
        ]

    real_process_harness.assert_tree_absent("compose-startup")
    up = next(item for item in invocations if "up" in item[0])
    down = next(item for item in invocations if "down" in item[0])
    assert up[1]["cancel"] is cancel
    assert up[1]["terminate_on_cancel"] is True
    assert up[1]["deadline"] is None
    assert down[1]["cancel"] is cancel
    assert down[1]["terminate_on_cancel"] is False
    assert down[1]["deadline"] == cancel.force_deadline
    assert resources
    assert runtime is not None and runtime.status is TaskRuntimeStatus.CLAIMED
    assert events[:2] == ["compose.starting", "compose.stopping"]
    assert set(events[2:]) == {"compose.stopped", "compose.cleanup_completed"}
    assert [activity.kind for activity in activities] == [
        AgentActivityKind.COMMAND,
        AgentActivityKind.COMMAND,
    ]
    assert " up " in f" {activities[0].detail} "
    assert " down " in f" {activities[1].detail} "


def test_compose_cleanup_deadline_reaps_tree_and_preserves_fence(
    execution_preflight_fixture,
    real_process_harness,
) -> None:
    fixture = execution_preflight_fixture()
    cancel = CancellationToken()
    fake = FakeComposeRunner()
    down_kwargs: dict[str, object] = {}

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        command = tuple(argv)
        if "down" in command:
            down_kwargs.update(kwargs)
            return run_captured(
                real_process_harness.resistant_argv("compose-cleanup"),
                **kwargs,
            )
        return fake(argv, **kwargs)

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        manager = fixture.compose_manager(runner)
        stack = manager.start_claimed_stack(
            store,
            _compose_plan(fixture.repository),
            claim,
            fixture.owner_token,
            cancel=cancel,
        )
        assert stack is not None
        cancel.cancel()
        with pytest.raises(ComposeStackError, match="teardown failed"):
            manager.stop_claimed_stack(
                store,
                stack,
                claim,
                fixture.owner_token,
                cancel=cancel,
            )
        resources = store.list_compose_resources(claim.task_id)
        runtime = store.get_task_runtime(claim.task_id)
        events = {
            event.kind for event in store.list_execution_events(fixture.run_id)
        }

    real_process_harness.assert_tree_absent("compose-cleanup")
    assert down_kwargs["cancel"] is cancel
    assert down_kwargs["terminate_on_cancel"] is False
    assert down_kwargs["deadline"] == cancel.force_deadline
    assert resources
    assert "compose.cleanup_failed" in events
    assert "compose.cleanup_completed" not in events
    assert runtime is not None and runtime.status is TaskRuntimeStatus.BLOCKED
    assert runtime.state_reason is not None
    assert "Compose teardown failed" in runtime.state_reason


def test_compose_cancellation_after_readiness_confirms_bounded_cleanup(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    cancel = CancellationToken()
    runner = FakeComposeRunner()

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        manager = fixture.compose_manager(runner)
        stack = manager.start_claimed_stack(
            store,
            _compose_plan(fixture.repository),
            claim,
            fixture.owner_token,
            cancel=cancel,
        )
        assert stack is not None
        runner.pause_down.add(stack.project_name)
        cancel.cancel()
        with ThreadPoolExecutor(max_workers=1) as executor:
            cleanup = executor.submit(
                manager.stop_claimed_stack,
                store,
                stack,
                claim,
                fixture.owner_token,
                cancel=cancel,
            )
            assert runner.down_entered.wait(timeout=2)
            before_release = [
                event.kind
                for event in store.list_execution_events(fixture.run_id)
                if event.kind.startswith("compose.")
            ]
            runner.release_down.set()
            cleanup.result(timeout=2)

        after_release = [
            event.kind
            for event in store.list_execution_events(fixture.run_id)
            if event.kind.startswith("compose.")
        ]

    assert before_release[-1] == "compose.stopping"
    assert "compose.cleanup_completed" not in before_release
    assert set(after_release[-2:]) == {
        "compose.stopped",
        "compose.cleanup_completed",
    }
    assert runner.active == set()
    assert runner.down_projects == [stack.project_name]


def test_compose_cleanup_metadata_excludes_resolved_env_file_secrets(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    runner = FakeComposeRunner()
    credential = "credential-from-service-env-file"
    worktree = fixture.worktree_paths[0]
    (worktree / "service.env").write_text(
        f"SERVICE_TOKEN={credential}\n", encoding="utf-8"
    )
    compose_file = worktree / "compose.yml"
    compose_file.write_text(
        compose_file.read_text(encoding="utf-8").replace(
            "  healthy:\n", "  healthy:\n    env_file: service.env\n"
        ),
        encoding="utf-8",
    )
    runner.config_services["healthy"]["environment"] = {
        "SERVICE_TOKEN": credential
    }

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        stack = fixture.compose_manager(runner).start_claimed_stack(
            store, _compose_plan(fixture.repository), claim, fixture.owner_token
        )
        assert stack is not None
        persisted = "\n".join(
            path.read_text(encoding="utf-8") for path in stack.compose_files
        )
        fixture.compose_manager(runner).stop_claimed_stack(
            store, stack, claim, fixture.owner_token
        )

    assert credential not in persisted
    assert "SERVICE_TOKEN" not in persisted
    assert stack.compose_files[0].name == "compose.cleanup.json"


def test_build_images_are_claim_owned_and_exactly_removed(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    runner = FakeComposeRunner()

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        stack = fixture.compose_manager(runner).start_claimed_stack(
            store, _compose_plan(fixture.repository), claim, fixture.owner_token
        )
        assert stack is not None
        override = stack.compose_files[-1].read_text(encoding="utf-8")
        cleanup = json.loads(stack.compose_files[0].read_text(encoding="utf-8"))
        resources = store.list_compose_resources(claim.task_id)
        fixture.compose_manager(runner).stop_claimed_stack(
            store, stack, claim, fixture.owner_token
        )

    assert len(stack.image_names) == 1
    image_name = stack.image_names[0]
    assert image_name.startswith(f"betterborg/{stack.project_name}-")
    assert f'image: "{image_name}"' in override
    assert "pull_policy: build" in override
    assert cleanup["services"]["healthy"]["image"] == image_name
    assert {
        resource.resource_name
        for resource in resources
        if resource.resource_type == "image"
    } == {image_name}
    assert runner.down_commands[-1][-2:] == ("--rmi", "all")


def test_distinct_dependencies_share_one_compose_service(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    runner = FakeComposeRunner()
    plan = replace(
        _compose_plan(fixture.repository),
        services=(
            HostService(
                name="application",
                kind="compose",
                evidence="fixture",
                compose_service="healthy",
                url_env="SERVICE_URL",
                port=8080,
                url_targets=(("SERVICE_URL", 8080, "tcp"),),
            ),
            HostService(
                name="metrics",
                kind="compose",
                evidence="fixture",
                compose_service="healthy",
                url_env="METRICS_HTTP_URL",
                port=8081,
                url_targets=(("METRICS_HTTP_URL", 8081, "tcp"),),
                port_targets=(("METRICS_URL", 8081, "tcp"),),
            ),
        ),
    )

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        stack = fixture.compose_manager(runner).start_claimed_stack(
            store, plan, claim, fixture.owner_token
        )
        assert stack is not None
        fixture.compose_manager(runner).stop_claimed_stack(
            store, stack, claim, fixture.owner_token
        )

    assert runner.started_services[stack.project_name] == ("healthy",)
    assert set(stack.environment) == {
        "SERVICE_URL",
        "METRICS_HTTP_URL",
        "METRICS_URL",
    }
    assert all(
        value.startswith("http://127.0.0.1:")
        for name, value in stack.environment.items()
        if name != "METRICS_URL"
    )
    assert stack.environment["METRICS_URL"].isdigit()
    assert {
        command[-1] for command in runner.port_commands
    } == {"8080", "8081"}


def test_udp_compose_endpoint_preserves_protocol_and_loopback_binding(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    runner = FakeComposeRunner()
    plan = replace(
        _compose_plan(fixture.repository),
        services=(
            HostService(
                name="dns-service",
                kind="compose",
                evidence="fixture",
                compose_service="healthy",
                url_env="DNS_URL",
                port=5353,
                url_targets=(("DNS_URL", 5353, "udp"),),
            ),
        ),
    )

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        stack = fixture.compose_manager(runner).start_claimed_stack(
            store, plan, claim, fixture.owner_token
        )
        assert stack is not None
        override = stack.compose_files[-1].read_text(encoding="utf-8")
        fixture.compose_manager(runner).stop_claimed_stack(
            store, stack, claim, fixture.owner_token
        )

    assert stack.environment["DNS_URL"].startswith("udp://127.0.0.1:")
    assert 'host_ip: "127.0.0.1"' in override
    assert 'protocol: "udp"' in override
    assert "--protocol" in runner.port_commands[-1]
    protocol_index = runner.port_commands[-1].index("--protocol")
    assert runner.port_commands[-1][protocol_index + 1] == "udp"


def test_startup_consumes_preflight_topology_without_revalidation(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    runner = FakeComposeRunner()
    runner.config_services["healthy"]["volumes"] = [
        {
            "type": "bind",
            "source": "/var/lib/example",
            "target": "/data",
        }
    ]
    plan = _compose_plan(fixture.repository)

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        stack = fixture.compose_manager(runner).start_claimed_stack(
            store, plan, claim, fixture.owner_token
        )
        assert stack is not None
        fixture.compose_manager(runner).stop_claimed_stack(
            store, stack, claim, fixture.owner_token
        )

    assert not any("config" in command for command in runner.commands)
    assert runner.active == set()


def test_unhealthy_compose_startup_blocks_and_tears_down_exact_project(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    runner = FakeComposeRunner()
    plan = _compose_plan(fixture.repository)
    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        expected_project = (
            f"betterborg-{claim.run_id.hex[:6]}-{claim.task_id.hex[:6]}-{claim.id.hex}"
        )
        runner.fail_up.add(expected_project)

        with pytest.raises(ComposeStackError, match="did not become healthy"):
            fixture.compose_manager(runner).start_claimed_stack(
                store, plan, claim, fixture.owner_token
            )

        runtime = store.get_task_runtime(claim.task_id)
        events = {
            event.kind for event in store.list_execution_events(fixture.run_id)
        }

    assert runtime is not None and runtime.status is TaskRuntimeStatus.BLOCKED
    assert runner.active == set()
    assert runner.down_commands[-1][3] == expected_project
    assert {"compose.starting", "compose.stopping", "compose.stopped"} <= events
    assert "compose.ready" not in events


def test_compose_service_without_health_status_never_becomes_ready(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    runner = FakeComposeRunner()
    runner.service_health["healthy"] = ""

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        with pytest.raises(
            ComposeStackError,
            match="Every selected Compose service must report healthy",
        ):
            fixture.compose_manager(runner).start_claimed_stack(
                store,
                _compose_plan(fixture.repository),
                claim,
                fixture.owner_token,
            )
        runtime = store.get_task_runtime(claim.task_id)
        events = {
            event.kind for event in store.list_execution_events(fixture.run_id)
        }

    assert runtime is not None and runtime.status is TaskRuntimeStatus.BLOCKED
    assert runner.active == set()
    assert "compose.ready" not in events


def test_external_service_urls_do_not_create_compose_inputs() -> None:
    services = (
        HostService(
            name="search",
            kind="external",
            evidence="fixture",
            url_env="SEARCH_URL",
            url="https://search.example.test/api",
        ),
    )

    assert service_url_environment(services) == {
        "SEARCH_URL": "https://search.example.test/api"
    }


def test_compose_url_scheme_uses_service_identity_with_loopback_host() -> None:
    services = (
        HostService(
            name="cache",
            kind="compose",
            evidence="fixture",
            compose_service="redis",
            url_env="CACHE_URL",
            port=6379,
            url_targets=(("CACHE_URL", 6379, "tcp"),),
        ),
    )

    assert service_url_environment(
        services,
        published_ports={("redis", 6379, "tcp"): 49153},
    ) == {"CACHE_URL": "redis://127.0.0.1:49153/0"}


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
        compose_files=(),
        services=(),
        package_managers=("pip",),
        secret_requirements=secrets,
    )


def _compose_plan(repository: Path) -> HostPreflightPlan:
    docker = Path(shutil.which("docker") or "/validated/docker")
    return HostPreflightPlan(
        repository_root=repository,
        commands=(),
        prepare_commands=(),
        materialize_commands=(),
        environment_files=(repository / "package.lock",),
        executables=(HostExecutable(name="docker", path=docker),),
        required_secret_names=(),
        compose_files=(repository / "compose.yml",),
        services=(
            HostService(
                name="http-service",
                kind="compose",
                evidence="fixture",
                compose_service="healthy",
                url_env="SERVICE_URL",
                port=8080,
                url_targets=(("SERVICE_URL", 8080, "tcp"),),
            ),
            HostService(
                name="search",
                kind="external",
                evidence="fixture",
                url_env="SEARCH_URL",
                url="https://search.example.test/api",
            ),
        ),
        compose_profiles=(),
        compose_networks=("default", "fixture"),
        compose_volumes=("fixture-data",),
        compose_build_services=("healthy",),
    )


class FakeComposeRunner:
    def __init__(self) -> None:
        self.active: set[str] = set()
        self.started_services: dict[str, tuple[str, ...]] = {}
        self.down_commands: list[tuple[str, ...]] = []
        self.port_commands: list[tuple[str, ...]] = []
        self.environments: list[dict[str, str]] = []
        self.timeouts: list[float] = []
        self.commands: list[tuple[str, ...]] = []
        self.fail_up: set[str] = set()
        self.fail_down: set[str] = set()
        self.fail_all_down = False
        self.timeout_down: set[str] = set()
        self.pause_up: set[str] = set()
        self.pause_down: set[str] = set()
        self.up_entered = threading.Event()
        self.release_up = threading.Event()
        self.down_entered = threading.Event()
        self.release_down = threading.Event()
        self.service_health: dict[str, str] = {"healthy": "healthy"}
        self.config_services: dict[str, dict[str, object]] = {
            "healthy": {"networks": {"default": None}},
            "unused": {"networks": {"default": None}},
        }
        self._lock = threading.Lock()

    @property
    def up_projects(self) -> list[str]:
        """Return Compose projects in startup order."""
        return [
            command[command.index("--project-name") + 1]
            for command in self.commands
            if "up" in command
        ]

    @property
    def down_projects(self) -> list[str]:
        """Return Compose projects in teardown order."""
        return [
            command[command.index("--project-name") + 1]
            for command in self.down_commands
        ]

    def __call__(self, argv, **kwargs):
        command = tuple(argv)
        self.commands.append(command)
        project = command[command.index("--project-name") + 1]
        if "up" in command and project in self.pause_up:
            self.up_entered.set()
            if not self.release_up.wait(timeout=10):
                return subprocess.CompletedProcess(
                    argv, 12, "", "timed out waiting for test startup release"
                )
        if "down" in command:
            self.down_entered.set()
            if project in self.pause_down and not self.release_down.wait(timeout=10):
                return subprocess.CompletedProcess(
                    argv, 13, "", "timed out waiting for test teardown release"
                )
        with self._lock:
            self.environments.append(dict(kwargs["env"]))
            self.timeouts.append(kwargs["timeout"])
            if "config" in command:
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    json.dumps(
                        {
                            "services": self.config_services,
                            "networks": {"default": {}},
                        }
                    ),
                    "",
                )
            if "up" in command:
                if project in self.fail_up:
                    return subprocess.CompletedProcess(
                        argv, 1, "", "service healthy is unhealthy"
                    )
                services = command[command.index("--no-deps") + 1 :]
                self.active.add(project)
                self.started_services[project] = services
                return subprocess.CompletedProcess(argv, 0, "healthy\n", "")
            if "ps" in command:
                records = [
                    {
                        "Service": service,
                        "State": "running",
                        "Health": self.service_health.get(service, ""),
                    }
                    for service in self.started_services.get(project, ())
                    if project in self.active
                ]
                return subprocess.CompletedProcess(
                    argv, 0, json.dumps(records), ""
                )
            if "port" in command:
                self.port_commands.append(command)
                port = 41000 + sum(project.encode()) % 20000
                return subprocess.CompletedProcess(
                    argv, 0, f"127.0.0.1:{port}\n", ""
                )
            if "down" in command:
                self.down_commands.append(command)
                if project in self.timeout_down:
                    raise subprocess.TimeoutExpired(command, kwargs["timeout"])
                file_paths = [
                    Path(command[index + 1])
                    for index, value in enumerate(command[:-1])
                    if value == "--file"
                ]
                if any(not path.is_file() for path in file_paths):
                    return subprocess.CompletedProcess(
                        argv, 11, "", "Compose file disappeared"
                    )
                if self.fail_all_down or project in self.fail_down:
                    return subprocess.CompletedProcess(
                        argv, 9, "", "simulated teardown failure"
                    )
                self.active.discard(project)
                return subprocess.CompletedProcess(argv, 0, "stopped\n", "")
        return subprocess.CompletedProcess(argv, 2, "", "unexpected command")


_COMPOSE_FIXTURE_SOURCE = r"""
#include <arpa/inet.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

int main(int argc, char **argv) {
    struct sockaddr_in address = {0};
    address.sin_family = AF_INET;
    address.sin_port = htons(8080);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (argc == 2 && strcmp(argv[1], "--health") == 0) {
        int probe = socket(AF_INET, SOCK_STREAM, 0);
        int result = connect(probe, (struct sockaddr *)&address, sizeof(address));
        close(probe);
        return result == 0 ? 0 : 1;
    }
    int server = socket(AF_INET, SOCK_STREAM, 0);
    int reuse = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(server, (struct sockaddr *)&address, sizeof(address)) != 0) return 2;
    if (listen(server, 16) != 0) return 3;
    for (;;) {
        int client = accept(server, 0, 0);
        if (client < 0) continue;
        char request[1024];
        read(client, request, sizeof(request));
        const char response[] =
            "HTTP/1.1 200 OK\r\nContent-Length: 8\r\n"
            "Connection: close\r\n\r\nhealthy\n";
        write(client, response, sizeof(response) - 1);
        close(client);
    }
}
"""


def _prepare_compose_fixture(fixture: ExecutionPreflightFixture) -> None:
    for worktree in fixture.worktree_paths:
        output = worktree / ".dependencies/compose-fixture"
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["gcc", "-static", "-Os", "-s", "-x", "c", "-o", str(output), "-"],
            input=_COMPOSE_FIXTURE_SOURCE,
            text=True,
            check=True,
            capture_output=True,
        )


def _http_body(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


def _run_real_compose(*args, **kwargs) -> subprocess.CompletedProcess[str]:
    """Run a Compose command for real, dropping reaping-only arguments."""
    for name in ("cancel", "terminate_on_cancel", "deadline"):
        kwargs.pop(name, None)
    kwargs["capture_output"] = True
    kwargs["text"] = True
    return subprocess.run(*args, **kwargs)


def _assert_project_released(project_name: str) -> None:
    """Assert one Compose project owns no network, volume or container."""
    for kind in ("network", "volume"):
        assert _project_resource_names(kind, project_name) == set()
    assert _compose_container_services(project_name) == set()
    assert _project_image_names(project_name) == set()


def _docker_resource_names(kind: str, name: str) -> set[str]:
    result = subprocess.run(
        ["docker", kind, "inspect", name, "--format", "{{.Name}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.splitlines())


def _compose_container_services(project_name: str) -> set[str]:
    result = subprocess.run(
        [
            "docker",
            "ps",
            "--all",
            "--filter",
            f"label=com.docker.compose.project={project_name}",
            "--format",
            '{{.Label "com.docker.compose.service"}}',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.splitlines()) - {""}


def _compose_container_images(project_name: str) -> set[str]:
    result = subprocess.run(
        [
            "docker",
            "ps",
            "--all",
            "--filter",
            f"label=com.docker.compose.project={project_name}",
            "--format",
            "{{.Image}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.splitlines()) - {""}


def _docker_resource_exists(kind: str, name: str) -> bool:
    return (
        subprocess.run(
            ["docker", kind, "inspect", name],
            check=False,
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )


def _project_resource_names(kind: str, project_name: str) -> set[str]:
    """Return the networks or volumes one Compose project still owns."""
    result = subprocess.run(
        [
            "docker",
            kind,
            "ls",
            "--filter",
            f"name={project_name}",
            "--format",
            "{{.Name}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    # The filter matches substrings, so a sibling "<project>-sanity" project
    # would answer to the base project's name. Keep only this project's own.
    return {
        name
        for name in result.stdout.splitlines()
        if name.startswith(f"{project_name}_")
    }


def _project_image_names(project_name: str) -> set[str]:
    """Return the claim-owned image tags one Compose project still holds."""
    result = subprocess.run(
        [
            "docker",
            "images",
            "--filter",
            f"reference=betterborg/{project_name}-*",
            "--format",
            "{{.Repository}}:{{.Tag}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.splitlines()) - {""}


def _release_project_resources(project_name: str) -> None:
    """Remove whatever a project left behind, so that one failing leak test
    cannot starve later runs of Docker's address pool."""
    # Containers first: Docker refuses to remove a network with a live
    # endpoint, so leaving them would silently defeat the removals below.
    containers = subprocess.run(
        [
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={project_name}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if containers:
        subprocess.run(
            ["docker", "rm", "--force", *containers],
            check=False,
            capture_output=True,
            text=True,
        )
    for kind in ("network", "volume"):
        names = _project_resource_names(kind, project_name)
        if names:
            subprocess.run(
                ["docker", kind, "rm", *sorted(names)],
                check=False,
                capture_output=True,
                text=True,
            )
    images = _project_image_names(project_name)
    if images:
        subprocess.run(
            ["docker", "image", "rm", *sorted(images)],
            check=False,
            capture_output=True,
            text=True,
        )


def _compose_published_host_ips(project_name: str) -> set[str]:
    containers = subprocess.run(
        [
            "docker",
            "ps",
            "--all",
            "--filter",
            f"label=com.docker.compose.project={project_name}",
            "--format",
            "{{.ID}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    inspected = subprocess.run(
        ["docker", "container", "inspect", *containers],
        check=True,
        capture_output=True,
        text=True,
    )
    records = json.loads(inspected.stdout)
    return {
        binding["HostIp"]
        for record in records
        for bindings in record["NetworkSettings"]["Ports"].values()
        if bindings is not None
        for binding in bindings
    }


def _execution_events(fixture: ExecutionPreflightFixture):
    with SqliteStore.open(fixture.database) as store:
        return store.list_execution_events(fixture.run_id)


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
