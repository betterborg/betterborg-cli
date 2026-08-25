"""Shared cache/materialization scaffold for host execution preflight."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from betterborg_cli.host_execution import (
    EnvironmentMaterializationError,
    HostCommand,
    HostEnvironmentManager,
    HostPreflightPlan,
    HostSecret,
    HostWorktreeManager,
    package_manager_cache_environment,
)
from betterborg_cli.planning import render_task_markdown, task_markdown_digest
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
    cache_root: Path
    preparation_root: Path
    worktree_paths: tuple[Path, ...]
    task_ids: tuple[UUID, ...]
    run_id: UUID
    owner_token: str

    def claim(self, store: SqliteStore) -> TaskClaim:
        claim = store.claim_dependency_ready_task(
            self.run_id,
            self.owner_token,
            lease_duration=timedelta(minutes=30),
        )
        assert claim is not None
        return claim

    def manager(self) -> HostEnvironmentManager:
        return HostEnvironmentManager(
            self.repository,
            cache_root=self.cache_root,
            preparation_root=self.preparation_root,
            environment={"PATH": os.environ["PATH"]},
        )


@pytest.fixture
def execution_preflight_fixture(tmp_path: Path):
    """Create claimed-worktree inputs with a fake package manager."""

    def create(*, task_count: int = 1) -> ExecutionPreflightFixture:
        repository = tmp_path / f"repository-{uuid4().hex}"
        repository.mkdir()
        _git(repository, "init", "--quiet", "--initial-branch=main")
        _git(repository, "config", "user.name", "BetterBorg Tests")
        _git(repository, "config", "user.email", "tests@betterborg.dev")
        (repository / "README.md").write_text("# Fixture\n", encoding="utf-8")
        (repository / "package.lock").write_text("lock-v1\n", encoding="utf-8")
        (repository / ".gitignore").write_text(
            ".dependencies/\n", encoding="utf-8"
        )
        _write_fake_package_manager(repository / "fake-package-manager")
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
            / ".borg/tasks"
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
                generation.id, durable_root=durable_root
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
            cache_root=repository / ".borg/state/environment-cache",
            preparation_root=tmp_path / f"preparation-{uuid4().hex}",
            worktree_paths=tuple(spec.path for spec in specs),
            task_ids=tuple(task.id for task in tasks),
            run_id=acquisition.run_id,
            owner_token=acquisition.owner_token,
        )

    return create


def test_prepares_once_per_fingerprint_and_materializes_every_worktree(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture(task_count=2)
    plan = _plan(fixture.repository)

    with SqliteStore.open(fixture.database) as store:
        first_claim = fixture.claim(store)
        first = fixture.manager().materialize_claimed_task(
            store, plan, first_claim, fixture.owner_token
        )
        assert first.preparation_reused is False
        assert first.materialization_reused is False

    # Reopening both the durable store and manager simulates a process restart.
    with SqliteStore.open(fixture.database) as reopened:
        second_claim = fixture.claim(reopened)
        second = fixture.manager().materialize_claimed_task(
            reopened, plan, second_claim, fixture.owner_token
        )
        assert second.fingerprint == first.fingerprint
        assert second.preparation_reused is True
        assert second.materialization_reused is False

    for worktree in fixture.worktree_paths:
        assert (worktree / ".dependencies/materialized").is_file()
    assert _preparation_count(fixture.cache_root) == 1
    assert _git(fixture.repository, "status", "--porcelain") == ""
    assert not any(fixture.preparation_root.iterdir())


def test_package_manager_caches_are_fingerprint_local(tmp_path: Path) -> None:
    cache = tmp_path / "fingerprint"

    environment = package_manager_cache_environment(
        cache,
        ("pnpm", "yarn", "uv", "poetry", "cargo", "go", "bundler"),
    )

    assert environment["XDG_CACHE_HOME"].startswith(str(cache))
    assert environment["PNPM_STORE_DIR"] == str(cache / "pnpm/store")
    assert environment["pnpm_config_store_dir"] == str(cache / "pnpm/store")
    assert environment["YARN_GLOBAL_FOLDER"] == str(cache / "yarn/berry")
    assert environment["UV_CACHE_DIR"] == str(cache / "uv/cache")
    assert environment["POETRY_CACHE_DIR"] == str(cache / "poetry/cache")
    assert environment["CARGO_HOME"] == str(cache / "cargo")
    assert environment["GOMODCACHE"] == str(cache / "go/pkg/mod")
    assert environment["BUNDLE_PATH"] == str(cache / "bundle")


def test_repository_local_cache_must_be_ignored(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    (fixture.repository / ".gitignore").write_text(
        ".dependencies/\n", encoding="utf-8"
    )

    with pytest.raises(EnvironmentMaterializationError, match="not ignored"):
        fixture.manager()


def test_falls_back_to_preparation_in_each_task_worktree(
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
    # One run happened in the disposable preparer and one in the task fallback.
    assert _preparation_count(fixture.cache_root) == 2


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

    assert resumed.preparation_reused is True
    assert resumed.materialization_reused is True
    assert _materialization_count(fixture.cache_root) == 1


def test_descriptor_change_rematerializes_before_coding(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture(task_count=2)
    plan = _plan(fixture.repository)

    with SqliteStore.open(fixture.database) as store:
        first_claim = fixture.claim(store)
        first = fixture.manager().materialize_claimed_task(
            store, plan, first_claim, fixture.owner_token
        )

        changed_worktree = fixture.worktree_paths[1]
        (changed_worktree / "package.lock").write_text(
            "lock-v2\n", encoding="utf-8"
        )
        _git(changed_worktree, "add", "package.lock")
        _git(changed_worktree, "commit", "--quiet", "-m", "update lock")

        second_claim = fixture.claim(store)
        second = fixture.manager().materialize_claimed_task(
            store, plan, second_claim, fixture.owner_token
        )

    assert second.fingerprint != first.fingerprint
    assert second.preparation_reused is False
    assert _preparation_count(fixture.cache_root) == 2


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
) -> HostPreflightPlan:
    def commands(action: str | None) -> tuple[HostCommand, ...]:
        if action is None:
            return ()
        return (
            HostCommand(
                stage="environment",
                argv=("./fake-package-manager", action),
                cwd=".",
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


def _write_fake_package_manager(path: Path) -> None:
    path.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "action=$1\n"
        "mkdir -p \"$XDG_CACHE_HOME\"\n"
        "case \"$action\" in\n"
        "  prepare)\n"
        "    printf 'prepare\\n' >> \"$XDG_CACHE_HOME/preparations.log\"\n"
        "    mkdir -p .dependencies\n"
        "    printf 'local\\n' > .dependencies/prepared\n"
        "    ;;\n"
        "  materialize)\n"
        "    printf 'materialize\\n' >> \"$XDG_CACHE_HOME/materializations.log\"\n"
        "    mkdir -p .dependencies\n"
        "    printf 'local\\n' > .dependencies/materialized\n"
        "    ;;\n"
        "  tracked)\n"
        "    printf 'changed\\n' > README.md\n"
        "    ;;\n"
        "  fail)\n"
        "    printf '%s\\n' \"${PACKAGE_TOKEN:-package failed}\" >&2\n"
        "    exit 7\n"
        "    ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _preparation_count(cache_root: Path) -> int:
    return sum(
        path.read_text(encoding="utf-8").count("prepare\n")
        for path in cache_root.rglob("preparations.log")
    )


def _materialization_count(cache_root: Path) -> int:
    return sum(
        path.read_text(encoding="utf-8").count("materialize\n")
        for path in cache_root.rglob("materializations.log")
    )


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
