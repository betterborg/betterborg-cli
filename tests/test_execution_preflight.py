"""Shared cache/materialization scaffold for host execution preflight."""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from betterborg_cli.host_execution import (
    ComposeStackError,
    EnvironmentMaterializationError,
    HostCommand,
    HostComposeManager,
    HostEnvironmentManager,
    HostPreflightPlan,
    HostSecret,
    HostService,
    HostWorktreeManager,
    package_manager_cache_environment,
    service_url_environment,
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
        _git(repository, "config", "user.name", "BetterBorg Tests")
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
    assert environment["BUNDLE_USER_CACHE"] == str(cache / "bundler")
    assert "BUNDLE_PATH" not in environment
    assert "GEM_HOME" not in environment


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


def test_same_task_descriptor_change_rematerializes_before_sanity(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    plan = _plan(fixture.repository)

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        first = fixture.manager().materialize_claimed_task(
            store, plan, claim, fixture.owner_token
        )

        changed_worktree = fixture.worktree_paths[0]
        (changed_worktree / "README.md").write_text(
            "coding work\n", encoding="utf-8"
        )
        (changed_worktree / "package.lock").write_text(
            "lock-v2\n", encoding="utf-8"
        )

        second = fixture.manager().materialize_claimed_task(
            store, plan, claim, fixture.owner_token
        )
        runtime = store.get_task_runtime(claim.task_id)

    assert second.fingerprint != first.fingerprint
    assert second.preparation_reused is False
    assert _preparation_count(fixture.cache_root) == 2
    assert (second.cache_path / "xdg/cache/prepared-lock").read_text() == (
        "lock-v2\n"
    )
    assert (changed_worktree / "README.md").read_text() == "coding work\n"
    assert (changed_worktree / "package.lock").read_text() == "lock-v2\n"
    assert runtime is not None and runtime.status is TaskRuntimeStatus.CODING


def test_reverted_fingerprint_rematerializes_checkout_local_dependencies(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    plan = _plan(fixture.repository)
    worktree = fixture.worktree_paths[0]

    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        first = fixture.manager().materialize_claimed_task(
            store, plan, claim, fixture.owner_token
        )
        assert (worktree / ".dependencies/materialized").read_text() == (
            "lock-v1\n"
        )

        (worktree / "package.lock").write_text("lock-v2\n", encoding="utf-8")
        second = fixture.manager().materialize_claimed_task(
            store, plan, claim, fixture.owner_token
        )
        assert second.fingerprint != first.fingerprint
        assert (worktree / ".dependencies/materialized").read_text() == (
            "lock-v2\n"
        )

        (worktree / "package.lock").write_text("lock-v1\n", encoding="utf-8")
        reverted = fixture.manager().materialize_claimed_task(
            store, plan, claim, fixture.owner_token
        )

    assert reverted.fingerprint == first.fingerprint
    assert reverted.materialization_reused is False
    assert (worktree / ".dependencies/materialized").read_text() == "lock-v1\n"
    assert _materialization_count(fixture.cache_root) == 3


def test_preparation_is_coordinated_across_processes(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture(task_count=2)
    plan = _plan(fixture.repository, prepare_action="prepare-slow")
    with SqliteStore.open(fixture.database) as store:
        claims = (fixture.claim(store), fixture.claim(store))

    context = multiprocessing.get_context("spawn")
    start = fixture.preparation_root.with_name(
        f"{fixture.preparation_root.name}-start"
    )
    result_paths = tuple(
        fixture.preparation_root.with_name(
            f"{fixture.preparation_root.name}-result-{index}"
        )
        for index in range(len(claims))
    )
    processes = [
        context.Process(
            target=_materialize_in_process,
            args=(
                fixture,
                plan,
                claim,
                start,
                result_path,
            ),
        )
        for claim, result_path in zip(claims, result_paths, strict=True)
    ]
    for process in processes:
        process.start()
    start.touch()
    for process in processes:
        process.join(timeout=20)

    assert [process.exitcode for process in processes] == [0, 0]
    outcomes = [
        json.loads(result_path.read_text(encoding="utf-8"))
        for result_path in result_paths
    ]
    assert all(outcome[0] == "ok" for outcome in outcomes), outcomes
    assert sorted(outcome[1] for outcome in outcomes) == [False, True]
    assert _preparation_count(fixture.cache_root) == 1


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
        cache_root=fixture.cache_root,
        preparation_root=fixture.preparation_root,
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
        result = subprocess.run(*args, **kwargs)
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
                        (fixture.repository / ".borg/state/compose").glob(
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
        for stack in (first, second):
            assert _compose_container_services(stack.project_name) == {"healthy"}
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
            ] == [{"project", "network"}, {"project", "network"}]
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
        assert _http_body(second.environment["SERVICE_URL"]) == "healthy\n"
        assert _compose_container_services(second.project_name) == {"healthy"}
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
    finally:
        for stack, claim in zip(stacks, claims, strict=False):
            if stack is None or not _compose_container_services(stack.project_name):
                continue
            with SqliteStore.open(fixture.database) as store:
                fixture.compose_manager(None).stop_claimed_stack(
                    store, stack, claim, fixture.owner_token
                )


def test_expired_compose_cleanup_failure_blocks_reclaim_until_retry(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    runner = _FakeComposeRunner()
    plan = _compose_plan(fixture.repository)
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

        runner.fail_down.add(stack.project_name)
        failed = fixture.compose_manager(runner).cleanup_stale_projects(store, stale)
        runtime = store.get_task_runtime(claim.task_id)
        persisted_claim = store.list_task_claims(fixture.run_id)[0]

        assert len(failed) == 1 and failed[0].stopped is False
        assert failed[0].project_name == stack.project_name
        assert failed[0].command == runner.down_commands[-1]
        assert runtime is not None
        assert runtime.status is TaskRuntimeStatus.BLOCKED
        assert runtime.state_reason is not None
        assert stack.project_name in runtime.state_reason
        assert "docker compose" in runtime.state_reason
        assert persisted_claim.released_at is None
        assert store.list_stale_compose_resources(fixture.run_id) == stale

        runner.fail_down.clear()
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
    assert str(source_compose) not in succeeded[0].command
    assert all(str(path) in succeeded[0].command for path in stack.compose_files)
    assert reclaimed is not None and reclaimed.task_id == claim.task_id
    assert artifact.read_text(encoding="utf-8") == "preserved\n"


def test_replayed_cleanup_failure_does_not_block_reclaimed_task(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    runner = _FakeComposeRunner()
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
    runner = _FakeComposeRunner()
    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        run = store.get_execution_run(fixture.run_id)
        assert run is not None
    project = (
        f"borg-{claim.run_id.hex[:6]}-{claim.task_id.hex[:6]}-{claim.id.hex}"
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
    runner = _FakeComposeRunner()
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


def test_compose_cleanup_metadata_excludes_resolved_env_file_secrets(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    runner = _FakeComposeRunner()
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


def test_distinct_dependencies_share_one_compose_service(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    runner = _FakeComposeRunner()
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
    runner = _FakeComposeRunner()
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


def test_writable_compose_bind_mount_blocks_before_start(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    runner = _FakeComposeRunner()
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
        with pytest.raises(
            ComposeStackError, match=r"writable bind mounts.*healthy\.volumes\[0\]"
        ):
            fixture.compose_manager(runner).start_claimed_stack(
                store, plan, claim, fixture.owner_token
            )
        runtime = store.get_task_runtime(claim.task_id)
        resources = store.list_compose_resources(claim.task_id)

    assert runtime is not None and runtime.status is TaskRuntimeStatus.BLOCKED
    assert runner.active == set()
    assert resources == []


def test_unhealthy_compose_startup_blocks_and_tears_down_exact_project(
    execution_preflight_fixture,
) -> None:
    fixture = execution_preflight_fixture()
    runner = _FakeComposeRunner()
    plan = _compose_plan(fixture.repository)
    with SqliteStore.open(fixture.database) as store:
        claim = fixture.claim(store)
        expected_project = (
            f"borg-{claim.run_id.hex[:6]}-{claim.task_id.hex[:6]}-{claim.id.hex}"
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
    runner = _FakeComposeRunner()
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


def _compose_plan(repository: Path) -> HostPreflightPlan:
    return HostPreflightPlan(
        repository_root=repository,
        commands=(),
        prepare_commands=(),
        materialize_commands=(),
        environment_files=(repository / "package.lock",),
        executables=(),
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
    )


class _FakeComposeRunner:
    def __init__(self) -> None:
        self.active: set[str] = set()
        self.started_services: dict[str, tuple[str, ...]] = {}
        self.down_commands: list[tuple[str, ...]] = []
        self.port_commands: list[tuple[str, ...]] = []
        self.environments: list[dict[str, str]] = []
        self.fail_up: set[str] = set()
        self.fail_down: set[str] = set()
        self.pause_up: set[str] = set()
        self.up_entered = threading.Event()
        self.release_up = threading.Event()
        self.down_entered = threading.Event()
        self.service_health: dict[str, str] = {"healthy": "healthy"}
        self.config_services: dict[str, dict[str, object]] = {
            "healthy": {"networks": {"default": None}},
            "unused": {"networks": {"default": None}},
        }
        self._lock = threading.Lock()

    def __call__(self, argv, **kwargs):
        command = tuple(argv)
        project = command[command.index("--project-name") + 1]
        if "up" in command and project in self.pause_up:
            self.up_entered.set()
            if not self.release_up.wait(timeout=10):
                return subprocess.CompletedProcess(
                    argv, 12, "", "timed out waiting for test startup release"
                )
        if "down" in command:
            self.down_entered.set()
        with self._lock:
            self.environments.append(dict(kwargs["env"]))
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
                file_paths = [
                    Path(command[index + 1])
                    for index, value in enumerate(command[:-1])
                    if value == "--file"
                ]
                if any(not path.is_file() for path in file_paths):
                    return subprocess.CompletedProcess(
                        argv, 11, "", "Compose file disappeared"
                    )
                if project in self.fail_down:
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
        "mkdir -p \"$XDG_CACHE_HOME\"\n"
        "case \"$action\" in\n"
        "  prepare|prepare-slow)\n"
        "    if [ \"$action\" = prepare-slow ]; then sleep 0.5; fi\n"
        "    printf 'prepare\\n' >> \"$XDG_CACHE_HOME/preparations.log\"\n"
        "    cp package.lock \"$XDG_CACHE_HOME/prepared-lock\"\n"
        "    mkdir -p .dependencies\n"
        "    printf 'local\\n' > .dependencies/prepared\n"
        "    ;;\n"
        "  materialize)\n"
        "    printf 'materialize\\n' >> \"$XDG_CACHE_HOME/materializations.log\"\n"
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


def _materialize_in_process(
    fixture: ExecutionPreflightFixture,
    plan: HostPreflightPlan,
    claim: TaskClaim,
    start: Path,
    result_path: Path,
) -> None:
    try:
        deadline = time.monotonic() + 10
        while not start.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("concurrent preparation did not start")
            time.sleep(0.01)
        with SqliteStore.open(fixture.database) as store:
            materialization = fixture.manager().materialize_claimed_task(
                store,
                plan,
                claim,
                fixture.owner_token,
            )
        outcome = ("ok", materialization.preparation_reused)
    except BaseException as error:
        outcome = ("error", repr(error))
    result_path.write_text(json.dumps(outcome), encoding="utf-8")


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
