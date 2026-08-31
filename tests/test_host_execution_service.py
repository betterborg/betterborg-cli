"""Integration contracts for the concrete host execution assembly."""

from __future__ import annotations

import hashlib
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from test_execution_preflight import FakeComposeRunner
from test_host_scheduler import FakeClock

from betterborg_cli.agent_runtime import (
    AgentResult,
    AgentStatus,
    CancellationToken,
    MockAdapter,
    MockResponse,
)
from betterborg_cli.host_execution import (
    EnvironmentMaterializationError,
    HostCodingConfig,
    HostCodingPhase,
    HostCommand,
    HostComposeManager,
    HostEnvironmentManager,
    HostExecutable,
    HostExecutionService,
    HostMergeConfig,
    HostMergePhase,
    HostMergeResult,
    HostPreflightBlock,
    HostPreflightFailure,
    HostPreflightPlan,
    HostReviewFixConfig,
    HostReviewFixPhase,
    HostSanityPhase,
    HostSanityResult,
    HostSchedulerConfig,
    HostSecret,
    HostService,
    HostTaskRuntime,
    HostWorktreeManager,
    MergeTip,
)
from betterborg_cli.planning import (
    approved_plan_digest,
    render_task_markdown,
    task_markdown_digest,
)
from betterborg_cli.repo_paths import RepoPaths, ensure_managed_gitignore
from betterborg_cli.store import (
    Borg,
    BorgState,
    ComposeResource,
    ExecutionAttemptStatus,
    ExecutionRunStatus,
    PlanApproval,
    Repository,
    RepositoryAnalysis,
    RepositoryPackage,
    SqliteStore,
    TaskBatch,
    TaskComplexity,
    TaskDependency,
    TaskGeneration,
    TaskRecord,
    TaskRuntimeStatus,
)


def _store_fixture(
    tmp_path: Path, task_count: int = 1
) -> tuple[SqliteStore, Borg, TaskGeneration, list[TaskRecord]]:
    repository = Repository(root=tmp_path / "repository")
    borg = Borg(repository_id=repository.id, name="Integration")
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
        manifest={"tasks": task_count},
    )
    generation = TaskGeneration(
        borg_id=borg.id,
        plan_approval_id=approval.id,
        batch_id=batch.id,
        digest="sha256:generation",
        manifest={"tasks": task_count},
    )
    records = []
    for position in range(1, task_count + 1):
        task_ref = f"task-{position}"
        digest = f"sha256:{hashlib.sha256(task_ref.encode()).hexdigest()}"
        records.append(
            TaskRecord(
                generation_id=generation.id,
                borg_id=borg.id,
                task_ref=task_ref,
                stage="07-host-execution",
                stem=f"{position:02d}-{task_ref}",
                position=position,
                title=f"Implement {task_ref}",
                complexity=TaskComplexity.SMALL,
                digest=digest,
                task={"acceptance_criteria": ["works"]},
                manifest={"task.md": digest},
            )
        )
    durable_root = repository.root / ".borg/tasks" / borg.name / str(generation.id)
    store = SqliteStore.open(tmp_path / "execution.sqlite3")
    store.add_repository(repository)
    store.add_borg(borg)
    store.append_plan_approval(approval)
    store.append_task_batch(batch)
    store.add_task_generation(generation, records, [])
    for record in records:
        path = durable_root / record.stage / f"{record.stem}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record.task_ref, encoding="utf-8")
    store._promote_published_task_generation(generation.id, durable_root=durable_root)
    return store, borg, generation, records


class _Preflight:
    def __init__(self, result, calls: list[str]) -> None:
        self.result = result
        self.calls = calls

    def validate(self, *args, **kwargs):
        self.calls.append("preflight")
        return self.result


class _Worktrees:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def prepare_current_task_worktrees(self, *args, **kwargs) -> list[object]:
        self.calls.append("worktrees")
        return []

    def refresh_unstarted_task_worktree(self, *args, **kwargs) -> bool:
        return False


class _Compose:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def cleanup_stale_projects(self, store, resources) -> tuple[object, ...]:
        self.calls.append("stale-cleanup")
        return ()

    def start_claimed_stack(self, *args, **kwargs):
        self.calls.append("services-start")
        return SimpleNamespace(environment={"SERVICE_URL": "http://127.0.0.1"})

    def stop_claimed_stack(self, *args, **kwargs) -> None:
        self.calls.append("services-stop")


class _Environment:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def materialize_claimed_task(self, store, plan, claim, owner_token, **kwargs):
        self.calls.append("environment")
        store.transition_task_runtime(
            claim.run_id,
            owner_token,
            claim.id,
            claim.claim_token,
            expected_status=TaskRuntimeStatus.CLAIMED,
            new_status=TaskRuntimeStatus.CODING,
        )
        return SimpleNamespace(environment={"CACHE": "prepared"})


class _Coding:
    def __init__(
        self,
        calls: list[str],
        expected_environment: dict[str, str] | None = None,
    ) -> None:
        self.calls = calls
        self.expected_environment = expected_environment or {
            "CACHE": "prepared",
            "SERVICE_URL": "http://127.0.0.1",
        }

    def run(
        self,
        context,
        *,
        environment=None,
    ) -> TaskRuntimeStatus:
        assert environment == self.expected_environment
        self.calls.append("coding")
        context.transition(TaskRuntimeStatus.CODING, TaskRuntimeStatus.REVIEW)
        return TaskRuntimeStatus.REVIEW


class _Review:
    def __init__(
        self,
        calls: list[str],
        expected_environment: dict[str, str] | None = None,
        expected_agent_environment: dict[str, str] | None = None,
    ) -> None:
        self.calls = calls
        self.expected_environment = expected_environment
        self.expected_agent_environment = expected_agent_environment

    def run(
        self,
        context,
        *,
        environment=None,
        review_environment=None,
        fix_environment=None,
    ) -> TaskRuntimeStatus:
        if self.expected_environment is None:
            assert environment["SERVICE_URL"] == "http://127.0.0.1"
        else:
            assert environment == self.expected_environment
        if self.expected_agent_environment is not None:
            assert review_environment == self.expected_agent_environment
            assert fix_environment == self.expected_agent_environment
        self.calls.append("review")
        context.transition(TaskRuntimeStatus.REVIEW, TaskRuntimeStatus.MERGING)
        return TaskRuntimeStatus.MERGING


class _Merge:
    def __init__(
        self,
        calls: list[str],
        expected_environment: dict[str, str] | None = None,
    ) -> None:
        self.calls = calls
        self.expected_environment = expected_environment

    def run(self, context, *, environment=None) -> HostMergeResult:
        if self.expected_environment is not None:
            assert environment == self.expected_environment
        self.calls.append("merge")
        return HostMergeResult(
            TaskRuntimeStatus.MERGING,
            "merged",
            MergeTip("task", "project/Integration", "a", "b", "c", False),
        )


class _Sanity:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def run(
        self,
        context,
        tip,
        *,
        secret_values=None,
        existing_stack=None,
    ) -> HostSanityResult:
        if existing_stack is not None:
            self.calls.append("services-stop")
            self.calls.append("services-start")
        self.calls.append("sanity")
        if existing_stack is not None:
            self.calls.append("services-stop")
        context.transition(TaskRuntimeStatus.MERGING, TaskRuntimeStatus.DONE)
        return HostSanityResult(TaskRuntimeStatus.DONE, "published", "c")


def _plan(tmp_path: Path) -> HostPreflightPlan:
    return HostPreflightPlan(
        repository_root=tmp_path / "repository",
        commands=(),
        prepare_commands=(),
        materialize_commands=(),
        environment_files=(),
        executables=(),
        required_secret_names=(),
        compose_files=(),
        services=(),
    )


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _advance_project_file(
    repository: Path,
    branch: str,
    filename: str,
    content: str,
    index_path: Path,
) -> str:
    """Create one project-branch commit without touching the primary checkout."""
    base_commit = _git(repository, "rev-parse", branch)
    environment = {**os.environ, "GIT_INDEX_FILE": str(index_path)}
    subprocess.run(
        ["git", "-C", str(repository), "read-tree", base_commit],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    blob = subprocess.run(
        ["git", "-C", str(repository), "hash-object", "-w", "--stdin"],
        input=content,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "update-index",
            "--add",
            "--cacheinfo",
            "100644",
            blob,
            filename,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    tree = subprocess.run(
        ["git", "-C", str(repository), "write-tree"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()
    commit = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "commit-tree",
            tree,
            "-p",
            base_commit,
            "-m",
            "advance project for merge conflict",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "update-ref",
            f"refs/heads/{branch}",
            commit,
            base_commit,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    index_path.unlink(missing_ok=True)
    return commit


def _concrete_task(
    generation_id: UUID,
    borg: Borg,
    position: int,
    *,
    dependencies: tuple[str, ...] = (),
    persisted_position: int | None = None,
) -> TaskRecord:
    stem = f"{position:02d}-integrated-task"
    body = {
        "stage": "07-host-execution",
        "stem": stem,
        "title": f"Implement integrated task {position}",
        "why": "Exercise the concrete host runtime.",
        "scope": ["Commit one feature through every host phase."],
        "implementation_notes": [],
        "acceptance_criteria": ["The project base advances."],
        "tests": ["Run the concrete integration fixture."],
        "dependencies": list(dependencies),
        "out_of_scope": [],
        "plan_refs": ["P1.deliverable.1"],
        "estimate_complexity": "small",
    }
    digest = task_markdown_digest(render_task_markdown(body))
    return TaskRecord(
        generation_id=generation_id,
        borg_id=borg.id,
        task_ref=f"07-host-execution/{stem}",
        stage=body["stage"],
        stem=stem,
        position=(
            persisted_position if persisted_position is not None else position
        ),
        title=body["title"],
        complexity=TaskComplexity.SMALL,
        digest=digest,
        task=body,
        manifest={"task.md": digest},
    )


def _coding_response(
    *, delay_seconds: float = 0, expected_existing_features: int | None = None
) -> MockResponse:
    def commit(spec):
        if expected_existing_features is not None:
            assert len(tuple(spec.cwd.glob("feature-*.txt"))) == (
                expected_existing_features
            )
        feature = spec.cwd / f"feature-{spec.cwd.name}.txt"
        feature.write_text("implemented\n", encoding="utf-8")
        _git(spec.cwd, "add", feature.name)
        _git(spec.cwd, "commit", "--quiet", "-m", "implement task")
        return MockResponse(
            payload={
                "task_file": ".betterborg-task/task.md",
                "status": "completed",
                "summary": "Implemented the task.",
                "changed_files": [feature.name],
                "tests_run": ["integration"],
                "follow_ups": [],
                "blockers": [],
            }
        )

    return MockResponse(dynamic=commit, delay_seconds=delay_seconds)


@dataclass(frozen=True)
class _ConcreteHostFixture:
    store: SqliteStore
    borg: Borg
    generation: TaskGeneration
    tasks: tuple[TaskRecord, ...]
    service: HostExecutionService
    coding: MockAdapter
    review: MockAdapter
    merge: MockAdapter
    compose: FakeComposeRunner
    environment: HostEnvironmentManager
    worktrees: HostWorktreeManager
    clock: FakeClock


def _concrete_host_fixture(
    tmp_path: Path,
    *,
    task_count: int = 1,
    coding_delay_seconds: float = 0,
    review_delay_seconds: float = 0,
    dependency_chain: bool = False,
    prerequisite_at_later_position: bool = False,
) -> _ConcreteHostFixture:
    repository_root = tmp_path / "concrete-repository"
    repository_root.mkdir()
    _git(repository_root, "init", "--quiet", "--initial-branch=main")
    _git(repository_root, "config", "user.name", "BetterBorg Tests")
    _git(repository_root, "config", "user.email", "tests@betterborg.dev")
    (repository_root / "README.md").write_text("# Fixture\n", encoding="utf-8")
    (repository_root / "compose.yml").write_text(
        "services:\n  healthy:\n    image: fixture\n",
        encoding="utf-8",
    )
    ensure_managed_gitignore(RepoPaths.discover(repository_root))
    _git(repository_root, "add", ".")
    _git(repository_root, "commit", "--quiet", "-m", "initial")

    repository = Repository(root=repository_root)
    borg = Borg(
        repository_id=repository.id,
        name="concrete-integration",
        state=BorgState.READY_TO_EXECUTE,
    )
    approval = PlanApproval(
        borg_id=borg.id,
        plan_digest="sha256:approved-plan",
        manifest={},
    )
    batch = TaskBatch(
        borg_id=borg.id,
        plan_approval_id=approval.id,
        round=1,
        digest="sha256:batch",
        manifest={},
    )
    generation_id = uuid4()
    if prerequisite_at_later_position:
        if task_count != 2 or not dependency_chain:
            raise ValueError(
                "a later-position prerequisite requires a two-task dependency chain"
            )
    task_list: list[TaskRecord] = []
    for position in range(1, task_count + 1):
        task_list.append(
            _concrete_task(
                generation_id,
                borg,
                position,
                dependencies=(task_list[-1].task_ref,)
                if dependency_chain and task_list
                else (),
                persisted_position=(3 - position)
                if prerequisite_at_later_position
                else None,
            )
        )
    tasks = tuple(task_list)
    dependencies = (
        tuple(
            TaskDependency(
                generation_id=generation_id,
                task_id=tasks[position].id,
                depends_on_task_id=tasks[position - 1].id,
            )
            for position in range(1, len(tasks))
        )
        if dependency_chain
        else ()
    )
    generation_manifest = {
        "approved_plan_digest": approval.plan_digest,
        "batch_digest": batch.digest,
        "dependencies": [
            {
                "task_ref": tasks[position].task_ref,
                "depends_on": tasks[position - 1].task_ref,
            }
            for position in range(1, len(tasks))
        ]
        if dependency_chain
        else [],
        "plan_approval_id": str(approval.id),
        "tasks": [
            {
                "digest": task.digest,
                "path": (
                    f".borg/tasks/{borg.name}/{generation_id}/"
                    f"{task.stage}/{task.stem}.md"
                ),
                "position": task.position,
                "task_ref": task.task_ref,
            }
            for task in sorted(tasks, key=lambda task: task.position)
        ],
    }
    generation = TaskGeneration(
        id=generation_id,
        borg_id=borg.id,
        plan_approval_id=approval.id,
        batch_id=batch.id,
        digest=approved_plan_digest(generation_manifest),
        manifest=generation_manifest,
    )
    analysis = RepositoryAnalysis(
        repository_id=repository.id,
        head_sha=_git(repository_root, "rev-parse", "HEAD"),
        summary="Concrete host integration fixture.",
        primary_language="Python",
        is_monorepo=False,
        overall_score=4,
        analysis_json={},
    )
    package = RepositoryPackage(
        repository_id=repository.id,
        analysis_id=analysis.id,
        package_path=".",
        package_name="fixture",
        primary_language="Python",
        rubric={},
        overall_score=4,
    )
    store = SqliteStore.open(tmp_path / "concrete.sqlite3")
    store.add_repository(repository)
    store.add_borg(borg)
    store.append_analysis(analysis, [package])
    for role in ("coding", "review", "merge"):
        store.append_generated_prompt(
            repository_id=repository.id,
            analysis_id=analysis.id,
            role=role,
            body_md=f"You are the generated {role} agent.\n",
        )
    store.append_plan_approval(approval)
    store.append_task_batch(batch)
    store.add_task_generation(generation, tasks, dependencies)
    durable_root = repository_root / ".borg/tasks" / borg.name / str(generation.id)
    for task in tasks:
        path = durable_root / task.stage / f"{task.stem}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_task_markdown(task.task), encoding="utf-8")
    store._promote_published_task_generation(generation.id, durable_root=durable_root)
    _git(repository_root, "add", ".")
    _git(repository_root, "commit", "--quiet", "-m", "publish tasks")

    clock = FakeClock()
    plan = HostPreflightPlan(
        repository_root=repository_root,
        commands=(HostCommand("test", ("git", "status", "--short"), "."),),
        prepare_commands=(HostCommand("prepare", ("git", "status", "--short"), "."),),
        materialize_commands=(),
        environment_files=(repository_root / "README.md",),
        executables=(HostExecutable("docker", Path("/validated/docker")),),
        required_secret_names=(),
        compose_files=(repository_root / "compose.yml",),
        services=(
            HostService(
                name="http-service",
                kind="compose",
                evidence="concrete fixture",
                compose_service="healthy",
                url_env="SERVICE_URL",
                port=8080,
                url_targets=(("SERVICE_URL", 8080, "tcp"),),
            ),
        ),
        compose_networks=("default",),
    )
    environment = HostEnvironmentManager(repository_root, clock=clock)
    compose_runner = FakeComposeRunner()
    compose = HostComposeManager(
        repository_root,
        command_runner=compose_runner,
        clock=clock,
    )
    worktrees = HostWorktreeManager(
        repository_root,
        tmp_path / "concrete-worktrees",
        source_branch="main",
    )
    coding = MockAdapter()
    for position, _ in enumerate(tasks):
        coding.queue(
            _coding_response(
                delay_seconds=coding_delay_seconds,
                expected_existing_features=(position if dependency_chain else None),
            )
        )
    review = MockAdapter()
    for _ in tasks:
        review.queue(
            MockResponse(
                payload={
                    "task_file": ".betterborg-task/task.md",
                    "status": "approved",
                    "summary": "Implementation approved.",
                    "issues_file": "",
                    "findings": [],
                },
                delay_seconds=review_delay_seconds,
            )
        )
    merge = MockAdapter()
    repository_lock = threading.RLock()
    runtime = HostTaskRuntime(
        plan,
        environment_manager=environment,
        compose_manager=compose,
        coding=HostCodingPhase(
            repository_root,
            coding,
            config=HostCodingConfig(model="coding-model"),
        ),
        review_fix=HostReviewFixPhase(
            repository_root,
            review,
            config=HostReviewFixConfig(review_model="review-model"),
        ),
        merge=HostMergePhase(
            repository_root,
            merge,
            config=HostMergeConfig(model="merge-model"),
            repository_lock=lambda: repository_lock,
        ),
        sanity=HostSanityPhase(
            repository_root,
            plan,
            environment_manager=environment,
            compose_manager=compose,
            worktree_manager=worktrees,
            repository_lock=lambda: repository_lock,
        ),
    )
    service = HostExecutionService(
        store,
        _Preflight(plan, []),
        runtime,
        worktree_manager=worktrees,
        compose_manager=compose,
        scheduler_config=HostSchedulerConfig(
            jobs=task_count,
            lease_duration=timedelta(minutes=5),
            heartbeat_interval=timedelta(minutes=1),
            poll_interval_seconds=0.005,
        ),
        clock=clock,
    )
    return _ConcreteHostFixture(
        store,
        borg,
        generation,
        tasks,
        service,
        coding,
        review,
        merge,
        compose_runner,
        environment,
        worktrees,
        clock,
    )


def test_service_runs_the_concrete_task_lifecycle_in_order(tmp_path: Path) -> None:
    store, borg, generation, records = _store_fixture(tmp_path)
    calls: list[str] = []
    plan = _plan(tmp_path)
    compose = _Compose(calls)
    runtime = HostTaskRuntime(
        plan,
        environment_manager=_Environment(calls),
        compose_manager=compose,
        coding=_Coding(calls),
        review_fix=_Review(calls),
        merge=_Merge(calls),
        sanity=_Sanity(calls),
    )
    try:
        result = HostExecutionService(
            store,
            _Preflight(plan, calls),
            runtime,
            worktree_manager=_Worktrees(calls),
            compose_manager=compose,
            scheduler_config=HostSchedulerConfig(poll_interval_seconds=0.005),
        ).run(borg.id, generation.id, {})

        assert result.status is ExecutionRunStatus.COMPLETED
        assert store.get_task_runtime(records[0].id).status is TaskRuntimeStatus.DONE
        assert calls == [
            "preflight",
            "worktrees",
            "environment",
            "services-start",
            "coding",
            "review",
            "merge",
            "services-stop",
            "services-start",
            "sanity",
            "services-stop",
        ]
    finally:
        store.close()


def test_cancellation_during_materialization_does_not_start_services(
    tmp_path: Path,
) -> None:
    store, borg, generation, records = _store_fixture(tmp_path)
    calls: list[str] = []
    plan = _plan(tmp_path)
    cancel = CancellationToken()
    materializing = threading.Event()

    class PausingEnvironment(_Environment):
        def materialize_claimed_task(self, store, plan, claim, owner_token, **kwargs):
            self.calls.append("environment")
            materializing.set()
            assert cancel.wait(timeout=2)
            store.transition_task_runtime(
                claim.run_id,
                owner_token,
                claim.id,
                claim.claim_token,
                expected_status=TaskRuntimeStatus.CLAIMED,
                new_status=TaskRuntimeStatus.CODING,
            )
            return SimpleNamespace(environment={"CACHE": "prepared"})

    compose = _Compose(calls)
    runtime = HostTaskRuntime(
        plan,
        environment_manager=PausingEnvironment(calls),
        compose_manager=compose,
        coding=_Coding(calls),
        review_fix=_Review(calls),
        merge=_Merge(calls),
        sanity=_Sanity(calls),
    )
    try:
        service = HostExecutionService(
            store,
            _Preflight(plan, calls),
            runtime,
            worktree_manager=_Worktrees(calls),
            compose_manager=compose,
            scheduler_config=HostSchedulerConfig(poll_interval_seconds=0.005),
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            running = executor.submit(
                service.run,
                borg.id,
                generation.id,
                {},
                cancel=cancel,
            )
            assert materializing.wait(timeout=2)
            cancel.cancel()
            result = running.result(timeout=2)

        assert result.status is ExecutionRunStatus.CANCELLED
        assert store.get_task_runtime(records[0].id).status is (
            TaskRuntimeStatus.PENDING
        )
        assert "services-start" not in calls
        assert "coding" not in calls
    finally:
        store.close()


def test_cancellation_during_compose_startup_does_not_start_coding(
    tmp_path: Path,
) -> None:
    store, borg, generation, records = _store_fixture(tmp_path)
    calls: list[str] = []
    plan = _plan(tmp_path)
    cancel = CancellationToken()
    starting_services = threading.Event()

    class PausingCompose(_Compose):
        def start_claimed_stack(self, *args, **kwargs):
            self.calls.append("services-start")
            starting_services.set()
            assert cancel.wait(timeout=2)
            return SimpleNamespace(environment={"SERVICE_URL": "http://127.0.0.1"})

    compose = PausingCompose(calls)
    runtime = HostTaskRuntime(
        plan,
        environment_manager=_Environment(calls),
        compose_manager=compose,
        coding=_Coding(calls),
        review_fix=_Review(calls),
        merge=_Merge(calls),
        sanity=_Sanity(calls),
    )
    try:
        service = HostExecutionService(
            store,
            _Preflight(plan, calls),
            runtime,
            worktree_manager=_Worktrees(calls),
            compose_manager=compose,
            scheduler_config=HostSchedulerConfig(poll_interval_seconds=0.005),
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            running = executor.submit(
                service.run,
                borg.id,
                generation.id,
                {},
                cancel=cancel,
            )
            assert starting_services.wait(timeout=2)
            cancel.cancel()
            result = running.result(timeout=2)

        assert result.status is ExecutionRunStatus.CANCELLED
        assert store.get_task_runtime(records[0].id).status is (
            TaskRuntimeStatus.PENDING
        )
        assert "services-stop" in calls
        assert "coding" not in calls
    finally:
        store.close()


def test_external_only_service_url_reaches_every_agent_phase(tmp_path: Path) -> None:
    store, borg, generation, records = _store_fixture(tmp_path)
    calls: list[str] = []
    registry_url = "https://registry.example.test"
    plan = replace(
        _plan(tmp_path),
        services=(
            HostService(
                name="registry",
                kind="external",
                evidence="validated external fixture",
                url_env="REGISTRY_URL",
                url=registry_url,
            ),
        ),
    )
    expected_environment = {
        "CACHE": "prepared",
        "REGISTRY_URL": registry_url,
    }

    class ExternalOnlyCompose(_Compose):
        def start_claimed_stack(self, *args, **kwargs):
            self.calls.append("services-start")
            return None

    compose = ExternalOnlyCompose(calls)
    runtime = HostTaskRuntime(
        plan,
        environment_manager=_Environment(calls),
        compose_manager=compose,
        coding=_Coding(calls, expected_environment),
        review_fix=_Review(calls, expected_environment),
        merge=_Merge(calls, expected_environment),
        sanity=_Sanity(calls),
    )
    try:
        result = HostExecutionService(
            store,
            _Preflight(plan, calls),
            runtime,
            worktree_manager=_Worktrees(calls),
            compose_manager=compose,
            scheduler_config=HostSchedulerConfig(poll_interval_seconds=0.005),
        ).run(borg.id, generation.id, {})

        assert result.status is ExecutionRunStatus.COMPLETED
        assert store.get_task_runtime(records[0].id).status is TaskRuntimeStatus.DONE
        assert calls == [
            "preflight",
            "worktrees",
            "environment",
            "services-start",
            "coding",
            "review",
            "merge",
            "sanity",
        ]
    finally:
        store.close()


def test_command_stage_agent_secret_reaches_every_agent_phase(tmp_path: Path) -> None:
    store, borg, generation, records = _store_fixture(tmp_path)
    calls: list[str] = []
    token = "agent-token"
    plan = replace(
        _plan(tmp_path),
        required_secret_names=("AGENT_TOKEN",),
        secret_requirements=(
            HostSecret(
                name="AGENT_TOKEN",
                scope="agent",
                used_by=("test",),
                evidence="validated command-stage fixture",
            ),
        ),
    )
    service_environment = {
        "CACHE": "prepared",
        "SERVICE_URL": "http://127.0.0.1",
    }
    agent_environment = {"AGENT_TOKEN": token}
    compose = _Compose(calls)
    runtime = HostTaskRuntime(
        plan,
        environment_manager=_Environment(calls),
        compose_manager=compose,
        coding=_Coding(calls, {**service_environment, **agent_environment}),
        review_fix=_Review(
            calls,
            service_environment,
            expected_agent_environment=agent_environment,
        ),
        merge=_Merge(calls, {**service_environment, **agent_environment}),
        sanity=_Sanity(calls),
    )
    try:
        result = HostExecutionService(
            store,
            _Preflight(plan, calls),
            runtime,
            worktree_manager=_Worktrees(calls),
            compose_manager=compose,
            scheduler_config=HostSchedulerConfig(poll_interval_seconds=0.005),
        ).run(
            borg.id,
            generation.id,
            {},
            secret_values={"AGENT_TOKEN": token},
        )

        assert result.status is ExecutionRunStatus.COMPLETED, (
            store.get_task_runtime(records[0].id).state_reason
        )
        assert store.get_task_runtime(records[0].id).status is TaskRuntimeStatus.DONE
        assert calls == [
            "preflight",
            "worktrees",
            "environment",
            "services-start",
            "coding",
            "review",
            "merge",
            "services-stop",
            "services-start",
            "sanity",
            "services-stop",
        ]
    finally:
        store.close()


def test_concrete_jobs_two_complete_and_resume_without_phase_replay(
    tmp_path: Path,
) -> None:
    fixture = _concrete_host_fixture(tmp_path, task_count=2)
    try:
        first = fixture.service.run(fixture.borg.id, fixture.generation.id, {})

        assert first.status is ExecutionRunStatus.COMPLETED, [
            fixture.store.get_task_runtime(task.id).state_reason
            for task in fixture.tasks
        ]
        assert len(fixture.store.list_task_claims(first.operation_id)) == 2
        assert all(
            fixture.store.get_task_runtime(task.id).status is TaskRuntimeStatus.DONE
            for task in fixture.tasks
        )
        assert len(fixture.coding.calls) == 2
        assert len(fixture.review.calls) == 2
        assert all(
            call.env["SERVICE_URL"].startswith("http://127.0.0.1:")
            for call in fixture.coding.calls
        )
        assert len(fixture.compose.up_projects) == 4
        assert sorted(fixture.compose.up_projects) == sorted(
            fixture.compose.down_projects
        )
        assert len(set(fixture.compose.up_projects)) == 4
        assert fixture.compose.active == set()
        environment_attempts = [
            attempt
            for task in fixture.tasks
            for attempt in fixture.store.list_environment_attempts(task.id)
        ]
        preparations = [
            attempt for attempt in environment_attempts if attempt.kind == "prepare"
        ]
        assert len(preparations) == 1
        assert preparations[0].result["prepared_before_dispatch"] is True
        assert (
            sum(attempt.kind == "materialize" for attempt in environment_attempts) == 2
        )
        project_tip = _git(
            fixture.store.get_repository(fixture.borg.repository_id).root,
            "rev-parse",
            f"project/{fixture.borg.name}",
        )

        resumed = fixture.service.run(fixture.borg.id, fixture.generation.id, {})

        assert resumed.status is ExecutionRunStatus.COMPLETED
        assert len(fixture.coding.calls) == 2
        assert len(fixture.review.calls) == 2
        assert (
            _git(
                fixture.store.get_repository(fixture.borg.repository_id).root,
                "rev-parse",
                f"project/{fixture.borg.name}",
            )
            == project_tip
        )
    finally:
        fixture.store.close()


def test_predispatch_preparation_failure_is_a_durable_environment_attempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _concrete_host_fixture(tmp_path)

    def fail_preparation(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv,
            23,
            "dependency setup output\n",
            "dependency setup failed\n",
        )

    monkeypatch.setattr(fixture.environment, "_run", fail_preparation)
    try:
        with pytest.raises(
            EnvironmentMaterializationError,
            match="dependency setup failed",
        ):
            fixture.service.run(fixture.borg.id, fixture.generation.id, {})

        task = fixture.tasks[0]
        attempts = fixture.store.list_environment_attempts(task.id)
        assert len(attempts) == 1
        attempt = attempts[0]
        assert attempt.claim_id is None
        assert attempt.status is ExecutionAttemptStatus.FAILED
        assert attempt.kind == "prepare"
        assert attempt.fingerprint.startswith("sha256:")
        assert attempt.commands == [["git", "status", "--short"]]
        assert attempt.error is not None
        assert "dependency setup failed" in attempt.error
        assert attempt.result is not None
        assert attempt.result["prepared_before_dispatch"] is True
        assert fixture.store.list_task_claims(attempt.run_id) == []
        assert fixture.coding.calls == []
    finally:
        fixture.store.close()


def test_cancellation_cannot_mask_primary_checkout_contamination(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _concrete_host_fixture(tmp_path)
    cancel = CancellationToken()
    repository = fixture.store.get_repository(fixture.borg.repository_id)
    assert repository is not None
    adapter_run = MockAdapter.run

    def cancel_after_contamination(self, spec, *, cancel=None):
        if self is not fixture.coding:
            return adapter_run(self, spec, cancel=cancel)
        self.calls.append(spec)
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        spec.log_path.write_text("cancelled after contamination\n", encoding="utf-8")
        (repository.root / "agent-contamination.txt").write_text(
            "unauthorized primary edit\n",
            encoding="utf-8",
        )
        assert cancel is not None
        cancel.cancel()
        return AgentResult(
            status=AgentStatus.CANCELLED,
            log_path=spec.log_path,
            error="coding cancelled",
        )

    monkeypatch.setattr(MockAdapter, "run", cancel_after_contamination)
    try:
        fixture.service.run(
            fixture.borg.id,
            fixture.generation.id,
            {},
            cancel=cancel,
        )

        runtime = fixture.store.get_task_runtime(fixture.tasks[0].id)
        assert runtime is not None and runtime.status is TaskRuntimeStatus.BLOCKED
        assert runtime.state_reason is not None
        assert "primary checkout" in runtime.state_reason
        assert "changed while it ran" in runtime.state_reason
    finally:
        fixture.store.close()


def test_cancellation_cannot_mask_coding_branch_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _concrete_host_fixture(tmp_path)
    cancel = CancellationToken()
    adapter_run = MockAdapter.run

    def cancel_after_branch_change(self, spec, *, cancel=None):
        if self is not fixture.coding:
            return adapter_run(self, spec, cancel=cancel)
        self.calls.append(spec)
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        spec.log_path.write_text("cancelled after branch change\n", encoding="utf-8")
        _git(spec.cwd, "checkout", "--quiet", "-b", "agent/unauthorized")
        assert cancel is not None
        cancel.cancel()
        return AgentResult(
            status=AgentStatus.CANCELLED,
            log_path=spec.log_path,
            error="coding cancelled",
        )

    monkeypatch.setattr(MockAdapter, "run", cancel_after_branch_change)
    try:
        fixture.service.run(
            fixture.borg.id,
            fixture.generation.id,
            {},
            cancel=cancel,
        )

        runtime = fixture.store.get_task_runtime(fixture.tasks[0].id)
        assert runtime is not None and runtime.status is TaskRuntimeStatus.BLOCKED
        assert runtime.state_reason == "coding agent changed the task branch"
    finally:
        fixture.store.close()


def test_cancellation_cannot_mask_review_worktree_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _concrete_host_fixture(tmp_path)
    cancel = CancellationToken()
    adapter_run = MockAdapter.run

    def cancel_after_mutation(self, spec, *, cancel=None):
        if self is not fixture.review:
            return adapter_run(self, spec, cancel=cancel)
        self.calls.append(spec)
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        spec.log_path.write_text("cancelled after mutation\n", encoding="utf-8")
        (spec.cwd / "unauthorized-review-edit.txt").write_text(
            "review agents are read-only\n",
            encoding="utf-8",
        )
        assert cancel is not None
        cancel.cancel()
        return AgentResult(
            status=AgentStatus.CANCELLED,
            log_path=spec.log_path,
            error="review cancelled",
        )

    monkeypatch.setattr(MockAdapter, "run", cancel_after_mutation)
    try:
        fixture.service.run(
            fixture.borg.id,
            fixture.generation.id,
            {},
            cancel=cancel,
        )

        runtime = fixture.store.get_task_runtime(fixture.tasks[0].id)
        assert runtime is not None and runtime.status is TaskRuntimeStatus.BLOCKED
        assert runtime.state_reason == "review agent modified the task worktree"
        assert runtime.worktree_path is not None
        assert (
            Path(runtime.worktree_path) / "unauthorized-review-edit.txt"
        ).is_file()
    finally:
        fixture.store.close()


def test_concrete_sanity_restarts_compose_after_merged_descriptor_change(
    tmp_path: Path,
) -> None:
    fixture = _concrete_host_fixture(tmp_path)

    def change_compose_descriptor(spec):  # noqa: ANN001
        descriptor = spec.cwd / "compose.yml"
        descriptor.write_text(
            "services:\n  healthy:\n    image: fixture-after-coding\n",
            encoding="utf-8",
        )
        _git(spec.cwd, "add", descriptor.name)
        _git(spec.cwd, "commit", "--quiet", "-m", "change service descriptor")
        return MockResponse(
            payload={
                "task_file": ".betterborg-task/task.md",
                "status": "completed",
                "summary": "Changed the service descriptor.",
                "changed_files": [descriptor.name],
                "tests_run": ["integration"],
                "follow_ups": [],
                "blockers": [],
            }
        )

    fixture.coding.responses.clear()
    fixture.coding.queue(MockResponse(dynamic=change_compose_descriptor))
    try:
        result = fixture.service.run(fixture.borg.id, fixture.generation.id, {})

        assert result.status is ExecutionRunStatus.COMPLETED
        assert len(fixture.compose.up_projects) == 2
        assert fixture.compose.up_projects[0] != fixture.compose.up_projects[1]
        assert fixture.compose.up_projects[1].endswith("-sanity")
        lifecycle = [
            "down" if "down" in command else "up"
            for command in fixture.compose.commands
            if "up" in command or "down" in command
        ]
        assert lifecycle == ["up", "down", "up", "down"]
        assert all(
            "--volumes" in command and command[-2:] == ("--rmi", "all")
            for command in fixture.compose.down_commands
        )
        repository = fixture.store.get_repository(fixture.borg.repository_id)
        assert repository is not None
        assert (
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository.root),
                    "show",
                    f"project/{fixture.borg.name}:compose.yml",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            == "services:\n  healthy:\n    image: fixture-after-coding\n"
        )
        assert fixture.compose.active == set()
    finally:
        fixture.store.close()


def test_concrete_dependent_starts_from_published_prerequisite(
    tmp_path: Path,
) -> None:
    fixture = _concrete_host_fixture(
        tmp_path,
        task_count=2,
        dependency_chain=True,
    )
    try:
        result = fixture.service.run(
            fixture.borg.id,
            fixture.generation.id,
            {},
        )

        assert result.status is ExecutionRunStatus.COMPLETED, [
            fixture.store.get_task_runtime(task.id).state_reason
            for task in fixture.tasks
        ]
        first = fixture.store.get_task_runtime(fixture.tasks[0].id)
        second = fixture.store.get_task_runtime(fixture.tasks[1].id)
        assert first is not None and first.branch is not None
        assert second is not None and second.branch is not None
        repository = fixture.store.get_repository(fixture.borg.repository_id)
        assert repository is not None
        assert (
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository.root),
                    "merge-base",
                    "--is-ancestor",
                    first.branch,
                    second.branch,
                ],
                check=False,
            ).returncode
            == 0
        )
    finally:
        fixture.store.close()


def test_concrete_dependent_with_earlier_position_refreshes_after_preparation(
    tmp_path: Path,
) -> None:
    fixture = _concrete_host_fixture(
        tmp_path,
        task_count=2,
        dependency_chain=True,
        prerequisite_at_later_position=True,
    )
    prerequisite, dependent = fixture.tasks
    try:
        result = fixture.service.run(
            fixture.borg.id,
            fixture.generation.id,
            {},
        )

        assert dependent.position < prerequisite.position
        assert dependent.stem > prerequisite.stem
        preparations = [
            attempt
            for attempt in fixture.store.list_environment_attempts(dependent.id)
            if attempt.kind == "prepare"
        ]
        assert len(preparations) == 1
        preparation = preparations[0]
        assert preparation.claim_id is None
        assert preparation.result is not None
        assert preparation.result["prepared_before_dispatch"] is True
        assert result.status is ExecutionRunStatus.COMPLETED, [
            fixture.store.get_task_runtime(task.id).state_reason
            for task in fixture.tasks
        ]
        dependent_runtime = fixture.store.get_task_runtime(dependent.id)
        prerequisite_runtime = fixture.store.get_task_runtime(prerequisite.id)
        assert dependent_runtime is not None and dependent_runtime.branch is not None
        assert prerequisite_runtime is not None
        assert prerequisite_runtime.branch is not None
        repository = fixture.store.get_repository(fixture.borg.repository_id)
        assert repository is not None
        assert (
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository.root),
                    "merge-base",
                    "--is-ancestor",
                    prerequisite_runtime.branch,
                    dependent_runtime.branch,
                ],
                check=False,
            ).returncode
            == 0
        )
    finally:
        fixture.store.close()


def test_concrete_agent_stack_teardown_failure_prevents_base_advance(
    tmp_path: Path,
) -> None:
    fixture = _concrete_host_fixture(tmp_path)
    repository = fixture.store.get_repository(fixture.borg.repository_id)
    assert repository is not None
    base_commit = _git(repository.root, "rev-parse", "main")
    fixture.compose.fail_all_down = True
    try:
        result = fixture.service.run(
            fixture.borg.id,
            fixture.generation.id,
            {},
        )

        runtime = fixture.store.get_task_runtime(fixture.tasks[0].id)
        assert result.status is ExecutionRunStatus.FAILED
        assert runtime is not None and runtime.status is TaskRuntimeStatus.BLOCKED
        assert "Compose teardown failed" in runtime.state_reason
        assert _git(
            repository.root,
            "rev-parse",
            f"project/{fixture.borg.name}",
        ) == base_commit
        assert fixture.store.list_task_execution_events(
            fixture.tasks[0].id,
            kind="sanity.completed",
        ) == []
        assert fixture.store.list_task_execution_events(
            fixture.tasks[0].id,
            kind="base.advance_started",
        ) == []
        assert set(fixture.compose.down_projects) == set(
            fixture.compose.up_projects
        )
        assert fixture.compose.active == set(fixture.compose.up_projects)
    finally:
        fixture.store.close()


def test_concrete_dependency_refresh_contamination_blocks_and_preserves_worktree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _concrete_host_fixture(
        tmp_path,
        task_count=2,
        dependency_chain=True,
    )
    refresh = fixture.worktrees.refresh_unstarted_task_worktree
    repository = fixture.store.get_repository(fixture.borg.repository_id)
    assert repository is not None

    def contaminate_before_second_refresh(runtime, *, project_name):
        if runtime.task_id == fixture.tasks[1].id:
            (repository.root / "contamination.txt").write_text(
                "primary checkout edit\n",
                encoding="utf-8",
            )
        return refresh(runtime, project_name=project_name)

    monkeypatch.setattr(
        fixture.worktrees,
        "refresh_unstarted_task_worktree",
        contaminate_before_second_refresh,
    )
    try:
        result = fixture.service.run(
            fixture.borg.id,
            fixture.generation.id,
            {},
        )

        first = fixture.store.get_task_runtime(fixture.tasks[0].id)
        blocked = fixture.store.get_task_runtime(fixture.tasks[1].id)
        assert result.status is ExecutionRunStatus.FAILED
        assert first is not None and first.status is TaskRuntimeStatus.DONE
        assert blocked is not None and blocked.status is TaskRuntimeStatus.BLOCKED
        assert "primary checkout" in blocked.state_reason
        assert "task work was preserved" in blocked.state_reason
        assert blocked.worktree_path is not None
        assert Path(blocked.worktree_path).is_dir()
        assert len(fixture.coding.calls) == 1
        assert len(fixture.compose.up_projects) == 2
        assert fixture.compose.up_projects == fixture.compose.down_projects
        assert fixture.compose.active == set()

    finally:
        fixture.store.close()


def test_concrete_cancellation_resumes_the_active_phase(tmp_path: Path) -> None:
    fixture = _concrete_host_fixture(tmp_path, coding_delay_seconds=2)
    cancel = CancellationToken()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            running = executor.submit(
                fixture.service.run,
                fixture.borg.id,
                fixture.generation.id,
                {},
                cancel=cancel,
            )
            assert fixture.coding.wait_for_response_consumption(timeout=1)
            cancel.cancel()
            cancelled = running.result(timeout=3)

        task = fixture.tasks[0]
        assert cancelled.status is ExecutionRunStatus.CANCELLED
        assert fixture.store.get_task_runtime(task.id).status is (
            TaskRuntimeStatus.PENDING
        )
        assert fixture.store.list_agent_attempts(task.id)[0].status.value == (
            "cancelled"
        )

        runtime = fixture.store.get_task_runtime(task.id)
        assert runtime is not None and runtime.worktree_path is not None
        preserved = Path(runtime.worktree_path) / "README.md"
        preserved.write_text("# Fixture\n\npreserved agent edit\n", encoding="utf-8")
        fixture.clock.advance(timedelta(seconds=1))

        def commit_preserved_edit(spec):
            feature = spec.cwd / "feature-resumed.txt"
            feature.write_text("implemented after resume\n", encoding="utf-8")
            _git(spec.cwd, "add", "README.md", feature.name)
            _git(spec.cwd, "commit", "--quiet", "-m", "finish resumed task")
            return MockResponse(
                payload={
                    "task_file": ".betterborg-task/task.md",
                    "status": "completed",
                    "summary": "Completed the preserved work.",
                    "changed_files": ["README.md", feature.name],
                    "tests_run": ["integration"],
                    "follow_ups": [],
                    "blockers": [],
                }
            )

        fixture.coding.queue(MockResponse(dynamic=commit_preserved_edit))
        resumed = fixture.service.run(fixture.borg.id, fixture.generation.id, {})

        assert resumed.status is ExecutionRunStatus.COMPLETED, (
            fixture.store.get_task_runtime(task.id).state_reason
        )
        assert fixture.store.get_task_runtime(task.id).status is (
            TaskRuntimeStatus.DONE
        )
        assert len(fixture.coding.calls) == 2
        assert len(fixture.review.calls) == 1
        repository = fixture.store.get_repository(fixture.borg.repository_id)
        assert repository is not None
        assert (
            _git(
                repository.root,
                "show",
                f"project/{fixture.borg.name}:README.md",
            )
            == "# Fixture\n\npreserved agent edit"
        )
    finally:
        fixture.store.close()


def test_concrete_retry_exhaustion_stops_and_resumes_coding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _concrete_host_fixture(tmp_path)
    cancel = CancellationToken()
    adapter_run = MockAdapter.run

    def exhaust_coding(self, spec, *, cancel=None):
        if self is not fixture.coding:
            return adapter_run(self, spec, cancel=cancel)
        self.calls.append(spec)
        spec.log_path.write_text("transient retries exhausted\n", encoding="utf-8")
        return AgentResult(
            status=AgentStatus.CANCELLED,
            log_path=spec.log_path,
            error="transient retry exhausted: provider unavailable",
            retryable=True,
        )

    monkeypatch.setattr(MockAdapter, "run", exhaust_coding)
    try:
        cancelled = fixture.service.run(
            fixture.borg.id,
            fixture.generation.id,
            {},
            cancel=cancel,
        )

        task = fixture.tasks[0]
        runtime = fixture.store.get_task_runtime(task.id)
        attempts = fixture.store.list_agent_attempts(task.id)
        assert cancel.is_set()
        assert cancelled.status is ExecutionRunStatus.CANCELLED
        assert runtime is not None and runtime.status is TaskRuntimeStatus.PENDING
        assert runtime.resume_phase == "coding"
        assert len(attempts) == 1 and attempts[0].status.value == "cancelled"
        assert "transient retry exhausted" in (
            attempts[0].result["_betterborg"]["outcome_reason"]
        )

        monkeypatch.setattr(MockAdapter, "run", adapter_run)
        fixture.clock.advance(timedelta(seconds=1))
        resumed = fixture.service.run(fixture.borg.id, fixture.generation.id, {})

        assert resumed.status is ExecutionRunStatus.COMPLETED
        assert fixture.store.get_task_runtime(task.id).status is TaskRuntimeStatus.DONE
        assert len(fixture.coding.calls) == 2
        assert len(fixture.review.calls) == 1
    finally:
        fixture.store.close()


def test_concrete_review_cancellation_resumes_without_replaying_coding(
    tmp_path: Path,
) -> None:
    fixture = _concrete_host_fixture(tmp_path, review_delay_seconds=2)
    cancel = CancellationToken()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            running = executor.submit(
                fixture.service.run,
                fixture.borg.id,
                fixture.generation.id,
                {},
                cancel=cancel,
            )
            for _ in range(200):
                if fixture.review.calls:
                    break
                threading.Event().wait(0.005)
            assert fixture.review.calls
            cancel.cancel()
            cancelled = running.result(timeout=3)

        task = fixture.tasks[0]
        assert cancelled.status is ExecutionRunStatus.CANCELLED
        assert fixture.store.get_task_runtime(task.id).status is (
            TaskRuntimeStatus.PENDING
        )
        attempts = fixture.store.list_agent_attempts(task.id)
        attempts_by_phase = {attempt.phase: attempt for attempt in attempts}
        assert set(attempts_by_phase) == {"coding", "review"}
        assert attempts_by_phase["coding"].status.value == "completed"
        assert attempts_by_phase["review"].status.value == "cancelled"

        fixture.review.queue(
            MockResponse(
                payload={
                    "task_file": ".betterborg-task/task.md",
                    "status": "approved",
                    "summary": "Implementation approved after resume.",
                    "issues_file": "",
                    "findings": [],
                }
            )
        )
        resumed = fixture.service.run(fixture.borg.id, fixture.generation.id, {})

        assert resumed.status is ExecutionRunStatus.COMPLETED, (
            fixture.store.get_task_runtime(task.id).state_reason
        )
        assert fixture.store.get_task_runtime(task.id).status is (
            TaskRuntimeStatus.DONE
        )
        assert len(fixture.coding.calls) == 1
        assert len(fixture.review.calls) == 2
    finally:
        fixture.store.close()


def test_concrete_retry_exhaustion_stops_and_resumes_review(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _concrete_host_fixture(tmp_path)
    cancel = CancellationToken()
    adapter_run = MockAdapter.run

    def exhaust_review(self, spec, *, cancel=None):
        if self is not fixture.review:
            return adapter_run(self, spec, cancel=cancel)
        self.calls.append(spec)
        spec.log_path.write_text("transient retries exhausted\n", encoding="utf-8")
        return AgentResult(
            status=AgentStatus.CANCELLED,
            log_path=spec.log_path,
            error="transient retry exhausted: provider unavailable",
            retryable=True,
        )

    monkeypatch.setattr(MockAdapter, "run", exhaust_review)
    try:
        cancelled = fixture.service.run(
            fixture.borg.id,
            fixture.generation.id,
            {},
            cancel=cancel,
        )

        task = fixture.tasks[0]
        runtime = fixture.store.get_task_runtime(task.id)
        attempts = fixture.store.list_agent_attempts(task.id)
        assert cancel.is_set()
        assert cancelled.status is ExecutionRunStatus.CANCELLED
        assert runtime is not None and runtime.status is TaskRuntimeStatus.PENDING
        assert runtime.resume_phase == "review"
        assert sorted(
            (attempt.phase, attempt.status.value) for attempt in attempts
        ) == [
            ("coding", "completed"),
            ("review", "cancelled"),
        ]

        monkeypatch.setattr(MockAdapter, "run", adapter_run)
        fixture.clock.advance(timedelta(seconds=1))
        resumed = fixture.service.run(fixture.borg.id, fixture.generation.id, {})

        assert resumed.status is ExecutionRunStatus.COMPLETED
        assert fixture.store.get_task_runtime(task.id).status is TaskRuntimeStatus.DONE
        assert len(fixture.coding.calls) == 1
        assert len(fixture.review.calls) == 2
    finally:
        fixture.store.close()


def test_concrete_fix_cancellation_resumes_without_replaying_review(
    tmp_path: Path,
) -> None:
    fixture = _concrete_host_fixture(tmp_path)
    cancel = CancellationToken()
    finding = "feature must include the reviewed fix"

    def commit_fix(spec):
        fixed = spec.cwd / "review-fix.txt"
        fixed.write_text("fixed\n", encoding="utf-8")
        _git(spec.cwd, "add", fixed.name)
        _git(spec.cwd, "commit", "--quiet", "-m", "fix review finding")
        return MockResponse(
            payload={
                "task_file": ".betterborg-task/task.md",
                "status": "completed",
                "summary": "Fixed the review finding.",
                "changed_files": [fixed.name],
                "tests_run": ["integration"],
                "follow_ups": [],
                "blockers": [],
            }
        )

    fixture.review.responses.clear()
    fixture.review.queue(
        MockResponse(
            payload={
                "task_file": ".betterborg-task/task.md",
                "status": "issues_found",
                "summary": "The implementation needs a fix.",
                "issues_file": ".betterborg-task/issues.md",
                "findings": [finding],
            }
        )
    ).queue(
        MockResponse(
            payload={
                "task_file": ".betterborg-task/task.md",
                "status": "completed",
                "summary": "This fix turn will be cancelled.",
                "changed_files": [],
                "tests_run": [],
                "follow_ups": [],
                "blockers": [],
            },
            delay_seconds=2,
        )
    ).queue(MockResponse(dynamic=commit_fix)).queue(
        MockResponse(
            payload={
                "task_file": ".betterborg-task/task.md",
                "status": "approved",
                "summary": "The fix is approved.",
                "issues_file": "",
                "findings": [],
            }
        )
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            running = executor.submit(
                fixture.service.run,
                fixture.borg.id,
                fixture.generation.id,
                {},
                cancel=cancel,
            )
            for _ in range(200):
                if len(fixture.review.calls) >= 2:
                    break
                threading.Event().wait(0.005)
            assert len(fixture.review.calls) == 2
            cancel.cancel()
            cancelled = running.result(timeout=3)

        task = fixture.tasks[0]
        assert cancelled.status is ExecutionRunStatus.CANCELLED
        assert fixture.store.get_task_runtime(task.id).status is (
            TaskRuntimeStatus.PENDING
        )
        cancelled_attempts = fixture.store.list_agent_attempts(task.id)
        assert sorted(
            (attempt.phase, attempt.review_round, attempt.status.value)
            for attempt in cancelled_attempts
        ) == [
            ("coding", 0, "completed"),
            ("fix", 1, "cancelled"),
            ("review", 0, "completed"),
        ]

        fixture.clock.advance(timedelta(seconds=1))
        resumed = fixture.service.run(fixture.borg.id, fixture.generation.id, {})

        assert resumed.status is ExecutionRunStatus.COMPLETED, (
            fixture.store.get_task_runtime(task.id).state_reason
        )
        assert fixture.store.get_task_runtime(task.id).status is (
            TaskRuntimeStatus.DONE
        )
        assert sorted(
            (attempt.phase, attempt.review_round, attempt.status.value)
            for attempt in fixture.store.list_agent_attempts(task.id)
        ) == [
            ("coding", 0, "completed"),
            ("fix", 1, "cancelled"),
            ("fix", 1, "completed"),
            ("review", 0, "completed"),
            ("review", 1, "completed"),
        ]
        assert len(fixture.coding.calls) == 1
        assert len(fixture.review.calls) == 4
        assert finding in fixture.review.calls[2].user_prompt
    finally:
        fixture.store.close()


def test_cancellation_after_fix_resumes_from_fixed_commit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _concrete_host_fixture(tmp_path)
    cancel = CancellationToken()
    coding_attempt_ids = iter([UUID(int=3)])
    review_attempt_ids = iter([UUID(int=2), UUID(int=1), UUID(int=4)])
    monkeypatch.setattr(
        "betterborg_cli.host_execution.coding.uuid4",
        lambda: next(coding_attempt_ids),
    )
    monkeypatch.setattr(
        "betterborg_cli.host_execution.review.uuid4",
        lambda: next(review_attempt_ids),
    )

    def commit_fix_then_cancel(spec):  # noqa: ANN001
        fixed = spec.cwd / "review-fix.txt"
        fixed.write_text("fixed before cancellation\n", encoding="utf-8")
        _git(spec.cwd, "add", fixed.name)
        _git(spec.cwd, "commit", "--quiet", "-m", "fix review finding")
        fixture.review.queue(
            MockResponse(
                payload={
                    "task_file": ".betterborg-task/task.md",
                    "status": "approved",
                    "summary": "The fixed commit is approved after resume.",
                    "issues_file": "",
                    "findings": [],
                }
            )
        )
        cancel.cancel()
        return MockResponse(
            payload={
                "task_file": ".betterborg-task/task.md",
                "status": "completed",
                "summary": "Committed the requested fix.",
                "changed_files": [fixed.name],
                "tests_run": ["integration"],
                "follow_ups": [],
                "blockers": [],
            }
        )

    fixture.review.responses.clear()
    fixture.review.queue(
        MockResponse(
            payload={
                "task_file": ".betterborg-task/task.md",
                "status": "issues_found",
                "summary": "The implementation needs a fix.",
                "issues_file": ".betterborg-task/issues.md",
                "findings": ["commit the reviewed fix"],
            }
        )
    ).queue(MockResponse(dynamic=commit_fix_then_cancel))

    try:
        cancelled = fixture.service.run(
            fixture.borg.id,
            fixture.generation.id,
            {},
            cancel=cancel,
        )

        task = fixture.tasks[0]
        runtime = fixture.store.get_task_runtime(task.id)
        assert cancelled.status is ExecutionRunStatus.CANCELLED
        assert runtime is not None and runtime.status is TaskRuntimeStatus.PENDING
        assert runtime.resume_phase == TaskRuntimeStatus.REVIEW.value
        assert runtime.worktree_path is not None
        fixed_commit = _git(Path(runtime.worktree_path), "rev-parse", "HEAD")
        attempts = fixture.store.list_agent_attempts(task.id)
        assert [attempt.phase for attempt in attempts] == [
            "fix",
            "review",
            "coding",
        ]
        assert sorted(
            (attempt.phase, attempt.review_round, attempt.status.value)
            for attempt in attempts
        ) == [
            ("coding", 0, "completed"),
            ("fix", 1, "completed"),
            ("review", 0, "completed"),
        ]

        fixture.clock.advance(timedelta(seconds=1))
        resumed = fixture.service.run(fixture.borg.id, fixture.generation.id, {})

        assert resumed.status is ExecutionRunStatus.COMPLETED, (
            fixture.store.get_task_runtime(task.id).state_reason
        )
        assert len(fixture.coding.calls) == 1
        assert len(fixture.review.calls) == 3
        approved = [
            attempt
            for attempt in fixture.store.list_agent_attempts(task.id)
            if attempt.phase == "review"
            and attempt.status is ExecutionAttemptStatus.COMPLETED
            and (attempt.result or {}).get("status") == "approved"
        ]
        assert approved[-1].result["_betterborg"]["commit_sha"] == fixed_commit
    finally:
        fixture.store.close()


def test_cancelled_fix_commit_is_not_a_resume_attestation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _concrete_host_fixture(tmp_path)
    cancel = CancellationToken()
    adapter_run = MockAdapter.run
    fixture.review.responses.clear()
    fixture.review.queue(
        MockResponse(
            payload={
                "task_file": ".betterborg-task/task.md",
                "status": "issues_found",
                "summary": "The implementation needs a fix.",
                "issues_file": ".betterborg-task/issues.md",
                "findings": ["commit the reviewed fix"],
            }
        )
    )

    def cancel_after_fix_commit(self, spec, *, cancel=None):  # noqa: ANN001
        if self is not fixture.review or not self.calls:
            return adapter_run(self, spec, cancel=cancel)
        self.calls.append(spec)
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        spec.log_path.write_text("cancelled after fix commit\n", encoding="utf-8")
        changed = spec.cwd / "unattested-fix.txt"
        changed.write_text("must not be trusted\n", encoding="utf-8")
        _git(spec.cwd, "add", changed.name)
        _git(spec.cwd, "commit", "--quiet", "-m", "unattested cancelled fix")
        assert cancel is not None
        cancel.cancel()
        return AgentResult(
            status=AgentStatus.CANCELLED,
            log_path=spec.log_path,
            error="fix cancelled after producing an unattested commit",
        )

    monkeypatch.setattr(MockAdapter, "run", cancel_after_fix_commit)
    try:
        cancelled = fixture.service.run(
            fixture.borg.id,
            fixture.generation.id,
            {},
            cancel=cancel,
        )

        task = fixture.tasks[0]
        runtime = fixture.store.get_task_runtime(task.id)
        assert cancelled.status is ExecutionRunStatus.CANCELLED
        assert runtime is not None and runtime.status is TaskRuntimeStatus.PENDING
        assert runtime.resume_phase == TaskRuntimeStatus.FIX.value
        assert sorted(
            (attempt.phase, attempt.status.value)
            for attempt in fixture.store.list_agent_attempts(task.id)
        ) == [
            ("coding", "completed"),
            ("fix", "cancelled"),
            ("review", "completed"),
        ]

        monkeypatch.setattr(MockAdapter, "run", adapter_run)
        fixture.clock.advance(timedelta(seconds=1))
        resumed = fixture.service.run(fixture.borg.id, fixture.generation.id, {})

        blocked = fixture.store.get_task_runtime(task.id)
        assert resumed.status is ExecutionRunStatus.FAILED
        assert blocked is not None and blocked.status is TaskRuntimeStatus.BLOCKED
        assert blocked.state_reason == (
            "declared coding/fix commit no longer matches task worktree"
        )
        assert len(fixture.coding.calls) == 1
        assert len(fixture.review.calls) == 2
    finally:
        fixture.store.close()


def test_concrete_retry_exhaustion_stops_and_resumes_fix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _concrete_host_fixture(tmp_path)
    cancel = CancellationToken()
    finding = "feature must include the reviewed fix"
    adapter_run = MockAdapter.run

    def commit_fix(spec):
        fixed = spec.cwd / "retry-fix.txt"
        fixed.write_text("fixed after retry exhaustion\n", encoding="utf-8")
        _git(spec.cwd, "add", fixed.name)
        _git(spec.cwd, "commit", "--quiet", "-m", "fix after retry exhaustion")
        return MockResponse(
            payload={
                "task_file": ".betterborg-task/task.md",
                "status": "completed",
                "summary": "Fixed the review finding.",
                "changed_files": [fixed.name],
                "tests_run": ["integration"],
                "follow_ups": [],
                "blockers": [],
            }
        )

    fixture.review.responses.clear()
    fixture.review.queue(
        MockResponse(
            payload={
                "task_file": ".betterborg-task/task.md",
                "status": "issues_found",
                "summary": "The implementation needs a fix.",
                "issues_file": ".betterborg-task/issues.md",
                "findings": [finding],
            }
        )
    ).queue(MockResponse(dynamic=commit_fix)).queue(
        MockResponse(
            payload={
                "task_file": ".betterborg-task/task.md",
                "status": "approved",
                "summary": "The resumed fix is approved.",
                "issues_file": "",
                "findings": [],
            }
        )
    )

    def exhaust_fix(self, spec, *, cancel=None):
        if self is not fixture.review or not self.calls:
            return adapter_run(self, spec, cancel=cancel)
        self.calls.append(spec)
        spec.log_path.write_text("transient retries exhausted\n", encoding="utf-8")
        return AgentResult(
            status=AgentStatus.CANCELLED,
            log_path=spec.log_path,
            error="transient retry exhausted: provider unavailable",
            retryable=True,
        )

    monkeypatch.setattr(MockAdapter, "run", exhaust_fix)
    try:
        cancelled = fixture.service.run(
            fixture.borg.id,
            fixture.generation.id,
            {},
            cancel=cancel,
        )

        task = fixture.tasks[0]
        runtime = fixture.store.get_task_runtime(task.id)
        attempts = fixture.store.list_agent_attempts(task.id)
        assert cancel.is_set()
        assert cancelled.status is ExecutionRunStatus.CANCELLED
        assert runtime is not None and runtime.status is TaskRuntimeStatus.PENDING
        assert runtime.resume_phase == "fix"
        assert sorted(
            (attempt.phase, attempt.status.value) for attempt in attempts
        ) == [
            ("coding", "completed"),
            ("fix", "cancelled"),
            ("review", "completed"),
        ]

        monkeypatch.setattr(MockAdapter, "run", adapter_run)
        fixture.clock.advance(timedelta(seconds=1))
        resumed = fixture.service.run(fixture.borg.id, fixture.generation.id, {})

        assert resumed.status is ExecutionRunStatus.COMPLETED
        assert fixture.store.get_task_runtime(task.id).status is TaskRuntimeStatus.DONE
        assert len(fixture.coding.calls) == 1
        assert len(fixture.review.calls) == 4
    finally:
        fixture.store.close()


def test_concrete_retry_exhaustion_stops_and_resumes_merge(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _concrete_host_fixture(tmp_path)
    cancel = CancellationToken()
    adapter_run = MockAdapter.run
    repository = fixture.store.get_repository(fixture.borg.repository_id)
    assert repository is not None

    def commit_conflicting_task(spec):
        (spec.cwd / "README.md").write_text(
            "# Fixture\n\ntask version\n",
            encoding="utf-8",
        )
        _git(spec.cwd, "add", "README.md")
        _git(spec.cwd, "commit", "--quiet", "-m", "change task readme")
        _advance_project_file(
            repository.root,
            f"project/{fixture.borg.name}",
            "README.md",
            "# Fixture\n\nproject version\n",
            tmp_path / "project-branch.index",
        )
        return MockResponse(
            payload={
                "task_file": ".betterborg-task/task.md",
                "status": "completed",
                "summary": "Created a conflicting task change.",
                "changed_files": ["README.md"],
                "tests_run": ["integration"],
                "follow_ups": [],
                "blockers": [],
            }
        )

    fixture.coding.responses.clear()
    fixture.coding.queue(MockResponse(dynamic=commit_conflicting_task))

    def exhaust_merge(self, spec, *, cancel=None):
        if self is not fixture.merge:
            return adapter_run(self, spec, cancel=cancel)
        self.calls.append(spec)
        spec.log_path.write_text("transient retries exhausted\n", encoding="utf-8")
        return AgentResult(
            status=AgentStatus.CANCELLED,
            log_path=spec.log_path,
            error="transient retry exhausted: provider unavailable",
            retryable=True,
        )

    monkeypatch.setattr(MockAdapter, "run", exhaust_merge)
    try:
        cancelled = fixture.service.run(
            fixture.borg.id,
            fixture.generation.id,
            {},
            cancel=cancel,
        )

        task = fixture.tasks[0]
        runtime = fixture.store.get_task_runtime(task.id)
        attempts = fixture.store.list_agent_attempts(task.id)
        assert cancel.is_set()
        assert cancelled.status is ExecutionRunStatus.CANCELLED
        assert runtime is not None and runtime.status is TaskRuntimeStatus.PENDING
        assert runtime.resume_phase == "merging"
        assert sorted(
            (attempt.phase, attempt.status.value) for attempt in attempts
        ) == [
            ("coding", "completed"),
            ("merge", "cancelled"),
            ("review", "completed"),
        ]

        def resolve_conflict(spec):
            (spec.cwd / "README.md").write_text(
                "# Fixture\n\ntask and project versions\n",
                encoding="utf-8",
            )
            _git(spec.cwd, "add", "README.md")
            _git(spec.cwd, "commit", "--quiet", "-m", "resolve readme conflict")
            return MockResponse(
                payload={
                    "task_file": ".betterborg-task/task.md",
                    "status": "completed",
                    "summary": "Resolved the project conflict.",
                    "changed_files": ["README.md"],
                    "tests_run": ["integration"],
                    "follow_ups": [],
                    "blockers": [],
                }
            )

        monkeypatch.setattr(MockAdapter, "run", adapter_run)
        fixture.merge.queue(MockResponse(dynamic=resolve_conflict))
        fixture.clock.advance(timedelta(seconds=1))
        resumed = fixture.service.run(fixture.borg.id, fixture.generation.id, {})

        assert resumed.status is ExecutionRunStatus.COMPLETED, (
            fixture.store.get_task_runtime(task.id).state_reason
        )
        assert fixture.store.get_task_runtime(task.id).status is TaskRuntimeStatus.DONE
        assert len(fixture.coding.calls) == 1
        assert len(fixture.review.calls) == 1
        assert len(fixture.merge.calls) == 2
        assert sorted(
            (attempt.phase, attempt.status.value)
            for attempt in fixture.store.list_agent_attempts(task.id)
        ) == [
            ("coding", "completed"),
            ("merge", "cancelled"),
            ("merge", "completed"),
            ("review", "completed"),
        ]
    finally:
        fixture.store.close()


def test_cancellation_after_merge_tip_resumes_from_merge_attestation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _concrete_host_fixture(tmp_path)
    cancel = CancellationToken()
    repository = fixture.store.get_repository(fixture.borg.repository_id)
    assert repository is not None

    def commit_task_and_advance_project(spec):  # noqa: ANN001
        response = _coding_response()
        assert response.dynamic is not None
        completed = response.dynamic(spec)
        _advance_project_file(
            repository.root,
            f"project/{fixture.borg.name}",
            "project-base.txt",
            "advanced while task was in progress\n",
            tmp_path / "post-coding-project.index",
        )
        return completed

    fixture.coding.responses.clear()
    fixture.coding.queue(MockResponse(dynamic=commit_task_and_advance_project))
    merge_phase = fixture.service._runtime._merge
    merge_run = merge_phase.run
    produced_tips: list[MergeTip] = []

    def cancel_after_merge_tip(context, **kwargs):  # noqa: ANN001, ANN003
        result = merge_run(context, **kwargs)
        assert result.tip is not None
        produced_tips.append(result.tip)
        cancel.cancel()
        return result

    monkeypatch.setattr(merge_phase, "run", cancel_after_merge_tip)
    try:
        cancelled = fixture.service.run(
            fixture.borg.id,
            fixture.generation.id,
            {},
            cancel=cancel,
        )

        task = fixture.tasks[0]
        runtime = fixture.store.get_task_runtime(task.id)
        assert cancelled.status is ExecutionRunStatus.CANCELLED
        assert runtime is not None and runtime.status is TaskRuntimeStatus.PENDING
        assert runtime.resume_phase == TaskRuntimeStatus.MERGING.value
        assert runtime.worktree_path is not None
        assert len(produced_tips) == 1
        tip = produced_tips[0]
        assert tip.commit_sha != tip.approved_commit
        assert _git(Path(runtime.worktree_path), "rev-parse", "HEAD") == tip.commit_sha
        completed_merges = fixture.store.list_task_execution_events(
            task.id,
            kind="merge.completed",
        )
        assert len(completed_merges) == 1
        assert completed_merges[0].payload["commit_sha"] == tip.commit_sha

        monkeypatch.setattr(merge_phase, "run", merge_run)
        fixture.clock.advance(timedelta(seconds=1))
        resumed = fixture.service.run(fixture.borg.id, fixture.generation.id, {})

        assert resumed.status is ExecutionRunStatus.COMPLETED, (
            fixture.store.get_task_runtime(task.id).state_reason
        )
        assert len(fixture.coding.calls) == 1
        assert len(fixture.review.calls) == 1
        assert fixture.merge.calls == []
        assert (
            _git(
                repository.root,
                "rev-parse",
                f"project/{fixture.borg.name}",
            )
            == tip.commit_sha
        )
        assert (
            len(
                fixture.store.list_task_execution_events(
                    task.id,
                    kind="merge.completed",
                )
            )
            == 1
        )
    finally:
        fixture.store.close()


def test_cancellation_during_base_advance_preserves_durable_attestation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _concrete_host_fixture(tmp_path)
    cancel = CancellationToken()
    advancing = threading.Event()
    release_advance = threading.Event()
    append_event = fixture.store.append_claim_execution_event

    def pause_after_fast_forward(event, owner_token, claim_token, *, now=None):
        if event.kind == "base.advanced":
            advancing.set()
            assert release_advance.wait(timeout=2)
        return append_event(
            event,
            owner_token,
            claim_token,
            now=now,
        )

    monkeypatch.setattr(
        fixture.store,
        "append_claim_execution_event",
        pause_after_fast_forward,
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            running = executor.submit(
                fixture.service.run,
                fixture.borg.id,
                fixture.generation.id,
                {},
                cancel=cancel,
            )
            assert advancing.wait(timeout=3)
            cancel.cancel()
            fixture.clock.advance(timedelta(minutes=1))
            # Give the scheduler multiple poll intervals to observe cancellation.
            # It must retain ownership until publication records its attestation.
            for _ in range(100):
                active_run = fixture.store.list_execution_runs(fixture.borg.id)[-1]
                if active_run.heartbeat_at == fixture.clock.now:
                    break
                threading.Event().wait(0.005)
            assert active_run.status is ExecutionRunStatus.RUNNING
            assert active_run.heartbeat_at == fixture.clock.now
            release_advance.set()
            cancelled = running.result(timeout=3)

        task = fixture.tasks[0]
        runtime = fixture.store.get_task_runtime(task.id)
        assert cancelled.status is ExecutionRunStatus.CANCELLED
        assert runtime is not None and runtime.status is TaskRuntimeStatus.DONE
        advanced = fixture.store.list_task_execution_events(
            task.id,
            kind="base.advanced",
        )
        assert len(advanced) == 1
        assert advanced[0].payload["commit_sha"] == _git(
            fixture.store.get_repository(fixture.borg.repository_id).root,
            "rev-parse",
            f"project/{fixture.borg.name}",
        )

        resumed = fixture.service.run(fixture.borg.id, fixture.generation.id, {})

        assert resumed.status is ExecutionRunStatus.COMPLETED
        assert len(fixture.coding.calls) == 1
        assert len(fixture.review.calls) == 1
        assert (
            len(fixture.store.list_task_execution_events(task.id, kind="base.advanced"))
            == 1
        )
    finally:
        release_advance.set()
        fixture.store.close()


@dataclass
class _ConcurrentRuntime:
    plan: HostPreflightPlan

    def with_secret_values(self, secret_values):
        return self

    def __call__(self, context) -> TaskRuntimeStatus:
        context.transition(TaskRuntimeStatus.CLAIMED, TaskRuntimeStatus.DONE)
        return TaskRuntimeStatus.DONE


def test_concrete_jobs_two_and_duplicate_callers_share_one_operation(
    tmp_path: Path,
) -> None:
    fixture = _concrete_host_fixture(tmp_path, task_count=2)
    release = threading.Event()

    def commit_after_duplicate_call(spec):
        assert release.wait(timeout=2)
        response = _coding_response()
        assert response.dynamic is not None
        return response.dynamic(spec)

    fixture.coding.responses.clear()
    for _ in fixture.tasks:
        fixture.coding.queue(MockResponse(dynamic=commit_after_duplicate_call))
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            owner = executor.submit(
                fixture.service.run,
                fixture.borg.id,
                fixture.generation.id,
                {},
            )
            for _ in range(200):
                if len(fixture.coding.calls) == 2:
                    break
                threading.Event().wait(0.005)
            assert len(fixture.coding.calls) == 2

            duplicate = fixture.service.run(
                fixture.borg.id,
                fixture.generation.id,
                {},
            )
            release.set()
            assert duplicate.acquired is False
            assert duplicate.active_operation_id is not None
            assert duplicate.status is ExecutionRunStatus.RUNNING
            completed = owner.result(timeout=3)

        assert completed.status is ExecutionRunStatus.COMPLETED
        assert completed.operation_id == duplicate.operation_id
        assert len(fixture.store.list_task_claims(completed.operation_id)) == 2
        assert len(fixture.coding.calls) == 2
        assert len(fixture.review.calls) == 2
        assert len(fixture.compose.up_projects) == 4
        assert sorted(fixture.compose.up_projects) == sorted(
            fixture.compose.down_projects
        )
        assert fixture.compose.active == set()
    finally:
        release.set()
        fixture.store.close()


def test_preflight_block_prevents_run_acquisition(tmp_path: Path) -> None:
    store, borg, generation, _ = _store_fixture(tmp_path)
    calls: list[str] = []
    block = HostPreflightBlock(
        (HostPreflightFailure("trusted workspace", "missing", "trust it"),)
    )
    try:
        result = HostExecutionService(
            store,
            _Preflight(block, calls),
            _ConcurrentRuntime(_plan(tmp_path)),
            worktree_manager=_Worktrees(calls),
            compose_manager=_Compose(calls),
        ).run(borg.id, generation.id, {})

        assert result.preflight is block
        assert result.operation_id is None
        assert store.list_execution_runs(borg.id) == []
        assert calls == ["preflight"]
    finally:
        store.close()


def test_cancelled_preflight_propagates_before_run_acquisition(tmp_path: Path) -> None:
    store, borg, generation, _ = _store_fixture(tmp_path)
    calls: list[str] = []
    cancel = CancellationToken()

    class CancelledPreflight:
        def validate(self, *args, **kwargs):
            calls.append("preflight")
            cancel.cancel()
            raise KeyboardInterrupt

    try:
        with pytest.raises(KeyboardInterrupt):
            HostExecutionService(
                store,
                CancelledPreflight(),
                _ConcurrentRuntime(_plan(tmp_path)),
                worktree_manager=_Worktrees(calls),
                compose_manager=_Compose(calls),
            ).run(borg.id, generation.id, {}, cancel=cancel)

        assert store.list_execution_runs(borg.id) == []
        assert calls == ["preflight"]
    finally:
        store.close()


def test_concrete_blocked_task_cleans_services_and_preserves_worktree(
    tmp_path: Path,
) -> None:
    fixture = _concrete_host_fixture(tmp_path)

    def leave_unfinished_work(spec):
        unfinished = spec.cwd / "unfinished.txt"
        unfinished.write_text("preserve me\n", encoding="utf-8")
        return MockResponse(
            payload={
                "task_file": ".betterborg-task/task.md",
                "status": "completed",
                "summary": "Work is unfinished.",
                "changed_files": [unfinished.name],
                "tests_run": [],
                "follow_ups": [],
                "blockers": [],
            }
        )

    fixture.coding.responses.clear()
    fixture.coding.queue(MockResponse(dynamic=leave_unfinished_work))
    try:
        result = fixture.service.run(
            fixture.borg.id,
            fixture.generation.id,
            {},
        )

        assert result.status is ExecutionRunStatus.FAILED
        blocked = fixture.store.get_task_runtime(fixture.tasks[0].id)
        assert blocked is not None and blocked.status is TaskRuntimeStatus.BLOCKED
        assert "without producing a commit" in blocked.state_reason
        assert blocked.worktree_path is not None
        worktree = Path(blocked.worktree_path)
        assert (worktree / "unfinished.txt").read_text(encoding="utf-8") == (
            "preserve me\n"
        )
        assert "?? unfinished.txt" in _git(worktree, "status", "--porcelain")
        assert len(fixture.store.list_task_claims(result.operation_id)) == 1
        assert len(fixture.coding.calls) == 1
        assert fixture.review.calls == []
        assert len(fixture.compose.up_projects) == 1
        assert fixture.compose.up_projects == fixture.compose.down_projects
        assert fixture.compose.active == set()

        resumed = fixture.service.run(
            fixture.borg.id,
            fixture.generation.id,
            {},
        )

        assert resumed.status is ExecutionRunStatus.FAILED
        assert fixture.store.get_task_runtime(fixture.tasks[0].id) == blocked
        assert len(fixture.coding.calls) == 1
        assert len(fixture.compose.up_projects) == 1
        assert fixture.store.list_task_claims(resumed.operation_id) == []
    finally:
        fixture.store.close()


def test_setup_heartbeats_keep_the_execution_lease_owned(tmp_path: Path) -> None:
    store, borg, generation, _ = _store_fixture(tmp_path)
    calls: list[str] = []
    plan = _plan(tmp_path)
    clock = FakeClock()

    class SlowWorktrees(_Worktrees):
        def prepare_current_task_worktrees(self, *args, **kwargs):
            result = super().prepare_current_task_worktrees(*args, **kwargs)
            clock.advance(timedelta(seconds=30))
            return result

    try:
        result = HostExecutionService(
            store,
            _Preflight(plan, calls),
            _ConcurrentRuntime(plan),
            worktree_manager=SlowWorktrees(calls),
            compose_manager=_Compose(calls),
            scheduler_config=HostSchedulerConfig(
                lease_duration=timedelta(seconds=10),
                heartbeat_interval=timedelta(seconds=2),
                poll_interval_seconds=0.005,
            ),
            clock=clock,
        ).run(borg.id, generation.id, {})

        assert result.status is ExecutionRunStatus.COMPLETED
    finally:
        store.close()


def test_acquisition_expiry_cleanup_precedes_new_task_dispatch(
    tmp_path: Path,
) -> None:
    store, borg, generation, records = _store_fixture(tmp_path)
    calls: list[str] = []
    plan = _plan(tmp_path)
    start = datetime(2026, 8, 26, 12, tzinfo=UTC)
    expired_at = start + timedelta(seconds=2)
    previous = store.acquire_execution_run(
        borg.id,
        generation.id,
        lease_duration=timedelta(seconds=1),
        now=start,
    )
    assert previous.owner_token is not None
    previous_claim = store.claim_dependency_ready_task(
        previous.run_id,
        previous.owner_token,
        lease_duration=timedelta(seconds=1),
        now=start,
    )
    assert previous_claim is not None
    resource = ComposeResource(
        run_id=previous.run_id,
        claim_id=previous_claim.id,
        task_id=records[0].id,
        project_name="expired-between-reconcile-and-acquire",
        resource_type="project",
        resource_name="expired-between-reconcile-and-acquire",
        created_at=start,
    )
    store.add_compose_resource(
        resource,
        previous.owner_token,
        previous_claim.claim_token,
        now=start,
    )

    class ExpiryRaceClock:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> datetime:
            self.calls += 1
            return start if self.calls == 1 else expired_at

    class CleanupCompose(_Compose):
        def cleanup_stale_projects(self, cleanup_store, resources):
            self.calls.append("stale-cleanup")
            for stale in resources:
                cleanup_store.confirm_compose_project_cleanup(
                    stale.run_id,
                    stale.task_id,
                    stale.project_name,
                    now=expired_at,
                )
            return ()

    clock = ExpiryRaceClock()
    compose = CleanupCompose(calls)
    try:
        result = HostExecutionService(
            store,
            _Preflight(plan, calls),
            _ConcurrentRuntime(plan),
            worktree_manager=_Worktrees(calls),
            compose_manager=compose,
            clock=clock,
        ).run(borg.id, generation.id, {})

        assert result.status is ExecutionRunStatus.COMPLETED
        assert calls.index("stale-cleanup") < calls.index("worktrees")
        assert store.list_stale_compose_resources(previous.run_id) == []
        assert store.list_task_claims(previous.run_id)[0].released_at == expired_at
    finally:
        store.close()


def test_reusable_cache_preparation_precedes_task_dispatch(tmp_path: Path) -> None:
    store, borg, generation, records = _store_fixture(tmp_path)
    calls: list[str] = []
    worktree = tmp_path / "prepared-worktree"
    worktree.mkdir()
    plan = HostPreflightPlan(
        repository_root=tmp_path / "repository",
        commands=(),
        prepare_commands=(HostCommand("prepare", ("prepare",), "."),),
        materialize_commands=(),
        environment_files=(),
        executables=(),
        required_secret_names=(),
        compose_files=(),
        services=(),
    )

    class PreparedWorktrees(_Worktrees):
        def prepare_current_task_worktrees(self, *args, **kwargs):
            super().prepare_current_task_worktrees(*args, **kwargs)
            return [SimpleNamespace(task_id=records[0].id, path=worktree)]

    @dataclass
    class PreparedRuntime(_ConcurrentRuntime):
        def prepare_reusable_caches(
            self, store, run_id, owner_token, worktrees, *, secret_values
        ):
            assert tuple(worktrees) == ((records[0].id, worktree),)
            calls.append("cache")
            return ("fingerprint",)

        def __call__(self, context) -> TaskRuntimeStatus:
            calls.append("dispatch")
            return super().__call__(context)

    try:
        result = HostExecutionService(
            store,
            _Preflight(plan, calls),
            PreparedRuntime(plan),
            worktree_manager=PreparedWorktrees(calls),
            compose_manager=_Compose(calls),
        ).run(borg.id, generation.id, {})

        assert result.status is ExecutionRunStatus.COMPLETED
        assert calls[:4] == ["preflight", "worktrees", "cache", "dispatch"]
    finally:
        store.close()


def test_cancelled_service_resumes_only_unfinished_tasks(tmp_path: Path) -> None:
    store, borg, generation, records = _store_fixture(tmp_path, task_count=2)
    calls: list[str] = []
    plan = _plan(tmp_path)
    cancel = CancellationToken()
    second_started = threading.Event()
    invocations: list[str] = []

    @dataclass
    class CancellingRuntime:
        plan: HostPreflightPlan

        def with_secret_values(self, secret_values):
            return self

        def __call__(self, context) -> TaskRuntimeStatus:
            task_id = context.claim.task_id
            invocations.append(str(task_id))
            if task_id == records[1].id:
                second_started.set()
                context.cancel.wait(timeout=2)
                return TaskRuntimeStatus.DONE
            context.transition(TaskRuntimeStatus.CLAIMED, TaskRuntimeStatus.DONE)
            return TaskRuntimeStatus.DONE

    try:
        first_service = HostExecutionService(
            store,
            _Preflight(plan, calls),
            CancellingRuntime(plan),
            worktree_manager=_Worktrees(calls),
            compose_manager=_Compose(calls),
            scheduler_config=HostSchedulerConfig(poll_interval_seconds=0.005),
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            running = executor.submit(
                first_service.run,
                borg.id,
                generation.id,
                {},
                cancel=cancel,
            )
            assert second_started.wait(timeout=2)
            cancel.cancel()
            cancelled = running.result(timeout=2)

        assert cancelled.status is ExecutionRunStatus.CANCELLED
        assert store.get_task_runtime(records[0].id).status is TaskRuntimeStatus.DONE
        assert store.get_task_runtime(records[1].id).status is (
            TaskRuntimeStatus.PENDING
        )

        resumed_ids: list[str] = []

        @dataclass
        class ResumeRuntime:
            plan: HostPreflightPlan

            def with_secret_values(self, secret_values):
                return self

            def __call__(self, context) -> TaskRuntimeStatus:
                resumed_ids.append(str(context.claim.task_id))
                context.transition(TaskRuntimeStatus.CLAIMED, TaskRuntimeStatus.DONE)
                return TaskRuntimeStatus.DONE

        resumed = HostExecutionService(
            store,
            _Preflight(plan, calls),
            ResumeRuntime(plan),
            worktree_manager=_Worktrees(calls),
            compose_manager=_Compose(calls),
        ).run(borg.id, generation.id, {})

        assert resumed.status is ExecutionRunStatus.COMPLETED
        assert resumed_ids == [str(records[1].id)]
        assert len(invocations) == 2
    finally:
        store.close()
