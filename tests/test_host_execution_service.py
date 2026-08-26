"""Integration contracts for the concrete host execution assembly."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

from betterborg_cli.agent_runtime import CancellationToken, MockAdapter, MockResponse
from betterborg_cli.host_execution import (
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
    ) -> None:
        self.calls = calls
        self.expected_environment = expected_environment

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
        self.calls.append("sanity")
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


def _concrete_task(
    generation_id: UUID,
    borg: Borg,
    position: int,
    *,
    dependencies: tuple[str, ...] = (),
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
        position=position,
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
    compose: _ConcreteComposeRunner
    clock: _FakeClock


class _ConcreteComposeRunner:
    def __init__(self) -> None:
        self.active: set[str] = set()
        self.up_projects: list[str] = []
        self.down_projects: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, argv, **kwargs):
        command = tuple(argv)
        project = command[command.index("--project-name") + 1]
        with self._lock:
            if "config" in command:
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    json.dumps(
                        {
                            "services": {"healthy": {"networks": {"default": None}}},
                            "networks": {"default": {}},
                        }
                    ),
                    "",
                )
            if "up" in command:
                self.active.add(project)
                self.up_projects.append(project)
                return subprocess.CompletedProcess(argv, 0, "started\n", "")
            if "ps" in command:
                records = (
                    [
                        {
                            "Service": "healthy",
                            "State": "running",
                            "Health": "healthy",
                        }
                    ]
                    if project in self.active
                    else []
                )
                return subprocess.CompletedProcess(argv, 0, json.dumps(records), "")
            if "port" in command:
                port = 41000 + sum(project.encode()) % 20000
                return subprocess.CompletedProcess(argv, 0, f"127.0.0.1:{port}\n", "")
            if "down" in command:
                self.active.discard(project)
                self.down_projects.append(project)
                return subprocess.CompletedProcess(argv, 0, "stopped\n", "")
        return subprocess.CompletedProcess(argv, 2, "", "unexpected command")


def _concrete_host_fixture(
    tmp_path: Path,
    *,
    task_count: int = 1,
    coding_delay_seconds: float = 0,
    review_delay_seconds: float = 0,
    dependency_chain: bool = False,
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
            for task in tasks
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

    clock = _FakeClock()
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
    compose_runner = _ConcreteComposeRunner()
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
            MockAdapter(),
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
        compose_runner,
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
            "sanity",
        ]
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
        assert len(fixture.compose.up_projects) == 2
        assert sorted(fixture.compose.up_projects) == sorted(
            fixture.compose.down_projects
        )
        assert len(set(fixture.compose.up_projects)) == 2
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
            for _ in range(200):
                if fixture.coding.calls:
                    break
                threading.Event().wait(0.005)
            assert fixture.coding.calls
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

        fixture.coding.queue(_coding_response())
        resumed = fixture.service.run(fixture.borg.id, fixture.generation.id, {})

        assert resumed.status is ExecutionRunStatus.COMPLETED, (
            fixture.store.get_task_runtime(task.id).state_reason
        )
        assert fixture.store.get_task_runtime(task.id).status is (
            TaskRuntimeStatus.DONE
        )
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
    started: threading.Barrier | None = None
    release: threading.Event | None = None
    block: bool = False

    def with_secret_values(self, secret_values):
        return self

    def __call__(self, context) -> TaskRuntimeStatus:
        if self.started is not None:
            self.started.wait(timeout=2)
        if self.release is not None:
            self.release.wait(timeout=2)
        outcome = TaskRuntimeStatus.BLOCKED if self.block else TaskRuntimeStatus.DONE
        context.transition(TaskRuntimeStatus.CLAIMED, outcome)
        return outcome


def test_service_jobs_two_and_duplicate_callers_share_one_operation(
    tmp_path: Path,
) -> None:
    store, borg, generation, _ = _store_fixture(tmp_path, task_count=2)
    calls: list[str] = []
    plan = _plan(tmp_path)
    release = threading.Event()
    service = HostExecutionService(
        store,
        _Preflight(plan, calls),
        _ConcurrentRuntime(plan, threading.Barrier(2), release),
        worktree_manager=_Worktrees(calls),
        compose_manager=_Compose(calls),
        scheduler_config=HostSchedulerConfig(jobs=2, poll_interval_seconds=0.005),
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            owner = executor.submit(service.run, borg.id, generation.id, {})
            while not store.list_execution_runs(borg.id):
                pass
            duplicate = service.run(borg.id, generation.id, {})
            assert duplicate.acquired is False
            assert duplicate.active_operation_id is not None
            release.set()
            completed = owner.result(timeout=2)

        assert completed.status is ExecutionRunStatus.COMPLETED
        assert completed.operation_id == duplicate.operation_id
        assert len(store.list_task_claims(completed.operation_id)) == 2
    finally:
        store.close()


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


def test_blocked_task_finishes_run_without_reclaim(tmp_path: Path) -> None:
    store, borg, generation, records = _store_fixture(tmp_path)
    calls: list[str] = []
    plan = _plan(tmp_path)
    try:
        result = HostExecutionService(
            store,
            _Preflight(plan, calls),
            _ConcurrentRuntime(plan, block=True),
            worktree_manager=_Worktrees(calls),
            compose_manager=_Compose(calls),
        ).run(borg.id, generation.id, {})

        assert result.status is ExecutionRunStatus.FAILED
        assert store.get_task_runtime(records[0].id).status is (
            TaskRuntimeStatus.BLOCKED
        )
        assert len(store.list_task_claims(result.operation_id)) == 1
    finally:
        store.close()


class _FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 26, 12, tzinfo=UTC)
        self.lock = threading.Lock()

    def __call__(self) -> datetime:
        with self.lock:
            return self.now

    def advance(self, delta: timedelta) -> None:
        with self.lock:
            self.now += delta


def test_setup_heartbeats_keep_the_execution_lease_owned(tmp_path: Path) -> None:
    store, borg, generation, _ = _store_fixture(tmp_path)
    calls: list[str] = []
    plan = _plan(tmp_path)
    clock = _FakeClock()

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
    store, borg, generation, _ = _store_fixture(tmp_path)
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
            return [SimpleNamespace(path=worktree)]

    @dataclass
    class PreparedRuntime(_ConcurrentRuntime):
        def prepare_reusable_caches(self, worktrees, *, secret_values):
            assert tuple(worktrees) == (worktree,)
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
