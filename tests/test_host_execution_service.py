"""Integration contracts for the concrete host execution assembly."""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from betterborg_cli.agent_runtime import CancellationToken
from betterborg_cli.host_execution import (
    HostExecutionService,
    HostMergeResult,
    HostPreflightBlock,
    HostPreflightFailure,
    HostPreflightPlan,
    HostSanityResult,
    HostSchedulerConfig,
    HostTaskRuntime,
    MergeTip,
)
from betterborg_cli.store import (
    Borg,
    ExecutionRunStatus,
    PlanApproval,
    Repository,
    SqliteStore,
    TaskBatch,
    TaskComplexity,
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
    store._promote_published_task_generation(
        generation.id, durable_root=durable_root
    )
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
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def run(
        self,
        context,
        *,
        environment=None,
    ) -> TaskRuntimeStatus:
        assert environment == {
            "CACHE": "prepared",
            "SERVICE_URL": "http://127.0.0.1",
        }
        self.calls.append("coding")
        context.transition(TaskRuntimeStatus.CODING, TaskRuntimeStatus.REVIEW)
        return TaskRuntimeStatus.REVIEW


class _Review:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def run(
        self,
        context,
        *,
        environment=None,
        review_environment=None,
        fix_environment=None,
    ) -> TaskRuntimeStatus:
        assert environment["SERVICE_URL"] == "http://127.0.0.1"
        self.calls.append("review")
        context.transition(TaskRuntimeStatus.REVIEW, TaskRuntimeStatus.MERGING)
        return TaskRuntimeStatus.MERGING


class _Merge:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def run(self, context, *, environment=None) -> HostMergeResult:
        self.calls.append("merge")
        return HostMergeResult(
            TaskRuntimeStatus.MERGING,
            "merged",
            MergeTip("task", "project/Integration", "a", "b", "c", False),
        )


class _Sanity:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def run(self, context, tip, *, secret_values=None) -> HostSanityResult:
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
        outcome = (
            TaskRuntimeStatus.BLOCKED if self.block else TaskRuntimeStatus.DONE
        )
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
                context.transition(
                    TaskRuntimeStatus.CLAIMED, TaskRuntimeStatus.DONE
                )
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
