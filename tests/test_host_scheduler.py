"""Behavior contracts for the lease-backed host task scheduler."""

from __future__ import annotations

import hashlib
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path

import pytest

from betterborg_cli.agent_runtime import AgentStatus, BillingMode, CancellationToken
from betterborg_cli.host_execution import (
    ActivitySink,
    HostSchedulerConfig,
    HostTaskScheduler,
    ScheduledTaskContext,
    TaskActivitySink,
)
from betterborg_cli.progress import (
    AgentActivity,
    AgentActivityKind,
    RunProgress,
    StageRecord,
    StageSpec,
    StageState,
)
from betterborg_cli.run_control import RunControl
from betterborg_cli.store import (
    AgentAttempt,
    Borg,
    ExecutionOwnershipError,
    ExecutionRunStatus,
    PlanApproval,
    Repository,
    SqliteStore,
    TaskBatch,
    TaskComplexity,
    TaskDependency,
    TaskGeneration,
    TaskRecord,
    TaskRuntimeStatus,
)


class FakeClock:
    """Small thread-safe controllable UTC clock."""

    def __init__(self) -> None:
        self._now = datetime(2026, 8, 26, 12, tzinfo=UTC)
        self._lock = threading.Lock()

    def __call__(self) -> datetime:
        with self._lock:
            return self._now

    @property
    def now(self) -> datetime:
        """Return the current instant for assertions."""
        return self()

    def advance(self, delta: timedelta) -> None:
        with self._lock:
            self._now += delta


def _scheduler_fixture(
    tmp_path: Path,
    *,
    task_refs: tuple[str, ...],
    dependencies: tuple[tuple[str, str], ...],
) -> tuple[Path, Borg, TaskGeneration, dict[str, TaskRecord]]:
    database = tmp_path / "scheduler.sqlite3"
    repository = Repository(root=tmp_path / "repository")
    borg = Borg(repository_id=repository.id, name="Scheduler")
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
        manifest={"tasks": list(task_refs)},
    )
    generation = TaskGeneration(
        borg_id=borg.id,
        plan_approval_id=approval.id,
        batch_id=batch.id,
        digest="sha256:generation",
        manifest={"tasks": list(task_refs)},
    )
    records: dict[str, TaskRecord] = {}
    for position, task_ref in enumerate(task_refs, start=1):
        digest = f"sha256:{hashlib.sha256(task_ref.encode()).hexdigest()}"
        records[task_ref] = TaskRecord(
            generation_id=generation.id,
            borg_id=borg.id,
            task_ref=task_ref,
            stage="07-host-execution",
            stem=f"{position:02d}-{task_ref}",
            position=position,
            title=f"Implement {task_ref}",
            complexity=TaskComplexity.SMALL,
            digest=digest,
            task={"acceptance_criteria": [f"{task_ref} works"]},
            manifest={"task.md": digest},
        )
    edges = [
        TaskDependency(
            generation_id=generation.id,
            task_id=records[task_ref].id,
            depends_on_task_id=records[dependency_ref].id,
        )
        for task_ref, dependency_ref in dependencies
    ]
    durable_root = repository.root / ".borg/tasks" / borg.name / str(generation.id)

    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.append_plan_approval(approval)
        store.append_task_batch(batch)
        store.add_task_generation(generation, list(records.values()), edges)
        for record in records.values():
            path = durable_root / record.stage / f"{record.stem}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(record.task_ref, encoding="utf-8")
        store._promote_published_task_generation(
            generation.id, durable_root=durable_root
        )
    return database, borg, generation, records


def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


def test_scheduler_limits_jobs_renews_claims_and_reports_active_operation(
    tmp_path: Path,
) -> None:
    database, borg, generation, records = _scheduler_fixture(
        tmp_path,
        task_refs=("foundation-a", "foundation-b", "consumer-a", "consumer-b"),
        dependencies=(
            ("consumer-a", "foundation-a"),
            ("consumer-b", "foundation-b"),
        ),
    )
    clock = FakeClock()
    release_foundations = threading.Event()
    foundations_started = threading.Event()
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    started: list[str] = []

    with SqliteStore.open(database) as owner_store, SqliteStore.open(
        database
    ) as observer_store:

        def behavior(context: ScheduledTaskContext) -> TaskRuntimeStatus:
            nonlocal active, maximum_active
            task_ref = next(
                ref
                for ref, record in records.items()
                if record.id == context.claim.task_id
            )
            if task_ref.startswith("consumer"):
                dependency_ref = task_ref.replace("consumer", "foundation")
                dependency = owner_store.get_task_runtime(records[dependency_ref].id)
                assert dependency is not None
                assert dependency.status is TaskRuntimeStatus.DONE
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
                started.append(task_ref)
                if sum(ref.startswith("foundation") for ref in started) == 2:
                    foundations_started.set()
            if task_ref.startswith("foundation"):
                assert release_foundations.wait(timeout=2)
            with lock:
                active -= 1
            return TaskRuntimeStatus.DONE

        config = HostSchedulerConfig(
            jobs=2,
            lease_duration=timedelta(seconds=30),
            heartbeat_interval=timedelta(seconds=10),
            poll_interval_seconds=0.005,
        )
        scheduler = HostTaskScheduler(
            owner_store, behavior, config=config, clock=clock
        )

        def unexpected_behavior(
            _context: ScheduledTaskContext,
        ) -> TaskRuntimeStatus:
            raise AssertionError("duplicate caller must not run tasks")

        with ThreadPoolExecutor(max_workers=1) as executor:
            running = executor.submit(scheduler.run, borg.id, generation.id)
            assert foundations_started.wait(timeout=2)
            run = observer_store.list_execution_runs(borg.id)[0]
            first_claims = observer_store.list_task_claims(run.id)
            assert {claim.task_id for claim in first_claims} == {
                records["foundation-a"].id,
                records["foundation-b"].id,
            }

            clock.advance(timedelta(seconds=11))
            _wait_until(
                lambda: any(
                    event.kind == "run.lease_renewed"
                    for event in observer_store.list_execution_events(run.id)
                )
            )
            renewed_claims = observer_store.list_task_claims(run.id)
            assert all(
                claim.lease_expires_at == clock() + timedelta(seconds=30)
                for claim in renewed_claims
            )

            duplicate = HostTaskScheduler(
                observer_store,
                unexpected_behavior,
                config=config,
                clock=clock,
            ).run(borg.id, generation.id)
            assert duplicate.acquired is False
            assert duplicate.active_operation_id == run.id
            assert duplicate.status is ExecutionRunStatus.RUNNING

            release_foundations.set()
            result = running.result(timeout=2)

        assert result.operation_id == run.id
        assert result.status is ExecutionRunStatus.COMPLETED
        assert result.done == result.total == 4
        assert maximum_active == 2
        assert set(started[:2]) == {"foundation-a", "foundation-b"}
        claims = owner_store.list_task_claims(run.id)
        assert len(claims) == 4
        assert {claim.run_id for claim in claims} == {run.id}


def test_scheduler_exposes_only_one_task_bound_activity_sink(tmp_path: Path) -> None:
    database, borg, generation, records = _scheduler_fixture(
        tmp_path,
        task_refs=("task",),
        dependencies=(),
    )
    reported: list[tuple[object, AgentActivity]] = []

    def report(task_id, activity: AgentActivity) -> None:  # noqa: ANN001
        reported.append((task_id, activity))

    task_reporter: TaskActivitySink = report

    with SqliteStore.open(database) as store:

        def behavior(context: ScheduledTaskContext) -> TaskRuntimeStatus:
            assert context.activity is context.activity_sink
            assert context.activity_sink is context.agent_activity_sink
            assert context.activity is not None
            activity_sink: ActivitySink = context.activity
            activity_sink(
                AgentActivity(AgentActivityKind.COMMAND, "bound activity")
            )
            assert "report" not in repr(context)
            context.transition(TaskRuntimeStatus.CLAIMED, TaskRuntimeStatus.DONE)
            return TaskRuntimeStatus.DONE

        result = HostTaskScheduler(store, behavior, activity=task_reporter).run(
            borg.id, generation.id
        )

    assert result.status is ExecutionRunStatus.COMPLETED
    assert reported == [
        (
            records["task"].id,
            AgentActivity(AgentActivityKind.COMMAND, "bound activity"),
        )
    ]


def test_scheduler_seeds_durable_rerun_before_claiming_pending_work(
    tmp_path: Path,
) -> None:
    database, borg, generation, records = _scheduler_fixture(
        tmp_path,
        task_refs=("done", "failed", "blocked", "pending"),
        dependencies=(),
    )
    clock = FakeClock()
    cancel = CancellationToken()
    durations = {"done": 1.5, "failed": 2.5, "blocked": 3.5}

    with SqliteStore.open(database) as store:

        def establish_durable_rows(
            context: ScheduledTaskContext,
        ) -> TaskRuntimeStatus:
            task_ref = next(
                ref
                for ref, record in records.items()
                if record.id == context.claim.task_id
            )
            assert task_ref != "pending"
            attempt = AgentAttempt(
                run_id=context.claim.run_id,
                claim_id=context.claim.id,
                task_id=context.claim.task_id,
                phase="coding",
                attempt_number=1,
                adapter="test",
                model="test-model",
                billing_mode=BillingMode.SUBSCRIPTION,
                status=AgentStatus.COMPLETED,
                log_path=f"artifacts/{task_ref}.log",
                duration_seconds=durations[task_ref],
                started_at=clock(),
                finished_at=clock(),
            )
            store.append_agent_attempt(
                attempt,
                context.owner_token,
                context.claim.claim_token,
                now=clock(),
            )
            status = {
                "done": TaskRuntimeStatus.DONE,
                "failed": TaskRuntimeStatus.FAILED,
                "blocked": TaskRuntimeStatus.BLOCKED,
            }[task_ref]
            reason = {
                "done": None,
                "failed": "durable failure",
                "blocked": "durable block",
            }[task_ref]
            context.transition(
                TaskRuntimeStatus.CLAIMED,
                status,
                state_reason=reason,
            )
            if task_ref == "blocked":
                cancel.cancel()
            return status

        first = HostTaskScheduler(
            store,
            establish_durable_rows,
            config=HostSchedulerConfig(poll_interval_seconds=0.005),
            clock=clock,
        ).run(borg.id, generation.id, cancel=cancel)
        assert first.status is ExecutionRunStatus.CANCELLED
        assert (first.done, first.failed, first.blocked, first.pending) == (1, 1, 1, 1)

        progress = RunProgress(stream=StringIO(), enabled=False)
        claimed: list[str] = []

        def resume_pending(context: ScheduledTaskContext) -> TaskRuntimeStatus:
            retained = [
                progress.stages[str(records[ref].id)]
                for ref in ("done", "failed", "blocked")
            ]
            assert [stage.state for stage in retained] == [
                StageState.COMPLETED,
                StageState.FAILED,
                StageState.FAILED,
            ]
            assert [stage.duration_seconds for stage in retained] == [1.5, 2.5, 3.5]
            claimed.append(str(context.claim.task_id))
            assert context.stage_key == str(records["pending"].id)
            assert progress.stages[context.stage_key].state is StageState.RUNNING
            context.transition(TaskRuntimeStatus.CLAIMED, TaskRuntimeStatus.DONE)
            return TaskRuntimeStatus.DONE

        rerun = HostTaskScheduler(
            store,
            resume_pending,
            config=HostSchedulerConfig(poll_interval_seconds=0.005),
            clock=clock,
            progress=progress,
        ).run(borg.id, generation.id)

        ordered_keys = [str(records[ref].id) for ref in records]
        assert list(progress.stages) == ordered_keys
        done, failed, blocked, pending = (
            progress.stages[str(records[ref].id)]
            for ref in ("done", "failed", "blocked", "pending")
        )
        assert (done.state, done.result, done.duration_seconds, done.started_at) == (
            StageState.COMPLETED,
            "done",
            1.5,
            None,
        )
        assert (
            failed.state,
            failed.result,
            failed.duration_seconds,
            failed.started_at,
        ) == (StageState.FAILED, "failed: durable failure", 2.5, None)
        assert (
            blocked.state,
            blocked.result,
            blocked.duration_seconds,
            blocked.started_at,
        ) == (StageState.FAILED, "blocked: durable block", 3.5, None)
        assert pending.state is StageState.COMPLETED
        assert pending.result == "done"
        assert pending.started_at is not None
        assert claimed == [str(records["pending"].id)]
        rerun_claims = store.list_task_claims(rerun.operation_id)
        assert [claim.task_id for claim in rerun_claims] == [records["pending"].id]
        assert (rerun.done, rerun.failed, rerun.blocked, rerun.pending) == (2, 1, 1, 0)


def test_scheduler_cancellation_preserves_done_work_for_durable_resume(
    tmp_path: Path,
) -> None:
    database, borg, generation, records = _scheduler_fixture(
        tmp_path,
        task_refs=("foundation", "consumer"),
        dependencies=(("consumer", "foundation"),),
    )
    clock = FakeClock()
    cancel = CancellationToken()
    consumer_started = threading.Event()
    first_invocations: list[str] = []
    progress = RunProgress(stream=StringIO(), enabled=False)

    with SqliteStore.open(database) as store:

        def cancelling_behavior(
            context: ScheduledTaskContext,
        ) -> TaskRuntimeStatus:
            task_ref = next(
                ref
                for ref, record in records.items()
                if record.id == context.claim.task_id
            )
            first_invocations.append(task_ref)
            if task_ref == "consumer":
                consumer_started.set()
                assert context.cancel.wait(timeout=2)
            return TaskRuntimeStatus.DONE

        scheduler = HostTaskScheduler(
            store,
            cancelling_behavior,
            config=HostSchedulerConfig(jobs=1, poll_interval_seconds=0.005),
            clock=clock,
            progress=progress,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            running = executor.submit(
                scheduler.run, borg.id, generation.id, cancel=cancel
            )
            assert consumer_started.wait(timeout=2)
            cancel.cancel()
            cancelled = running.result(timeout=2)

        assert cancelled.status is ExecutionRunStatus.CANCELLED
        assert first_invocations == ["foundation", "consumer"]
        assert store.get_task_runtime(records["foundation"].id).status is (
            TaskRuntimeStatus.DONE
        )
        assert store.get_task_runtime(records["consumer"].id).status is (
            TaskRuntimeStatus.PENDING
        )
        foundation_stage = progress.stages[str(records["foundation"].id)]
        consumer_stage = progress.stages[str(records["consumer"].id)]
        assert foundation_stage.state is StageState.COMPLETED
        assert consumer_stage.state is StageState.STOPPED
        assert consumer_stage.result == "execution cancelled"
        assert progress.cancelling is True
        cancelled_claims = store.list_task_claims(cancelled.operation_id)
        assert all(claim.released_at is not None for claim in cancelled_claims)
        assert any(
            event.kind == "run.interrupted"
            for event in store.list_execution_events(cancelled.operation_id)
        )

        resumed_invocations: list[str] = []

        def resumed_behavior(
            context: ScheduledTaskContext,
        ) -> TaskRuntimeStatus | None:
            task_ref = next(
                ref
                for ref, record in records.items()
                if record.id == context.claim.task_id
            )
            resumed_invocations.append(task_ref)
            context.transition(TaskRuntimeStatus.CLAIMED, TaskRuntimeStatus.DONE)
            return None

        resumed = HostTaskScheduler(
            store,
            resumed_behavior,
            config=HostSchedulerConfig(jobs=1, poll_interval_seconds=0.005),
            clock=clock,
        ).run(borg.id, generation.id)

        assert resumed.status is ExecutionRunStatus.COMPLETED
        assert resumed_invocations == ["consumer"]
        assert resumed.operation_id != cancelled.operation_id
        runs_by_id = {run.id: run for run in store.list_execution_runs(borg.id)}
        assert runs_by_id[cancelled.operation_id].status is (
            ExecutionRunStatus.CANCELLED
        )
        assert runs_by_id[resumed.operation_id].status is (
            ExecutionRunStatus.COMPLETED
        )


def test_scheduler_cancels_claim_accepted_as_progress_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, borg, generation, records = _scheduler_fixture(
        tmp_path,
        task_refs=("task",),
        dependencies=(),
    )
    cancel = CancellationToken()
    progress = RunProgress(stream=StringIO(), enabled=False)
    original_start = progress.start
    cancelled_before_start = False

    def cancel_then_start(stage_key: str) -> StageRecord:
        nonlocal cancelled_before_start
        if not cancelled_before_start:
            cancelled_before_start = True
            cancel.cancel()
            progress.begin_cancellation()
        return original_start(stage_key)

    monkeypatch.setattr(progress, "start", cancel_then_start)

    with SqliteStore.open(database) as store:

        def unexpected_behavior(
            _context: ScheduledTaskContext,
        ) -> TaskRuntimeStatus:
            raise AssertionError("cancelled claimed work must not start")

        result = HostTaskScheduler(
            store,
            unexpected_behavior,
            config=HostSchedulerConfig(poll_interval_seconds=0.005),
            progress=progress,
        ).run(borg.id, generation.id, cancel=cancel)

        assert cancelled_before_start is True
        assert result.status is ExecutionRunStatus.CANCELLED
        assert (result.done, result.failed, result.blocked, result.pending) == (
            0,
            0,
            0,
            1,
        )
        runtime = store.get_task_runtime(records["task"].id)
        assert runtime is not None
        assert runtime.status is TaskRuntimeStatus.PENDING
        stage = progress.stages[str(records["task"].id)]
        assert stage.state is StageState.PENDING
        claims = store.list_task_claims(result.operation_id)
        assert len(claims) == 1
        assert claims[0].released_at is not None
        assert any(
            event.kind == "run.interrupted"
            for event in store.list_execution_events(result.operation_id)
        )


def test_scheduler_cancels_when_progress_stops_during_seeding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, borg, generation, _records = _scheduler_fixture(
        tmp_path,
        task_refs=("task",),
        dependencies=(),
    )
    cancel = CancellationToken()
    progress = RunProgress(stream=StringIO(), enabled=False)
    original_declare = progress.declare

    def cancel_then_declare(spec: StageSpec) -> StageRecord:
        cancel.cancel()
        progress.begin_cancellation()
        return original_declare(spec)

    monkeypatch.setattr(progress, "declare", cancel_then_declare)

    with SqliteStore.open(database) as store:

        def unexpected_behavior(
            _context: ScheduledTaskContext,
        ) -> TaskRuntimeStatus:
            raise AssertionError("cancelled work must not be claimed")

        result = HostTaskScheduler(
            store,
            unexpected_behavior,
            progress=progress,
        ).run(borg.id, generation.id, cancel=cancel)

        assert result.status is ExecutionRunStatus.CANCELLED
        assert result.pending == 1
        assert store.list_task_claims(result.operation_id) == []
        assert any(
            event.kind == "run.interrupted"
            for event in store.list_execution_events(result.operation_id)
        )


def test_scheduler_durably_cancels_when_progress_output_fails(
    tmp_path: Path,
) -> None:
    database, borg, generation, _records = _scheduler_fixture(
        tmp_path,
        task_refs=("task",),
        dependencies=(),
    )
    cancel = CancellationToken()
    cancel.cancel()

    class FailingCancellationStream(StringIO):
        def write(self, value: str) -> int:
            if "stopping..." in value:
                raise OSError("progress output unavailable")
            return super().write(value)

    progress = RunProgress(stream=FailingCancellationStream())

    with SqliteStore.open(database) as store:

        def unexpected_behavior(
            _context: ScheduledTaskContext,
        ) -> TaskRuntimeStatus:
            raise AssertionError("cancelled work must not be claimed")

        with pytest.raises(OSError, match="progress output unavailable"):
            HostTaskScheduler(
                store,
                unexpected_behavior,
                progress=progress,
            ).run(borg.id, generation.id, cancel=cancel)

        run = store.list_execution_runs(borg.id)[0]
        assert run.status is ExecutionRunStatus.CANCELLED
        assert store.list_task_claims(run.id) == []
        assert any(
            event.kind == "run.interrupted"
            for event in store.list_execution_events(run.id)
        )


def test_scheduler_sigint_durably_interrupts_before_keyboard_interrupt_escapes(
    tmp_path: Path,
) -> None:
    database, borg, generation, records = _scheduler_fixture(
        tmp_path,
        task_refs=("task",),
        dependencies=(),
    )
    cancel = CancellationToken()
    started = threading.Event()
    progress = RunProgress(stream=StringIO(), enabled=False)

    with SqliteStore.open(database) as store:

        def behavior(context: ScheduledTaskContext) -> TaskRuntimeStatus:
            started.set()
            assert context.cancel.wait(timeout=2)
            return context.runtime.status

        def interrupt() -> None:
            assert started.wait(timeout=2)
            os.kill(os.getpid(), signal.SIGINT)

        interrupter = threading.Thread(target=interrupt)
        control = RunControl(cancel, progress=progress)
        with control:
            interrupter.start()
            with pytest.raises(KeyboardInterrupt):
                HostTaskScheduler(
                    store,
                    behavior,
                    config=HostSchedulerConfig(poll_interval_seconds=0.005),
                    progress=progress,
                ).run(borg.id, generation.id, cancel=cancel)
        interrupter.join(timeout=2)

        assert not interrupter.is_alive()
        assert control.wait_for_cancellation(timeout=2)
        run = store.list_execution_runs(borg.id)[0]
        assert run.status is ExecutionRunStatus.CANCELLED
        runtime = store.get_task_runtime(records["task"].id)
        assert runtime is not None
        assert runtime.status is TaskRuntimeStatus.PENDING
        assert progress.stages[str(records["task"].id)].state is StageState.STOPPED
        claims = store.list_task_claims(run.id)
        assert len(claims) == 1
        assert claims[0].released_at is not None
        assert any(
            event.kind == "run.interrupted"
            for event in store.list_execution_events(run.id)
        )


@pytest.mark.parametrize(
    ("durable_status", "expected_stage_state", "expected_result"),
    [
        (TaskRuntimeStatus.DONE, StageState.COMPLETED, "done"),
        (
            TaskRuntimeStatus.FAILED,
            StageState.FAILED,
            "failed: durable failure",
        ),
        (
            TaskRuntimeStatus.BLOCKED,
            StageState.FAILED,
            "blocked: durable block",
        ),
    ],
)
def test_scheduler_interrupt_reconciles_terminal_stage_during_seeding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    durable_status: TaskRuntimeStatus,
    expected_stage_state: StageState,
    expected_result: str,
) -> None:
    database, borg, generation, records = _scheduler_fixture(
        tmp_path,
        task_refs=("retained",),
        dependencies=(),
    )
    clock = FakeClock()

    with SqliteStore.open(database) as store:

        def complete_with_duration(
            context: ScheduledTaskContext,
        ) -> TaskRuntimeStatus:
            store.append_agent_attempt(
                AgentAttempt(
                    run_id=context.claim.run_id,
                    claim_id=context.claim.id,
                    task_id=context.claim.task_id,
                    phase="coding",
                    attempt_number=1,
                    adapter="test",
                    model="test-model",
                    billing_mode=BillingMode.SUBSCRIPTION,
                    status=AgentStatus.COMPLETED,
                    log_path="artifacts/retained.log",
                    duration_seconds=4.25,
                    started_at=clock(),
                    finished_at=clock(),
                ),
                context.owner_token,
                context.claim.claim_token,
                now=clock(),
            )
            reason = {
                TaskRuntimeStatus.DONE: None,
                TaskRuntimeStatus.FAILED: "durable failure",
                TaskRuntimeStatus.BLOCKED: "durable block",
            }[durable_status]
            context.transition(
                TaskRuntimeStatus.CLAIMED,
                durable_status,
                state_reason=reason,
            )
            return durable_status

        completed = HostTaskScheduler(
            store,
            complete_with_duration,
            clock=clock,
        ).run(borg.id, generation.id)
        progress = RunProgress(stream=StringIO(), enabled=False)
        seed_method_name = (
            "seed_completed"
            if durable_status is TaskRuntimeStatus.DONE
            else "seed_failed"
        )
        original_seed = getattr(progress, seed_method_name)
        interrupted = False

        def interrupt_first_seed(
            stage_key: str,
            result: object,
            duration_seconds: float | None = None,
        ) -> object:
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            return original_seed(stage_key, result, duration_seconds)

        monkeypatch.setattr(progress, seed_method_name, interrupt_first_seed)

        def unexpected_behavior(
            _context: ScheduledTaskContext,
        ) -> TaskRuntimeStatus:
            raise AssertionError("retained work must not be claimed")

        with pytest.raises(KeyboardInterrupt):
            HostTaskScheduler(
                store,
                unexpected_behavior,
                clock=clock,
                progress=progress,
            ).run(borg.id, generation.id)

        interrupted_run = next(
            run
            for run in store.list_execution_runs(borg.id)
            if run.id != completed.operation_id
        )
        assert interrupted_run.status is ExecutionRunStatus.CANCELLED
        runtime = store.get_task_runtime(records["retained"].id)
        assert runtime is not None
        assert runtime.status is durable_status
        stage = progress.stages[str(records["retained"].id)]
        assert (stage.state, stage.result, stage.duration_seconds) == (
            expected_stage_state,
            expected_result,
            4.25,
        )
        assert stage.retained is True
        assert progress.cancelling is True
        assert store.list_task_claims(interrupted_run.id) == []


def test_scheduler_cancels_inflight_behavior_when_run_lease_expires(
    tmp_path: Path,
) -> None:
    database, borg, generation, records = _scheduler_fixture(
        tmp_path,
        task_refs=("task",),
        dependencies=(),
    )
    clock = FakeClock()
    started = threading.Event()
    cancellation_observed = threading.Event()

    with SqliteStore.open(database) as store:

        def behavior(context: ScheduledTaskContext) -> TaskRuntimeStatus:
            started.set()
            if context.cancel.wait(timeout=2):
                cancellation_observed.set()
            return TaskRuntimeStatus.DONE

        scheduler = HostTaskScheduler(
            store,
            behavior,
            config=HostSchedulerConfig(
                lease_duration=timedelta(seconds=10),
                heartbeat_interval=timedelta(seconds=2),
                poll_interval_seconds=0.005,
            ),
            clock=clock,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            running = executor.submit(scheduler.run, borg.id, generation.id)
            assert started.wait(timeout=2)
            clock.advance(timedelta(seconds=11))

            with pytest.raises(ExecutionOwnershipError, match="lease expired"):
                running.result(timeout=2)

        assert cancellation_observed.is_set()
        run = store.list_execution_runs(borg.id)[0]
        assert run.status is ExecutionRunStatus.CANCELLED
        runtime = store.get_task_runtime(records["task"].id)
        assert runtime is not None
        assert runtime.status is TaskRuntimeStatus.PENDING


def test_scheduler_isolates_task_failure_and_finishes_run(tmp_path: Path) -> None:
    database, borg, generation, records = _scheduler_fixture(
        tmp_path,
        task_refs=("foundation", "consumer"),
        dependencies=(("consumer", "foundation"),),
    )

    def failing_behavior(_context: ScheduledTaskContext) -> TaskRuntimeStatus:
        raise RuntimeError("injected task crash")

    with SqliteStore.open(database) as store:
        result = HostTaskScheduler(store, failing_behavior).run(
            borg.id, generation.id
        )

        assert result.status is ExecutionRunStatus.FAILED
        assert result.failed == 1
        assert result.pending == 1
        failed = store.get_task_runtime(records["foundation"].id)
        assert failed is not None
        assert failed.status is TaskRuntimeStatus.FAILED
        assert failed.state_reason == "task behavior failed: injected task crash"
        consumer = store.get_task_runtime(records["consumer"].id)
        assert consumer is not None
        assert consumer.status is TaskRuntimeStatus.PENDING
