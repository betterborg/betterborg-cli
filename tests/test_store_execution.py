"""Contracts for durable host-execution ownership state."""

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Barrier

import pytest

from betterborg_cli.agent_runtime import AgentStatus, AgentUsage, BillingMode
from betterborg_cli.planning import render_task_markdown
from betterborg_cli.store import (
    AgentAttempt,
    Borg,
    ComposeResource,
    EnvironmentAttempt,
    ExecutionAttemptStatus,
    ExecutionEvent,
    ExecutionOwnershipError,
    ExecutionRun,
    ExecutionRunStatus,
    PlanApproval,
    Repository,
    SqliteStore,
    TaskBatch,
    TaskClaim,
    TaskComplexity,
    TaskDependency,
    TaskGeneration,
    TaskRecord,
    TaskRuntime,
    TaskRuntimeStatus,
)
from betterborg_cli.store.models import utcnow


def _execution_fixture(tmp_path: Path, approved_task_generation):
    database = tmp_path / "state.sqlite3"
    repository = Repository(root=tmp_path / "repository")
    borg = Borg(repository_id=repository.id, name="Executor")
    approval = PlanApproval(
        borg_id=borg.id,
        plan_digest="sha256:plan",
        manifest={"plan.md": "sha256:plan"},
    )
    body = {
        "stage": "07-host-execution",
        "stem": "01-foundation",
        "title": "Build execution foundation",
        "why": "Host execution needs durable ownership state.",
        "scope": ["Store execution ownership records."],
        "implementation_notes": [],
        "acceptance_criteria": ["Execution state is durable."],
        "tests": ["Reopen the disk store."],
        "dependencies": [],
        "out_of_scope": [],
        "plan_refs": ["P1.deliverable.1"],
        "estimate_complexity": "medium",
    }

    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.append_plan_approval(approval)
        fixture = approved_task_generation(
            store,
            borg,
            approval,
            body=body,
            round_number=1,
            task_ref="foundation",
        )
        generation, task = fixture.generation, fixture.task
        durable_root = (
            repository.root / ".borg/tasks" / borg.name / str(generation.id)
        )
        task_path = durable_root / task.stage / f"{task.stem}.md"
        task_path.parent.mkdir(parents=True)
        task_path.write_text(render_task_markdown(body), encoding="utf-8")
        store._promote_published_task_generation(
            generation.id, durable_root=durable_root
        )

    return database, borg, generation, task


def _dependency_execution_fixture(tmp_path: Path):
    database = tmp_path / "state.sqlite3"
    repository = Repository(root=tmp_path / "repository")
    borg = Borg(repository_id=repository.id, name="DependencyExecutor")
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
        manifest={"tasks": ["foundation", "consumer"]},
    )
    generation = TaskGeneration(
        borg_id=borg.id,
        plan_approval_id=approval.id,
        batch_id=batch.id,
        digest="sha256:generation",
        manifest={"tasks": ["foundation", "consumer"]},
    )

    def task(task_ref: str, position: int) -> TaskRecord:
        digest = hashlib.sha256(task_ref.encode()).hexdigest()
        return TaskRecord(
            generation_id=generation.id,
            borg_id=borg.id,
            task_ref=task_ref,
            stage="07-host-execution",
            stem=f"{position:02d}-{task_ref}",
            position=position,
            title=f"Implement {task_ref}",
            complexity=TaskComplexity.SMALL,
            digest=f"sha256:{digest}",
            task={"acceptance_criteria": [f"{task_ref} works"]},
            manifest={"task.md": f"sha256:{digest}"},
        )

    foundation = task("foundation", 1)
    consumer = task("consumer", 2)
    dependency = TaskDependency(
        generation_id=generation.id,
        task_id=consumer.id,
        depends_on_task_id=foundation.id,
    )
    durable_root = repository.root / ".borg/tasks" / borg.name / str(generation.id)

    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.append_plan_approval(approval)
        store.append_task_batch(batch)
        store.add_task_generation(
            generation,
            [foundation, consumer],
            [dependency],
        )
        for record in (foundation, consumer):
            path = durable_root / record.stage / f"{record.stem}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(record.task_ref, encoding="utf-8")
        store._promote_published_task_generation(
            generation.id, durable_root=durable_root
        )

    return database, borg, generation, foundation, consumer


def test_execution_ownership_records_round_trip_after_reopen(
    tmp_path: Path, approved_task_generation
) -> None:
    database, borg, generation, task = _execution_fixture(
        tmp_path, approved_task_generation
    )
    started_at = utcnow()
    run = ExecutionRun(
        borg_id=borg.id,
        generation_id=generation.id,
        started_at=started_at,
        heartbeat_at=started_at,
        lease_expires_at=started_at + timedelta(minutes=5),
    )
    runtime = TaskRuntime(
        generation_id=generation.id,
        task_id=task.id,
        status=TaskRuntimeStatus.REVIEW,
        resume_phase="review",
        review_round=2,
        branch="betterborg-tasks/07-host-execution/01-foundation",
        worktree_path="worktrees/01-foundation",
        last_run_id=run.id,
    )
    claim = TaskClaim(
        run_id=run.id,
        task_id=task.id,
        resume_phase=runtime.resume_phase,
        claimed_at=started_at,
        lease_expires_at=started_at + timedelta(minutes=2),
    )
    environment = EnvironmentAttempt(
        run_id=run.id,
        claim_id=claim.id,
        task_id=task.id,
        kind="materialize",
        attempt_number=1,
        fingerprint="sha256:environment",
        status=AgentStatus.COMPLETED,
        commands=[["make", "sync"]],
        result={"cache": "hit"},
        duration_seconds=1.5,
        started_at=started_at,
        finished_at=started_at + timedelta(seconds=2),
    )
    agent = AgentAttempt(
        run_id=run.id,
        claim_id=claim.id,
        task_id=task.id,
        phase="review",
        review_round=2,
        attempt_number=1,
        adapter="codex",
        model="test-model",
        billing_mode=BillingMode.SUBSCRIPTION,
        status=AgentStatus.COMPLETED,
        log_path="artifacts/review.log",
        result_path="artifacts/review.json",
        result={"verdict": "approved"},
        summary="Approved.",
        duration_seconds=3.0,
        usage=AgentUsage(
            cost_usd=0.25,
            tokens_input=100,
            tokens_output=20,
            tokens_cache_read=80,
            num_turns=1,
        ),
        started_at=started_at,
        finished_at=started_at + timedelta(seconds=3),
    )
    event = ExecutionEvent(
        run_id=run.id,
        task_id=task.id,
        attempt_id=agent.id,
        kind="task.review.completed",
        payload={"review_round": 2},
    )
    compose = ComposeResource(
        run_id=run.id,
        claim_id=claim.id,
        task_id=task.id,
        project_name="borg-foundation-1234",
        resource_type="network",
        resource_name="borg-foundation-1234_default",
        labels={"betterborg.task_id": str(task.id)},
    )

    with SqliteStore.open(database) as store:
        store.add_execution_run(run)
        store.add_task_runtime(runtime)
        store.append_task_claim(claim)
        store.append_environment_attempt(
            environment, run.owner_token, claim.claim_token, now=started_at
        )
        store.append_agent_attempt(
            agent, run.owner_token, claim.claim_token, now=started_at
        )
        store.append_execution_event(event)
        store.add_compose_resource(
            compose, run.owner_token, claim.claim_token, now=started_at
        )

        assert store.execution_run_owned_by(run.id, run.owner_token)
        assert not store.execution_run_owned_by(run.id, "wrong-token")
        assert store.task_claim_owned_by(claim.id, claim.claim_token)
        assert not store.task_claim_owned_by(claim.id, "wrong-token")

    with SqliteStore.open(database) as reopened:
        assert reopened.applied_migrations() == tuple(range(1, 12))
        assert reopened.get_execution_run(run.id) == run
        assert reopened.list_execution_runs(borg.id) == [run]
        assert reopened.get_task_runtime(task.id) == runtime
        assert reopened.list_task_runtimes(generation.id) == [runtime]
        assert reopened.list_task_claims(run.id) == [claim]
        assert reopened.list_environment_attempts(task.id) == [environment]
        assert reopened.list_agent_attempts(task.id) == [agent]
        assert reopened.list_execution_events(run.id) == [event]
        assert reopened.list_compose_resources(task.id) == [compose]


@pytest.mark.parametrize(
    ("attempts", "api_spend_usd", "api_spend_unknown", "included", "duration"),
    [
        ([(BillingMode.API, 0.25, 3.0)], 0.25, False, False, 3.0),
        ([(BillingMode.API, None, 4.0)], None, True, False, 4.0),
        ([(BillingMode.SUBSCRIPTION, 9.0, 5.0)], None, False, True, 5.0),
        (
            [
                (BillingMode.API, 0.5, 6.0),
                (BillingMode.SUBSCRIPTION, 12.0, 7.0),
            ],
            0.5,
            False,
            True,
            13.0,
        ),
        ([], None, True, False, None),
    ],
)
def test_current_task_runtime_projection_preserves_billing_semantics(
    tmp_path: Path,
    approved_task_generation,
    attempts,
    api_spend_usd: float | None,
    api_spend_unknown: bool,
    included: bool,
    duration: float | None,
) -> None:
    database, borg, generation, task = _execution_fixture(
        tmp_path, approved_task_generation
    )
    started_at = utcnow()
    run = ExecutionRun(
        borg_id=borg.id,
        generation_id=generation.id,
        started_at=started_at,
        heartbeat_at=started_at,
        lease_expires_at=started_at + timedelta(minutes=5),
    )
    runtime = TaskRuntime(
        generation_id=generation.id,
        task_id=task.id,
        status=TaskRuntimeStatus.BLOCKED,
        state_reason="review requested changes",
        review_round=3,
    )
    claim = TaskClaim(
        run_id=run.id,
        task_id=task.id,
        resume_phase="review",
        claimed_at=started_at,
        lease_expires_at=started_at + timedelta(minutes=2),
    )

    with SqliteStore.open(database) as store:
        store.add_execution_run(run)
        store.add_task_runtime(runtime)
        store.append_task_claim(claim)
        for index, (billing_mode, cost, attempt_duration) in enumerate(
            attempts, start=1
        ):
            store.append_agent_attempt(
                AgentAttempt(
                    run_id=run.id,
                    claim_id=claim.id,
                    task_id=task.id,
                    phase=f"phase-{index}",
                    attempt_number=1,
                    adapter="mock",
                    model="test-model",
                    billing_mode=billing_mode,
                    status=AgentStatus.COMPLETED,
                    log_path=f"artifacts/{index}.log",
                    duration_seconds=attempt_duration,
                    usage=AgentUsage(cost_usd=cost),
                    started_at=started_at,
                    finished_at=started_at + timedelta(seconds=attempt_duration),
                ),
                run.owner_token,
                claim.claim_token,
                now=started_at,
            )

        rows = store.list_task_runtime(borg.id)

    assert len(rows) == 1
    row = rows[0]
    assert row.generation_id == generation.id
    assert row.task_id == task.id
    assert (row.stage, row.stem, row.title) == (
        task.stage,
        task.stem,
        task.title,
    )
    assert row.status is TaskRuntimeStatus.BLOCKED
    assert row.state_reason == "review requested changes"
    assert row.review_round == 3
    assert row.attempt_count == len(attempts)
    assert row.duration_seconds == duration
    assert row.cost.api_spend_usd == api_spend_usd
    assert row.cost.api_spend_unknown is api_spend_unknown
    assert row.cost.subscription_included is included


def test_live_run_claim_and_token_ownership_are_database_enforced(
    tmp_path: Path, approved_task_generation
) -> None:
    database, borg, generation, task = _execution_fixture(
        tmp_path, approved_task_generation
    )
    started_at = utcnow()
    run = ExecutionRun(
        borg_id=borg.id,
        generation_id=generation.id,
        started_at=started_at,
        lease_expires_at=started_at + timedelta(minutes=5),
    )
    claim = TaskClaim(
        run_id=run.id,
        task_id=task.id,
        resume_phase="coding",
        claimed_at=started_at,
        lease_expires_at=started_at + timedelta(minutes=2),
    )

    assert len(run.owner_token) >= 32
    assert len(claim.claim_token) >= 32

    with SqliteStore.open(database) as store:
        store.add_execution_run(run)
        store.append_task_claim(claim)

        duplicate_run = ExecutionRun(
            borg_id=borg.id,
            generation_id=generation.id,
            started_at=started_at,
            lease_expires_at=started_at + timedelta(minutes=5),
        )
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            store.add_execution_run(duplicate_run)

        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            with store.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO task_claims(
                        id, run_id, generation_id, task_id, claim_token,
                        resume_phase, claimed_at, lease_expires_at, released_at
                    ) SELECT ?, id, generation_id, ?, ?, ?, ?, ?, NULL
                      FROM execution_runs WHERE id = ?
                    """,
                    (
                        "second-claim",
                        str(task.id),
                        "x" * 32,
                        "coding",
                        started_at.isoformat(),
                        (started_at + timedelta(minutes=2)).isoformat(),
                        str(run.id),
                    ),
                )

        with pytest.raises(sqlite3.IntegrityError, match="ownership is immutable"):
            with store.transaction() as connection:
                connection.execute(
                    "UPDATE execution_runs SET owner_token = ? WHERE id = ?",
                    ("replacement-token" * 3, str(run.id)),
                )
        with pytest.raises(sqlite3.IntegrityError, match="ownership is immutable"):
            with store.transaction() as connection:
                connection.execute(
                    "UPDATE task_claims SET claim_token = ? WHERE id = ?",
                    ("replacement-token" * 3, str(claim.id)),
                )


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("environment_attempts", "fingerprint"),
        ("agent_attempts", "summary"),
        ("execution_events", "kind"),
        ("compose_resources", "resource_name"),
    ],
)
@pytest.mark.parametrize("statement", ["UPDATE", "DELETE", "REPLACE"])
def test_execution_history_and_resource_ownership_are_immutable(
    tmp_path: Path,
    approved_task_generation,
    table: str,
    column: str,
    statement: str,
) -> None:
    database, borg, generation, task = _execution_fixture(
        tmp_path, approved_task_generation
    )
    started_at = utcnow()
    run = ExecutionRun(
        borg_id=borg.id,
        generation_id=generation.id,
        started_at=started_at,
        lease_expires_at=started_at + timedelta(minutes=5),
    )
    claim = TaskClaim(
        run_id=run.id,
        task_id=task.id,
        resume_phase="environment",
        claimed_at=started_at,
        lease_expires_at=started_at + timedelta(minutes=2),
    )
    environment = EnvironmentAttempt(
        run_id=run.id,
        claim_id=claim.id,
        task_id=task.id,
        kind="prepare",
        attempt_number=1,
        fingerprint="sha256:environment",
        status=AgentStatus.COMPLETED,
        started_at=started_at,
        finished_at=started_at,
    )
    agent = AgentAttempt(
        run_id=run.id,
        claim_id=claim.id,
        task_id=task.id,
        phase="coding",
        attempt_number=1,
        adapter="openai",
        model="test-model",
        billing_mode=BillingMode.API,
        status=AgentStatus.COMPLETED,
        log_path="artifacts/coding.log",
        started_at=started_at,
        finished_at=started_at,
    )
    event = ExecutionEvent(run_id=run.id, kind="task.claimed", task_id=task.id)
    compose = ComposeResource(
        run_id=run.id,
        claim_id=claim.id,
        task_id=task.id,
        project_name="borg-foundation-1234",
        resource_type="project",
        resource_name="borg-foundation-1234",
    )
    records = {
        "environment_attempts": environment,
        "agent_attempts": agent,
        "execution_events": event,
        "compose_resources": compose,
    }

    with SqliteStore.open(database) as store:
        store.add_execution_run(run)
        store.append_task_claim(claim)
        store.append_environment_attempt(
            environment, run.owner_token, claim.claim_token, now=started_at
        )
        store.append_agent_attempt(
            agent, run.owner_token, claim.claim_token, now=started_at
        )
        store.append_execution_event(event)
        store.add_compose_resource(
            compose, run.owner_token, claim.claim_token, now=started_at
        )

        sql, parameters = {
            "UPDATE": (
                f"UPDATE {table} SET {column} = ? WHERE id = ?",
                ("changed", str(records[table].id)),
            ),
            "DELETE": (
                f"DELETE FROM {table} WHERE id = ?",
                (str(records[table].id),),
            ),
            "REPLACE": (
                f"INSERT OR REPLACE INTO {table} "
                f"SELECT * FROM {table} WHERE id = ?",
                (str(records[table].id),),
            ),
        }[statement]

        with pytest.raises(
            sqlite3.IntegrityError, match="immutable|append-only|durable"
        ):
            with store.transaction() as connection:
                connection.execute(sql, parameters)

        persisted_records = {
            "environment_attempts": store.list_environment_attempts(task.id),
            "agent_attempts": store.list_agent_attempts(task.id),
            "execution_events": store.list_execution_events(run.id),
            "compose_resources": store.list_compose_resources(task.id),
        }
        assert persisted_records[table] == [records[table]]


def test_execution_acquisition_is_atomic_across_store_connections(
    tmp_path: Path, approved_task_generation
) -> None:
    database, borg, generation, _task = _execution_fixture(
        tmp_path, approved_task_generation
    )
    now = utcnow()
    barrier = Barrier(2)

    def acquire():
        with SqliteStore.open(database) as store:
            barrier.wait()
            return store.acquire_execution_run(
                borg.id,
                generation.id,
                lease_duration=timedelta(minutes=5),
                now=now,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        acquisitions = list(executor.map(lambda _index: acquire(), range(2)))

    assert len({result.operation_id for result in acquisitions}) == 1
    assert [result.acquired for result in acquisitions].count(True) == 1
    owner = next(result for result in acquisitions if result.acquired)
    observer = next(result for result in acquisitions if not result.acquired)
    assert owner.owner_token is not None
    assert observer.owner_token is None

    claim_barrier = Barrier(2)

    def claim():
        with SqliteStore.open(database) as store:
            claim_barrier.wait()
            return store.claim_dependency_ready_task(
                owner.run_id,
                owner.owner_token,
                lease_duration=timedelta(minutes=2),
                now=now,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(lambda _index: claim(), range(2)))

    assert sum(result is not None for result in claims) == 1
    with SqliteStore.open(database) as store:
        assert len(store.list_execution_runs(borg.id)) == 1
        assert len(store.list_task_claims(owner.run_id)) == 1
        assert {event.kind for event in store.list_execution_events(owner.run_id)} == {
            "run.acquired",
            "task.claimed",
        }


def test_claims_only_select_dependency_ready_tasks(tmp_path: Path) -> None:
    database, borg, generation, foundation, consumer = (
        _dependency_execution_fixture(tmp_path)
    )
    now = utcnow()

    with SqliteStore.open(database) as store:
        acquisition = store.acquire_execution_run(
            borg.id,
            generation.id,
            lease_duration=timedelta(minutes=5),
            now=now,
        )
        assert acquisition.owner_token is not None
        first = store.claim_dependency_ready_task(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=2),
            now=now,
        )
        assert first is not None
        assert first.task_id == foundation.id
        assert (
            store.claim_dependency_ready_task(
                acquisition.run_id,
                acquisition.owner_token,
                lease_duration=timedelta(minutes=2),
                now=now + timedelta(seconds=1),
            )
            is None
        )

        store.transition_task_runtime(
            acquisition.run_id,
            acquisition.owner_token,
            first.id,
            first.claim_token,
            expected_status=TaskRuntimeStatus.CLAIMED,
            new_status=TaskRuntimeStatus.DONE,
            now=now + timedelta(seconds=2),
        )
        second = store.claim_dependency_ready_task(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=2),
            now=now + timedelta(seconds=3),
        )
        assert second is not None
        assert second.task_id == consumer.id


def test_completed_task_samples_include_agent_work_and_preserve_missing_usage(
    tmp_path: Path, approved_task_generation
) -> None:
    database, borg, generation, task = _execution_fixture(
        tmp_path, approved_task_generation
    )
    now = utcnow()
    with SqliteStore.open(database) as store:
        acquisition = store.acquire_execution_run(
            borg.id,
            generation.id,
            lease_duration=timedelta(minutes=5),
            now=now,
        )
        assert acquisition.owner_token is not None
        claim = store.claim_dependency_ready_task(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=4),
            now=now,
        )
        assert claim is not None

        statuses = {
            "coding": AgentStatus.COMPLETED,
            "review": AgentStatus.COMPLETED,
            "fix": AgentStatus.FAILED,
            "merge": AgentStatus.CANCELLED,
        }
        for offset, phase in enumerate(statuses, 1):
            attempt = AgentAttempt(
                run_id=acquisition.run_id,
                claim_id=claim.id,
                task_id=task.id,
                phase=phase,
                attempt_number=1,
                adapter="openai",
                model="gpt-5",
                billing_mode=BillingMode.API,
                status=ExecutionAttemptStatus.RUNNING,
                log_path=f"artifacts/{phase}.log",
                started_at=now + timedelta(seconds=offset),
                finished_at=None,
            )
            store.append_agent_attempt(
                attempt,
                acquisition.owner_token,
                claim.claim_token,
                now=now + timedelta(seconds=offset),
            )
            store.complete_agent_attempt(
                attempt.id,
                acquisition.owner_token,
                claim.claim_token,
                status=statuses[phase],
                duration_seconds=float(offset * 10),
                usage=(
                    AgentUsage(
                        tokens_input=offset,
                        tokens_output=offset,
                        tokens_cache_read=offset,
                        tokens_cache_write=offset,
                    )
                    if phase != "fix"
                    else AgentUsage(tokens_input=offset)
                ),
                now=now + timedelta(seconds=offset + 10),
            )

        store.transition_task_runtime(
            acquisition.run_id,
            acquisition.owner_token,
            claim.id,
            claim.claim_token,
            expected_status=TaskRuntimeStatus.CLAIMED,
            new_status=TaskRuntimeStatus.DONE,
            now=now + timedelta(seconds=30),
        )
        samples = store.list_task_completion_samples()

    assert len(samples) == 1
    sample = samples[0]
    assert sample.complexity is TaskComplexity.MEDIUM
    assert sample.duration_seconds == 100.0
    assert sample.coding_usage == AgentUsage(
        tokens_input=1,
        tokens_output=1,
        tokens_cache_read=1,
        tokens_cache_write=1,
    )
    assert sample.review_usage == AgentUsage(
        tokens_input=5,
        tokens_output=None,
        tokens_cache_read=None,
        tokens_cache_write=None,
        cost_usd=None,
        num_turns=None,
    )
    assert sample.merge_usage == AgentUsage(
        tokens_input=4,
        tokens_output=4,
        tokens_cache_read=4,
        tokens_cache_write=4,
    )


def test_expiry_closes_open_attempt_and_blocks_reclaim_until_compose_cleanup(
    tmp_path: Path, approved_task_generation
) -> None:
    database, borg, generation, task = _execution_fixture(
        tmp_path, approved_task_generation
    )
    now = utcnow()

    with SqliteStore.open(database) as store:
        acquisition = store.acquire_execution_run(
            borg.id,
            generation.id,
            lease_duration=timedelta(minutes=5),
            now=now,
        )
        assert acquisition.owner_token is not None
        claim = store.claim_dependency_ready_task(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=2),
            now=now,
        )
        assert claim is not None
        runtime = store.transition_task_runtime(
            acquisition.run_id,
            acquisition.owner_token,
            claim.id,
            claim.claim_token,
            expected_status=TaskRuntimeStatus.CLAIMED,
            new_status=TaskRuntimeStatus.CODING,
            now=now + timedelta(seconds=10),
        )
        assert runtime.status is TaskRuntimeStatus.CODING
        attempt = AgentAttempt(
            run_id=acquisition.run_id,
            claim_id=claim.id,
            task_id=task.id,
            phase="coding",
            attempt_number=1,
            adapter="codex",
            model="test-model",
            billing_mode=BillingMode.SUBSCRIPTION,
            status=ExecutionAttemptStatus.RUNNING,
            log_path="artifacts/coding.log",
            started_at=now + timedelta(seconds=10),
            finished_at=None,
        )
        resource = ComposeResource(
            run_id=acquisition.run_id,
            claim_id=claim.id,
            task_id=task.id,
            project_name="borg-foundation-exact",
            resource_type="network",
            resource_name="borg-foundation-exact_default",
            labels={"com.docker.compose.project": "borg-foundation-exact"},
            created_at=now + timedelta(seconds=15),
        )
        store.append_agent_attempt(
            attempt,
            acquisition.owner_token,
            claim.claim_token,
            now=now + timedelta(seconds=10),
        )
        store.add_compose_resource(
            resource,
            acquisition.owner_token,
            claim.claim_token,
            now=now + timedelta(seconds=15),
        )

        renewed = store.renew_execution_run(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=5),
            now=now + timedelta(minutes=1),
        )
        assert renewed.lease_expires_at == now + timedelta(minutes=6)
        assert store.reconcile_expired_execution_runs(
            now=now + timedelta(minutes=5)
        ) == []

        stale = store.reconcile_expired_execution_runs(
            now=now + timedelta(minutes=7)
        )
        assert stale == [resource]
        interrupted = store.get_execution_run(acquisition.run_id)
        assert interrupted is not None
        assert interrupted.status is ExecutionRunStatus.CANCELLED
        assert store.get_task_runtime(task.id).status is TaskRuntimeStatus.CODING
        persisted_attempts = store.list_agent_attempts(task.id)
        assert len(persisted_attempts) == 1
        assert persisted_attempts[0].status is ExecutionAttemptStatus.CANCELLED
        assert persisted_attempts[0].finished_at == now + timedelta(minutes=7)
        assert persisted_attempts[0].duration_seconds == 410
        assert store.list_task_claims(acquisition.run_id)[0].released_at is None

        replacement = store.acquire_execution_run(
            borg.id,
            generation.id,
            lease_duration=timedelta(minutes=5),
            now=now + timedelta(minutes=7),
        )
        assert replacement.owner_token is not None
        assert (
            store.claim_dependency_ready_task(
                replacement.run_id,
                replacement.owner_token,
                lease_duration=timedelta(minutes=2),
                now=now + timedelta(minutes=7),
            )
            is None
        )

        cleaned = store.confirm_compose_project_cleanup(
            acquisition.run_id,
            task.id,
            resource.project_name,
            now=now + timedelta(minutes=7, seconds=1),
        )
        assert cleaned == [resource]
        assert store.list_stale_compose_resources(acquisition.run_id) == []
        released = store.list_task_claims(acquisition.run_id)[0]
        assert released.released_at == now + timedelta(minutes=7, seconds=1)
        assert store.get_task_runtime(task.id).status is TaskRuntimeStatus.PENDING
        reclaimed = store.claim_dependency_ready_task(
            replacement.run_id,
            replacement.owner_token,
            lease_duration=timedelta(minutes=2),
            now=now + timedelta(minutes=7, seconds=2),
        )
        assert reclaimed is not None
        assert reclaimed.task_id == task.id

        kinds = {
            event.kind
            for event in store.list_execution_events(acquisition.run_id)
        }
        assert {
            "run.acquired",
            "run.lease_renewed",
            "run.expired",
            "task.claimed",
            "task.interrupted",
            "task.phase_transitioned",
            "agent.attempt_interrupted",
            "compose.cleanup_completed",
        } <= kinds


def test_interruption_closes_open_environment_and_agent_attempts(
    tmp_path: Path, approved_task_generation
) -> None:
    database, borg, generation, task = _execution_fixture(
        tmp_path, approved_task_generation
    )
    now = utcnow()

    with SqliteStore.open(database) as store:
        acquisition = store.acquire_execution_run(
            borg.id,
            generation.id,
            lease_duration=timedelta(minutes=5),
            now=now,
        )
        assert acquisition.owner_token is not None
        claim = store.claim_dependency_ready_task(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=2),
            now=now,
        )
        assert claim is not None
        environment = EnvironmentAttempt(
            run_id=acquisition.run_id,
            claim_id=claim.id,
            task_id=task.id,
            kind="materialize",
            attempt_number=1,
            fingerprint="sha256:open-environment",
            status=ExecutionAttemptStatus.RUNNING,
            commands=[["make", "sync"]],
            started_at=now + timedelta(seconds=5),
            finished_at=None,
        )
        agent = AgentAttempt(
            run_id=acquisition.run_id,
            claim_id=claim.id,
            task_id=task.id,
            phase="coding",
            attempt_number=1,
            adapter="codex",
            model="test-model",
            billing_mode=BillingMode.SUBSCRIPTION,
            status=ExecutionAttemptStatus.RUNNING,
            log_path="artifacts/coding.log",
            started_at=now + timedelta(seconds=10),
            finished_at=None,
        )
        store.append_environment_attempt(
            environment,
            acquisition.owner_token,
            claim.claim_token,
            now=now + timedelta(seconds=10),
        )
        store.append_agent_attempt(
            agent,
            acquisition.owner_token,
            claim.claim_token,
            now=now + timedelta(seconds=10),
        )

        interrupted_at = now + timedelta(seconds=30)
        assert (
            store.interrupt_execution_run(
                acquisition.run_id,
                acquisition.owner_token,
                reason="operator requested stop",
                now=interrupted_at,
            )
            == []
        )

        closed_environment = store.list_environment_attempts(task.id)
        closed_agent = store.list_agent_attempts(task.id)
        assert len(closed_environment) == len(closed_agent) == 1
        assert closed_environment[0].status is ExecutionAttemptStatus.CANCELLED
        assert closed_environment[0].finished_at == interrupted_at
        assert closed_environment[0].duration_seconds == 25
        assert closed_agent[0].status is ExecutionAttemptStatus.CANCELLED
        assert closed_agent[0].finished_at == interrupted_at
        assert closed_agent[0].duration_seconds == 20

        attempt_events = {
            (event.kind, event.attempt_id)
            for event in store.list_execution_events(acquisition.run_id)
            if event.attempt_id is not None
        }
        assert attempt_events == {
            ("environment.attempt_interrupted", environment.id),
            ("agent.attempt_interrupted", agent.id),
        }
        with pytest.raises(ExecutionOwnershipError, match="no longer running"):
            store.append_agent_attempt(
                agent,
                acquisition.owner_token,
                claim.claim_token,
                now=interrupted_at,
            )


def test_cleanup_confirmation_does_not_cover_later_project_resources(
    tmp_path: Path, approved_task_generation
) -> None:
    database, borg, generation, task = _execution_fixture(
        tmp_path, approved_task_generation
    )
    now = utcnow()

    with SqliteStore.open(database) as store:
        acquisition = store.acquire_execution_run(
            borg.id,
            generation.id,
            lease_duration=timedelta(minutes=5),
            now=now,
        )
        assert acquisition.owner_token is not None
        claim = store.claim_dependency_ready_task(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=2),
            now=now,
        )
        assert claim is not None
        first = ComposeResource(
            run_id=acquisition.run_id,
            claim_id=claim.id,
            task_id=task.id,
            project_name="borg-shared-project",
            resource_type="network",
            resource_name="borg-shared-project_default",
            created_at=now,
        )
        store.add_compose_resource(
            first, acquisition.owner_token, claim.claim_token, now=now
        )
        assert store.confirm_compose_project_cleanup(
            acquisition.run_id,
            task.id,
            first.project_name,
            now=now + timedelta(seconds=1),
        ) == [first]

        later = ComposeResource(
            run_id=acquisition.run_id,
            claim_id=claim.id,
            task_id=task.id,
            project_name=first.project_name,
            resource_type="volume",
            resource_name="borg-shared-project_data",
            created_at=now + timedelta(milliseconds=500),
        )
        store.add_compose_resource(
            later,
            acquisition.owner_token,
            claim.claim_token,
            now=now + timedelta(milliseconds=1500),
        )
        assert store.interrupt_execution_run(
            acquisition.run_id,
            acquisition.owner_token,
            now=now + timedelta(seconds=2),
        ) == [later]
        assert store.list_task_claims(acquisition.run_id)[0].released_at is None

        assert store.confirm_compose_project_cleanup(
            acquisition.run_id,
            task.id,
            first.project_name,
            now=now + timedelta(seconds=3),
        ) == [first, later]
        assert store.list_stale_compose_resources(acquisition.run_id) == []
        assert store.list_task_claims(acquisition.run_id)[0].released_at == (
            now + timedelta(seconds=3)
        )
        cleanup_events = [
            event
            for event in store.list_execution_events(acquisition.run_id)
            if event.kind == "compose.cleanup_completed"
        ]
        assert [event.payload["resource_ids"] for event in cleanup_events] == [
            [str(first.id)],
            [str(later.id)],
        ]


def test_interruption_preserves_completed_task_and_guards_phase_ownership(
    tmp_path: Path, approved_task_generation
) -> None:
    database, borg, generation, task = _execution_fixture(
        tmp_path, approved_task_generation
    )
    now = utcnow()

    with SqliteStore.open(database) as store:
        acquisition = store.acquire_execution_run(
            borg.id,
            generation.id,
            lease_duration=timedelta(minutes=5),
            now=now,
        )
        assert acquisition.owner_token is not None
        claim = store.claim_dependency_ready_task(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=2),
            now=now,
        )
        assert claim is not None

        with pytest.raises(ExecutionOwnershipError, match="no longer owned"):
            store.transition_task_runtime(
                acquisition.run_id,
                acquisition.owner_token,
                claim.id,
                "wrong-claim-token",
                expected_status=TaskRuntimeStatus.CLAIMED,
                new_status=TaskRuntimeStatus.DONE,
                now=now + timedelta(seconds=1),
            )

        resource = ComposeResource(
            run_id=acquisition.run_id,
            claim_id=claim.id,
            task_id=task.id,
            project_name="borg-completed-exact",
            resource_type="project",
            resource_name="borg-completed-exact",
            created_at=now + timedelta(seconds=1),
        )
        store.add_compose_resource(
            resource,
            acquisition.owner_token,
            claim.claim_token,
            now=now + timedelta(seconds=1),
        )
        completed = store.transition_task_runtime(
            acquisition.run_id,
            acquisition.owner_token,
            claim.id,
            claim.claim_token,
            expected_status=TaskRuntimeStatus.CLAIMED,
            new_status=TaskRuntimeStatus.DONE,
            now=now + timedelta(seconds=2),
        )
        assert completed.status is TaskRuntimeStatus.DONE

        assert store.interrupt_execution_run(
            acquisition.run_id,
            acquisition.owner_token,
            reason="operator requested stop",
            now=now + timedelta(seconds=3),
        ) == [resource]
        assert store.get_task_runtime(task.id).status is TaskRuntimeStatus.DONE
        assert store.list_task_claims(acquisition.run_id)[0].released_at is None

        store.confirm_compose_project_cleanup(
            acquisition.run_id,
            task.id,
            resource.project_name,
            now=now + timedelta(seconds=4),
        )
        assert store.get_task_runtime(task.id).status is TaskRuntimeStatus.DONE
        assert store.list_task_claims(acquisition.run_id)[0].released_at == (
            now + timedelta(seconds=4)
        )


def test_renewal_reconciles_an_expired_open_claim_before_reclaim(
    tmp_path: Path, approved_task_generation
) -> None:
    database, borg, generation, task = _execution_fixture(
        tmp_path, approved_task_generation
    )
    now = utcnow()

    with SqliteStore.open(database) as store:
        acquisition = store.acquire_execution_run(
            borg.id,
            generation.id,
            lease_duration=timedelta(minutes=5),
            now=now,
        )
        assert acquisition.owner_token is not None
        expired_claim = store.claim_dependency_ready_task(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=1),
            now=now,
        )
        assert expired_claim is not None
        environment = EnvironmentAttempt(
            run_id=acquisition.run_id,
            claim_id=expired_claim.id,
            task_id=task.id,
            kind="materialize",
            attempt_number=1,
            fingerprint="sha256:expired-claim",
            status=ExecutionAttemptStatus.RUNNING,
            commands=[["make", "sync"]],
            started_at=now + timedelta(seconds=5),
            finished_at=None,
        )
        agent = AgentAttempt(
            run_id=acquisition.run_id,
            claim_id=expired_claim.id,
            task_id=task.id,
            phase="coding",
            attempt_number=1,
            adapter="codex",
            model="test-model",
            billing_mode=BillingMode.SUBSCRIPTION,
            status=ExecutionAttemptStatus.RUNNING,
            log_path="artifacts/coding.log",
            started_at=now + timedelta(seconds=10),
            finished_at=None,
        )
        store.append_environment_attempt(
            environment,
            acquisition.owner_token,
            expired_claim.claim_token,
            now=now + timedelta(seconds=10),
        )
        store.append_agent_attempt(
            agent,
            acquisition.owner_token,
            expired_claim.claim_token,
            now=now + timedelta(seconds=10),
        )
        resource = ComposeResource(
            run_id=acquisition.run_id,
            claim_id=expired_claim.id,
            task_id=task.id,
            project_name="borg-expired-claim",
            resource_type="project",
            resource_name="borg-expired-claim",
            created_at=now + timedelta(seconds=1),
        )
        store.add_compose_resource(
            resource,
            acquisition.owner_token,
            expired_claim.claim_token,
            now=now,
        )

        store.renew_execution_run(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=5),
            now=now + timedelta(minutes=2),
        )
        persisted = store.list_task_claims(acquisition.run_id)
        assert persisted[0].released_at is None
        assert store.get_task_runtime(task.id).status is TaskRuntimeStatus.CLAIMED
        assert store.list_stale_compose_resources(acquisition.run_id) == [resource]
        closed_environment = store.list_environment_attempts(task.id)
        closed_agent = store.list_agent_attempts(task.id)
        assert closed_environment[0].status is ExecutionAttemptStatus.CANCELLED
        assert closed_environment[0].finished_at == now + timedelta(minutes=2)
        assert closed_environment[0].duration_seconds == 115
        assert closed_agent[0].status is ExecutionAttemptStatus.CANCELLED
        assert closed_agent[0].finished_at == now + timedelta(minutes=2)
        assert closed_agent[0].duration_seconds == 110

        store.confirm_compose_project_cleanup(
            acquisition.run_id,
            task.id,
            resource.project_name,
            now=now + timedelta(minutes=2, seconds=1),
        )
        persisted = store.list_task_claims(acquisition.run_id)
        assert persisted[0].released_at == now + timedelta(minutes=2, seconds=1)
        assert store.get_task_runtime(task.id).status is TaskRuntimeStatus.PENDING
        assert store.list_stale_compose_resources(acquisition.run_id) == []

        replacement_claim = store.claim_dependency_ready_task(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=1),
            now=now + timedelta(minutes=2, seconds=2),
        )
        assert replacement_claim is not None
        assert replacement_claim.id != expired_claim.id
        replacement_resource = ComposeResource(
            run_id=acquisition.run_id,
            claim_id=replacement_claim.id,
            task_id=task.id,
            project_name=resource.project_name,
            resource_type=resource.resource_type,
            resource_name=resource.resource_name,
            created_at=now + timedelta(minutes=2, seconds=3),
        )
        store.add_compose_resource(
            replacement_resource,
            acquisition.owner_token,
            replacement_claim.claim_token,
            now=now + timedelta(minutes=2, seconds=3),
        )

        store.renew_execution_run(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=5),
            now=now + timedelta(minutes=3, seconds=3),
        )
        claims = store.list_task_claims(acquisition.run_id)
        assert claims[1].released_at is None
        assert store.list_compose_resources(task.id) == [
            resource,
            replacement_resource,
        ]

        assert store.confirm_compose_project_cleanup(
            acquisition.run_id,
            task.id,
            resource.project_name,
            now=now + timedelta(minutes=3, seconds=4),
        ) == [resource, replacement_resource]
        claims = store.list_task_claims(acquisition.run_id)
        assert claims[1].released_at == now + timedelta(minutes=3, seconds=4)
        assert "task.claim_expired" in {
            event.kind
            for event in store.list_execution_events(acquisition.run_id)
        }


def test_cleanup_reconciles_expired_claim_attempts_before_reclaim(
    tmp_path: Path, approved_task_generation
) -> None:
    database, borg, generation, task = _execution_fixture(
        tmp_path, approved_task_generation
    )
    now = utcnow()

    with SqliteStore.open(database) as store:
        acquisition = store.acquire_execution_run(
            borg.id,
            generation.id,
            lease_duration=timedelta(minutes=5),
            now=now,
        )
        assert acquisition.owner_token is not None
        claim = store.claim_dependency_ready_task(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=1),
            now=now,
        )
        assert claim is not None
        environment = EnvironmentAttempt(
            run_id=acquisition.run_id,
            claim_id=claim.id,
            task_id=task.id,
            kind="materialize",
            attempt_number=1,
            fingerprint="sha256:cleanup-expiry",
            status=ExecutionAttemptStatus.RUNNING,
            commands=[["make", "sync"]],
            started_at=now + timedelta(seconds=5),
            finished_at=None,
        )
        agent = AgentAttempt(
            run_id=acquisition.run_id,
            claim_id=claim.id,
            task_id=task.id,
            phase="coding",
            attempt_number=1,
            adapter="codex",
            model="test-model",
            billing_mode=BillingMode.SUBSCRIPTION,
            status=ExecutionAttemptStatus.RUNNING,
            log_path="artifacts/coding.log",
            started_at=now + timedelta(seconds=10),
            finished_at=None,
        )
        store.append_environment_attempt(
            environment,
            acquisition.owner_token,
            claim.claim_token,
            now=now + timedelta(seconds=10),
        )
        store.append_agent_attempt(
            agent,
            acquisition.owner_token,
            claim.claim_token,
            now=now + timedelta(seconds=10),
        )
        resource = ComposeResource(
            run_id=acquisition.run_id,
            claim_id=claim.id,
            task_id=task.id,
            project_name="borg-cleanup-after-expiry",
            resource_type="project",
            resource_name="borg-cleanup-after-expiry",
            created_at=now + timedelta(seconds=15),
        )
        store.add_compose_resource(
            resource,
            acquisition.owner_token,
            claim.claim_token,
            now=now + timedelta(seconds=15),
        )

        cleaned_at = now + timedelta(minutes=2)
        assert store.confirm_compose_project_cleanup(
            acquisition.run_id,
            task.id,
            resource.project_name,
            now=cleaned_at,
        ) == [resource]

        persisted_claim = store.list_task_claims(acquisition.run_id)[0]
        assert persisted_claim.released_at == cleaned_at
        assert store.get_task_runtime(task.id).status is TaskRuntimeStatus.PENDING
        assert store.list_stale_compose_resources(acquisition.run_id) == []

        closed_environment = store.list_environment_attempts(task.id)[0]
        closed_agent = store.list_agent_attempts(task.id)[0]
        assert closed_environment.status is ExecutionAttemptStatus.CANCELLED
        assert closed_environment.finished_at == cleaned_at
        assert closed_environment.duration_seconds == 115
        assert closed_agent.status is ExecutionAttemptStatus.CANCELLED
        assert closed_agent.finished_at == cleaned_at
        assert closed_agent.duration_seconds == 110

        events = store.list_execution_events(acquisition.run_id)
        claim_expired = [
            event for event in events if event.kind == "task.claim_expired"
        ]
        assert len(claim_expired) == 1
        assert claim_expired[0].payload == {"claim_id": str(claim.id)}
        assert {
            (event.kind, event.attempt_id)
            for event in events
            if event.attempt_id is not None
        } == {
            ("environment.attempt_interrupted", environment.id),
            ("agent.attempt_interrupted", agent.id),
        }

        replacement = store.claim_dependency_ready_task(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=1),
            now=cleaned_at + timedelta(seconds=1),
        )
        assert replacement is not None
        assert replacement.id != claim.id
        assert replacement.task_id == task.id


def test_attempts_cannot_open_without_live_run_and_claim_authority(
    tmp_path: Path, approved_task_generation
) -> None:
    database, borg, generation, task = _execution_fixture(
        tmp_path, approved_task_generation
    )
    now = utcnow()

    with SqliteStore.open(database) as store:
        acquisition = store.acquire_execution_run(
            borg.id,
            generation.id,
            lease_duration=timedelta(minutes=5),
            now=now,
        )
        assert acquisition.owner_token is not None
        claim = store.claim_dependency_ready_task(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=1),
            now=now,
        )
        assert claim is not None
        environment = EnvironmentAttempt(
            run_id=acquisition.run_id,
            claim_id=claim.id,
            task_id=task.id,
            kind="materialize",
            attempt_number=1,
            fingerprint="sha256:late-environment",
            status=ExecutionAttemptStatus.RUNNING,
            commands=[["make", "sync"]],
            started_at=now + timedelta(seconds=10),
            finished_at=None,
        )
        agent = AgentAttempt(
            run_id=acquisition.run_id,
            claim_id=claim.id,
            task_id=task.id,
            phase="coding",
            attempt_number=1,
            adapter="codex",
            model="test-model",
            billing_mode=BillingMode.SUBSCRIPTION,
            status=ExecutionAttemptStatus.RUNNING,
            log_path="artifacts/coding.log",
            started_at=now + timedelta(seconds=10),
            finished_at=None,
        )

        with pytest.raises(ExecutionOwnershipError, match="ownership changed"):
            store.append_environment_attempt(
                environment,
                "wrong-owner-token",
                claim.claim_token,
                now=now + timedelta(seconds=10),
            )
        with pytest.raises(ExecutionOwnershipError, match="no longer owned"):
            store.append_agent_attempt(
                agent,
                acquisition.owner_token,
                "wrong-claim-token",
                now=now + timedelta(seconds=10),
            )

        with pytest.raises(ExecutionOwnershipError, match="no longer owned"):
            store.append_environment_attempt(
                environment,
                acquisition.owner_token,
                claim.claim_token,
                now=now + timedelta(minutes=2),
            )
        with pytest.raises(ExecutionOwnershipError, match="lease expired"):
            store.append_agent_attempt(
                agent,
                acquisition.owner_token,
                claim.claim_token,
                now=now + timedelta(minutes=6),
            )
        assert store.list_environment_attempts(task.id) == []
        assert store.list_agent_attempts(task.id) == []


def test_compose_resource_cannot_be_persisted_after_expiry_releases_claim(
    tmp_path: Path, approved_task_generation
) -> None:
    database, borg, generation, task = _execution_fixture(
        tmp_path, approved_task_generation
    )
    now = utcnow()

    with SqliteStore.open(database) as stale_worker:
        acquisition = stale_worker.acquire_execution_run(
            borg.id,
            generation.id,
            lease_duration=timedelta(minutes=1),
            now=now,
        )
        assert acquisition.owner_token is not None
        claim = stale_worker.claim_dependency_ready_task(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=1),
            now=now,
        )
        assert claim is not None
        late_resource = ComposeResource(
            run_id=acquisition.run_id,
            claim_id=claim.id,
            task_id=task.id,
            project_name="borg-too-late",
            resource_type="project",
            resource_name="borg-too-late",
            created_at=now + timedelta(minutes=2),
        )

        with SqliteStore.open(database) as reconciler:
            assert reconciler.reconcile_expired_execution_runs(
                now=now + timedelta(minutes=2)
            ) == []
            replacement = reconciler.acquire_execution_run(
                borg.id,
                generation.id,
                lease_duration=timedelta(minutes=5),
                now=now + timedelta(minutes=2),
            )
            assert replacement.owner_token is not None
            replacement_claim = reconciler.claim_dependency_ready_task(
                replacement.run_id,
                replacement.owner_token,
                lease_duration=timedelta(minutes=2),
                now=now + timedelta(minutes=2),
            )
            assert replacement_claim is not None

        with pytest.raises(ExecutionOwnershipError, match="no longer running"):
            stale_worker.add_compose_resource(
                late_resource,
                acquisition.owner_token,
                claim.claim_token,
                now=now + timedelta(minutes=2),
            )
        assert stale_worker.list_compose_resources(task.id) == []


def test_terminal_attempt_events_require_guarded_owned_transition(
    tmp_path: Path, approved_task_generation
) -> None:
    database, borg, generation, task = _execution_fixture(
        tmp_path, approved_task_generation
    )
    now = utcnow()

    with SqliteStore.open(database) as store:
        acquisition = store.acquire_execution_run(
            borg.id,
            generation.id,
            lease_duration=timedelta(minutes=5),
            now=now,
        )
        assert acquisition.owner_token is not None
        claim = store.claim_dependency_ready_task(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=1),
            now=now,
        )
        assert claim is not None
        attempt = EnvironmentAttempt(
            run_id=acquisition.run_id,
            claim_id=claim.id,
            task_id=task.id,
            kind="materialize",
            attempt_number=1,
            fingerprint="sha256:guarded-terminal-event",
            status=ExecutionAttemptStatus.RUNNING,
            commands=[["make", "sync"]],
            started_at=now,
            finished_at=None,
        )
        store.append_environment_attempt(
            attempt,
            acquisition.owner_token,
            claim.claim_token,
            now=now,
        )

        stale_terminal = ExecutionEvent(
            run_id=acquisition.run_id,
            task_id=task.id,
            attempt_id=attempt.id,
            kind="environment.attempt_finished",
            payload={"status": "completed"},
            created_at=now + timedelta(minutes=2),
        )
        wrong_kind = ExecutionEvent(
            run_id=acquisition.run_id,
            task_id=task.id,
            attempt_id=attempt.id,
            kind="agent.attempt_finished",
            payload={"status": "completed"},
            created_at=now + timedelta(seconds=10),
        )
        for event in (stale_terminal, wrong_kind):
            with pytest.raises(
                ValueError, match="require guarded completion"
            ):
                store.append_execution_event(event)

        with pytest.raises(
            sqlite3.IntegrityError, match="does not match attempt"
        ):
            with store.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO execution_events(
                        id, run_id, task_id, attempt_id, kind,
                        payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(wrong_kind.id),
                        str(wrong_kind.run_id),
                        str(wrong_kind.task_id),
                        str(wrong_kind.attempt_id),
                        wrong_kind.kind,
                        "{}",
                        wrong_kind.created_at.isoformat(),
                    ),
                )

        completed = store.complete_environment_attempt(
            attempt.id,
            acquisition.owner_token,
            claim.claim_token,
            status=ExecutionAttemptStatus.COMPLETED,
            result={"exit_code": 0},
            now=now + timedelta(seconds=20),
        )
        assert completed.status is ExecutionAttemptStatus.COMPLETED
        attempt_events = [
            event
            for event in store.list_execution_events(acquisition.run_id)
            if event.attempt_id == attempt.id
        ]
        assert [event.kind for event in attempt_events] == [
            "environment.attempt_finished"
        ]


def test_open_attempts_finish_with_durable_results_and_usage(
    tmp_path: Path, approved_task_generation
) -> None:
    database, borg, generation, task = _execution_fixture(
        tmp_path, approved_task_generation
    )
    now = utcnow()

    with SqliteStore.open(database) as store:
        acquisition = store.acquire_execution_run(
            borg.id,
            generation.id,
            lease_duration=timedelta(minutes=5),
            now=now,
        )
        assert acquisition.owner_token is not None
        claim = store.claim_dependency_ready_task(
            acquisition.run_id,
            acquisition.owner_token,
            lease_duration=timedelta(minutes=2),
            now=now,
        )
        assert claim is not None
        environment = EnvironmentAttempt(
            run_id=acquisition.run_id,
            claim_id=claim.id,
            task_id=task.id,
            kind="materialize",
            attempt_number=1,
            fingerprint="sha256:normal-failure",
            status=ExecutionAttemptStatus.RUNNING,
            commands=[["make", "sync"]],
            started_at=now + timedelta(seconds=5),
            finished_at=None,
        )
        agent = AgentAttempt(
            run_id=acquisition.run_id,
            claim_id=claim.id,
            task_id=task.id,
            phase="coding",
            attempt_number=1,
            adapter="codex",
            model="test-model",
            billing_mode=BillingMode.API,
            status=ExecutionAttemptStatus.RUNNING,
            log_path="artifacts/coding.log",
            started_at=now + timedelta(seconds=10),
            finished_at=None,
        )
        store.append_environment_attempt(
            environment,
            acquisition.owner_token,
            claim.claim_token,
            now=now + timedelta(seconds=10),
        )
        store.append_agent_attempt(
            agent,
            acquisition.owner_token,
            claim.claim_token,
            now=now + timedelta(seconds=10),
        )

        with pytest.raises(ExecutionOwnershipError, match="no longer owned"):
            store.complete_environment_attempt(
                environment.id,
                acquisition.owner_token,
                "wrong-claim-token",
                status=ExecutionAttemptStatus.FAILED,
                now=now + timedelta(seconds=14),
            )
        failed_environment = store.complete_environment_attempt(
            environment.id,
            acquisition.owner_token,
            claim.claim_token,
            status=ExecutionAttemptStatus.FAILED,
            result={"exit_code": 2},
            error="materialization failed",
            now=now + timedelta(seconds=15),
        )
        usage = AgentUsage(
            cost_usd=0.5,
            tokens_input=120,
            tokens_output=30,
            tokens_cache_read=80,
            tokens_cache_write=10,
            num_turns=2,
        )
        completed_agent = store.complete_agent_attempt(
            agent.id,
            acquisition.owner_token,
            claim.claim_token,
            status=AgentStatus.COMPLETED,
            result_path="artifacts/coding.json",
            result={"status": "completed"},
            summary="Implementation completed.",
            duration_seconds=18.5,
            usage=usage,
            now=now + timedelta(seconds=30),
        )

        assert failed_environment.status is ExecutionAttemptStatus.FAILED
        assert failed_environment.result == {"exit_code": 2}
        assert failed_environment.error == "materialization failed"
        assert failed_environment.finished_at == now + timedelta(seconds=15)
        assert failed_environment.duration_seconds == 10
        assert completed_agent.status is ExecutionAttemptStatus.COMPLETED
        assert completed_agent.result_path == "artifacts/coding.json"
        assert completed_agent.result == {"status": "completed"}
        assert completed_agent.summary == "Implementation completed."
        assert completed_agent.duration_seconds == 18.5
        assert completed_agent.usage == usage
        assert completed_agent.finished_at == now + timedelta(seconds=30)

        with pytest.raises(ValueError, match="already finished"):
            store.complete_agent_attempt(
                agent.id,
                acquisition.owner_token,
                claim.claim_token,
                status=ExecutionAttemptStatus.FAILED,
                now=now + timedelta(seconds=31),
            )

        assert store.interrupt_execution_run(
            acquisition.run_id,
            acquisition.owner_token,
            now=now + timedelta(seconds=40),
        ) == []
        attempt_events = [
            event
            for event in store.list_execution_events(acquisition.run_id)
            if event.attempt_id is not None
        ]
        assert [(event.kind, event.attempt_id) for event in attempt_events] == [
            ("environment.attempt_finished", environment.id),
            ("agent.attempt_finished", agent.id),
        ]
        with store.locked_connection() as connection:
            raw_statuses = (
                connection.execute(
                    "SELECT status FROM environment_attempts WHERE id = ?",
                    (str(environment.id),),
                ).fetchone()[0],
                connection.execute(
                    "SELECT status FROM agent_attempts WHERE id = ?",
                    (str(agent.id),),
                ).fetchone()[0],
            )
        assert raw_statuses == ("running", "running")

    with SqliteStore.open(database) as reopened:
        assert reopened.list_environment_attempts(task.id) == [failed_environment]
        assert reopened.list_agent_attempts(task.id) == [completed_agent]
