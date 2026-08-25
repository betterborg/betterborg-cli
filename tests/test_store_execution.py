"""Contracts for migration-006 host-execution ownership state."""

import hashlib
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from betterborg_cli.agent_runtime import AgentStatus, BillingMode
from betterborg_cli.store import (
    AgentAttempt,
    Borg,
    ComposeResource,
    EnvironmentAttempt,
    ExecutionEvent,
    ExecutionRun,
    PlanApproval,
    Repository,
    SqliteStore,
    TaskBatch,
    TaskClaim,
    TaskComplexity,
    TaskGeneration,
    TaskRecord,
    TaskRuntime,
    TaskRuntimeStatus,
)
from betterborg_cli.store.models import utcnow


def _execution_fixture(tmp_path: Path) -> tuple[
    Repository, Borg, PlanApproval, TaskBatch, TaskGeneration, TaskRecord
]:
    repository = Repository(root=tmp_path / "repository")
    borg = Borg(repository_id=repository.id, name="Executor")
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
        manifest={"tasks": ["foundation"]},
    )
    generation = TaskGeneration(
        borg_id=borg.id,
        plan_approval_id=approval.id,
        batch_id=batch.id,
        digest="sha256:generation",
        manifest={"07-host-execution": ["foundation"]},
    )
    body = b"foundation"
    digest = f"sha256:{hashlib.sha256(body).hexdigest()}"
    task = TaskRecord(
        generation_id=generation.id,
        borg_id=borg.id,
        task_ref="foundation",
        stage="07-host-execution",
        stem="01-foundation",
        position=1,
        title="Build execution foundation",
        complexity=TaskComplexity.MEDIUM,
        digest=digest,
        task={"acceptance_criteria": ["execution is durable"]},
        manifest={"task.md": digest},
    )
    durable_root = repository.root / ".borg/tasks" / borg.name / str(generation.id)
    task_path = durable_root / task.stage / f"{task.stem}.md"
    task_path.parent.mkdir(parents=True)
    task_path.write_bytes(body)
    return repository, borg, approval, batch, generation, task


def _store_execution_fixture(
    store: SqliteStore,
    fixture: tuple[
        Repository, Borg, PlanApproval, TaskBatch, TaskGeneration, TaskRecord
    ],
) -> None:
    repository, borg, approval, batch, generation, task = fixture
    store.add_repository(repository)
    store.add_borg(borg)
    store.append_plan_approval(approval)
    store.append_task_batch(batch)
    store.add_task_generation(generation, [task])
    durable_root = repository.root / ".borg/tasks" / borg.name / str(generation.id)
    store._promote_published_task_generation(
        generation.id, durable_root=durable_root
    )


def test_execution_ownership_records_round_trip_after_reopen(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"
    fixture = _execution_fixture(tmp_path)
    _, borg, _, _, generation, task = fixture
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
        cost_usd=0.25,
        tokens_input=100,
        tokens_output=20,
        tokens_cache_read=80,
        num_turns=1,
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
        _store_execution_fixture(store, fixture)
        store.add_execution_run(run)
        store.add_task_runtime(runtime)
        store.append_task_claim(claim)
        store.append_environment_attempt(environment)
        store.append_agent_attempt(agent)
        store.append_execution_event(event)
        store.add_compose_resource(compose)

        assert store.execution_run_owned_by(run.id, run.owner_token)
        assert not store.execution_run_owned_by(run.id, "wrong-token")
        assert store.task_claim_owned_by(claim.id, claim.claim_token)
        assert not store.task_claim_owned_by(claim.id, "wrong-token")

    with SqliteStore.open(database) as reopened:
        assert reopened.applied_migrations() == (1, 2, 3, 4, 5, 6)
        assert reopened.get_execution_run(run.id) == run
        assert reopened.list_execution_runs(borg.id) == [run]
        assert reopened.get_task_runtime(task.id) == runtime
        assert reopened.list_task_runtimes(generation.id) == [runtime]
        assert reopened.list_task_claims(run.id) == [claim]
        assert reopened.list_environment_attempts(task.id) == [environment]
        assert reopened.list_agent_attempts(task.id) == [agent]
        assert reopened.list_execution_events(run.id) == [event]
        assert reopened.list_compose_resources(task.id) == [compose]


def test_live_run_claim_and_token_ownership_are_database_enforced(
    tmp_path: Path,
) -> None:
    fixture = _execution_fixture(tmp_path)
    _, borg, _, _, generation, task = fixture
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

    with SqliteStore.open(tmp_path / "state.sqlite3") as store:
        _store_execution_fixture(store, fixture)
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
def test_execution_history_and_resource_ownership_are_immutable(
    tmp_path: Path, table: str, column: str
) -> None:
    fixture = _execution_fixture(tmp_path)
    _, borg, _, _, generation, task = fixture
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

    with SqliteStore.open(tmp_path / "state.sqlite3") as store:
        _store_execution_fixture(store, fixture)
        store.add_execution_run(run)
        store.append_task_claim(claim)
        store.append_environment_attempt(environment)
        store.append_agent_attempt(agent)
        store.append_execution_event(event)
        store.add_compose_resource(compose)

        with pytest.raises(sqlite3.IntegrityError, match="immutable|append-only"):
            with store.transaction() as connection:
                connection.execute(
                    f"UPDATE {table} SET {column} = ? WHERE id = ?",
                    ("changed", str(records[table].id)),
                )
