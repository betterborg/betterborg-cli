"""Contracts for generation-bound execution decisions."""

import hashlib
import sqlite3
from pathlib import Path

import pytest

from betterborg_cli.store import (
    Borg,
    ExecutionDecision,
    PlanApproval,
    Repository,
    SqliteStore,
    TaskBatch,
    TaskComplexity,
    TaskGeneration,
    TaskRecord,
)


def _publish_generation(
    store: SqliteStore,
    repository: Repository,
    borg: Borg,
    approval: PlanApproval,
    *,
    round: int,
) -> tuple[TaskBatch, TaskGeneration]:
    label = f"generation-{round}"
    batch = TaskBatch(
        borg_id=borg.id,
        plan_approval_id=approval.id,
        round=round,
        digest=f"sha256:batch-{round}",
        manifest={"tasks": [label]},
    )
    generation = TaskGeneration(
        borg_id=borg.id,
        plan_approval_id=approval.id,
        batch_id=batch.id,
        digest=f"sha256:{label}",
        manifest={"tasks": [label]},
    )
    body = label.encode()
    task = TaskRecord(
        generation_id=generation.id,
        borg_id=borg.id,
        task_ref=label,
        stage="08-estimate-publish",
        stem=f"{round:02d}-{label}",
        position=1,
        title=f"Publish {label}",
        complexity=TaskComplexity.SMALL,
        digest=f"sha256:{hashlib.sha256(body).hexdigest()}",
        task={"acceptance_criteria": ["published"]},
        manifest={"task.md": label},
    )
    store.append_task_batch(batch)
    store.add_task_generation(generation, [task])
    durable_root = (
        repository.root / ".borg" / "tasks" / borg.name / str(generation.id)
    )
    task_path = durable_root / task.stage / f"{task.stem}.md"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_bytes(body)
    return batch, store._promote_published_task_generation(
        generation.id, durable_root=durable_root
    )


def test_decisions_are_complete_unique_and_current_generation_bound(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state" / "borg.sqlite3"
    repository = Repository(root=tmp_path / "repository")
    borg = Borg(repository_id=repository.id, name="Estimator")
    approval = PlanApproval(
        borg_id=borg.id,
        plan_digest="sha256:approved-plan",
        manifest={"plan.md": "sha256:approved-plan"},
    )

    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.append_plan_approval(approval)
        first_batch, first_generation = _publish_generation(
            store, repository, borg, approval, round=1
        )
        first = ExecutionDecision(
            borg_id=borg.id,
            generation_id=first_generation.id,
            approved_plan_digest=approval.plan_digest,
            task_batch_digest=first_batch.digest,
            estimate_version="execution-estimate-v1",
            source="interactive",
            snapshot={
                "generation_id": str(first_generation.id),
                "time": {"p50": 1800.0, "p80": 3600.0, "unit": "seconds"},
                "billing": {"subscription": {"included": True}},
            },
            decision="approved",
        )
        store.append_execution_decision(first)

        assert store.get_execution_decision(borg.id, first_generation.id) == first
        assert store.get_current_execution_decision(borg.id) == first
        with store.locked_connection() as connection:
            migration_applied_at = connection.execute(
                "SELECT applied_at FROM schema_version WHERE version = 11"
            ).fetchone()[0]

        duplicate = ExecutionDecision(
            borg_id=borg.id,
            generation_id=first_generation.id,
            approved_plan_digest=approval.plan_digest,
            task_batch_digest=first_batch.digest,
            estimate_version="execution-estimate-v1",
            source="noninteractive",
            snapshot=first.snapshot,
            decision="approved",
        )
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            store.append_execution_decision(duplicate)

        second_batch, second_generation = _publish_generation(
            store, repository, borg, approval, round=2
        )
        assert store.get_current_execution_decision(borg.id) is None
        assert store.get_execution_decision(borg.id, first_generation.id) == first

        stale = ExecutionDecision(
            borg_id=borg.id,
            generation_id=first_generation.id,
            approved_plan_digest=approval.plan_digest,
            task_batch_digest=first_batch.digest,
            estimate_version="execution-estimate-v2",
            source="interactive",
            snapshot={"generation_id": str(first_generation.id)},
            decision="approved",
        )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="does not match the current generation",
        ):
            store.append_execution_decision(stale)

        second = ExecutionDecision(
            borg_id=borg.id,
            generation_id=second_generation.id,
            approved_plan_digest=approval.plan_digest,
            task_batch_digest=second_batch.digest,
            estimate_version="execution-estimate-v2",
            source="noninteractive",
            snapshot={"generation_id": str(second_generation.id)},
            decision="approved",
        )
        store.append_execution_decision(second)
        assert store.get_current_execution_decision(borg.id) == second

        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with store.transaction() as connection:
                connection.execute(
                    "UPDATE execution_decisions SET decision = 'rejected' "
                    "WHERE id = ?",
                    (str(second.id),),
                )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with store.transaction() as connection:
                connection.execute(
                    "DELETE FROM execution_decisions WHERE id = ?",
                    (str(second.id),),
                )

    with SqliteStore.open(database) as reopened:
        assert reopened.applied_migrations() == tuple(range(1, 12))
        assert reopened.get_execution_decision(borg.id, first_generation.id) == first
        assert reopened.get_current_execution_decision(borg.id) == second
        with reopened.locked_connection() as connection:
            reopened_applied_at = connection.execute(
                "SELECT applied_at FROM schema_version WHERE version = 11"
            ).fetchone()[0]

    assert reopened_applied_at == migration_applied_at


def test_decision_digests_must_match_the_current_generation(tmp_path: Path) -> None:
    repository = Repository(root=tmp_path / "repository")
    borg = Borg(repository_id=repository.id, name="DigestBound")
    approval = PlanApproval(
        borg_id=borg.id,
        plan_digest="sha256:approved-plan",
        manifest={},
    )

    with SqliteStore.open(tmp_path / "borg.sqlite3") as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.append_plan_approval(approval)
        batch, generation = _publish_generation(
            store, repository, borg, approval, round=1
        )
        mismatched = ExecutionDecision(
            borg_id=borg.id,
            generation_id=generation.id,
            approved_plan_digest=approval.plan_digest,
            task_batch_digest=f"{batch.digest}-changed",
            estimate_version="execution-estimate-v1",
            source="interactive",
            snapshot={"generation_id": str(generation.id)},
            decision="approved",
        )

        with pytest.raises(
            sqlite3.IntegrityError,
            match="does not match the current generation",
        ):
            store.append_execution_decision(mismatched)

        assert store.get_current_execution_decision(borg.id) is None
