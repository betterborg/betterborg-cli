"""Contracts for generation-bound execution decisions."""

import sqlite3
from pathlib import Path

import pytest

from betterborg_cli.planning import TaskPublisher
from betterborg_cli.store import (
    Borg,
    BorgState,
    ExecutionDecision,
    PlanApproval,
    Repository,
    SqliteStore,
    TaskGeneration,
)


def _task_body(round_number: int) -> dict:
    label = f"generation-{round_number}"
    return {
        "stage": "08-estimate-publish",
        "stem": f"{round_number:02d}-{label}",
        "title": f"Publish {label}",
        "why": "Execution decisions must follow the published generation.",
        "scope": [f"Publish {label}."],
        "implementation_notes": [],
        "acceptance_criteria": ["The generation is published."],
        "tests": ["Verify the generation-bound decision."],
        "dependencies": [],
        "out_of_scope": [],
        "plan_refs": ["P1.deliverable.1"],
        "estimate_complexity": "small",
    }


def _publish_generation(
    store: SqliteStore,
    repository: Repository,
    borg: Borg,
    approval: PlanApproval,
    approved_task_generation,
    *,
    round_number: int,
) -> tuple[str, TaskGeneration]:
    fixture = approved_task_generation(
        store,
        borg,
        approval,
        body=_task_body(round_number),
        round_number=round_number,
    )
    generation = (
        TaskPublisher(repository, store).publish(fixture.generation.id).generation
    )
    batch = next(
        batch
        for batch in store.list_task_batches(borg.id)
        if batch.id == generation.batch_id
    )
    return batch.digest, generation


def test_decisions_are_complete_unique_and_current_generation_bound(
    committed_git_repo: Path,
    approved_task_generation,
) -> None:
    database = committed_git_repo.parent / "state" / "borg.sqlite3"
    repository = Repository(root=committed_git_repo)
    borg = Borg(
        repository_id=repository.id,
        name="Estimator",
        state=BorgState.SUPERVISOR_WORKING,
    )
    approval = PlanApproval(
        borg_id=borg.id,
        plan_digest="sha256:approved-plan",
        manifest={"plan.md": "sha256:approved-plan"},
    )

    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.append_plan_approval(approval)
        first_batch_digest, first_generation = _publish_generation(
            store,
            repository,
            borg,
            approval,
            approved_task_generation,
            round_number=1,
        )
        first = ExecutionDecision(
            borg_id=borg.id,
            generation_id=first_generation.id,
            approved_plan_digest=approval.plan_digest,
            task_batch_digest=first_batch_digest,
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
            task_batch_digest=first_batch_digest,
            estimate_version="execution-estimate-v1",
            source="noninteractive",
            snapshot=first.snapshot,
            decision="approved",
        )
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            store.append_execution_decision(duplicate)

        second_batch_digest, second_generation = _publish_generation(
            store,
            repository,
            borg,
            approval,
            approved_task_generation,
            round_number=2,
        )
        assert store.get_current_execution_decision(borg.id) is None
        assert store.get_execution_decision(borg.id, first_generation.id) == first

        stale = ExecutionDecision(
            borg_id=borg.id,
            generation_id=first_generation.id,
            approved_plan_digest=approval.plan_digest,
            task_batch_digest=first_batch_digest,
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
            task_batch_digest=second_batch_digest,
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
                    "UPDATE execution_decisions SET decision = 'rejected' WHERE id = ?",
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


def test_decision_digests_must_match_the_current_generation(
    committed_git_repo: Path,
    approved_task_generation,
) -> None:
    repository = Repository(root=committed_git_repo)
    borg = Borg(
        repository_id=repository.id,
        name="DigestBound",
        state=BorgState.SUPERVISOR_WORKING,
    )
    approval = PlanApproval(
        borg_id=borg.id,
        plan_digest="sha256:approved-plan",
        manifest={},
    )

    with SqliteStore.open(committed_git_repo.parent / "borg.sqlite3") as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.append_plan_approval(approval)
        batch_digest, generation = _publish_generation(
            store,
            repository,
            borg,
            approval,
            approved_task_generation,
            round_number=1,
        )
        mismatched = ExecutionDecision(
            borg_id=borg.id,
            generation_id=generation.id,
            approved_plan_digest=approval.plan_digest,
            task_batch_digest=f"{batch_digest}-changed",
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
