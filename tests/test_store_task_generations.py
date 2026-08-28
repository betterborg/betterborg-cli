"""Contracts for immutable, SQLite-current task generations."""

import sqlite3
from pathlib import Path

import pytest

from betterborg_cli.store import (
    Borg,
    PlanApproval,
    Repository,
    SqliteStore,
    TaskBatch,
    TaskComplexity,
    TaskDependency,
    TaskFinding,
    TaskGeneration,
    TaskGenerationStatus,
    TaskRecord,
)


def _task(
    generation: TaskGeneration,
    *,
    task_ref: str,
    position: int,
    complexity: TaskComplexity,
) -> TaskRecord:
    return TaskRecord(
        generation_id=generation.id,
        borg_id=generation.borg_id,
        task_ref=task_ref,
        stage="06-task-decomposition",
        stem=f"{position:02d}-{task_ref}",
        position=position,
        title=f"Implement {task_ref}",
        complexity=complexity,
        digest=f"sha256:task-{task_ref}",
        task={"acceptance_criteria": [f"{task_ref} works"]},
        manifest={"task.md": f"sha256:markdown-{task_ref}"},
    )


def test_generation_transitions_preserve_immutable_history_after_reopen(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.sqlite3"
    repository = Repository(root=tmp_path / "repository")
    borg = Borg(repository_id=repository.id, name="TaskPlanner")
    approval = PlanApproval(
        borg_id=borg.id,
        plan_digest="sha256:approved-plan",
        manifest={"plan.json": "sha256:approved-plan"},
        approved_by="operator",
    )
    first_batch = TaskBatch(
        borg_id=borg.id,
        plan_approval_id=approval.id,
        round=1,
        digest="sha256:first-batch",
        manifest={"task-refs": ["foundation", "consumer"]},
        summary="Initial decomposition.",
    )
    finding = TaskFinding(
        borg_id=borg.id,
        batch_id=first_batch.id,
        round=1,
        severity="minor",
        message="Keep the dependency explicit.",
        task_ref="consumer",
    )
    first_generation = TaskGeneration(
        borg_id=borg.id,
        plan_approval_id=approval.id,
        batch_id=first_batch.id,
        digest="sha256:first-generation",
        manifest={"06-task-decomposition": ["foundation", "consumer"]},
    )
    foundation = _task(
        first_generation,
        task_ref="foundation",
        position=1,
        complexity=TaskComplexity.SMALL,
    )
    consumer = _task(
        first_generation,
        task_ref="consumer",
        position=2,
        complexity=TaskComplexity.MEDIUM,
    )
    dependency = TaskDependency(
        generation_id=first_generation.id,
        task_id=consumer.id,
        depends_on_task_id=foundation.id,
    )

    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.append_plan_approval(approval)
        store.append_task_batch(first_batch)
        store.append_task_finding(finding)
        store.add_task_generation(
            first_generation, [foundation, consumer], [dependency]
        )

        assert store.get_current_task_generation(borg.id) is None
        first_current = store.promote_task_generation(first_generation.id)
        assert first_current.status is TaskGenerationStatus.CURRENT
        assert first_current.current_at is not None

        second_batch = TaskBatch(
            borg_id=borg.id,
            plan_approval_id=approval.id,
            round=2,
            digest="sha256:second-batch",
            manifest={"task-refs": ["replacement"]},
        )
        store.append_task_batch(second_batch)
        second_generation = TaskGeneration(
            borg_id=borg.id,
            plan_approval_id=approval.id,
            batch_id=second_batch.id,
            digest="sha256:second-generation",
            manifest={"06-task-decomposition": ["replacement"]},
        )
        replacement = _task(
            second_generation,
            task_ref="replacement",
            position=1,
            complexity=TaskComplexity.LARGE,
        )
        store.add_task_generation(second_generation, [replacement])
        assert store.get_current_task_generation(borg.id) == first_current

        second_current = store.promote_task_generation(second_generation.id)
        generations = store.list_task_generations(borg.id)
        first_superseded = generations[0]
        assert first_superseded.status is TaskGenerationStatus.SUPERSEDED
        assert first_superseded.current_at == first_current.current_at
        assert first_superseded.superseded_at == second_current.current_at
        assert generations[1] == second_current
        assert store.get_current_task_generation(borg.id) == second_current

    with SqliteStore.open(database) as reopened:
        assert reopened.applied_migrations() == (1, 2, 3, 4, 5)
        assert reopened.list_plan_approvals(borg.id) == [approval]
        assert reopened.list_task_batches(borg.id) == [first_batch, second_batch]
        assert reopened.list_task_findings(borg.id) == [finding]
        assert reopened.list_task_generations(borg.id) == generations
        assert reopened.list_task_records(first_generation.id) == [
            foundation,
            consumer,
        ]
        assert reopened.list_task_dependencies(first_generation.id) == [dependency]


def test_generation_rows_and_current_visibility_are_database_enforced(
    tmp_path: Path,
) -> None:
    repository = Repository(root=tmp_path / "repository")
    borg = Borg(repository_id=repository.id, name="ImmutableTasks")
    approval = PlanApproval(
        borg_id=borg.id,
        plan_digest="sha256:plan",
        manifest={},
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
    first = _task(
        generation,
        task_ref="first",
        position=1,
        complexity=TaskComplexity.SMALL,
    )
    second = _task(
        generation,
        task_ref="second",
        position=2,
        complexity=TaskComplexity.SMALL,
    )
    dependency = TaskDependency(
        generation_id=generation.id,
        task_id=second.id,
        depends_on_task_id=first.id,
    )

    with SqliteStore.open(tmp_path / "state.sqlite3") as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.append_plan_approval(approval)
        store.append_task_batch(batch)
        store.add_task_generation(generation, [first, second], [dependency])
        current = store.promote_task_generation(generation.id)

        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with store.transaction() as connection:
                connection.execute(
                    "UPDATE task_records SET title = ? WHERE id = ?",
                    ("Changed", str(first.id)),
                )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            with store.transaction() as connection:
                connection.execute(
                    "DELETE FROM task_dependencies WHERE id = ?",
                    (str(dependency.id),),
                )
        with pytest.raises(sqlite3.IntegrityError, match="no longer preparing"):
            with store.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO task_records(
                        id, generation_id, borg_id, task_ref, stage, stem,
                        position, title, complexity, task_json, manifest_json,
                        digest, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "late-task",
                        str(generation.id),
                        str(borg.id),
                        "late",
                        "stage",
                        "late",
                        3,
                        "Late task",
                        "small",
                        "{}",
                        "{}",
                        "sha256:late",
                        current.current_at.isoformat(),
                    ),
                )
        with pytest.raises(ValueError, match="only a preparing"):
            store.promote_task_generation(generation.id)

        next_batch = TaskBatch(
            borg_id=borg.id,
            plan_approval_id=approval.id,
            round=2,
            digest="sha256:next-batch",
            manifest={},
        )
        store.append_task_batch(next_batch)
        next_generation = TaskGeneration(
            borg_id=borg.id,
            plan_approval_id=approval.id,
            batch_id=next_batch.id,
            digest="sha256:next-generation",
            manifest={},
        )
        store.add_task_generation(next_generation)
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            with store.transaction() as connection:
                connection.execute(
                    """
                    UPDATE task_generations
                    SET status = 'current', current_at = ?
                    WHERE id = ?
                    """,
                    (current.current_at.isoformat(), str(next_generation.id)),
                )

        assert store.get_current_task_generation(borg.id) == current
        assert store.get_task_generation(next_generation.id) == next_generation
        assert store.list_task_records(generation.id) == [first, second]
        assert store.list_task_dependencies(generation.id) == [dependency]
