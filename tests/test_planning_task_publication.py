"""Durability contracts for immutable published task Markdown."""

from __future__ import annotations

import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from betterborg_cli.planning import (
    TaskDigestDriftError,
    TaskPublisher,
    approved_plan_digest,
    render_task_markdown,
    task_markdown_digest,
)
from betterborg_cli.store import (
    Borg,
    BorgState,
    PlanApproval,
    PlanningAttempt,
    PlanningAttemptStatus,
    Repository,
    SqliteStore,
    TaskBatch,
    TaskComplexity,
    TaskGeneration,
    TaskGenerationStatus,
    TaskRecord,
)


class InjectedPublicationFailure(RuntimeError):
    pass


def _task_body(stem: str) -> dict:
    return {
        "stage": "01-foundation",
        "stem": stem,
        "title": f"Implement {stem}",
        "why": "The approved plan needs a durable foundation.",
        "scope": ["Add the smallest production behavior."],
        "implementation_notes": [],
        "acceptance_criteria": ["The behavior is externally observable."],
        "tests": ["Assert the public behavior."],
        "dependencies": [],
        "out_of_scope": [],
        "plan_refs": ["P1.deliverable.1"],
        "estimate_complexity": "small",
    }


def _add_approved_generation(
    store: SqliteStore,
    borg: Borg,
    approval: PlanApproval,
    *,
    stem: str,
    round_number: int,
) -> TaskGeneration:
    attempt = PlanningAttempt(
        borg_id=borg.id,
        phase="supervisor_review",
        round=round_number,
        adapter="mock",
        model="test-model",
    )
    batch = TaskBatch(
        borg_id=borg.id,
        plan_approval_id=approval.id,
        attempt_id=attempt.id,
        round=round_number,
        digest=f"sha256:batch-{round_number}",
        manifest={},
    )
    generation_id = uuid4()
    body = _task_body(stem)
    digest = task_markdown_digest(render_task_markdown(body))
    task = TaskRecord(
        generation_id=generation_id,
        borg_id=borg.id,
        task_ref=f"T-{generation_id.hex}",
        stage=body["stage"],
        stem=stem,
        position=1,
        title=body["title"],
        complexity=TaskComplexity.SMALL,
        digest=digest,
        task=body,
        manifest={"approved_plan_digest": approval.plan_digest, "task.md": digest},
    )
    relative_path = (
        f".borg/tasks/{borg.name}/{generation_id}/{task.stage}/{task.stem}.md"
    )
    manifest = {
        "approved_plan_digest": approval.plan_digest,
        "batch_digest": batch.digest,
        "dependencies": [],
        "plan_approval_id": str(approval.id),
        "tasks": [
            {
                "digest": digest,
                "path": relative_path,
                "position": 1,
                "task_ref": task.task_ref,
            }
        ],
    }
    generation = TaskGeneration(
        id=generation_id,
        borg_id=borg.id,
        plan_approval_id=approval.id,
        batch_id=batch.id,
        digest=approved_plan_digest(manifest),
        manifest=manifest,
    )
    attempt = PlanningAttempt(
        id=attempt.id,
        borg_id=borg.id,
        phase=attempt.phase,
        round=attempt.round,
        adapter=attempt.adapter,
        model=attempt.model,
        request={
            "batch_id": str(batch.id),
            "generation_id": str(generation.id),
        },
    )
    store.append_planning_attempt(attempt)
    store.append_task_batch(batch)
    store.add_task_generation(generation, [task])
    store.complete_planning_attempt(
        attempt.id,
        status=PlanningAttemptStatus.COMPLETED,
        result={"decision": "approve", "summary": "Ready.", "findings": []},
        summary="Ready.",
    )
    return generation


def _publication_context(
    root: Path, database: Path
) -> tuple[Repository, Borg, PlanApproval]:
    repository = Repository(root=root)
    borg = Borg(
        repository_id=repository.id,
        name="durable-tasks",
        state=BorgState.TASKS_APPROVAL_PENDING,
    )
    approval = PlanApproval(
        borg_id=borg.id,
        plan_digest="sha256:approved-plan",
        manifest={},
    )
    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.append_plan_approval(approval)
    return repository, borg, approval


def test_renders_canonical_inspectable_markdown() -> None:
    body = _task_body("01-publish")

    rendered = render_task_markdown(dict(reversed(tuple(body.items()))))

    assert rendered == """# Implement 01-publish

## Why
The approved plan needs a durable foundation.

## Scope
- Add the smallest production behavior.

## Implementation Notes
(none)

## Acceptance Criteria
- The behavior is externally observable.

## Tests
- Assert the public behavior.

## Dependencies
(none)

## Out of Scope
(none)
"""


def test_publishes_exact_tracked_generation_and_blocks_digest_drift(
    committed_git_repo: Path,
) -> None:
    database = committed_git_repo.parent / "task-publication.sqlite3"
    repository, borg, approval = _publication_context(committed_git_repo, database)
    with SqliteStore.open(database) as store:
        generation = _add_approved_generation(
            store, borg, approval, stem="01-publish", round_number=1
        )
        publication = TaskPublisher(repository, store).publish(generation.id)

        expected = (
            committed_git_repo
            / ".borg/tasks/durable-tasks"
            / str(generation.id)
            / "01-foundation/01-publish.md"
        )
        assert publication.generation.status is TaskGenerationStatus.CURRENT
        assert [item.path for item in publication.files] == [expected]
        assert expected.read_text(encoding="utf-8") == render_task_markdown(
            _task_body("01-publish")
        )
        assert store.get_current_task_generation(borg.id) == publication.generation
        assert not (committed_git_repo / ".borg/state/task-staging").is_symlink()
        ignored = subprocess.run(
            [
                "git",
                "-C",
                str(committed_git_repo),
                "check-ignore",
                ".borg/state/task-staging/probe",
            ],
            check=False,
        )
        tracked = subprocess.run(
            ["git", "-C", str(committed_git_repo), "check-ignore", str(expected)],
            check=False,
        )
        assert ignored.returncode == 0
        assert tracked.returncode == 1

        expected.write_text("# drifted\n", encoding="utf-8")
        with pytest.raises(TaskDigestDriftError, match="digest drifted"):
            TaskPublisher(repository, store).current_task_files(borg.id)


@pytest.mark.parametrize(
    "failure_point",
    [
        "after_preparing",
        "during_staging",
        "after_file_fsync",
        "after_rename",
        "during_parent_fsync",
        "after_parent_fsync",
        "before_db_commit",
        "after_db_commit",
    ],
)
def test_reopens_and_resumes_every_publication_boundary(
    committed_git_repo: Path,
    failure_point: str,
) -> None:
    database = committed_git_repo.parent / f"failure-{failure_point}.sqlite3"
    repository, borg, approval = _publication_context(committed_git_repo, database)
    with SqliteStore.open(database) as store:
        prior = _add_approved_generation(
            store, borg, approval, stem="01-prior", round_number=1
        )
        prior_current = TaskPublisher(repository, store).publish(prior.id).generation
        replacement = _add_approved_generation(
            store, borg, approval, stem="02-replacement", round_number=2
        )

        def fail_at(point: str) -> None:
            if point == failure_point:
                raise InjectedPublicationFailure(point)

        with pytest.raises(InjectedPublicationFailure, match=failure_point):
            TaskPublisher(
                repository, store, failure_injector=fail_at
            ).publish(replacement.id)

        current = store.get_current_task_generation(borg.id)
        if failure_point == "after_db_commit":
            assert current is not None and current.id == replacement.id
        else:
            assert current == prior_current

    with SqliteStore.open(database) as reopened:
        resumed = TaskPublisher(repository, reopened).reconcile(borg.id)

        assert resumed is not None
        assert resumed.generation.status is TaskGenerationStatus.CURRENT
        assert reopened.get_current_task_generation(borg.id) == resumed.generation
        assert [item.task.stem for item in resumed.files] == ["02-replacement"]
        generation_root = committed_git_repo / ".borg/tasks/durable-tasks"
        assert [path.name for path in generation_root.iterdir()] == [
            str(replacement.id)
        ]
        assert task_markdown_digest(resumed.files[0].path.read_bytes()) == (
            resumed.files[0].task.digest
        )
        assert reopened.get_task_generation(prior.id).status is (
            TaskGenerationStatus.SUPERSEDED
        )


def test_reconcile_resumes_interrupted_first_publication(
    committed_git_repo: Path,
) -> None:
    database = committed_git_repo.parent / "reconcile-first.sqlite3"
    repository, borg, approval = _publication_context(committed_git_repo, database)
    with SqliteStore.open(database) as store:
        generation = _add_approved_generation(
            store, borg, approval, stem="01-first", round_number=1
        )

        def fail_after_rename(point: str) -> None:
            if point == "after_rename":
                raise InjectedPublicationFailure(point)

        with pytest.raises(InjectedPublicationFailure, match="after_rename"):
            TaskPublisher(
                repository, store, failure_injector=fail_after_rename
            ).publish(generation.id)

        assert store.get_current_task_generation(borg.id) is None

    with SqliteStore.open(database) as reopened:
        resumed = TaskPublisher(repository, reopened).reconcile(borg.id)

        assert resumed is not None
        assert resumed.generation.id == generation.id
        assert resumed.generation.status is TaskGenerationStatus.CURRENT
        assert reopened.get_current_task_generation(borg.id) == resumed.generation
        assert [item.task.stem for item in resumed.files] == ["01-first"]
