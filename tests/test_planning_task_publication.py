"""Durability contracts for immutable published task Markdown."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from betterborg_cli.agent_runtime import CancellationToken, run_captured
from betterborg_cli.planning import (
    TaskDigestDriftError,
    TaskPublicationError,
    TaskPublisher,
    render_task_markdown,
    task_markdown_digest,
)
from betterborg_cli.planning import task_publication as publication_module
from betterborg_cli.repository_files import require_git_trackable
from betterborg_cli.run_control import RunControl
from betterborg_cli.store import (
    Borg,
    BorgState,
    PlanApproval,
    Repository,
    SqliteStore,
    TaskGenerationStatus,
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
    approved_task_generation,
) -> None:
    database = committed_git_repo.parent / "task-publication.sqlite3"
    repository, borg, approval = _publication_context(committed_git_repo, database)
    with SqliteStore.open(database) as store:
        generation = approved_task_generation(
            store,
            borg,
            approval,
            body=_task_body("01-publish"),
            round_number=1,
        ).generation
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


def test_publication_visibility_uses_run_token_before_database_promotion(
    committed_git_repo: Path,
    approved_task_generation,
    real_process_harness: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = committed_git_repo.parent / "cancelled-publication.sqlite3"
    repository, borg, approval = _publication_context(committed_git_repo, database)
    with SqliteStore.open(database) as store:
        generation = approved_task_generation(
            store,
            borg,
            approval,
            body=_task_body("01-cancelled"),
            round_number=1,
        ).generation

    publication_cancel = CancellationToken()
    errors: list[BaseException] = []
    checkpoints: list[str] = []

    def blocked_visibility(path: Path, *, root: Path, cancel=None) -> None:
        assert cancel is publication_cancel

        def runner(
            _command: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            return run_captured(
                real_process_harness.resistant_argv("publication-visibility"),
                check=kwargs["check"],
                cancel=kwargs["cancel"],
            )

        require_git_trackable(
            path,
            root=root,
            cancel=cancel,
            command_runner=runner,
        )

    monkeypatch.setattr(
        publication_module,
        "require_git_trackable",
        blocked_visibility,
    )

    def publish() -> None:
        try:
            with SqliteStore.open(database) as store:
                TaskPublisher(
                    repository,
                    store,
                    failure_injector=checkpoints.append,
                    cancel=publication_cancel,
                ).publish(generation.id)
        except BaseException as error:
            errors.append(error)

    exits: list[int] = []
    control = RunControl(publication_cancel, exit_function=exits.append).install()
    worker = threading.Thread(target=publish)
    try:
        worker.start()
        real_process_harness.wait_for_marker("publication-visibility.parent.pid")
        real_process_harness.wait_for_marker("publication-visibility.child.pid")
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except KeyboardInterrupt:
            pass
        assert control.wait_for_cancellation(timeout=1)
        worker.join(timeout=2)
    finally:
        control.close()
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], TaskPublicationError)
    assert "before_db_commit" not in checkpoints
    assert exits == [130]
    with SqliteStore.open(database) as store:
        persisted = store.get_task_generation(generation.id)
    assert persisted is not None
    assert persisted.status is TaskGenerationStatus.PREPARING
    real_process_harness.assert_tree_absent("publication-visibility")


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
    approved_task_generation,
) -> None:
    database = committed_git_repo.parent / f"failure-{failure_point}.sqlite3"
    repository, borg, approval = _publication_context(committed_git_repo, database)
    with SqliteStore.open(database) as store:
        prior = approved_task_generation(
            store,
            borg,
            approval,
            body=_task_body("01-prior"),
            round_number=1,
        ).generation
        prior_current = TaskPublisher(repository, store).publish(prior.id).generation
        replacement = approved_task_generation(
            store,
            borg,
            approval,
            body=_task_body("02-replacement"),
            round_number=2,
        ).generation

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
    approved_task_generation,
) -> None:
    database = committed_git_repo.parent / "reconcile-first.sqlite3"
    repository, borg, approval = _publication_context(committed_git_repo, database)
    with SqliteStore.open(database) as store:
        generation = approved_task_generation(
            store,
            borg,
            approval,
            body=_task_body("01-first"),
            round_number=1,
        ).generation

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
