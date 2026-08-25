"""CLI contracts for inspecting the SQLite-current task generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from click.testing import CliRunner

from betterborg_cli.cli import cli
from betterborg_cli.planning import (
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
    SqliteStore,
    TaskBatch,
    TaskComplexity,
    TaskGeneration,
    TaskGenerationStatus,
    TaskRecord,
)


@dataclass(frozen=True)
class GenerationFixture:
    generation: TaskGeneration
    task: TaskRecord


def _task_body(stem: str, title: str) -> dict:
    return {
        "stage": "01-foundation",
        "stem": stem,
        "title": title,
        "why": "The approved plan needs an executable task.",
        "scope": [f"Implement {title.casefold()}."],
        "implementation_notes": [],
        "acceptance_criteria": [f"{title} is complete."],
        "tests": [f"Verify {title.casefold()}."],
        "dependencies": [],
        "out_of_scope": [],
        "plan_refs": ["P1.deliverable.1"],
        "estimate_complexity": "small",
    }


def _add_generation(
    store: SqliteStore,
    borg: Borg,
    approval: PlanApproval,
    *,
    stem: str,
    title: str,
    round_number: int,
) -> GenerationFixture:
    attempt_id = uuid4()
    attempt = PlanningAttempt(
        id=attempt_id,
        borg_id=borg.id,
        phase="supervisor_review",
        round=round_number,
        adapter="mock",
        model="test-model",
        request={},
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
    body = _task_body(stem, title)
    digest = task_markdown_digest(render_task_markdown(body))
    record = TaskRecord(
        generation_id=generation_id,
        borg_id=borg.id,
        task_ref=f"T-{round_number}",
        stage=body["stage"],
        stem=stem,
        position=1,
        title=title,
        complexity=TaskComplexity.SMALL,
        digest=digest,
        task=body,
        manifest={"approved_plan_digest": approval.plan_digest, "task.md": digest},
    )
    relative_path = (
        f".borg/tasks/{borg.name}/{generation_id}/{record.stage}/{record.stem}.md"
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
                "position": record.position,
                "task_ref": record.task_ref,
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
        borg_id=attempt.borg_id,
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
    store.add_task_generation(generation, [record])
    store.complete_planning_attempt(
        attempt.id,
        status=PlanningAttemptStatus.COMPLETED,
        result={"decision": "approve", "summary": "Ready.", "findings": []},
        summary="Ready.",
    )
    return GenerationFixture(generation=generation, task=record)


def _multiple_generations(
    root: Path,
    planning_cli_repository,
) -> tuple[Path, GenerationFixture, GenerationFixture, GenerationFixture]:
    repository, paths = planning_cli_repository(root, "inspect-tasks")
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "inspect-tasks")
        assert borg is not None
        borg = store.compare_and_set_borg_state(
            borg.id,
            expected_state=borg.state,
            expected_version=borg.state_version,
            new_state=BorgState.READY_TO_EXECUTE,
        )
        approval = PlanApproval(
            borg_id=borg.id,
            plan_digest="sha256:approved-plan",
            manifest={},
        )
        store.append_plan_approval(approval)
        superseded = _add_generation(
            store,
            borg,
            approval,
            stem="01-superseded",
            title="Superseded task",
            round_number=1,
        )
        TaskPublisher(repository, store).publish(superseded.generation.id)
        current = _add_generation(
            store,
            borg,
            approval,
            stem="02-current",
            title="Current task",
            round_number=2,
        )
        TaskPublisher(repository, store).publish(current.generation.id)
        preparing = _add_generation(
            store,
            borg,
            approval,
            stem="03-preparing",
            title="Preparing task",
            round_number=3,
        )
    return paths.state_dir / "borg.sqlite3", superseded, current, preparing


def test_task_list_exposes_only_current_and_does_not_reconcile_other_trees(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    monkeypatch,
) -> None:
    database, superseded, current, preparing = _multiple_generations(
        committed_git_repo, planning_cli_repository
    )
    stale_tree = (
        committed_git_repo
        / ".borg/tasks/inspect-tasks"
        / str(superseded.generation.id)
    )
    stale_tree.mkdir()
    (stale_tree / "stale.md").write_text("stale\n", encoding="utf-8")
    monkeypatch.chdir(committed_git_repo)

    result = cli_runner.invoke(cli, ["task", "list", "inspect-tasks"])

    assert result.exit_code == 0, result.output
    assert str(current.generation.id) in result.output
    assert "01-foundation/02-current [small] Current task" in result.output
    assert current.task.task_ref in result.output
    assert current.generation.manifest["tasks"][0]["path"] in result.output
    assert "Superseded task" not in result.output
    assert "Preparing task" not in result.output
    assert stale_tree.is_dir()
    with SqliteStore.open(database) as store:
        generations = store.list_task_generations(current.task.borg_id)
        assert [item.status for item in generations] == [
            TaskGenerationStatus.SUPERSEDED,
            TaskGenerationStatus.CURRENT,
            TaskGenerationStatus.PREPARING,
        ]
        assert preparing.generation in generations


def test_task_list_json_describes_only_verified_current_records(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    monkeypatch,
) -> None:
    _database, _superseded, current, _preparing = _multiple_generations(
        committed_git_repo, planning_cli_repository
    )
    monkeypatch.chdir(committed_git_repo)

    result = cli_runner.invoke(
        cli, ["task", "list", "inspect-tasks", "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
        "approved_plan_digest": "sha256:approved-plan",
        "borg": "inspect-tasks",
        "generation_digest": current.generation.digest,
        "generation_id": str(current.generation.id),
        "tasks": [
            {
                "complexity": "small",
                "dependencies": [],
                "digest": current.task.digest,
                "path": current.generation.manifest["tasks"][0]["path"],
                "position": 1,
                "stage": "01-foundation",
                "stem": "02-current",
                "task_ref": current.task.task_ref,
                "title": "Current task",
            }
        ],
    }


def test_task_show_renders_current_markdown_and_rejects_noncurrent_refs(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    monkeypatch,
) -> None:
    _database, superseded, current, preparing = _multiple_generations(
        committed_git_repo, planning_cli_repository
    )
    monkeypatch.chdir(committed_git_repo)

    shown = cli_runner.invoke(
        cli, ["task", "show", "inspect-tasks", "01-foundation/02-current"]
    )

    assert shown.exit_code == 0, shown.output
    assert shown.output == render_task_markdown(current.task.task)
    for hidden in (superseded, preparing):
        result = cli_runner.invoke(
            cli, ["task", "show", "inspect-tasks", hidden.task.task_ref]
        )
        assert result.exit_code == 1
        assert "current task" in result.output
        assert hidden.task.title not in result.output


def test_task_list_and_show_block_digest_mismatch(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    monkeypatch,
) -> None:
    _database, _superseded, current, _preparing = _multiple_generations(
        committed_git_repo, planning_cli_repository
    )
    task_path = committed_git_repo / current.generation.manifest["tasks"][0]["path"]
    task_path.write_text("# drifted\n", encoding="utf-8")
    monkeypatch.chdir(committed_git_repo)

    for arguments in (
        ["task", "list", "inspect-tasks"],
        ["task", "show", "inspect-tasks", current.task.task_ref],
    ):
        result = cli_runner.invoke(cli, arguments)
        assert result.exit_code == 1
        assert "digest drifted" in result.output
        assert "Current task" not in result.output
