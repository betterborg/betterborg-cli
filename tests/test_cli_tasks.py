"""CLI contracts for inspecting the SQLite-current task generation."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from click.testing import CliRunner

from betterborg_cli.agent_runtime import AgentStatus, AgentUsage, BillingMode
from betterborg_cli.cli import cli
from betterborg_cli.planning import (
    TaskPublisher,
    render_task_markdown,
)
from betterborg_cli.store import (
    AgentAttempt,
    BorgState,
    ExecutionRun,
    PlanApproval,
    SqliteStore,
    TaskClaim,
    TaskGenerationStatus,
    TaskRuntime,
    TaskRuntimeStatus,
)
from betterborg_cli.store.models import utcnow


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


def _multiple_generations(
    root: Path,
    planning_cli_repository,
    approved_task_generation,
):
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
        superseded = approved_task_generation(
            store,
            borg,
            approval,
            body=_task_body("01-superseded", "Superseded task"),
            round_number=1,
            task_ref="T-1",
        )
        TaskPublisher(repository, store).publish(superseded.generation.id)
        current = approved_task_generation(
            store,
            borg,
            approval,
            body=_task_body("02-current", "Current task"),
            round_number=2,
            task_ref="T-2",
        )
        TaskPublisher(repository, store).publish(current.generation.id)
        preparing = approved_task_generation(
            store,
            borg,
            approval,
            body=_task_body("03-preparing", "Preparing task"),
            round_number=3,
            task_ref="T-3",
        )
    return paths.state_dir / "borg.sqlite3", superseded, current, preparing


def test_task_list_exposes_only_current_and_does_not_reconcile_other_trees(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch,
) -> None:
    database, superseded, current, preparing = _multiple_generations(
        committed_git_repo, planning_cli_repository, approved_task_generation
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
    approved_task_generation,
    monkeypatch,
) -> None:
    _database, _superseded, current, _preparing = _multiple_generations(
        committed_git_repo, planning_cli_repository, approved_task_generation
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
                "attempt_count": 0,
                "cost": {
                    "api_spend_unknown": True,
                    "api_spend_usd": None,
                    "subscription_included": False,
                },
                "duration_seconds": None,
                "path": current.generation.manifest["tasks"][0]["path"],
                "position": 1,
                "review_round": 0,
                "stage": "01-foundation",
                "state_reason": None,
                "status": "pending",
                "stem": "02-current",
                "task_ref": current.task.task_ref,
                "title": "Current task",
            }
        ],
        "totals": {
            "attempt_count": 0,
            "cost": {
                "api_spend_unknown": True,
                "api_spend_usd": None,
                "subscription_included": False,
            },
            "duration_seconds": None,
        },
    }


def test_task_estimate_reports_generation_dummy_source_and_unknown_billing(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch,
) -> None:
    _database, _superseded, current, _preparing = _multiple_generations(
        committed_git_repo, planning_cli_repository, approved_task_generation
    )
    monkeypatch.chdir(committed_git_repo)

    terminal = cli_runner.invoke(cli, ["task", "estimate", "inspect-tasks"])
    machine = cli_runner.invoke(
        cli, ["task", "estimate", "inspect-tasks", "--json"]
    )

    assert terminal.exit_code == 0, terminal.output
    assert terminal.output.startswith("DUMMY DATA")
    assert f"Execution estimate for Borg 'inspect-tasks': {current.generation.id}" in (
        terminal.output
    )
    assert "Task mix: 1 small, 0 medium, 0 large, 0 unsized" in terminal.output
    assert "Total agent work (not calendar time): P50 30.0m, P80 1.0h" in (
        terminal.output
    )
    assert "n=0, source=dummy_prior" in terminal.output
    assert "API estimate: unknown" in terminal.output
    assert "Billing mode unknown for: coding, merge, review" in terminal.output

    assert machine.exit_code == 0, machine.output
    estimate = json.loads(machine.output)
    assert estimate["generation_id"] == str(current.generation.id)
    assert estimate["sample_size"] == 0
    assert estimate["billing"]["api"]["estimate"] is None
    assert estimate["billing"]["api"]["unknown"] is True
    assert estimate["billing"]["subscription"]["usd"] is None
    assert estimate["provenance"]["prior_label"].startswith("DUMMY DATA")


def test_task_list_terminal_and_json_share_runtime_totals(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch,
) -> None:
    database, superseded, current, _preparing = _multiple_generations(
        committed_git_repo, planning_cli_repository, approved_task_generation
    )
    started_at = utcnow()
    run = ExecutionRun(
        borg_id=current.task.borg_id,
        generation_id=current.generation.id,
        started_at=started_at,
        heartbeat_at=started_at,
        lease_expires_at=started_at + timedelta(minutes=5),
    )
    runtime = TaskRuntime(
        generation_id=current.generation.id,
        task_id=current.task.id,
        status=TaskRuntimeStatus.FIX,
        state_reason="review requested changes",
        review_round=2,
    )
    claim = TaskClaim(
        run_id=run.id,
        task_id=current.task.id,
        resume_phase="fix",
        claimed_at=started_at,
        lease_expires_at=started_at + timedelta(minutes=2),
    )
    with SqliteStore.open(database) as store:
        store.add_execution_run(run)
        store.add_task_runtime(runtime)
        store.append_task_claim(claim)
        for index, (billing, cost, duration) in enumerate(
            (
                (BillingMode.API, 0.75, 4.0),
                (BillingMode.SUBSCRIPTION, 8.0, 6.0),
            ),
            start=1,
        ):
            store.append_agent_attempt(
                AgentAttempt(
                    run_id=run.id,
                    claim_id=claim.id,
                    task_id=current.task.id,
                    phase=f"phase-{index}",
                    attempt_number=1,
                    adapter="mock",
                    model="test-model",
                    billing_mode=billing,
                    status=AgentStatus.COMPLETED,
                    log_path=f"artifacts/{index}.log",
                    duration_seconds=duration,
                    usage=AgentUsage(cost_usd=cost),
                    started_at=started_at,
                    finished_at=started_at + timedelta(seconds=duration),
                ),
                run.owner_token,
                claim.claim_token,
                now=started_at,
            )
    monkeypatch.chdir(committed_git_repo)

    terminal = cli_runner.invoke(cli, ["task", "list", "inspect-tasks"])
    machine = cli_runner.invoke(
        cli, ["task", "list", "inspect-tasks", "--json"]
    )

    assert terminal.exit_code == 0, terminal.output
    assert "Status: fix" in terminal.output
    assert "Reason: review requested changes" in terminal.output
    assert "Review rounds: 2" in terminal.output
    assert "Attempts: 2" in terminal.output
    assert "Duration: 10s" in terminal.output
    assert "Cost: $0.7500 API + subscription included" in terminal.output
    assert "Totals: 2 attempt(s), 10s, $0.7500 API + subscription included" in (
        terminal.output
    )
    assert superseded.task.title not in terminal.output

    assert machine.exit_code == 0, machine.output
    payload = json.loads(machine.output)
    runtime_fields = {
        "attempt_count": 2,
        "cost": {
            "api_spend_unknown": False,
            "api_spend_usd": 0.75,
            "subscription_included": True,
        },
        "duration_seconds": 10.0,
        "review_round": 2,
        "state_reason": "review requested changes",
        "status": "fix",
    }
    assert {key: payload["tasks"][0][key] for key in runtime_fields} == (
        runtime_fields
    )
    assert payload["totals"] == {
        "attempt_count": 2,
        "cost": runtime_fields["cost"],
        "duration_seconds": 10.0,
    }
    assert superseded.task.title not in machine.output


def test_task_show_renders_current_markdown_and_json_and_rejects_noncurrent_refs(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    approved_task_generation,
    monkeypatch,
) -> None:
    _database, superseded, current, preparing = _multiple_generations(
        committed_git_repo, planning_cli_repository, approved_task_generation
    )
    monkeypatch.chdir(committed_git_repo)

    shown = cli_runner.invoke(
        cli, ["task", "show", "inspect-tasks", "01-foundation/02-current"]
    )

    assert shown.exit_code == 0, shown.output
    assert shown.output == render_task_markdown(current.task.task)

    shown_json = cli_runner.invoke(
        cli,
        [
            "task",
            "show",
            "inspect-tasks",
            current.task.task_ref,
            "--json",
        ],
    )
    assert shown_json.exit_code == 0, shown_json.output
    assert json.loads(shown_json.output) == {
        "complexity": "small",
        "dependencies": [],
        "digest": current.task.digest,
        "path": current.generation.manifest["tasks"][0]["path"],
        "position": 1,
        "stage": "01-foundation",
        "stem": "02-current",
        "task": current.task.task,
        "task_ref": current.task.task_ref,
        "title": "Current task",
    }

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
    approved_task_generation,
    monkeypatch,
) -> None:
    _database, _superseded, current, _preparing = _multiple_generations(
        committed_git_repo, planning_cli_repository, approved_task_generation
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
