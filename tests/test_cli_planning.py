"""CLI contracts for starting and resuming terminal planning."""

import json
from collections.abc import Iterator
from pathlib import Path

from click.testing import CliRunner

from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.cli import cli
from betterborg_cli.planning import render_plan_markdown, validate_plan
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.store import (
    BorgState,
    PlanChangeRequest,
    PlanningAttempt,
    PlanningAttemptStatus,
    PlanningFinding,
    PlanningQuestion,
    SqliteStore,
)


def test_plan_start_answers_inline_and_reaches_approval_pending(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    tech_lead_approval_response,
    configure_interactive_cli,
) -> None:
    adapter = MockAdapter(name="openai")
    adapter.queue(
        MockResponse(
            payload={
                "decision": "ask_more",
                "questions": [
                    {
                        "id": "q1",
                        "question": "Which platforms are required?",
                        "why": "This controls the test matrix.",
                    }
                ],
            }
        )
    )
    adapter.queue(MockResponse(payload={"decision": "ready_to_plan"}))
    adapter.queue(MockResponse(payload=planning_plan_response()))
    adapter.queue(MockResponse(payload=tech_lead_approval_response()))
    prompts: list[str] = []
    outputs: list[str] = []

    repository, paths = planning_cli_repository(committed_git_repo, "inline-plan")
    configure_interactive_cli(
        repository.root,
        adapter,
        InteractiveIO(
            prompt=lambda message: prompts.append(message) or "Linux and macOS.",
            confirm=lambda _message, _default: False,
            write=outputs.append,
        ),
        state_home=repository.root.parent / f".{repository.root.name}-state",
    )

    result = cli_runner.invoke(cli, ["plan", "start", "inline-plan", "--yes"])

    assert result.exit_code == 0, result.output
    assert prompts == ["Which platforms are required?"]
    assert outputs == ["Why this matters: This controls the test matrix."]
    assert "Plan approval pending" in result.output
    assert "borg plan show inline-plan" in result.output
    assert len(adapter.calls) == 4
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "inline-plan")
        assert borg is not None
        assert borg.state is BorgState.PLAN_APPROVAL_PENDING
        assert store.list_planning_questions(borg.id)[0].answers == [
            {"q_id": "q1", "answer": "Linux and macOS."}
        ]


def test_plan_start_interruption_preserves_question_and_same_command_resumes(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    tech_lead_approval_response,
    configure_interactive_cli,
) -> None:
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "decision": "ask_more",
                "questions": [
                    {"id": "q1", "question": "Which users are in scope?"}
                ],
            }
        )
    )
    answers: Iterator[str | None] = iter((None, "Repository maintainers."))
    repository, paths = planning_cli_repository(committed_git_repo, "resume-plan")
    configure_interactive_cli(
        repository.root,
        adapter,
        InteractiveIO(
            prompt=lambda _message: next(answers),
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        state_home=repository.root.parent / f".{repository.root.name}-state",
    )

    interrupted = cli_runner.invoke(
        cli, ["plan", "start", "resume-plan", "--yes"]
    )

    assert interrupted.exit_code == 1
    assert "was interrupted" in interrupted.output
    assert "borg plan start resume-plan" in interrupted.output
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "resume-plan")
        assert borg is not None
        assert borg.state is BorgState.ARCHITECT_AWAITING_ANSWERS
        assert store.list_planning_questions(borg.id)[0].answers is None

    adapter.queue(MockResponse(payload={"decision": "ready_to_plan"}))
    adapter.queue(MockResponse(payload=planning_plan_response()))
    adapter.queue(MockResponse(payload=tech_lead_approval_response()))
    resumed = cli_runner.invoke(cli, ["plan", "start", "resume-plan", "--yes"])

    assert resumed.exit_code == 0, resumed.output
    assert "Plan approval pending" in resumed.output
    assert len(adapter.calls) == 4
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "resume-plan")
        assert borg is not None
        assert borg.state is BorgState.PLAN_APPROVAL_PENDING
        assert store.list_planning_questions(borg.id)[0].answers == [
            {"q_id": "q1", "answer": "Repository maintainers."}
        ]


def test_plan_start_reports_review_cap_as_blocked(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    tech_lead_change_request_response,
    configure_interactive_cli,
) -> None:
    adapter = MockAdapter(name="openai")
    for payload in (
        {"decision": "ready_to_plan"},
        planning_plan_response(),
        tech_lead_change_request_response("Clarify rollback behavior."),
        planning_plan_response(summary="Clarify rollback behavior."),
        tech_lead_change_request_response("Name the rollback checks."),
        planning_plan_response(summary="Name the rollback checks."),
        tech_lead_change_request_response("Cover a partial rollback."),
    ):
        adapter.queue(MockResponse(payload=payload))
    repository, paths = planning_cli_repository(committed_git_repo, "blocked-plan")
    configure_interactive_cli(
        repository.root,
        adapter,
        InteractiveIO(
            prompt=lambda _message: None,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        state_home=repository.root.parent / f".{repository.root.name}-state",
    )

    result = cli_runner.invoke(cli, ["plan", "start", "blocked-plan", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Planning blocked" in result.output
    assert "borg plan show blocked-plan" in result.output
    assert len(adapter.calls) == 7
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "blocked-plan")
        assert borg is not None
        assert borg.state is BorgState.BLOCKED
        assert len(store.list_planning_findings(borg.id)) == 3


def test_plan_show_survives_checkout_drift_without_mutating_planning_history(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    monkeypatch,
) -> None:
    repository, paths = planning_cli_repository(committed_git_repo, "show-plan")
    plan = planning_plan_response()
    plan["phases"][0]["files_touched"].append(
        {
            "path": "CHANGELOG.md",
            "role": "new",
            "description": "Record release changes.",
        }
    )
    validate_plan(plan, repository.root)
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "show-plan")
        assert borg is not None
        plan_attempt = PlanningAttempt(
            borg_id=borg.id,
            phase="architect_plan",
            round=1,
            adapter="mock",
            model="test-model",
        )
        review_attempt = PlanningAttempt(
            borg_id=borg.id,
            phase="tech_review",
            round=1,
            adapter="mock",
            model="test-model",
        )
        store.append_planning_attempt(plan_attempt)
        store.complete_planning_attempt(
            plan_attempt.id,
            status=PlanningAttemptStatus.COMPLETED,
            result=plan,
            summary="Plan ready for review.",
        )
        question = PlanningQuestion(
            borg_id=borg.id,
            attempt_id=plan_attempt.id,
            round=1,
            questions=[{"id": "scope", "question": "Which platforms?"}],
        )
        store.append_planning_question(question)
        store.answer_planning_question(
            question.id,
            [{"q_id": "scope", "answer": "Linux and macOS."}],
        )
        store.append_planning_attempt(review_attempt)
        store.complete_planning_attempt(
            review_attempt.id,
            status=PlanningAttemptStatus.COMPLETED,
            result={"decision": "request_changes"},
        )
        store.append_planning_finding(
            PlanningFinding(
                borg_id=borg.id,
                attempt_id=review_attempt.id,
                round=1,
                severity="minor",
                message="Name the supported platforms.",
            )
        )
        store.append_plan_change_request(
            PlanChangeRequest(
                borg_id=borg.id,
                round=1,
                note="Keep the plan portable.",
            )
        )
        store.compare_and_set_borg_state(
            borg.id,
            expected_state=borg.state,
            expected_version=borg.state_version,
            new_state=BorgState.PLAN_APPROVAL_PENDING,
        )
        before = _planning_snapshot(store, borg.id)

    (repository.root / "README.md").unlink()
    (repository.root / "CHANGELOG.md").write_text("# Changes\n", encoding="utf-8")
    monkeypatch.chdir(repository.root)

    markdown_result = cli_runner.invoke(cli, ["plan", "show", "show-plan"])

    assert markdown_result.exit_code == 0, markdown_result.output
    assert markdown_result.output == render_plan_markdown(plan)
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        assert _planning_snapshot(store, borg.id) == before

    json_result = cli_runner.invoke(
        cli, ["plan", "show", "show-plan", "--json"]
    )

    assert json_result.exit_code == 0, json_result.output
    assert json.loads(json_result.output) == plan
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        assert _planning_snapshot(store, borg.id) == before


def test_plan_show_reports_when_no_plan_is_stored(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    monkeypatch,
) -> None:
    repository, _paths = planning_cli_repository(committed_git_repo, "missing-plan")
    monkeypatch.chdir(repository.root)

    result = cli_runner.invoke(cli, ["plan", "show", "missing-plan"])

    assert result.exit_code == 1
    assert "does not have a stored plan" in result.output
    assert "borg plan start missing-plan" in result.output


def test_plan_change_preserves_history_and_drains_revision_loop_to_gate(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    tech_lead_approval_response,
    tech_lead_change_request_response,
    configure_interactive_cli,
) -> None:
    original_plan = planning_plan_response(summary="Original plan.")
    reviewed_plan = planning_plan_response(summary="Address the original finding.")
    requested_plan = planning_plan_response(summary="Add staged rollout checks.")
    final_plan = planning_plan_response(summary="Add rollback checks too.")
    adapter = MockAdapter(name="openai")
    for payload in (
        {"decision": "ready_to_plan"},
        original_plan,
        tech_lead_change_request_response("Clarify the original rollout."),
        reviewed_plan,
        tech_lead_approval_response(),
    ):
        adapter.queue(MockResponse(payload=payload))
    repository, paths = planning_cli_repository(committed_git_repo, "change-plan")
    configure_interactive_cli(
        repository.root,
        adapter,
        InteractiveIO(
            prompt=lambda _message: None,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        state_home=repository.root.parent / f".{repository.root.name}-state",
    )

    started = cli_runner.invoke(cli, ["plan", "start", "change-plan", "--yes"])

    assert started.exit_code == 0, started.output
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "change-plan")
        assert borg is not None
        before_attempts = store.list_planning_attempts(borg.id)
        before_findings = store.list_planning_findings(borg.id)
        assert len(before_findings) == 1

    def requested_revision(spec):
        manifest = json.loads(
            (
                spec.cwd / ".borg/state/planning/context/manifest.json"
            ).read_text(encoding="utf-8")
        )
        current_plan = json.loads(
            (spec.cwd / manifest["current_plan"]).read_text(encoding="utf-8")
        )
        changes = json.loads(
            (spec.cwd / manifest["change_requests"]).read_text(encoding="utf-8")
        )
        assert current_plan == reviewed_plan
        assert [item["note"] for item in changes] == [
            "Add staged rollout safety."
        ]
        return requested_plan

    adapter.queue(MockResponse(dynamic=requested_revision))
    for payload in (
        tech_lead_change_request_response("Add rollback verification."),
        final_plan,
        tech_lead_approval_response(),
    ):
        adapter.queue(MockResponse(payload=payload))

    changed = cli_runner.invoke(
        cli,
        [
            "plan",
            "change",
            "change-plan",
            "--note",
            "  Add staged rollout safety.  ",
            "--yes",
        ],
    )

    assert changed.exit_code == 0, changed.output
    assert "Plan approval pending" in changed.output
    assert "borg plan show change-plan" in changed.output
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "change-plan")
        assert borg is not None
        assert borg.state is BorgState.PLAN_APPROVAL_PENDING
        attempts = store.list_planning_attempts(borg.id)
        assert attempts[: len(before_attempts)] == before_attempts
        assert [
            item.result["summary"]
            for item in attempts
            if item.phase == "architect_plan"
        ] == [
            "Original plan.",
            "Address the original finding.",
            "Add staged rollout checks.",
            "Add rollback checks too.",
        ]
        findings = store.list_planning_findings(borg.id)
        assert findings[: len(before_findings)] == before_findings
        assert [item.message for item in findings] == [
            "Clarify the original rollout.",
            "Add rollback verification.",
        ]
        assert [item.round for item in findings] == [1, 1]
        requests = store.list_plan_change_requests(borg.id)
        assert [item.note for item in requests] == ["Add staged rollout safety."]

    shown = cli_runner.invoke(cli, ["plan", "show", "change-plan", "--json"])

    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.output) == final_plan
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        assert store.list_planning_attempts(borg.id) == attempts
        assert store.list_planning_findings(borg.id) == findings
        assert store.list_plan_change_requests(borg.id) == requests


def test_plan_change_rejects_empty_note_without_mutating_gate(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    tech_lead_approval_response,
    configure_interactive_cli,
) -> None:
    adapter = MockAdapter(name="openai")
    for payload in (
        {"decision": "ready_to_plan"},
        planning_plan_response(),
        tech_lead_approval_response(),
    ):
        adapter.queue(MockResponse(payload=payload))
    repository, paths = planning_cli_repository(committed_git_repo, "empty-change")
    configure_interactive_cli(
        repository.root,
        adapter,
        InteractiveIO(
            prompt=lambda _message: None,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        state_home=repository.root.parent / f".{repository.root.name}-state",
    )
    started = cli_runner.invoke(cli, ["plan", "start", "empty-change", "--yes"])
    assert started.exit_code == 0, started.output
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "empty-change")
        assert borg is not None
        before = _planning_snapshot(store, borg.id)

    result = cli_runner.invoke(
        cli,
        ["plan", "change", "empty-change", "--yes"],
        input="   \n",
    )

    assert result.exit_code == 1
    assert "plan change note must not be empty" in result.output
    assert len(adapter.calls) == 3
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        assert _planning_snapshot(store, borg.id) == before


def test_plan_change_runtime_failure_is_actionably_resumable(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    tech_lead_approval_response,
    configure_interactive_cli,
) -> None:
    adapter = MockAdapter(name="openai")
    for payload in (
        {"decision": "ready_to_plan"},
        planning_plan_response(summary="Original plan."),
        tech_lead_approval_response(),
    ):
        adapter.queue(MockResponse(payload=payload))
    repository, paths = planning_cli_repository(committed_git_repo, "resume-change")
    configure_interactive_cli(
        repository.root,
        adapter,
        InteractiveIO(
            prompt=lambda _message: None,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        state_home=repository.root.parent / f".{repository.root.name}-state",
    )
    started = cli_runner.invoke(cli, ["plan", "start", "resume-change", "--yes"])
    assert started.exit_code == 0, started.output

    adapter.queue(
        MockResponse(raise_error=RuntimeError("planning provider unavailable"))
    )
    failed = cli_runner.invoke(
        cli,
        [
            "plan",
            "change",
            "resume-change",
            "--note",
            "Add rollback verification.",
            "--yes",
        ],
    )

    assert failed.exit_code == 1
    assert "could not continue" in failed.output
    assert "planning provider unavailable" in failed.output
    assert "borg plan start resume-change" in failed.output
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "resume-change")
        assert borg is not None
        assert borg.state is BorgState.ARCHITECT_WORKING
        assert [
            request.note for request in store.list_plan_change_requests(borg.id)
        ] == ["Add rollback verification."]

    adapter.queue(
        MockResponse(payload=planning_plan_response(summary="Revised plan."))
    )
    adapter.queue(MockResponse(payload=tech_lead_approval_response()))
    resumed = cli_runner.invoke(
        cli, ["plan", "start", "resume-change", "--yes"]
    )

    assert resumed.exit_code == 0, resumed.output
    assert "Plan approval pending" in resumed.output
    with SqliteStore.open(paths.state_dir / "borg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "resume-change")
        assert borg is not None
        assert borg.state is BorgState.PLAN_APPROVAL_PENDING
        assert [
            request.note for request in store.list_plan_change_requests(borg.id)
        ] == ["Add rollback verification."]


def test_plan_exposes_start_show_and_change_commands(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(cli, ["plan", "--help"])

    assert result.exit_code == 0
    assert "start" in result.output
    assert "show" in result.output
    assert "change" in result.output
    assert "question" not in result.output
    assert "answer" not in result.output


def _planning_snapshot(store: SqliteStore, borg_id):
    return (
        store.get_borg(borg_id),
        store.list_planning_attempts(borg_id),
        store.list_planning_questions(borg_id),
        store.list_planning_findings(borg_id),
        store.list_plan_change_requests(borg_id),
    )
