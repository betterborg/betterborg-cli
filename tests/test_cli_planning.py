"""CLI contracts for starting and resuming terminal planning."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from pytest import MonkeyPatch

from betterborg_cli import cli as cli_module
from betterborg_cli.agent_runtime import CancellationToken
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.cli import CliRunContext, cli
from betterborg_cli.planning import render_plan_markdown, validate_plan
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.progress import RunProgress, StageState
from betterborg_cli.repository_config import AgentStage
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
    monkeypatch: MonkeyPatch,
) -> None:
    architect_adapter = MockAdapter(name="openai")
    tech_lead_adapter = MockAdapter(name="openai")
    architect_adapter.queue(
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
    architect_adapter.queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect_adapter.queue(MockResponse(payload=planning_plan_response()))
    tech_lead_adapter.queue(
        MockResponse(payload=tech_lead_approval_response())
    )
    prompts: list[str] = []
    outputs: list[str] = []

    repository, paths = planning_cli_repository(committed_git_repo, "inline-plan")
    configure_interactive_cli(
        repository.root,
        architect_adapter,
        InteractiveIO(
            prompt=lambda message: prompts.append(message) or "Linux and macOS.",
            confirm=lambda _message, _default: False,
            write=outputs.append,
        ),
        state_home=repository.root.parent / f".{repository.root.name}-state",
    )
    selected_stages = _select_planning_agents(
        monkeypatch,
        architect=architect_adapter,
        tech_lead=tech_lead_adapter,
    )

    result = cli_runner.invoke(cli, ["plan", "start", "inline-plan", "--yes"])

    assert result.exit_code == 0, result.output
    assert prompts == ["Which platforms are required?"]
    assert outputs == ["Why this matters: This controls the test matrix."]
    assert "Plan approval pending" in result.output
    assert "betterborg plan show inline-plan" in result.output
    assert result.output.count("none failed or stopped.") == 1
    assert result.output.index("none failed or stopped.") < result.output.index(
        "Plan approval pending"
    )
    assert architect_adapter is not tech_lead_adapter
    assert selected_stages == [AgentStage.ARCHITECT, AgentStage.TECH_LEAD]
    assert len(architect_adapter.calls) == 3
    assert len(tech_lead_adapter.calls) == 1
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
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
    monkeypatch: MonkeyPatch,
) -> None:
    architect_adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "decision": "ask_more",
                "questions": [
                    {"id": "q1", "question": "Which users are in scope?"}
                ],
            }
        )
    )
    tech_lead_adapter = MockAdapter(name="openai")
    answers: Iterator[str | None] = iter((None, "Repository maintainers."))
    repository, paths = planning_cli_repository(committed_git_repo, "resume-plan")
    configure_interactive_cli(
        repository.root,
        architect_adapter,
        InteractiveIO(
            prompt=lambda _message: next(answers),
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        state_home=repository.root.parent / f".{repository.root.name}-state",
    )
    selected_stages = _select_planning_agents(
        monkeypatch,
        architect=architect_adapter,
        tech_lead=tech_lead_adapter,
    )

    interrupted = cli_runner.invoke(
        cli, ["plan", "start", "resume-plan", "--yes"]
    )

    assert interrupted.exit_code == 1
    assert "was interrupted" in interrupted.output
    assert "betterborg plan start resume-plan" in interrupted.output
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "resume-plan")
        assert borg is not None
        assert borg.state is BorgState.ARCHITECT_AWAITING_ANSWERS
        assert store.list_planning_questions(borg.id)[0].answers is None

    architect_adapter.queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect_adapter.queue(MockResponse(payload=planning_plan_response()))
    tech_lead_adapter.queue(
        MockResponse(payload=tech_lead_approval_response())
    )
    resumed = cli_runner.invoke(cli, ["plan", "start", "resume-plan", "--yes"])

    assert resumed.exit_code == 0, resumed.output
    assert "Plan approval pending" in resumed.output
    assert selected_stages == [
        AgentStage.ARCHITECT,
        AgentStage.TECH_LEAD,
        AgentStage.ARCHITECT,
        AgentStage.TECH_LEAD,
    ]
    assert len(architect_adapter.calls) == 3
    assert len(tech_lead_adapter.calls) == 1
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "resume-plan")
        assert borg is not None
        assert borg.state is BorgState.PLAN_APPROVAL_PENDING
        assert store.list_planning_questions(borg.id)[0].answers == [
            {"q_id": "q1", "answer": "Repository maintainers."}
        ]


def test_plan_start_resumes_directly_with_tech_lead_agent(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    tech_lead_approval_response,
    configure_interactive_cli,
    monkeypatch: MonkeyPatch,
) -> None:
    architect_adapter = MockAdapter(name="openai")
    architect_adapter.queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect_adapter.queue(MockResponse(payload=planning_plan_response()))
    tech_lead_adapter = MockAdapter(name="openai").queue(
        MockResponse(raise_error=RuntimeError("review provider unavailable"))
    )
    repository, paths = planning_cli_repository(committed_git_repo, "resume-review")
    configure_interactive_cli(
        repository.root,
        architect_adapter,
        InteractiveIO(
            prompt=lambda _message: None,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        state_home=repository.root.parent / f".{repository.root.name}-state",
    )
    selected_stages = _select_planning_agents(
        monkeypatch,
        architect=architect_adapter,
        tech_lead=tech_lead_adapter,
    )

    interrupted = cli_runner.invoke(
        cli, ["plan", "start", "resume-review", "--yes"]
    )

    assert interrupted.exit_code == 1
    assert "review provider unavailable" in interrupted.output
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "resume-review")
        assert borg is not None
        assert borg.state is BorgState.TECH_REVIEW_WORKING

    tech_lead_adapter.queue(
        MockResponse(payload=tech_lead_approval_response())
    )
    resumed = cli_runner.invoke(
        cli, ["plan", "start", "resume-review", "--yes"]
    )

    assert resumed.exit_code == 0, resumed.output
    assert "Plan approval pending" in resumed.output
    assert selected_stages == [
        AgentStage.ARCHITECT,
        AgentStage.TECH_LEAD,
        AgentStage.ARCHITECT,
        AgentStage.TECH_LEAD,
    ]
    assert len(architect_adapter.calls) == 2
    assert len(tech_lead_adapter.calls) == 2
    assert all(
        "You are the Architect" in call.system_prompt
        for call in architect_adapter.calls
    )
    assert all(
        "You are the Tech Lead" in call.system_prompt
        for call in tech_lead_adapter.calls
    )


def test_plan_start_reports_review_cap_as_blocked(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    tech_lead_change_request_response,
    configure_interactive_cli,
    monkeypatch: MonkeyPatch,
) -> None:
    architect_adapter = MockAdapter(name="openai")
    for payload in (
        {"decision": "ready_to_plan"},
        planning_plan_response(),
        planning_plan_response(summary="Clarify rollback behavior."),
        planning_plan_response(summary="Name the rollback checks."),
    ):
        architect_adapter.queue(MockResponse(payload=payload))
    tech_lead_adapter = MockAdapter(name="openai")
    for payload in (
        tech_lead_change_request_response("Clarify rollback behavior."),
        tech_lead_change_request_response("Name the rollback checks."),
        tech_lead_change_request_response("Cover a partial rollback."),
    ):
        tech_lead_adapter.queue(MockResponse(payload=payload))
    repository, paths = planning_cli_repository(committed_git_repo, "blocked-plan")
    configure_interactive_cli(
        repository.root,
        architect_adapter,
        InteractiveIO(
            prompt=lambda _message: None,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        state_home=repository.root.parent / f".{repository.root.name}-state",
    )
    selected_stages = _select_planning_agents(
        monkeypatch,
        architect=architect_adapter,
        tech_lead=tech_lead_adapter,
    )

    result = cli_runner.invoke(cli, ["plan", "start", "blocked-plan", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Planning blocked" in result.output
    assert "betterborg plan show blocked-plan" in result.output
    assert selected_stages == [AgentStage.ARCHITECT, AgentStage.TECH_LEAD]
    assert len(architect_adapter.calls) == 4
    assert len(tech_lead_adapter.calls) == 3
    assert all(
        "You are the Architect" in call.system_prompt
        for call in architect_adapter.calls
    )
    assert all(
        "You are the Tech Lead" in call.system_prompt
        for call in tech_lead_adapter.calls
    )
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
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
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
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

    markdown_progress = _SuspensionRecorder()
    markdown_result = cli_runner.invoke(
        cli,
        ["plan", "show", "show-plan"],
        obj=CliRunContext(CancellationToken(), markdown_progress),
    )

    assert markdown_result.exit_code == 0, markdown_result.output
    assert markdown_result.output == render_plan_markdown(plan)
    assert markdown_progress.entries == 1
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        assert _planning_snapshot(store, borg.id) == before

    json_progress = _SuspensionRecorder()
    json_result = cli_runner.invoke(
        cli,
        ["plan", "show", "show-plan", "--json"],
        obj=CliRunContext(CancellationToken(), json_progress),
    )

    assert json_result.exit_code == 0, json_result.output
    assert json.loads(json_result.output) == plan
    assert json_progress.entries == 1
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
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
    assert "betterborg plan start missing-plan" in result.output


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
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "change-plan")
        assert borg is not None
        before_attempts = store.list_planning_attempts(borg.id)
        before_findings = store.list_planning_findings(borg.id)
        assert len(before_findings) == 1

    def requested_revision(spec):
        manifest = json.loads(
            (
                spec.cwd / ".betterborg/state/planning/context/manifest.json"
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
    assert "betterborg plan show change-plan" in changed.output
    assert changed.output.count("none failed or stopped.") == 1
    assert changed.output.index("none failed or stopped.") < changed.output.index(
        "Plan approval pending"
    )
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
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
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
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
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
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
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        assert _planning_snapshot(store, borg.id) == before


def test_plan_change_runtime_failure_is_actionably_resumable(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    planning_cli_repository,
    planning_plan_response,
    tech_lead_approval_response,
    configure_interactive_cli,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture,
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

    primary_error = RuntimeError("planning provider unavailable")
    adapter.queue(MockResponse(raise_error=primary_error))
    close_error = RuntimeError("progress close failed")
    shown_errors: list[click.ClickException] = []
    reporters: list[RunProgress] = []
    original_progress = cli_module.RunProgress
    original_show = click.ClickException.show

    class CloseFailingProgress(RunProgress):
        stop_calls = 0

        def close(self) -> None:
            raise close_error

        def stop_display(self) -> None:
            self.stop_calls += 1
            super().stop_display()

    def progress_factory(**kwargs: object) -> CloseFailingProgress:
        progress = CloseFailingProgress(enabled=False, **kwargs)
        reporters.append(progress)
        return progress

    def capture_show(error: click.ClickException, *args, **kwargs) -> None:
        shown_errors.append(error)
        original_show(error, *args, **kwargs)

    monkeypatch.setattr(cli_module, "RunProgress", progress_factory)
    monkeypatch.setattr(cli_module.click.ClickException, "show", capture_show)
    failed_exit_code = cli_module.main(
        [
            "plan",
            "change",
            "resume-change",
            "--note",
            "Add rollback verification.",
            "--yes",
        ],
        prog_name="betterborg",
    )

    captured = capsys.readouterr()
    expected_error = (
        "Error: Plan change for Borg 'resume-change' could not continue "
        "(Architect architect_plan turn crashed: planning provider unavailable). "
        "Run 'betterborg plan start "
        "resume-change' to resume.\n"
    )
    assert failed_exit_code == 1
    assert captured.out == ""
    assert captured.err == expected_error
    assert "progress close failed" not in captured.err
    assert shown_errors[0].__cause__ is not None
    assert shown_errors[0].__cause__.__cause__ is primary_error
    assert shown_errors[0].__notes__ == [
        "progress finalization also failed: progress close failed"
    ]
    assert reporters[0].stop_calls == 1
    assert reporters[0]._cadence_worker is None
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "resume-change")
        assert borg is not None
        assert borg.state is BorgState.ARCHITECT_WORKING
        assert [
            request.note for request in store.list_plan_change_requests(borg.id)
        ] == ["Add rollback verification."]

    monkeypatch.setattr(cli_module, "RunProgress", original_progress)
    adapter.queue(
        MockResponse(payload=planning_plan_response(summary="Revised plan."))
    )
    adapter.queue(MockResponse(payload=tech_lead_approval_response()))
    resumed = cli_runner.invoke(
        cli, ["plan", "start", "resume-change", "--yes"]
    )

    assert resumed.exit_code == 0, resumed.output
    assert "Plan approval pending" in resumed.output
    with SqliteStore.open(paths.state_dir / "betterborg.sqlite3") as store:
        borg = store.get_borg_by_name(repository.id, "resume-change")
        assert borg is not None
        assert borg.state is BorgState.PLAN_APPROVAL_PENDING
        assert [
            request.note for request in store.list_plan_change_requests(borg.id)
        ] == ["Add rollback verification."]


def test_plan_start_primary_error_survives_root_progress_close_failure(
    committed_git_repo: Path,
    planning_cli_repository,
    configure_interactive_cli,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    repository, _paths = planning_cli_repository(
        committed_git_repo, "start-close-failure"
    )
    primary_error = RuntimeError("planning provider unavailable")
    adapter = MockAdapter(name="openai").queue(
        MockResponse(raise_error=primary_error)
    )
    configure_interactive_cli(
        repository.root,
        adapter,
        InteractiveIO(
            prompt=lambda _message: None,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
        state_home=repository.root.parent / ".start-close-state",
    )
    close_error = RuntimeError("progress close failed")
    shown_errors: list[click.ClickException] = []
    original_show = click.ClickException.show

    class CloseFailingProgress(RunProgress):
        stop_calls = 0

        def close(self) -> None:
            raise close_error

        def stop_display(self) -> None:
            self.stop_calls += 1
            super().stop_display()

    reporters: list[CloseFailingProgress] = []

    def progress_factory(**kwargs: object) -> CloseFailingProgress:
        progress = CloseFailingProgress(enabled=False, **kwargs)
        reporters.append(progress)
        return progress

    def capture_show(error: click.ClickException, *args, **kwargs) -> None:
        shown_errors.append(error)
        original_show(error, *args, **kwargs)

    monkeypatch.setattr(cli_module, "RunProgress", progress_factory)
    monkeypatch.setattr(cli_module.click.ClickException, "show", capture_show)

    exit_code = cli_module.main(
        ["plan", "start", "start-close-failure", "--yes"],
        prog_name="betterborg",
    )

    captured = capsys.readouterr()
    progress = reporters[0]
    expected_error = (
        "Error: Planning for Borg 'start-close-failure' could not continue "
        "(Architect architect_questions turn crashed: planning provider "
        "unavailable). "
        "Run 'betterborg plan start start-close-failure' to resume.\n"
    )
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == expected_error
    assert "progress close failed" not in captured.err
    assert shown_errors[0].__notes__ == [
        "progress finalization also failed: progress close failed"
    ]
    assert shown_errors[0].__cause__ is not None
    assert shown_errors[0].__cause__.__cause__ is primary_error
    assert progress.stop_calls == 1
    assert progress._cadence_worker is None
    assert progress.stages["architect"].state is StageState.FAILED


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


class _SuspensionRecorder:
    def __init__(self) -> None:
        self.entries = 0

    @contextmanager
    def suspend(self):
        self.entries += 1
        yield self


def _select_planning_agents(
    monkeypatch: MonkeyPatch,
    *,
    architect: MockAdapter,
    tech_lead: MockAdapter,
) -> list[AgentStage]:
    selected_stages: list[AgentStage] = []
    adapters = {
        AgentStage.ARCHITECT: architect,
        AgentStage.TECH_LEAD: tech_lead,
    }

    def select(_config, stage, _paths, *, interactive):
        assert interactive is True
        selected_stages.append(stage)
        return adapters[stage]

    monkeypatch.setattr(cli_module, "select_agent", select)
    return selected_stages
