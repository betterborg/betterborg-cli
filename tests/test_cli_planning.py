"""CLI contracts for starting and resuming terminal planning."""

import shutil
from collections.abc import Iterator
from pathlib import Path

from click.testing import CliRunner

from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.cli import cli
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import BorgState, SqliteStore


def test_plan_start_answers_inline_and_reaches_approval_pending(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    persist_planning_context,
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

    repository, paths = _planning_cli_repository(
        committed_git_repo, persist_planning_context, "inline-plan"
    )
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
    persist_planning_context,
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
    repository, paths = _planning_cli_repository(
        committed_git_repo, persist_planning_context, "resume-plan"
    )
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
    persist_planning_context,
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
    repository, paths = _planning_cli_repository(
        committed_git_repo, persist_planning_context, "blocked-plan"
    )
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


def test_plan_exposes_only_start_command(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(cli, ["plan", "--help"])

    assert result.exit_code == 0
    assert "start" in result.output
    assert "question" not in result.output
    assert "answer" not in result.output


def _planning_cli_repository(root: Path, persist_planning_context, name: str):
    paths = RepoPaths.discover(root)
    fixture_database = root.parent / f"{name}.sqlite3"
    with SqliteStore.open(fixture_database) as store:
        repository, _borg = persist_planning_context(root, store, name)
    paths.state_dir.mkdir(parents=True)
    shutil.copyfile(fixture_database, paths.state_dir / "borg.sqlite3")
    return repository, paths
