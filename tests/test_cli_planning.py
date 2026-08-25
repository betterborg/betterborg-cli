"""CLI contracts for starting and resuming terminal planning."""

import shutil
from collections.abc import Iterator
from pathlib import Path

from click.testing import CliRunner
from pytest import MonkeyPatch

from betterborg_cli import cli as cli_module
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.cli import cli
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.store import BorgState, SqliteStore


def test_plan_start_answers_inline_and_reaches_approval_pending(
    cli_runner: CliRunner,
    committed_git_repo: Path,
    monkeypatch: MonkeyPatch,
    persist_planning_context,
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
    adapter.queue(MockResponse(payload=_plan()))
    adapter.queue(MockResponse(payload=_approve()))
    prompts: list[str] = []
    outputs: list[str] = []

    repository, paths = _planning_cli_repository(
        committed_git_repo, persist_planning_context, "inline-plan"
    )
    _configure_cli(
        monkeypatch,
        repository.root,
        adapter,
        InteractiveIO(
            prompt=lambda message: prompts.append(message) or "Linux and macOS.",
            confirm=lambda _message, _default: False,
            write=outputs.append,
        ),
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
    monkeypatch: MonkeyPatch,
    persist_planning_context,
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
    _configure_cli(
        monkeypatch,
        repository.root,
        adapter,
        InteractiveIO(
            prompt=lambda _message: next(answers),
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
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
    adapter.queue(MockResponse(payload=_plan()))
    adapter.queue(MockResponse(payload=_approve()))
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
    monkeypatch: MonkeyPatch,
    persist_planning_context,
) -> None:
    adapter = MockAdapter(name="openai")
    for payload in (
        {"decision": "ready_to_plan"},
        _plan(),
        _request_changes("Clarify rollback behavior."),
        _plan(summary="Clarify rollback behavior."),
        _request_changes("Name the rollback checks."),
        _plan(summary="Name the rollback checks."),
        _request_changes("Cover a partial rollback."),
    ):
        adapter.queue(MockResponse(payload=payload))
    repository, paths = _planning_cli_repository(
        committed_git_repo, persist_planning_context, "blocked-plan"
    )
    _configure_cli(
        monkeypatch,
        repository.root,
        adapter,
        InteractiveIO(
            prompt=lambda _message: None,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        ),
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


def _configure_cli(
    monkeypatch: MonkeyPatch,
    root: Path,
    adapter: MockAdapter,
    io: InteractiveIO,
) -> None:
    monkeypatch.chdir(root)
    monkeypatch.setenv("XDG_STATE_HOME", str(root.parent / f".{root.name}-state"))
    monkeypatch.setattr(cli_module, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr(cli_module, "select_agent", lambda *_args, **_kwargs: adapter)
    monkeypatch.setattr(cli_module, "_interactive_io", lambda: io)


def _approve() -> dict:
    return {
        "decision": "approve",
        "summary": "The plan is ready for approval.",
        "findings": [],
    }


def _request_changes(message: str) -> dict:
    return {
        "decision": "request_changes",
        "summary": message,
        "findings": [
            {
                "severity": "major",
                "message": message,
                "suggestion": "Make the verification explicit.",
            }
        ],
    }


def _plan(*, summary: str = "Add a small, tested release workflow.") -> dict:
    return {
        "title": "Release workflow",
        "summary": summary,
        "overall_approach": "Extend the existing repository conventions.",
        "phases": [
            {
                "name": "01-release-workflow",
                "title": "Add release workflow",
                "goal": "Document and test the release path.",
                "technical_approach": "Update the tracked README convention.",
                "files_touched": [
                    {
                        "path": "README.md",
                        "role": "modified",
                        "description": "Document the release workflow.",
                    }
                ],
                "test_strategy": "Assert the documented public workflow.",
                "acceptance_criteria": ["The release path is documented."],
                "deliverables": ["Release workflow documentation."],
                "dependencies_on": [],
            }
        ],
        "code_pointers": [
            {"path": "README.md", "why": "It owns repository guidance."}
        ],
        "risks": [],
        "open_questions": [],
    }
