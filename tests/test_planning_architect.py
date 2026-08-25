"""Terminal Architect lifecycle, durability, and resume contracts."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pytest

from betterborg_cli.agent_runtime import ApiAgentRole, CodexAdapter, select_agent
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.planning import ArchitectCancelled, ArchitectError, ArchitectLoop
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import load_repository_config
from betterborg_cli.store import (
    BorgState,
    PlanningAttempt,
    PlanningAttemptStatus,
    PlanningQuestion,
    SqliteStore,
)


def test_answers_product_questions_inline_and_persists_plan(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    answers = iter(["Linux and macOS in the first release."])
    output: list[str] = []
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "decision": "ask_more",
                "questions": [
                    {
                        "id": "q1",
                        "question": "Which platforms are required at launch?",
                        "why": "This determines the packaging test matrix.",
                        "hint": "Name the required operating systems.",
                    }
                ],
            }
        )
    )
    adapter.queue(MockResponse(payload={"decision": "ready_to_plan"}))
    def plan_after_answer(spec):
        questions = json.loads(
            (
                spec.cwd / ".borg/state/planning/context/questions.json"
            ).read_text(encoding="utf-8")
        )
        assert questions[0]["answers"] == [
            {"q_id": "q1", "answer": "Linux and macOS in the first release."}
        ]
        return _plan()

    adapter.queue(MockResponse(dynamic=plan_after_answer))

    database = committed_git_repo.parent / "architect-inline.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "inline"
        )
        result = ArchitectLoop(
            repository,
            borg,
            store,
            adapter,
            io=_io(answers, output),
        ).run()

        assert result.plan == _plan()
        assert result.borg.state is BorgState.TECH_REVIEW_WORKING
        assert len(adapter.calls) == 3
        assert all(call.cwd != committed_git_repo for call in adapter.calls)
        assert all(not call.cwd.exists() for call in adapter.calls)
        assert "Why this matters" in output[0]
        assert "Answer guidance" in output[1]

        questions = store.list_planning_questions(borg.id)
        assert len(questions) == 1
        assert questions[0].answers == [
            {"q_id": "q1", "answer": "Linux and macOS in the first release."}
        ]
        attempts = store.list_planning_attempts(borg.id)
        assert [attempt.phase for attempt in attempts] == [
            "architect_questions",
            "architect_questions",
            "architect_plan",
        ]
        assert all(
            attempt.status is PlanningAttemptStatus.COMPLETED for attempt in attempts
        )
        assert attempts[-1].result == _plan()


def test_selected_codex_agent_runs_architect_in_read_only_sandbox(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    commands: list[list[str]] = []
    trusted_worktrees: list[Path] = []
    payloads = iter([{"decision": "ready_to_plan"}, _plan()])

    def runner(
        command: Sequence[str],
        _cwd: Path,
        _stdin_text: str,
        _log_path: Path,
        _cancel: object,
        _env: Mapping[str, str] | None,
    ) -> int:
        commands.append(list(command))
        invocation_result = Path(command[command.index("-o") + 1])
        invocation_result.write_text(
            json.dumps(next(payloads)), encoding="utf-8"
        )
        return 0

    database = committed_git_repo.parent / "architect-codex.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "codex"
        )
        paths = RepoPaths.discover(committed_git_repo)
        selected = select_agent(
            load_repository_config(paths),
            ApiAgentRole.PLANNING,
            paths,
            interactive=True,
            credentials={},
            executable_lookup=lambda binary: (
                "/bin/codex" if binary == "codex" else None
            ),
            trust_requirement=lambda run_paths, **_kwargs: (
                trusted_worktrees.append(run_paths.root)
            ),
        )
        assert isinstance(selected.adapter, CodexAdapter)
        assert not selected.capabilities.tool_allowlist
        assert selected.capabilities.read_only_sandbox
        selected.adapter.proc_runner = runner

        result = ArchitectLoop(
            repository,
            borg,
            store,
            selected,
            io=_io(iter(()), []),
        ).run()

        assert result.plan == _plan()
        assert result.borg.state is BorgState.TECH_REVIEW_WORKING
        assert len(commands) == 2
        assert all(
            command[command.index("-s") + 1] == "read-only"
            for command in commands
        )
        assert len(trusted_worktrees) == 2
        assert all(not worktree.exists() for worktree in trusted_worktrees)


def test_answers_final_question_round_before_forcing_plan(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    prompts: list[str] = []
    answers = iter(
        [
            "Small internal teams.",
            "Linux and macOS.",
            "Use the stable channel.",
        ]
    )
    adapter = MockAdapter(name="openai")
    for question in (
        "Which users are in scope?",
        "Which platforms are required?",
        "Which release channel should be the default?",
    ):
        adapter.queue(
            MockResponse(
                payload={
                    "decision": "ask_more",
                    "questions": [{"id": "q1", "question": question}],
                }
            )
        )

    def plan_after_final_answer(spec):
        questions = json.loads(
            (
                spec.cwd / ".borg/state/planning/context/questions.json"
            ).read_text(encoding="utf-8")
        )
        assert [item["answers"] for item in questions] == [
            [{"q_id": "q1", "answer": "Small internal teams."}],
            [{"q_id": "q1", "answer": "Linux and macOS."}],
            [{"q_id": "q1", "answer": "Use the stable channel."}],
        ]
        return _plan()

    adapter.queue(MockResponse(dynamic=plan_after_final_answer))
    database = committed_git_repo.parent / "architect-final-question.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "final-question"
        )
        result = ArchitectLoop(
            repository,
            borg,
            store,
            adapter,
            io=InteractiveIO(
                prompt=lambda message: prompts.append(message) or next(answers, None),
                confirm=lambda _message, _default: False,
                write=lambda _message: None,
            ),
        ).run()

        assert result.plan == _plan()
        assert prompts == [
            "Which users are in scope?",
            "Which platforms are required?",
            "Which release channel should be the default?",
        ]
        assert len(adapter.calls) == 4
        assert "question round 3 of 3" in adapter.calls[2].user_prompt
        questions = store.list_planning_questions(borg.id)
        assert len(questions) == 3
        assert questions[-1].answers == [
            {"q_id": "q1", "answer": "Use the stable channel."}
        ]


def test_resumes_a_stored_question_before_invoking_the_agent(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    prompts: list[str] = []
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            dynamic=lambda spec: (
                prompts.append(spec.user_prompt) or {"decision": "ready_to_plan"}
            )
        )
    )
    adapter.queue(MockResponse(payload=_plan()))
    database = committed_git_repo.parent / "architect-question-resume.sqlite3"

    with SqliteStore.open(database) as store:
        repository, draft = persist_planning_context(
            committed_git_repo, store, "resume"
        )
        working = store.compare_and_set_borg_state(
            draft.id,
            expected_state=BorgState.DRAFT,
            expected_version=0,
            new_state=BorgState.ARCHITECT_WORKING,
        )
        attempt = PlanningAttempt(
            borg_id=working.id,
            phase="architect_questions",
            round=1,
            adapter="mock",
            model="test-model",
            status=PlanningAttemptStatus.COMPLETED,
            result={
                "decision": "ask_more",
                "questions": [{"id": "q1", "question": "Which users are first?"}],
            },
            started_at=working.created_at,
            finished_at=working.created_at,
        )
        question = PlanningQuestion(
            borg_id=working.id,
            attempt_id=attempt.id,
            round=1,
            questions=[{"id": "q1", "question": "Which users are first?"}],
        )
        with store.transaction():
            store.append_planning_attempt(attempt)
            store.append_planning_question(question)
            awaiting = store.compare_and_set_borg_state(
                working.id,
                expected_state=working.state,
                expected_version=working.state_version,
                new_state=BorgState.ARCHITECT_AWAITING_ANSWERS,
            )

        result = ArchitectLoop(
            repository,
            awaiting,
            store,
            adapter,
            io=_io(iter(["Small internal teams."]), []),
        ).run()

        assert result.plan == _plan()
        assert len(adapter.calls) == 2
        assert "question round 2" in prompts[0]
        assert store.list_planning_questions(working.id)[0].answers == [
            {"q_id": "q1", "answer": "Small internal teams."}
        ]


def test_recovers_provider_completed_plan_without_duplicate_invocation_or_cost(
    committed_git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    persist_planning_context,
) -> None:
    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    adapter.queue(MockResponse(payload=_plan()))
    database = committed_git_repo.parent / "architect-interrupted.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "interrupted"
        )
        loop = ArchitectLoop(repository, borg, store, adapter, io=_io(iter(()), []))
        original_complete = store.complete_planning_attempt
        interrupted = False

        def interrupt_after_plan_result(attempt_id, **kwargs):
            nonlocal interrupted
            attempt = next(
                item
                for item in store.list_planning_attempts(borg.id)
                if item.id == attempt_id
            )
            if attempt.phase == "architect_plan" and not interrupted:
                interrupted = True
                raise RuntimeError("simulated terminal interruption")
            return original_complete(attempt_id, **kwargs)

        with monkeypatch.context() as interruption:
            interruption.setattr(
                store, "complete_planning_attempt", interrupt_after_plan_result
            )
            with pytest.raises(RuntimeError, match="terminal interruption"):
                loop.run()

        assert len(adapter.calls) == 2
        running_plan = store.list_planning_attempts(borg.id)[-1]
        assert running_plan.phase == "architect_plan"
        assert running_plan.status is PlanningAttemptStatus.RUNNING
        assert Path(running_plan.request["result_path"]).is_file()

        resumed = ArchitectLoop(
            repository,
            store.get_borg(borg.id),
            store,
            adapter,
            io=_io(iter(()), []),
        ).run()

        assert resumed.plan == _plan()
        assert resumed.borg.state is BorgState.TECH_REVIEW_WORKING
        assert len(adapter.calls) == 2
        assert len(store.list_planning_attempts(borg.id)) == 2


def test_plan_open_questions_are_answered_inline_before_replanning(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    ambiguous_plan = _plan()
    ambiguous_plan["open_questions"] = [
        "Should releases default to the stable or preview channel?"
    ]

    def replan_after_answer(spec):
        manifest = json.loads(
            (
                spec.cwd / ".borg/state/planning/context/manifest.json"
            ).read_text(encoding="utf-8")
        )
        current_plan_path = manifest["current_plan"]
        assert current_plan_path is not None
        assert json.loads(
            (spec.cwd / current_plan_path).read_text(encoding="utf-8")
        ) == ambiguous_plan
        questions = json.loads(
            (
                spec.cwd / ".borg/state/planning/context/questions.json"
            ).read_text(encoding="utf-8")
        )
        assert questions[-1]["answers"] == [
            {"q_id": "q1", "answer": "Use the stable channel."}
        ]
        return _plan()

    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    adapter.queue(MockResponse(payload=ambiguous_plan))
    adapter.queue(MockResponse(dynamic=replan_after_answer))
    database = committed_git_repo.parent / "architect-plan-question.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "plan-question"
        )
        with pytest.raises(ArchitectCancelled, match="awaiting answers"):
            ArchitectLoop(
                repository,
                borg,
                store,
                adapter,
                io=_io(iter(()), []),
            ).run()

        assert len(adapter.calls) == 2
        assert store.get_borg(borg.id).state is BorgState.ARCHITECT_AWAITING_ANSWERS
        result = ArchitectLoop(
            repository,
            store.get_borg(borg.id),
            store,
            adapter,
            io=_io(iter(["Use the stable channel."]), []),
        ).run()

        assert result.plan == _plan()
        assert result.borg.state is BorgState.TECH_REVIEW_WORKING
        questions = store.list_planning_questions(borg.id)
        assert len(questions) == 1
        assert questions[0].questions == [
            {
                "id": "q1",
                "question": "Should releases default to the stable or preview channel?",
            }
        ]
        plan_attempts = [
            attempt
            for attempt in store.list_planning_attempts(borg.id)
            if attempt.phase == "architect_plan"
        ]
        assert [attempt.result for attempt in plan_attempts] == [
            ambiguous_plan,
            _plan(),
        ]
        assert len(adapter.calls) == 3


def test_recovered_semantically_invalid_questions_are_failed_and_retryable(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    database = committed_git_repo.parent / "architect-invalid-resume.sqlite3"
    result_path = committed_git_repo.parent / "invalid-question-result.json"
    result_path.write_text(
        json.dumps({"decision": "ask_more"}) + "\n", encoding="utf-8"
    )
    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    adapter.queue(MockResponse(payload=_plan()))

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "invalid"
        )
        interrupted = PlanningAttempt(
            borg_id=borg.id,
            phase="architect_questions",
            round=1,
            adapter="openai",
            model="test-model",
            request={"result_path": str(result_path)},
        )
        store.append_planning_attempt(interrupted)

        with pytest.raises(
            ArchitectError, match="ask_more result must contain questions"
        ):
            ArchitectLoop(
                repository,
                borg,
                store,
                adapter,
                io=_io(iter(()), []),
            ).run()

        failed = store.list_planning_attempts(borg.id)[0]
        assert failed.status is PlanningAttemptStatus.FAILED
        assert failed.result == {"decision": "ask_more"}
        assert len(adapter.calls) == 0

        resumed = ArchitectLoop(
            repository,
            store.get_borg(borg.id),
            store,
            adapter,
            io=_io(iter(()), []),
        ).run()

        assert resumed.plan == _plan()
        question_attempts = [
            attempt
            for attempt in store.list_planning_attempts(borg.id)
            if attempt.phase == "architect_questions"
        ]
        assert [attempt.round for attempt in question_attempts] == [1, 2]
        assert [attempt.status for attempt in question_attempts] == [
            PlanningAttemptStatus.FAILED,
            PlanningAttemptStatus.COMPLETED,
        ]
        assert len(adapter.calls) == 2


def test_cancelled_and_failed_attempts_do_not_consume_question_round_cap(
    committed_git_repo: Path,
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
    adapter.queue(MockResponse(payload={"decision": "ready_to_plan"}))
    adapter.queue(MockResponse(payload=_plan()))
    database = committed_git_repo.parent / "architect-failed-budget.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "budget"
        )
        for round_number, status in enumerate(
            (
                PlanningAttemptStatus.CANCELLED,
                PlanningAttemptStatus.FAILED,
                PlanningAttemptStatus.CANCELLED,
            ),
            start=1,
        ):
            attempt = PlanningAttempt(
                borg_id=borg.id,
                phase="architect_questions",
                round=round_number,
                adapter="openai",
                model="test-model",
            )
            store.append_planning_attempt(attempt)
            store.complete_planning_attempt(
                attempt.id,
                status=status,
                summary="interrupted provider invocation",
            )

        result = ArchitectLoop(
            repository,
            borg,
            store,
            adapter,
            io=_io(iter(["Small internal teams."]), []),
        ).run()

        assert result.plan == _plan()
        assert "question round 1" in adapter.calls[0].user_prompt
        question_attempts = [
            attempt
            for attempt in store.list_planning_attempts(borg.id)
            if attempt.phase == "architect_questions"
        ]
        assert [attempt.round for attempt in question_attempts] == [1, 2, 3, 4, 5]
        assert [attempt.status for attempt in question_attempts[-2:]] == [
            PlanningAttemptStatus.COMPLETED,
            PlanningAttemptStatus.COMPLETED,
        ]
        questions = store.list_planning_questions(borg.id)
        assert len(questions) == 1
        assert questions[0].round == 1
        assert len(adapter.calls) == 3


def _io(answers: Iterator[str], output: list[str]) -> InteractiveIO:
    return InteractiveIO(
        prompt=lambda _message: next(answers, None),
        confirm=lambda _message, _default: False,
        write=output.append,
    )


def _plan() -> dict[str, object]:
    return {
        "title": "Release workflow",
        "summary": "Add a small, tested release workflow.",
        "overall_approach": (
            "Extend the existing repository conventions and verify the public "
            "behavior."
        ),
        "phases": [
            {
                "name": "01-release-workflow",
                "title": "Add release workflow",
                "goal": "Ship a repeatable release path.",
                "technical_approach": (
                    "Add the workflow beside existing automation and exercise "
                    "its contract."
                ),
                "files_touched": [
                    {
                        "path": ".github/workflows/release.yml",
                        "role": "new",
                        "description": "Builds release artifacts.",
                    }
                ],
                "test_strategy": (
                    "Run the repository's declared checks with a release fixture."
                ),
                "acceptance_criteria": ["A tagged release builds the package."],
                "deliverables": ["Release workflow"],
            }
        ],
        "code_pointers": [{"path": "README.md", "why": "Repository overview."}],
        "risks": [],
        "open_questions": [],
    }
