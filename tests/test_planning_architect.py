"""Terminal Architect lifecycle, durability, and resume contracts."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import pytest
from planning_progress_test_support import BoundaryInterruptProgress

from betterborg_cli.agent_runtime import ApiAgentRole, CodexAdapter, select_agent
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.planning import ArchitectCancelled, ArchitectError, ArchitectLoop
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.progress import RunProgress, StageState
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import load_repository_config
from betterborg_cli.store import (
    BorgState,
    PlanningAttempt,
    PlanningAttemptStatus,
    PlanningQuestion,
    SqliteStore,
)


class _TrackingProgress(RunProgress):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_suspended = False
        self.suspension_count = 0

    @contextmanager
    def suspend(self):
        self.suspension_count += 1
        with super().suspend():
            self.output_suspended = True
            try:
                yield self
            finally:
                self.output_suspended = False


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
                spec.cwd / ".betterborg/state/planning/context/questions.json"
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


def test_clarification_output_is_suspended_and_stops_active_architect(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "decision": "ask_more",
                "questions": [
                    {
                        "id": "q1",
                        "question": "Which users are in scope?",
                        "why": "The answer bounds the workflow.",
                    }
                ],
            }
        )
    )
    progress = _TrackingProgress(stream=StringIO())
    output_states: list[bool] = []
    database = committed_git_repo.parent / "architect-progress-prompt.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "progress-prompt"
        )
        with pytest.raises(ArchitectCancelled, match="awaiting answers"):
            ArchitectLoop(
                repository,
                borg,
                store,
                adapter,
                io=InteractiveIO(
                    prompt=lambda _message: output_states.append(
                        progress.output_suspended
                    ),
                    confirm=lambda _message, _default: False,
                    write=lambda _message: output_states.append(
                        progress.output_suspended
                    ),
                ),
                progress=progress,
            ).run()

    assert output_states == [True, True]
    assert progress.suspension_count == 1
    assert progress.stages["architect"].state is StageState.STOPPED


@pytest.mark.parametrize(
    ("interrupt_at", "expected_state", "expected_calls"),
    [
        pytest.param("after-start", StageState.STOPPED, 0, id="start"),
        pytest.param("before-complete", StageState.COMPLETED, 2, id="complete"),
    ],
)
def test_architect_progress_boundary_interrupt_reconciles_durable_state(
    committed_git_repo: Path,
    persist_planning_context,
    interrupt_at: str,
    expected_state: StageState,
    expected_calls: int,
) -> None:
    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    adapter.queue(MockResponse(payload=_plan()))
    progress = BoundaryInterruptProgress(interrupt_at, stream=StringIO())
    database = committed_git_repo.parent / f"architect-{interrupt_at}.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, f"architect-{interrupt_at}"
        )
        with pytest.raises(KeyboardInterrupt, match="interrupted"):
            ArchitectLoop(
                repository,
                borg,
                store,
                adapter,
                io=_io(iter(()), []),
                progress=progress,
            ).run()

        assert len(adapter.calls) == expected_calls
        assert progress.stages["architect"].state is expected_state
        if interrupt_at == "before-complete":
            assert store.get_borg(borg.id).state is BorgState.TECH_REVIEW_WORKING
            assert store.list_planning_attempts(borg.id)[-1].status is (
                PlanningAttemptStatus.COMPLETED
            )
        else:
            assert store.get_borg(borg.id).state is BorgState.DRAFT
        progress.close()


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
        _on_line: Callable[[str], None] | None,
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
                spec.cwd / ".betterborg/state/planning/context/questions.json"
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
                spec.cwd / ".betterborg/state/planning/context/manifest.json"
            ).read_text(encoding="utf-8")
        )
        current_plan_path = manifest["current_plan"]
        assert current_plan_path is not None
        assert json.loads(
            (spec.cwd / current_plan_path).read_text(encoding="utf-8")
        ) == ambiguous_plan
        questions = json.loads(
            (
                spec.cwd / ".betterborg/state/planning/context/questions.json"
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


def test_invalid_plan_contract_fails_before_tech_review(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    invalid_plan = _plan()
    invalid_plan["phases"][0]["name"] = "02-release-workflow"
    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    adapter.queue(MockResponse(payload=invalid_plan))
    database = committed_git_repo.parent / "architect-invalid-plan.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "invalid-plan"
        )

        with pytest.raises(ArchitectError, match="deterministic validation"):
            ArchitectLoop(
                repository,
                borg,
                store,
                adapter,
                io=_io(iter(()), []),
            ).run()

        assert store.get_borg(borg.id).state is BorgState.ARCHITECT_WORKING
        plan_attempt = store.list_planning_attempts(borg.id)[-1]
        assert plan_attempt.status is PlanningAttemptStatus.FAILED
        assert plan_attempt.result == invalid_plan
        assert "expected number 01" in plan_attempt.summary


def test_plan_validation_rejects_untracked_source_the_architect_did_not_see(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    invalid_plan = _plan()
    invalid_plan["code_pointers"] = [
        {"path": "untracked.py", "why": "This file is not in the planning snapshot."}
    ]

    def create_untracked_source(_spec):
        (committed_git_repo / "untracked.py").write_text(
            "print('dirty source')\n", encoding="utf-8"
        )
        return invalid_plan

    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    adapter.queue(MockResponse(dynamic=create_untracked_source))
    database = committed_git_repo.parent / "architect-untracked-source.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "untracked-source"
        )

        with pytest.raises(ArchitectError, match="not grounded in the repository"):
            ArchitectLoop(
                repository,
                borg,
                store,
                adapter,
                io=_io(iter(()), []),
            ).run()

        plan_attempt = store.list_planning_attempts(borg.id)[-1]
        assert plan_attempt.status is PlanningAttemptStatus.FAILED
        assert plan_attempt.result == invalid_plan


def test_plan_validation_ignores_dirty_source_drift_after_architect_turn(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    def remove_tracked_source_after_inspection(spec):
        assert (spec.cwd / "README.md").is_file()
        (committed_git_repo / "README.md").unlink()
        return _plan()

    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    adapter.queue(MockResponse(dynamic=remove_tracked_source_after_inspection))
    database = committed_git_repo.parent / "architect-dirty-drift.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "dirty-drift"
        )

        result = ArchitectLoop(
            repository,
            borg,
            store,
            adapter,
            io=_io(iter(()), []),
        ).run()

        assert result.plan == _plan()
        assert result.borg.state is BorgState.TECH_REVIEW_WORKING


def test_revalidates_a_durable_completed_plan_before_resuming(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    invalid_plan = _plan()
    invalid_plan["phases"][0]["name"] = "02-release-workflow"
    adapter = MockAdapter(name="openai")
    database = committed_git_repo.parent / "architect-invalid-completed-plan.sqlite3"

    with SqliteStore.open(database) as store:
        repository, draft = persist_planning_context(
            committed_git_repo, store, "invalid-completed-plan"
        )
        working = store.compare_and_set_borg_state(
            draft.id,
            expected_state=BorgState.DRAFT,
            expected_version=0,
            new_state=BorgState.ARCHITECT_WORKING,
        )
        attempt = PlanningAttempt(
            borg_id=working.id,
            phase="architect_plan",
            round=1,
            adapter="openai",
            model="test-model",
        )
        store.append_planning_attempt(attempt)
        store.complete_planning_attempt(
            attempt.id,
            status=PlanningAttemptStatus.COMPLETED,
            result=invalid_plan,
            summary=str(invalid_plan["title"]),
        )

        with pytest.raises(
            ArchitectError,
            match="Stored Architect plan failed deterministic validation",
        ):
            ArchitectLoop(
                repository,
                working,
                store,
                adapter,
                io=_io(iter(()), []),
            ).run()

        assert store.get_borg(working.id).state is BorgState.ARCHITECT_WORKING
        assert len(adapter.calls) == 0


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
                        "path": "CHANGELOG.md",
                        "role": "new",
                        "description": "Documents the release workflow.",
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
