"""Terminal Architect lifecycle, durability, and resume contracts."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import timedelta
from io import StringIO
from pathlib import Path

import pytest
from planning_progress_test_support import BoundaryInterruptProgress

from betterborg_cli.agent_runtime import CodexAdapter, select_agent
from betterborg_cli.agent_runtime.base import CancellationToken
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.planning import (
    ARCHITECT_PLAN_CONTRACT_ROUND_CAP,
    ARCHITECT_QUESTION_ROUND_CAP,
    ArchitectCancelled,
    ArchitectError,
    ArchitectLoop,
    render_plan_markdown,
)
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.progress import RunProgress, StageState
from betterborg_cli.repo_paths import RepoPaths
from betterborg_cli.repository_config import AgentStage, load_repository_config
from betterborg_cli.store import (
    BorgState,
    PlanningAttempt,
    PlanningAttemptStatus,
    PlanningFinding,
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


def test_unattended_planning_assumes_its_own_answers_and_carries_them(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    prompts: list[str] = []
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

    def assume_the_answer(spec):
        assert "q1: Which platforms are required at launch?" in spec.user_prompt
        assert (
            "Why this matters: This determines the packaging test matrix."
            in spec.user_prompt
        )
        assert (
            "Answer guidance: Name the required operating systems."
            in spec.user_prompt
        )
        return {
            "answers": [
                {
                    "q_id": "q1",
                    "answer": "Linux and macOS in the first release.",
                }
            ]
        }

    adapter.queue(MockResponse(dynamic=assume_the_answer))
    adapter.queue(MockResponse(payload={"decision": "ready_to_plan"}))

    def plan_after_assumption(spec):
        questions = json.loads(
            (
                spec.cwd / ".betterborg/state/planning/context/questions.json"
            ).read_text(encoding="utf-8")
        )
        assert questions[0]["answers"] == [
            {
                "q_id": "q1",
                "answer": "Linux and macOS in the first release.",
                "assumed": True,
            }
        ]
        return _plan()

    adapter.queue(MockResponse(dynamic=plan_after_assumption))

    database = committed_git_repo.parent / "architect-unattended.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "unattended"
        )
        result = ArchitectLoop(
            repository,
            borg,
            store,
            adapter,
            io=InteractiveIO(
                prompt=lambda message: prompts.append(message) or "Never asked.",
                confirm=lambda _message, _default: False,
                write=lambda _message: None,
            ),
            unattended=True,
        ).run()

        assert prompts == []
        assert result.borg.state is BorgState.TECH_REVIEW_WORKING
        assert result.plan["assumptions"] == [
            {
                "question": "Which platforms are required at launch?",
                "assumption": "Linux and macOS in the first release.",
            }
        ]
        assert "## Assumptions" in render_plan_markdown(result.plan)
        assert [attempt.phase for attempt in store.list_planning_attempts(borg.id)] == [
            "architect_questions",
            "architect_answers",
            "architect_questions",
            "architect_plan",
        ]
        assert all(
            attempt.status is PlanningAttemptStatus.COMPLETED
            for attempt in store.list_planning_attempts(borg.id)
        )
        questions = store.list_planning_questions(borg.id)
        assert questions[0].answers == [
            {
                "q_id": "q1",
                "answer": "Linux and macOS in the first release.",
                "assumed": True,
            }
        ]


def test_an_attended_plan_cannot_claim_an_assumption_nobody_assumed(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    """With a person present, an assumption is a claim nobody made.

    Every requirement an attended run did not read from the PRD it got by
    asking, so a plan that mints an assumption is describing a conversation
    that did not happen, and sends the reader to audit settled ground.
    """
    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    forged = dict(_plan())
    forged["assumptions"] = [
        {"question": "Invented?", "assumption": "Nobody was ever asked this."}
    ]
    adapter.queue(MockResponse(payload=forged))

    database = committed_git_repo.parent / "architect-forged.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "forged"
        )
        result = ArchitectLoop(
            repository,
            borg,
            store,
            adapter,
            io=InteractiveIO(
                prompt=lambda _message: None,
                confirm=lambda _message, _default: False,
                write=lambda _message: None,
            ),
        ).run()

        assert "assumptions" not in result.plan
        assert "## Assumptions" not in render_plan_markdown(result.plan)


def test_unattended_planning_is_told_to_decide_rather_than_ask(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    """Answering a question it never had to ask is the cheaper path.

    A question round costs a turn to ask and another to answer, and ends the
    run outright once the round cap is reached, so the instruction that stops
    it being asked is worth more than the machinery that recovers from it.
    """
    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    adapter.queue(MockResponse(payload=_plan()))

    database = committed_git_repo.parent / "architect-directive.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "directive"
        )
        io = InteractiveIO(
            prompt=lambda _message: None,
            confirm=lambda _message, _default: False,
            write=lambda _message: None,
        )
        ArchitectLoop(
            repository, borg, store, adapter, io=io, unattended=True
        ).run()

        questions_prompt = " ".join(adapter.calls[0].system_prompt.split())
        assert "Nobody is available to answer questions on this run." in (
            questions_prompt
        )
        assert "return ready_to_plan" in questions_prompt
        plan_prompt = " ".join(adapter.calls[1].system_prompt.split())
        assert "List each one under assumptions" in plan_prompt


def test_attended_planning_is_not_told_to_answer_its_own_questions(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    """A run with somebody to ask keeps the instruction to ask them."""
    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    adapter.queue(MockResponse(payload=_plan()))

    database = committed_git_repo.parent / "architect-attended.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "attended"
        )
        ArchitectLoop(
            repository,
            borg,
            store,
            adapter,
            io=InteractiveIO(
                prompt=lambda _message: None,
                confirm=lambda _message, _default: False,
                write=lambda _message: None,
            ),
        ).run()

        assert "Nobody is available" not in adapter.calls[0].system_prompt
        assert "under assumptions" not in adapter.calls[1].system_prompt


def test_an_unattended_plan_carries_the_requirements_it_settled_itself(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    """A decision taken instead of a question is the one nobody else knows."""
    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    decided = dict(_plan())
    decided["assumptions"] = [
        {
            "question": "Which platforms are required at launch?",
            "assumption": "Linux and macOS, because CI builds only those.",
        }
    ]
    adapter.queue(MockResponse(payload=decided))

    database = committed_git_repo.parent / "architect-decided.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "decided"
        )
        result = ArchitectLoop(
            repository,
            borg,
            store,
            adapter,
            io=InteractiveIO(
                prompt=lambda _message: None,
                confirm=lambda _message, _default: False,
                write=lambda _message: None,
            ),
            unattended=True,
        ).run()

        assert result.plan["assumptions"] == [
            {
                "question": "Which platforms are required at launch?",
                "assumption": "Linux and macOS, because CI builds only those.",
            }
        ]
        rendered = render_plan_markdown(result.plan)
        assert "## Assumptions" in rendered
        assert "Linux and macOS, because CI builds only those." in rendered


def test_a_recorded_assumption_survives_a_plan_that_leaves_it_out(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    """Betterborg recorded the round, so the plan cannot decide to forget it.

    The plan below names one new decision and restates the recorded one in
    its own words. Dropping the recorded wording would let a plan quietly
    soften the question it was actually asked.
    """
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "decision": "ask_more",
                "questions": [
                    {
                        "id": "q1",
                        "question": "Which platforms are required at launch?",
                        "why": "This determines the packaging test matrix.",
                    }
                ],
            }
        )
    )
    adapter.queue(
        MockResponse(
            payload={
                "answers": [
                    {"q_id": "q1", "answer": "Linux and macOS in the first release."}
                ]
            }
        )
    )
    adapter.queue(MockResponse(payload={"decision": "ready_to_plan"}))
    partial = dict(_plan())
    partial["assumptions"] = [
        {
            "question": "which platforms are required at launch?",
            "assumption": "Every platform the team uses.",
        },
        {
            "question": "Where does the changelog live?",
            "assumption": "At the repository root, beside the README.",
        },
    ]
    adapter.queue(MockResponse(payload=partial))

    database = committed_git_repo.parent / "architect-merged.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "merged"
        )
        result = ArchitectLoop(
            repository,
            borg,
            store,
            adapter,
            io=InteractiveIO(
                prompt=lambda _message: None,
                confirm=lambda _message, _default: False,
                write=lambda _message: None,
            ),
            unattended=True,
        ).run()

        assert result.plan["assumptions"] == [
            {
                "question": "Which platforms are required at launch?",
                "assumption": "Linux and macOS in the first release.",
            },
            {
                "question": "Where does the changelog live?",
                "assumption": "At the repository root, beside the README.",
            },
        ]


def test_an_answered_question_is_not_an_assumption(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    """A person's answer is a requirement, and must not read as a guess.

    The marking has to separate the two in both directions: labelling a given
    requirement an assumption sends the reader to audit settled ground.
    """
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
    adapter.queue(MockResponse(payload=_plan()))

    database = committed_git_repo.parent / "architect-answered.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "answered"
        )
        result = ArchitectLoop(
            repository,
            borg,
            store,
            adapter,
            io=InteractiveIO(
                prompt=lambda _message: "Linux only.",
                confirm=lambda _message, _default: False,
                write=lambda _message: None,
            ),
        ).run()

        answers = store.list_planning_questions(borg.id)[0].answers or []
        assert answers == [{"q_id": "q1", "answer": "Linux only."}]
        assert "assumptions" not in result.plan
        assert "## Assumptions" not in render_plan_markdown(result.plan)


def test_unattended_planning_ends_when_questions_pass_the_round_cap(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    adapter = MockAdapter(name="openai")
    for index in range(ARCHITECT_QUESTION_ROUND_CAP):
        adapter.queue(
            MockResponse(
                payload={
                    "decision": "ask_more",
                    "questions": [
                        {"id": "q1", "question": f"Question {index + 1}?"}
                    ],
                }
            )
        )
        adapter.queue(
            MockResponse(
                payload={
                    "answers": [
                        {"q_id": "q1", "answer": f"Assumption {index + 1}."}
                    ]
                }
            )
        )
    adapter.queue(
        MockResponse(
            payload={
                **_plan(),
                "open_questions": ["Which release channel is the default?"],
            }
        )
    )

    database = committed_git_repo.parent / "architect-unattended-cap.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "unattended-cap"
        )
        with pytest.raises(ArchitectError, match="asked past question round"):
            ArchitectLoop(
                repository,
                borg,
                store,
                adapter,
                io=_io(iter(()), []),
                unattended=True,
            ).run()

        assert len(adapter.calls) == 2 * ARCHITECT_QUESTION_ROUND_CAP + 1
        questions = store.list_planning_questions(borg.id)
        assert len(questions) == ARCHITECT_QUESTION_ROUND_CAP + 1
        assert questions[-1].answers is None
        current = store.get_borg(borg.id)
        assert current is not None
        assert current.state is BorgState.ARCHITECT_AWAITING_ANSWERS


def test_assumed_answers_must_cover_every_question_in_their_round(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "decision": "ask_more",
                "questions": [
                    {"id": "q1", "question": "Which platforms are required?"},
                    {"id": "q2", "question": "Which release channel is default?"},
                ],
            }
        )
    )
    adapter.queue(
        MockResponse(
            payload={"answers": [{"q_id": "q1", "answer": "Linux and macOS."}]}
        )
    )

    database = committed_git_repo.parent / "architect-partial-answers.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "partial-answers"
        )
        with pytest.raises(ArchitectError, match="one answer for each question"):
            ArchitectLoop(
                repository,
                borg,
                store,
                adapter,
                io=_io(iter(()), []),
                unattended=True,
            ).run()

        attempts = store.list_planning_attempts(borg.id)
        assert [attempt.phase for attempt in attempts] == [
            "architect_questions",
            "architect_answers",
        ]
        assert attempts[-1].status is PlanningAttemptStatus.FAILED
        assert store.list_planning_questions(borg.id)[0].answers is None


def test_assumed_answers_must_answer_each_question_once(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    """Two answers to one question leave the round with no single decision.

    The set of ids answered still matches the set asked, so only counting
    them separates this from a complete round.
    """
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "decision": "ask_more",
                "questions": [
                    {"id": "q1", "question": "Which platforms are required?"}
                ],
            }
        )
    )
    adapter.queue(
        MockResponse(
            payload={
                "answers": [
                    {"q_id": "q1", "answer": "Linux and macOS."},
                    {"q_id": "q1", "answer": "Windows only."},
                ]
            }
        )
    )

    database = committed_git_repo.parent / "architect-duplicate-answers.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "duplicate-answers"
        )
        with pytest.raises(ArchitectError, match="answer IDs must be unique"):
            ArchitectLoop(
                repository,
                borg,
                store,
                adapter,
                io=_io(iter(()), []),
                unattended=True,
            ).run()

        assert store.list_planning_questions(borg.id)[0].answers is None


def test_an_assumed_answer_of_whitespace_is_not_a_decision(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    """Blank is what the schema's minimum length cannot catch.

    A single space satisfies it, strips to nothing, and would be stored as an
    assumption saying the Architect decided the question and decided nothing.
    """
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "decision": "ask_more",
                "questions": [
                    {"id": "q1", "question": "Which platforms are required?"}
                ],
            }
        )
    )
    adapter.queue(MockResponse(payload={"answers": [{"q_id": "q1", "answer": "   "}]}))

    database = committed_git_repo.parent / "architect-blank-answer.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "blank-answer"
        )
        with pytest.raises(ArchitectError, match="answers must not be empty"):
            ArchitectLoop(
                repository,
                borg,
                store,
                adapter,
                io=_io(iter(()), []),
                unattended=True,
            ).run()

        assert store.list_planning_questions(borg.id)[0].answers is None


def test_an_abandoned_answers_turn_is_not_recovered_for_a_later_round(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    """One round's answer must never be recovered as another round's.

    Question ids restart at q1 every round, so an abandoned attempt from an
    earlier round validates perfectly against a later one and would be stored
    as its assumption, silently and with the wrong text. The request context
    is the only thing keeping the two apart.
    """
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "decision": "ask_more",
                "questions": [
                    {"id": "q1", "question": "Which release channel is default?"}
                ],
            }
        )
    )
    adapter.queue(
        MockResponse(payload={"answers": [{"q_id": "q1", "answer": "The stable one."}]})
    )
    adapter.queue(MockResponse(payload={"decision": "ready_to_plan"}))
    adapter.queue(MockResponse(payload=_plan()))

    database = committed_git_repo.parent / "architect-stale-answers.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "stale-answers"
        )
        # An answers turn killed before it could record its result: the row
        # stays RUNNING with a recoverable payload beside it, and it belongs
        # to a question this run will never ask.
        store.append_planning_attempt(
            PlanningAttempt(
                borg_id=borg.id,
                phase="architect_answers",
                round=1,
                adapter="mock",
                model="test-model",
                status=PlanningAttemptStatus.RUNNING,
                request={"question_id": "an-earlier-round"},
                result={
                    "answers": [{"q_id": "q1", "answer": "Whatever was asked."}]
                },
            )
        )

        result = ArchitectLoop(
            repository,
            borg,
            store,
            adapter,
            io=_io(iter(()), []),
            unattended=True,
        ).run()

        stale = next(
            item
            for item in store.list_planning_attempts(borg.id)
            if item.request.get("question_id") == "an-earlier-round"
        )
        assert stale.status is PlanningAttemptStatus.FAILED
        assert "stale request context" in (stale.summary or "")
        assert store.list_planning_questions(borg.id)[0].answers == [
            {"q_id": "q1", "answer": "The stable one.", "assumed": True}
        ]
        assert result.plan["assumptions"] == [
            {
                "question": "Which release channel is default?",
                "assumption": "The stable one.",
            }
        ]


def test_planning_survives_an_architect_result_that_misses_the_schema(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    adapter = MockAdapter(name="openai").queue(
        MockResponse(
            payload={
                "decision": "ask_more",
                "questions": [
                    {
                        "id": "q01",
                        "question": "Which platforms are required at launch?",
                    }
                ],
            }
        )
    )
    adapter.queue(MockResponse(payload={"decision": "ready_to_plan"}))
    adapter.queue(MockResponse(payload=_plan()))
    database = committed_git_repo.parent / "architect-schema-retry.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "schema-retry"
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
        assert len(adapter.calls) == 3
        assert '"^q[1-9][0-9]?$"' in adapter.calls[1].user_prompt
        attempts = store.list_planning_attempts(borg.id)
        assert [attempt.phase for attempt in attempts] == [
            "architect_questions",
            "architect_plan",
        ]
        assert all(
            attempt.status is PlanningAttemptStatus.COMPLETED for attempt in attempts
        )


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
            AgentStage.ARCHITECT,
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


def test_a_plan_that_passes_its_contract_costs_one_turn(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    def plan_without_a_rejected_predecessor(spec):
        manifest = json.loads(
            (
                spec.cwd / ".betterborg/state/planning/context/manifest.json"
            ).read_text(encoding="utf-8")
        )
        assert manifest["current_plan"] is None
        return _plan()

    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    adapter.queue(MockResponse(dynamic=plan_without_a_rejected_predecessor))
    database = committed_git_repo.parent / "architect-valid-plan.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "valid-plan"
        )

        result = ArchitectLoop(
            repository,
            borg,
            store,
            adapter,
            io=_io(iter(()), []),
        ).run()

        assert result.plan == _plan()
        assert len(adapter.calls) == 2
        assert "Rejected plan" not in adapter.calls[1].user_prompt
        attempts = store.list_planning_attempts(borg.id)
        assert [(attempt.phase, attempt.status) for attempt in attempts] == [
            ("architect_questions", PlanningAttemptStatus.COMPLETED),
            ("architect_plan", PlanningAttemptStatus.COMPLETED),
        ]


def test_planning_survives_a_plan_that_fails_its_contract(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    invalid_plan = _plan()
    invalid_plan["phases"][0]["name"] = "02-release-workflow"

    def plan_after_correction(spec):
        assert (
            json.loads(
                (spec.cwd / ".betterborg/plans/failed-contract.md").read_text(
                    encoding="utf-8"
                )
            )
            == invalid_plan
        )
        return _plan()

    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    adapter.queue(MockResponse(payload=invalid_plan))
    adapter.queue(MockResponse(dynamic=plan_after_correction))
    database = committed_git_repo.parent / "architect-failed-contract.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "failed-contract"
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
        assert len(adapter.calls) == 3
        assert "expected number 01" in adapter.calls[2].user_prompt
        attempts = store.list_planning_attempts(borg.id)
        assert [(attempt.phase, attempt.status) for attempt in attempts] == [
            ("architect_questions", PlanningAttemptStatus.COMPLETED),
            ("architect_plan", PlanningAttemptStatus.FAILED),
            ("architect_plan", PlanningAttemptStatus.COMPLETED),
        ]
        assert [attempt.round for attempt in attempts[1:]] == [1, 2]


def test_plan_correction_directs_a_whole_plan_pass(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    invalid_plan = _plan()
    invalid_plan["phases"][0]["name"] = "02-release-workflow"
    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    adapter.queue(MockResponse(payload=invalid_plan))
    adapter.queue(MockResponse(payload=_plan()))
    database = committed_git_repo.parent / "architect-whole-plan-pass.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "whole-plan-pass"
        )

        ArchitectLoop(
            repository,
            borg,
            store,
            adapter,
            io=_io(iter(()), []),
        ).run()

        correction = " ".join(adapter.calls[2].user_prompt.split())
        assert "expected number 01" in correction
        sweep = "re-check the whole plan for anything else these checks would reject"
        assert sweep in correction
        assert "they stop at the first value they reject" in correction
        # The rejection here is a renumber, not a lookup, so the repair verb has
        # to fit every check rather than only the three about grounding a path.
        assert "ground" not in correction.lower()


def test_a_cancelled_run_stops_before_it_validates_the_plan_it_received(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    invalid_plan = _plan()
    invalid_plan["phases"][0]["name"] = "02-release-workflow"
    cancel = CancellationToken()

    class _CancelOnceTheTurnIsDone:
        """Cancels once the turn is done, before the plan is validated."""

        def __init__(self, inner: MockAdapter) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

        def run(self, spec: object, **kwargs: object) -> object:
            result = self._inner.run(spec, **kwargs)
            if len(self._inner.calls) == 2:
                cancel.cancel()
            return result

    inner = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    inner.queue(MockResponse(payload=invalid_plan))
    adapter = _CancelOnceTheTurnIsDone(inner)
    database = committed_git_repo.parent / "architect-cancelled-correction.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "cancelled-correction"
        )

        # Without the check, this iteration's own validation materializes a
        # worktree and the cancellation surfaces as a missing git repository.
        with pytest.raises(ArchitectCancelled, match="cancelled"):
            ArchitectLoop(
                repository,
                borg,
                store,
                adapter,
                io=_io(iter(()), []),
                cancel=cancel,
            ).run()

        assert len(adapter.calls) == 2


def test_a_correction_does_not_outlive_the_turn_it_was_built_for(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    invalid_plan = _plan()
    invalid_plan["phases"][0]["name"] = "02-release-workflow"
    ambiguous_plan = _plan()
    ambiguous_plan["open_questions"] = [
        "Should releases default to the stable or preview channel?"
    ]

    def plan_after_the_question_was_answered(spec):
        prompt = " ".join(spec.user_prompt.split())
        assert "Rejected plan" not in prompt
        manifest = json.loads(
            (
                spec.cwd / ".betterborg/state/planning/context/manifest.json"
            ).read_text(encoding="utf-8")
        )
        current = manifest["current_plan"]
        assert current is not None
        assert json.loads(
            (spec.cwd / current).read_text(encoding="utf-8")
        ) == ambiguous_plan
        return _plan()

    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    adapter.queue(MockResponse(payload=invalid_plan))
    adapter.queue(MockResponse(payload=ambiguous_plan))
    adapter.queue(MockResponse(dynamic=plan_after_the_question_was_answered))
    database = committed_git_repo.parent / "architect-spent-correction.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "spent-correction"
        )

        result = ArchitectLoop(
            repository,
            borg,
            store,
            adapter,
            io=_io(iter(["Use the stable channel."]), []),
        ).run()

        assert result.plan == _plan()
        assert len(adapter.calls) == 4


def test_several_violations_are_correctable_within_the_budget(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    placeholders = _three_phase_plan(
        "<the domain module the manifest names>",
        "<the service module the manifest names>",
        "<the docs page the manifest names>",
    )
    grounded = _three_phase_plan("README.md", "README.md", "README.md")
    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    adapter.queue(MockResponse(payload=placeholders))
    adapter.queue(MockResponse(payload=grounded))
    database = committed_git_repo.parent / "architect-placeholder-paths.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "placeholder-paths"
        )

        result = ArchitectLoop(
            repository,
            borg,
            store,
            adapter,
            io=_io(iter(()), []),
        ).run()

        assert result.plan == grounded
        assert result.borg.state is BorgState.TECH_REVIEW_WORKING
        assert len(adapter.calls) == 3
        attempts = store.list_planning_attempts(borg.id)
        assert [(attempt.phase, attempt.status) for attempt in attempts] == [
            ("architect_questions", PlanningAttemptStatus.COMPLETED),
            ("architect_plan", PlanningAttemptStatus.FAILED),
            ("architect_plan", PlanningAttemptStatus.COMPLETED),
        ]


def test_invalid_plan_contract_fails_before_tech_review(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    invalid_plan = _plan()
    invalid_plan["phases"][0]["name"] = "02-release-workflow"
    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    for _round in range(ARCHITECT_PLAN_CONTRACT_ROUND_CAP):
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
        assert len(adapter.calls) == 1 + ARCHITECT_PLAN_CONTRACT_ROUND_CAP
        plan_attempt = store.list_planning_attempts(borg.id)[-1]
        assert plan_attempt.status is PlanningAttemptStatus.FAILED
        assert plan_attempt.result == invalid_plan
        assert "expected number 01" in plan_attempt.summary


def test_exhausted_plan_corrections_report_the_last_failure(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    out_of_sequence = _plan()
    out_of_sequence["phases"][0]["name"] = "02-release-workflow"
    ungrounded = _plan()
    ungrounded["phases"][0]["files_touched"] = [
        {
            "path": "<the release module the manifest names>",
            "role": "modified",
            "description": "A description where a repository path belongs.",
        }
    ]
    adapter = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    adapter.queue(MockResponse(payload=out_of_sequence))
    for _round in range(ARCHITECT_PLAN_CONTRACT_ROUND_CAP - 1):
        adapter.queue(MockResponse(payload=ungrounded))
    database = committed_git_repo.parent / "architect-last-failure.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "last-failure"
        )

        with pytest.raises(ArchitectError) as failure:
            ArchitectLoop(
                repository,
                borg,
                store,
                adapter,
                io=_io(iter(()), []),
            ).run()

        assert "is not a repository file" in str(failure.value)
        assert "expected number 01" not in str(failure.value)
        assert len(adapter.calls) == 1 + ARCHITECT_PLAN_CONTRACT_ROUND_CAP
        plan_attempt = store.list_planning_attempts(borg.id)[-1]
        assert plan_attempt.status is PlanningAttemptStatus.FAILED
        assert plan_attempt.result == ungrounded


def test_a_rejected_revision_is_corrected_against_its_persisted_findings(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    reviewed_plan = _plan()
    rejected_revision = _plan()
    rejected_revision["summary"] = "Document the tested rollback behavior."
    rejected_revision["phases"][0]["name"] = "02-release-workflow"
    corrected_revision = _plan()
    corrected_revision["summary"] = "Document the tested rollback behavior."

    def revise_the_reviewed_plan(spec):
        assert _materialized_plan(spec, "revision-correction") == reviewed_plan
        return rejected_revision

    def correct_the_rejected_revision(spec):
        assert _materialized_plan(spec, "revision-correction") == rejected_revision
        assert [item["message"] for item in _materialized_findings(spec)] == [
            "Define rollback behavior."
        ]
        return corrected_revision

    adapter = MockAdapter(name="openai")
    adapter.queue(MockResponse(dynamic=revise_the_reviewed_plan))
    adapter.queue(MockResponse(dynamic=correct_the_rejected_revision))
    database = committed_git_repo.parent / "architect-revision-correction.sqlite3"

    with SqliteStore.open(database) as store:
        repository, draft = persist_planning_context(
            committed_git_repo, store, "revision-correction"
        )
        working = store.compare_and_set_borg_state(
            draft.id,
            expected_state=BorgState.DRAFT,
            expected_version=draft.state_version,
            new_state=BorgState.ARCHITECT_WORKING,
        )
        review = _seed_reviewed_plan(
            store, working, reviewed_plan, "Define rollback behavior."
        )

        result = ArchitectLoop(
            repository,
            working,
            store,
            adapter,
            io=_io(iter(()), []),
        ).run()

        assert result.plan == corrected_revision
        assert result.plan["summary"] != reviewed_plan["summary"]
        assert result.borg.state is BorgState.TECH_REVIEW_WORKING
        assert len(adapter.calls) == 2
        # The correcting turn of a revision carries both instructions: the
        # revision still stands, and the plan to revise is now the rejected one.
        correction = " ".join(adapter.calls[1].user_prompt.split())
        assert "addressing every persisted Tech Lead finding" in correction
        assert "That rejected plan is the current plan in your context" in correction
        assert [item.attempt_id for item in store.list_planning_findings(draft.id)] == [
            review.id
        ]
        attempts = store.list_planning_attempts(draft.id)
        assert [(attempt.phase, attempt.status) for attempt in attempts] == [
            ("architect_questions", PlanningAttemptStatus.COMPLETED),
            ("architect_plan", PlanningAttemptStatus.COMPLETED),
            ("tech_review", PlanningAttemptStatus.COMPLETED),
            ("architect_plan", PlanningAttemptStatus.FAILED),
            ("architect_plan", PlanningAttemptStatus.COMPLETED),
        ]


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
    for _round in range(ARCHITECT_PLAN_CONTRACT_ROUND_CAP):
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


def _materialized_plan(spec, borg_name: str) -> dict:
    return json.loads(
        (spec.cwd / f".betterborg/plans/{borg_name}.md").read_text(encoding="utf-8")
    )


def _materialized_findings(spec) -> list[dict]:
    return json.loads(
        (
            spec.cwd / ".betterborg/state/planning/context/findings.json"
        ).read_text(encoding="utf-8")
    )


def _seed_reviewed_plan(
    store: SqliteStore, borg, plan: dict[str, object], finding: str
) -> PlanningAttempt:
    """Persist a reviewed plan whose Tech Lead round requested a revision."""
    review: PlanningAttempt | None = None
    for offset, (phase, result) in enumerate(
        (
            ("architect_questions", {"decision": "ready_to_plan"}),
            ("architect_plan", plan),
            (
                "tech_review",
                {
                    "decision": "request_changes",
                    "summary": finding,
                    "findings": [{"severity": "major", "message": finding}],
                },
            ),
        )
    ):
        # Microsecond steps keep the seeded order stable while leaving every
        # seeded attempt earlier than the ones this run appends.
        started_at = borg.created_at + timedelta(microseconds=offset)
        review = PlanningAttempt(
            borg_id=borg.id,
            phase=phase,
            round=1,
            adapter="openai",
            model="test-model",
            status=PlanningAttemptStatus.COMPLETED,
            result=result,
            started_at=started_at,
            finished_at=started_at,
        )
        store.append_planning_attempt(review)
    assert review is not None
    store.append_planning_finding(
        PlanningFinding(
            borg_id=borg.id,
            attempt_id=review.id,
            round=1,
            severity="major",
            message=finding,
        )
    )
    return review


def _io(answers: Iterator[str], output: list[str]) -> InteractiveIO:
    return InteractiveIO(
        prompt=lambda _message: next(answers, None),
        confirm=lambda _message, _default: False,
        write=output.append,
    )


def _three_phase_plan(*paths: str) -> dict[str, object]:
    """Build a three-phase plan whose phases each modify one supplied path."""
    plan = _plan()
    template = plan["phases"][0]
    names = ("01-release-groundwork", "02-release-workflow", "03-release-docs")
    plan["phases"] = [
        {
            **template,
            "name": name,
            "files_touched": [
                {
                    "path": path,
                    "role": "modified",
                    "description": "The file this phase changes.",
                }
            ],
        }
        for name, path in zip(names, paths, strict=True)
    ]
    return plan


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
