"""Tech Lead findings, revisions, durability, and convergence contracts."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from io import StringIO
from pathlib import Path
from uuid import uuid4

import pytest
from planning_progress_test_support import BoundaryInterruptProgress
from steering_test_support import CancellingAgent

from betterborg_cli.agent_runtime import CancellationToken
from betterborg_cli.agent_runtime.api_tools import READ_ONLY_API_TOOLS
from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.planning import (
    ARCHITECT_QUESTION_ROUND_CAP,
    TECH_REVIEW_ROUND_MINIMUM,
    ArchitectCancelled,
    ArchitectError,
    ArchitectLoop,
    TechLeadCancelled,
    TechLeadError,
    TechLeadLoop,
)
from betterborg_cli.planning.cycles import INITIAL_PLANNING_CYCLE
from betterborg_cli.planning.findings_ledger import open_planning_findings
from betterborg_cli.planning.steering import (
    STEERING_NOTE_SCHEMA,
    STEERING_SYSTEM_PROMPT,
    SteeringScope,
    SteeringSubject,
    assembled_steering_note,
    render_steering_prompt,
)
from betterborg_cli.planning.tech_lead import tech_lead_grant_account
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.progress import (
    AgentActivity,
    AgentActivityKind,
    ChildRecord,
    RunProgress,
    StageRecord,
    StageState,
)
from betterborg_cli.repository_config import PlanningLimits
from betterborg_cli.store import (
    BorgState,
    FindingStatus,
    PlanChangeRequest,
    PlanningAttempt,
    PlanningAttemptStatus,
    PlanningLedgerFinding,
    ReviewAssessment,
    SqliteStore,
)


class _SeedOrderProgress(RunProgress):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seed_parent_states: list[StageState] = []

    def seed_child_completed(
        self,
        stage_key: str,
        child_key: str,
        result: object,
        duration_seconds: float | None = None,
    ) -> ChildRecord:
        self.seed_parent_states.append(self.stages[stage_key].state)
        return super().seed_child_completed(
            stage_key, child_key, result, duration_seconds
        )


class _LifecycleProgress(RunProgress):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.events: list[tuple[str, str]] = []

    def start(self, stage_key: str) -> StageRecord:
        record = super().start(stage_key)
        self.events.append(("start", stage_key))
        return record

    def seed_completed(
        self,
        stage_key: str,
        result: object,
        duration_seconds: float | None = None,
    ) -> StageRecord:
        record = super().seed_completed(stage_key, result, duration_seconds)
        self.events.append(("seed", stage_key))
        return record

    def complete(
        self, stage_key: str, result: object | None = None
    ) -> StageRecord:
        record = super().complete(stage_key, result)
        self.events.append(("complete", stage_key))
        return record


def test_fresh_progress_finishes_architect_before_tech_lead_starts(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_approval_response,
) -> None:
    architect = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect.queue(MockResponse(payload=planning_plan_response()))
    reviewer = MockAdapter(name="openai").queue(
        MockResponse(payload=tech_lead_approval_response())
    )
    progress = _LifecycleProgress(stream=StringIO())
    database = committed_git_repo.parent / "fresh-role-progress.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "fresh-role-progress"
        )
        handoff = ArchitectLoop(
            repository,
            borg,
            store,
            architect,
            io=_io(),
            progress=progress,
        ).run()
        result = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            reviewer,
            io=_io(),
            progress=progress,
        ).run()

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        assert progress.events == [
            ("start", "architect"),
            ("complete", "architect"),
            ("start", "tech-lead"),
            ("complete", "tech-lead"),
        ]
        progress.close()


def test_retained_architect_uses_durable_duration_without_starting(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_approval_response,
) -> None:
    architect = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect.queue(MockResponse(payload=planning_plan_response()))
    reviewer = MockAdapter(name="openai").queue(
        MockResponse(payload=tech_lead_approval_response())
    )
    database = committed_git_repo.parent / "retained-role-progress.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "retained-role-progress"
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        durable_duration = (
            handoff.attempt.finished_at - handoff.attempt.started_at
        ).total_seconds()
        progress = _LifecycleProgress(stream=StringIO())

        result = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            reviewer,
            io=_io(),
            progress=progress,
        ).run()

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        architect_record = progress.stages["architect"]
        assert architect_record.retained is True
        assert architect_record.started_at is None
        assert architect_record.duration_seconds == pytest.approx(durable_duration)
        assert progress.events == [
            ("seed", "architect"),
            ("start", "tech-lead"),
            ("complete", "tech-lead"),
        ]
        progress.close()


def test_findings_drive_bounded_revision_then_exact_approval_transition(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_approval_response,
    tech_lead_change_request_response,
) -> None:
    initial_plan = planning_plan_response()
    revised_plan = planning_plan_response(
        summary="Clarify the tested release failure behavior."
    )
    architect = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect.queue(MockResponse(payload=initial_plan))

    def request_revision(spec):
        assert _current_plan(spec) == initial_plan
        assert _findings(spec) == []
        return tech_lead_change_request_response("Define rollback behavior.")

    def revise_with_persisted_finding(spec):
        findings = _findings(spec)
        assert [item["message"] for item in findings] == [
            "Define rollback behavior."
        ]
        assert _current_plan(spec) == initial_plan
        return revised_plan

    def approve_revision(spec):
        assert _current_plan(spec) == revised_plan
        assert [item["message"] for item in _findings(spec)] == [
            "Define rollback behavior."
        ]
        return tech_lead_approval_response()

    reviewer = MockAdapter(name="openai")
    reviewer.queue(MockResponse(dynamic=request_revision))
    reviewer.queue(MockResponse(dynamic=approve_revision))
    database = committed_git_repo.parent / "tech-lead-approval.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "review-approval"
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        assert handoff.borg.state is BorgState.TECH_REVIEW_WORKING

        architect.queue(MockResponse(dynamic=revise_with_persisted_finding))
        result = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            reviewer,
            architect_agent=architect,
            io=_io(),
        ).run()

        assert result.plan == revised_plan
        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        assert result.borg.state_version == 5
        assert len(reviewer.calls) == 2
        assert len(architect.calls) == 3
        attempts = store.list_planning_attempts(borg.id)
        assert [item.phase for item in attempts] == [
            "architect_questions",
            "architect_plan",
            "tech_review",
            "architect_plan",
            "tech_review",
        ]
        assert all(
            item.status is PlanningAttemptStatus.COMPLETED for item in attempts
        )
        findings = store.list_planning_findings(borg.id)
        assert [(item.round, item.message) for item in findings] == [
            (1, "Define rollback behavior.")
        ]


@pytest.mark.parametrize(
    ("interrupt_at", "expected_state", "expected_calls"),
    [
        pytest.param("after-start", StageState.STOPPED, 0, id="start"),
        pytest.param("before-complete", StageState.COMPLETED, 1, id="complete"),
    ],
)
def test_tech_lead_progress_boundary_interrupt_reconciles_durable_state(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_approval_response,
    interrupt_at: str,
    expected_state: StageState,
    expected_calls: int,
) -> None:
    architect = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect.queue(MockResponse(payload=planning_plan_response()))
    reviewer = MockAdapter(name="openai").queue(
        MockResponse(payload=tech_lead_approval_response())
    )
    progress = BoundaryInterruptProgress(interrupt_at, stream=StringIO())
    database = committed_git_repo.parent / f"tech-lead-{interrupt_at}.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, f"tech-lead-{interrupt_at}"
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()

        with pytest.raises(KeyboardInterrupt, match="interrupted"):
            TechLeadLoop(
                repository,
                handoff.borg,
                store,
                reviewer,
                io=_io(),
                progress=progress,
            ).run()

        assert len(reviewer.calls) == expected_calls
        assert progress.stages["tech-lead"].state is expected_state
        if interrupt_at == "before-complete":
            assert store.get_borg(borg.id).state is BorgState.PLAN_APPROVAL_PENDING
            review = store.list_planning_attempts(borg.id)[-1]
            assert review.status is PlanningAttemptStatus.COMPLETED
        else:
            assert store.get_borg(borg.id).state is BorgState.TECH_REVIEW_WORKING
        progress.close()


def test_completed_revision_child_is_not_stopped_by_completion_interrupt(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_change_request_response,
) -> None:
    architect = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect.queue(MockResponse(payload=planning_plan_response()))
    architect.queue(
        MockResponse(payload=planning_plan_response(summary="Durable revision."))
    )
    reviewer = MockAdapter(name="openai").queue(
        MockResponse(
            payload=tech_lead_change_request_response("Revise the rollout plan.")
        )
    )
    progress = BoundaryInterruptProgress(
        "before-complete-child", stream=StringIO()
    )
    database = committed_git_repo.parent / "revision-completion-interrupt.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "revision-completion-interrupt"
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()

        with pytest.raises(KeyboardInterrupt, match="complete-child"):
            TechLeadLoop(
                repository,
                handoff.borg,
                store,
                reviewer,
                architect_agent=architect,
                io=_io(),
                progress=progress,
            ).run()

        review = next(
            item
            for item in store.list_planning_attempts(borg.id)
            if item.phase == "tech_review"
        )
        child = progress.stages["tech-lead"].children[
            f"architect-revision:{review.id}"
        ]
        assert child.state is StageState.COMPLETED
        assert progress.stages["tech-lead"].state is StageState.STOPPED
        assert store.get_borg(borg.id).state is BorgState.TECH_REVIEW_WORKING
        progress.close()


def test_recovers_completed_provider_review_without_duplicate_turn(
    committed_git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    persist_planning_context,
    planning_plan_response,
    tech_lead_approval_response,
) -> None:
    database = committed_git_repo.parent / "tech-lead-interruption.sqlite3"
    architect = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect.queue(MockResponse(payload=planning_plan_response()))
    reviewer = MockAdapter(name="openai").queue(
        MockResponse(payload=tech_lead_approval_response())
    )

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "review-interruption"
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        loop = TechLeadLoop(
            repository, handoff.borg, store, reviewer, io=_io()
        )
        original_complete = store.complete_planning_attempt
        interrupted = False

        def interrupt_after_review_result(attempt_id, **kwargs):
            nonlocal interrupted
            attempt = next(
                item
                for item in store.list_planning_attempts(borg.id)
                if item.id == attempt_id
            )
            if attempt.phase == "tech_review" and not interrupted:
                interrupted = True
                raise RuntimeError("simulated terminal interruption")
            return original_complete(attempt_id, **kwargs)

        with monkeypatch.context() as interruption:
            interruption.setattr(
                store, "complete_planning_attempt", interrupt_after_review_result
            )
            with pytest.raises(RuntimeError, match="terminal interruption"):
                loop.run()

        running = store.list_planning_attempts(borg.id)[-1]
        assert running.phase == "tech_review"
        assert running.status is PlanningAttemptStatus.RUNNING
        assert Path(running.request["result_path"]).is_file()
        assert len(reviewer.calls) == 1

        resumed = loop.run()

        assert resumed.borg.state is BorgState.PLAN_APPROVAL_PENDING
        assert len(reviewer.calls) == 1
        assert [
            item.status
            for item in store.list_planning_attempts(borg.id)
            if item.phase == "tech_review"
        ] == [PlanningAttemptStatus.COMPLETED]
        assert loop.run() == resumed
        assert len(reviewer.calls) == 1


def test_unattended_revision_assumes_the_questions_it_raises(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_approval_response,
    tech_lead_change_request_response,
) -> None:
    """A revision the Tech Lead asked for must not stop on its own question.

    Later question rounds arise here rather than in the first Architect pass,
    and this loop builds the Architect that answers them, so an unattended run
    that does not reach this one dies after the review it already paid for.
    """
    initial_plan = planning_plan_response()
    ambiguous_plan = planning_plan_response(
        summary="Choose a concrete rollback strategy."
    )
    ambiguous_plan["open_questions"] = ["Which rollback strategy should be used?"]
    revised_plan = planning_plan_response(
        summary="Use retries before rolling back the release."
    )
    # The plan names the decision the run made for it, which is what
    # the unattended directive asks of a real one.
    revised_plan["assumptions"] = [
        {
            "question": "Which rollback strategy should be used?",
            "assumption": "Retry twice, then roll back.",
        }
    ]
    database = committed_git_repo.parent / "tech-lead-unattended.sqlite3"
    architect = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect.queue(MockResponse(payload=initial_plan))
    reviewer = MockAdapter(name="openai")
    reviewer.queue(
        MockResponse(
            payload=tech_lead_change_request_response("Define rollback behavior.")
        )
    )
    reviewer.queue(MockResponse(payload=tech_lead_approval_response()))

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "review-unattended"
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io(), unattended=True
        ).run()

        architect.queue(MockResponse(payload=ambiguous_plan))
        architect.queue(
            MockResponse(
                payload={
                    "answers": [
                        {"q_id": "q1", "answer": "Retry twice, then roll back."}
                    ]
                }
            )
        )
        architect.queue(MockResponse(payload=revised_plan))

        resumed = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            reviewer,
            architect_agent=architect,
            io=_io(),
            unattended=True,
        ).run()

    assert resumed.borg.state is BorgState.PLAN_APPROVAL_PENDING
    assert resumed.plan["assumptions"] == [
        {
            "question": "Which rollback strategy should be used?",
            "assumption": "Retry twice, then roll back.",
        }
    ]


def test_an_assumption_survives_the_revision_that_does_not_revisit_it(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_approval_response,
    tech_lead_change_request_response,
) -> None:
    """A revision addresses a finding; it does not re-argue settled ground.

    The Architect names an assumption once, in the plan that made it. Nothing
    obliges the revision that answers an unrelated finding to restate it, and
    a plan that quietly stops carrying one leaves a decision nobody took
    reading like a requirement somebody gave.
    """
    initial_plan = planning_plan_response()
    initial_plan["assumptions"] = [
        {
            "question": "Where does the changelog live?",
            "assumption": "At the repository root, beside the README.",
        }
    ]
    # The revision addresses the finding and says nothing about assumptions,
    # which is what a schema making the field optional invites.
    revised_plan = planning_plan_response(
        summary="Define the rollback behavior the review asked for."
    )
    assert "assumptions" not in revised_plan

    database = committed_git_repo.parent / "tech-lead-carried.sqlite3"
    architect = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect.queue(MockResponse(payload=initial_plan))
    reviewer = MockAdapter(name="openai")
    reviewer.queue(
        MockResponse(
            payload=tech_lead_change_request_response("Define rollback behavior.")
        )
    )
    reviewer.queue(MockResponse(payload=tech_lead_approval_response()))

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "review-carried"
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io(), unattended=True
        ).run()
        assert handoff.plan["assumptions"] == initial_plan["assumptions"]

        architect.queue(MockResponse(payload=revised_plan))
        resumed = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            reviewer,
            architect_agent=architect,
            io=_io(),
            unattended=True,
        ).run()

    assert resumed.borg.state is BorgState.PLAN_APPROVAL_PENDING
    assert resumed.plan["assumptions"] == [
        {
            "question": "Where does the changelog live?",
            "assumption": "At the repository root, beside the README.",
        }
    ]


def test_a_revision_that_names_its_assumptions_replaces_them_rather_than_adding(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_approval_response,
    tech_lead_change_request_response,
) -> None:
    """The plan names the set it rests on, not every wording it has used.

    A revision restating an assumption in different words is the same
    decision, and keeping both wordings turns the section into a history of
    the review. After a few rounds that buries the decisions it exists to
    surface, and a reviewer reads the duplication as the defect it is.
    """
    initial_plan = planning_plan_response()
    initial_plan["assumptions"] = [
        {
            "question": "Where does the changelog live?",
            "assumption": "At the repository root.",
        }
    ]
    revised_plan = planning_plan_response(summary="Define the rollback behavior.")
    # The same decision asked in different words, which is what defeats a
    # merge that can only tell two entries apart by their question text.
    revised_plan["assumptions"] = [
        {
            "question": "Which directory holds the changelog?",
            "assumption": "The repository root, beside the README.",
        }
    ]

    database = committed_git_repo.parent / "tech-lead-restated.sqlite3"
    architect = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect.queue(MockResponse(payload=initial_plan))
    reviewer = MockAdapter(name="openai")
    reviewer.queue(
        MockResponse(
            payload=tech_lead_change_request_response("Define rollback behavior.")
        )
    )
    reviewer.queue(MockResponse(payload=tech_lead_approval_response()))

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "review-restated"
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io(), unattended=True
        ).run()

        architect.queue(MockResponse(payload=revised_plan))
        resumed = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            reviewer,
            architect_agent=architect,
            io=_io(),
            unattended=True,
        ).run()

    assert resumed.plan["assumptions"] == [
        {
            "question": "Which directory holds the changelog?",
            "assumption": "The repository root, beside the README.",
        }
    ]


def test_two_question_raising_revisions_strand_no_decision(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_approval_response,
    tech_lead_change_request_response,
) -> None:
    """A stored plan's assumptions key does not say who wrote it.

    Every payload passes through the merge before it is stored, so a plan that
    named nothing still holds what it inherited. Reading that as having spoken
    would let each question-raising revision close the window over the answer
    the one before it produced, and those decisions would reach neither the
    correction nor the published plan.
    """
    initial = planning_plan_response()
    initial["assumptions"] = [
        {
            "question": "Where does the changelog live?",
            "assumption": "At the repository root.",
        }
    ]
    first_questions = planning_plan_response(summary="Stage the rollout.")
    first_questions["open_questions"] = ["Which rollback strategy should be used?"]
    second_questions = planning_plan_response(summary="Stage it again.")
    second_questions["open_questions"] = ["Which changelog format is required?"]
    silent = planning_plan_response(summary="Address the finding.")
    assert "assumptions" not in silent

    database = committed_git_repo.parent / "tech-lead-two-question-plans.sqlite3"
    architect = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect.queue(MockResponse(payload=initial))
    reviewer = MockAdapter(name="openai").queue(
        MockResponse(
            payload=tech_lead_change_request_response("Define rollback behavior.")
        )
    )
    reviewer.queue(MockResponse(payload=tech_lead_approval_response()))

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "two-question-plans"
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io(), unattended=True
        ).run()

        architect.queue(MockResponse(payload=first_questions))
        architect.queue(
            MockResponse(
                payload={
                    "answers": [
                        {"q_id": "q1", "answer": "Retry twice, then roll back."}
                    ]
                }
            )
        )
        architect.queue(MockResponse(payload=second_questions))
        architect.queue(
            MockResponse(
                payload={"answers": [{"q_id": "q1", "answer": "Keep a Changelog."}]}
            )
        )
        architect.queue(MockResponse(payload=silent))
        architect.queue(MockResponse(payload=silent))

        resumed = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            reviewer,
            architect_agent=architect,
            io=_io(),
            unattended=True,
        ).run()

    published = {
        item["assumption"] for item in resumed.plan.get("assumptions", [])
    }
    assert "Retry twice, then roll back." in published
    assert "Keep a Changelog." in published


def test_a_question_raised_by_a_plan_is_answered_against_that_plan(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_approval_response,
    tech_lead_change_request_response,
) -> None:
    """The turn deciding a plan's open question needs the plan that raised it.

    It is a fresh agent holding none of the reasoning that produced the
    question. Given a workspace whose manifest says no plan exists, it answers
    from the PRD alone, and that answer is recorded as a decision the plan
    rests on.
    """
    initial_plan = planning_plan_response()
    ambiguous_plan = planning_plan_response(
        summary="Choose a concrete rollback strategy."
    )
    ambiguous_plan["open_questions"] = ["Which rollback strategy should be used?"]
    revised_plan = planning_plan_response(
        summary="Use retries before rolling back the release."
    )
    revised_plan["assumptions"] = [
        {
            "question": "Which rollback strategy should be used?",
            "assumption": "Retry twice, then roll back.",
        }
    ]
    seen: dict[str, object] = {}

    def answer_against_the_plan(spec):
        manifest = json.loads(
            (
                spec.cwd / ".betterborg/state/planning/context/manifest.json"
            ).read_text(encoding="utf-8")
        )
        seen["current_plan"] = manifest.get("current_plan")
        seen["plan_text"] = (spec.cwd / str(manifest["current_plan"])).read_text(
            encoding="utf-8"
        )
        seen["user_prompt"] = spec.user_prompt
        return {"answers": [{"q_id": "q1", "answer": "Retry twice, then roll back."}]}

    database = committed_git_repo.parent / "tech-lead-plan-context.sqlite3"
    architect = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect.queue(MockResponse(payload=initial_plan))
    reviewer = MockAdapter(name="openai")
    reviewer.queue(
        MockResponse(
            payload=tech_lead_change_request_response("Define rollback behavior.")
        )
    )
    reviewer.queue(MockResponse(payload=tech_lead_approval_response()))

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "review-plan-context"
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io(), unattended=True
        ).run()

        architect.queue(MockResponse(payload=ambiguous_plan))
        architect.queue(MockResponse(dynamic=answer_against_the_plan))
        architect.queue(MockResponse(payload=revised_plan))

        TechLeadLoop(
            repository,
            handoff.borg,
            store,
            reviewer,
            architect_agent=architect,
            io=_io(),
            unattended=True,
        ).run()

    assert seen["current_plan"] is not None
    assert "Choose a concrete rollback strategy." in str(seen["plan_text"])
    assert "a question a plan raises is a question about that plan" in str(
        seen["user_prompt"]
    )


def test_resumes_committed_change_request_through_architect_pause(
    committed_git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    persist_planning_context,
    planning_plan_response,
    tech_lead_approval_response,
    tech_lead_change_request_response,
) -> None:
    initial_plan = planning_plan_response()
    ambiguous_plan = planning_plan_response(
        summary="Choose a concrete rollback strategy."
    )
    ambiguous_plan["open_questions"] = ["Which rollback strategy should be used?"]
    revised_plan = planning_plan_response(
        summary="Use retries before rolling back the release."
    )
    database = committed_git_repo.parent / "tech-lead-architect-resume.sqlite3"
    architect = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect.queue(MockResponse(payload=initial_plan))
    reviewer = MockAdapter(name="openai")
    reviewer.queue(
        MockResponse(
            payload=tech_lead_change_request_response("Define rollback behavior.")
        )
    )
    reviewer.queue(MockResponse(payload=tech_lead_approval_response()))

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "review-architect-resume"
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        loop = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            reviewer,
            architect_agent=architect,
            io=_io(),
        )

        def interrupt_architect(_loop: ArchitectLoop) -> None:
            raise RuntimeError("simulated Architect interruption")

        with monkeypatch.context() as interruption:
            interruption.setattr(ArchitectLoop, "run", interrupt_architect)
            with pytest.raises(RuntimeError, match="Architect interruption"):
                loop.run()

        assert store.get_borg(borg.id).state is BorgState.ARCHITECT_WORKING
        assert len(reviewer.calls) == 1

        architect.queue(MockResponse(payload=ambiguous_plan))
        with pytest.raises(ArchitectCancelled, match="awaiting answers"):
            loop.run()

        assert store.get_borg(borg.id).state is BorgState.ARCHITECT_AWAITING_ANSWERS
        assert len(reviewer.calls) == 1

        architect.queue(MockResponse(payload=revised_plan))
        resumed = TechLeadLoop(
            repository,
            store.get_borg(borg.id),
            store,
            reviewer,
            architect_agent=architect,
            io=_io(iter(["Retry twice, then roll back."])),
        ).run()

        assert resumed.borg.state is BorgState.PLAN_APPROVAL_PENDING
        assert resumed.plan == revised_plan
        assert len(reviewer.calls) == 2
        assert [
            item.result["decision"]
            for item in store.list_planning_attempts(borg.id)
            if item.phase == "tech_review"
        ] == ["request_changes", "approve"]


def test_third_change_request_blocks_with_durable_resumable_history(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_change_request_response,
) -> None:
    database = committed_git_repo.parent / "tech-lead-cap.sqlite3"
    plans = [
        planning_plan_response(),
        planning_plan_response(summary="Revision one."),
        planning_plan_response(summary="Revision two."),
    ]
    architect = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect.queue(MockResponse(payload=plans[0]))

    def revise(index: int):
        def response(spec):
            assert len(_findings(spec)) == index
            return plans[index]

        return response

    architect.queue(MockResponse(dynamic=revise(1)))
    architect.queue(MockResponse(dynamic=revise(2)))
    reviewer = MockAdapter(name="openai")
    for round_number in range(1, 4):
        reviewer.queue(
            MockResponse(
                dynamic=lambda spec, round_number=round_number: (
                    _assert_prior_finding_count(spec, round_number - 1)
                    or tech_lead_change_request_response(
                        f"Finding round {round_number}."
                    )
                )
            )
        )

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "review-cap"
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        loop = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            reviewer,
            architect_agent=architect,
            io=_io(),
            grant_budget=0,
        )

        result = loop.run()

        assert result.borg.state is BorgState.BLOCKED
        assert result.plan == plans[-1]
        assert len(reviewer.calls) == 3
        assert len(architect.calls) == 4
        assert [item.round for item in store.list_planning_findings(borg.id)] == [
            1,
            2,
            3,
        ]
        assert loop.run() == result
        assert len(reviewer.calls) == 3


def test_unconfigured_repository_keeps_the_default_review_round_budget() -> None:
    assert PlanningLimits().review_rounds == TECH_REVIEW_ROUND_MINIMUM


def test_raised_review_budget_approves_on_a_round_the_default_denies(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_approval_response,
    tech_lead_change_request_response,
) -> None:
    database = committed_git_repo.parent / "tech-lead-raised-budget.sqlite3"
    plans = [
        planning_plan_response(),
        planning_plan_response(summary="Revision one."),
        planning_plan_response(summary="Revision two."),
        planning_plan_response(summary="Revision three."),
    ]
    architect = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    for plan in plans:
        architect.queue(MockResponse(payload=plan))
    reviewer = MockAdapter(name="openai")
    for round_number in range(1, 4):
        reviewer.queue(
            MockResponse(
                payload=tech_lead_change_request_response(
                    f"Finding round {round_number}."
                )
            )
        )
    reviewer.queue(MockResponse(payload=tech_lead_approval_response()))

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "review-raised-budget"
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()

        result = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            reviewer,
            architect_agent=architect,
            io=_io(),
            review_rounds=4,
            grant_budget=0,
        ).run()

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        assert result.plan == plans[-1]
        assert len(reviewer.calls) == 4
        assert "review round 1." in reviewer.calls[0].user_prompt
        assert "review round 4." in reviewer.calls[-1].user_prompt
        assert [item.round for item in store.list_planning_findings(borg.id)] == [
            1,
            2,
            3,
        ]


def test_lowering_the_budget_does_not_strand_a_revision_already_under_way(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_change_request_response,
    tech_lead_approval_response,
) -> None:
    """The budget bounds what happens next, never what already happened.

    A run interrupted mid-revision is resumable, and the CLI says so. Reading
    the record through a budget lowered since would hide the rejection the
    revision belongs to, and the advertised resume could never succeed: the
    only way back would be restoring a number nothing names.
    """
    database = committed_git_repo.parent / "tech-lead-lowered-midflight.sqlite3"
    architect = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect.queue(MockResponse(payload=planning_plan_response()))
    reviewer = MockAdapter(name="openai").queue(
        MockResponse(
            payload=tech_lead_change_request_response("Define rollback behavior.")
        )
    )

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "review-lowered-midflight"
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()

        # The review asks for a revision, then the run dies before the
        # Architect can answer it.
        with pytest.raises((TechLeadError, ArchitectError)):
            TechLeadLoop(
                repository,
                handoff.borg,
                store,
                reviewer,
                architect_agent=architect,
                io=_io(),
                review_rounds=3,
            ).run()
        interrupted = store.get_borg(borg.id)
        assert interrupted is not None
        assert interrupted.state is BorgState.ARCHITECT_WORKING

        # The operator lowers the budget below the round already spent, then
        # resumes as the CLI told them to.
        architect.queue(MockResponse(payload=planning_plan_response()))
        reviewer.queue(MockResponse(payload=tech_lead_approval_response()))
        resumed = TechLeadLoop(
            repository,
            interrupted,
            store,
            reviewer,
            architect_agent=architect,
            io=_io(),
            review_rounds=1,
        ).run()

        assert resumed.borg.state is BorgState.PLAN_APPROVAL_PENDING
        # The round the stranded revision leads to is an ordinary granted one,
        # and it is told its number and nothing else.
        assert "review round 1." in reviewer.calls[0].user_prompt
        assert "review round 2." in reviewer.calls[-1].user_prompt
        assert "of 1" not in reviewer.calls[-1].user_prompt


def test_lowered_review_budget_blocks_after_its_only_round(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_change_request_response,
) -> None:
    database = committed_git_repo.parent / "tech-lead-lowered-budget.sqlite3"
    architect = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect.queue(MockResponse(payload=planning_plan_response()))
    reviewer = MockAdapter(name="openai").queue(
        MockResponse(
            payload=tech_lead_change_request_response("Define rollback behavior.")
        )
    )

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "review-lowered-budget"
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        loop = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            reviewer,
            architect_agent=architect,
            io=_io(),
            review_rounds=1,
            grant_budget=0,
        )

        result = loop.run()

        assert result.borg.state is BorgState.BLOCKED
        assert "review round 1." in reviewer.calls[0].user_prompt
        assert len(reviewer.calls) == 1
        assert len(architect.calls) == 2
        assert [
            (item.round, item.message)
            for item in store.list_planning_findings(borg.id)
        ] == [(1, "Define rollback behavior.")]
        assert loop.run() == result
        assert len(reviewer.calls) == 1


def test_two_revision_children_reconstruct_once_from_durable_attempt_ids(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_approval_response,
    tech_lead_change_request_response,
) -> None:
    plans = [
        planning_plan_response(summary="Initial plan."),
        planning_plan_response(summary="First revision."),
        planning_plan_response(summary="Second revision."),
    ]
    architect = MockAdapter(name="openai")
    architect.queue(MockResponse(payload={"decision": "ready_to_plan"}))
    architect.queue(MockResponse(payload=plans[0]))
    architect.queue(MockResponse(payload=plans[1]))
    architect.queue(MockResponse(raise_error=RuntimeError("revision interrupted")))
    reviewer = MockAdapter(name="openai")
    reviewer.queue(
        MockResponse(payload=tech_lead_change_request_response("First finding."))
    )
    reviewer.queue(
        MockResponse(payload=tech_lead_change_request_response("Second finding."))
    )
    database = committed_git_repo.parent / "tech-lead-progress-resume.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "review-progress-resume"
        )
        interrupted_progress = RunProgress(stream=StringIO())
        handoff = ArchitectLoop(
            repository,
            borg,
            store,
            architect,
            io=_io(),
            progress=interrupted_progress,
        ).run()
        with pytest.raises(ArchitectError, match="revision interrupted"):
            TechLeadLoop(
                repository,
                handoff.borg,
                store,
                reviewer,
                architect_agent=architect,
                io=_io(),
                progress=interrupted_progress,
            ).run()

        reviews = [
            item
            for item in store.list_planning_attempts(borg.id)
            if item.phase == "tech_review"
        ]
        keys = [f"architect-revision:{item.id}" for item in reviews]
        assert len(keys) == 2
        assert len(set(keys)) == 2
        interrupted_children = interrupted_progress.stages["tech-lead"].children
        assert interrupted_children[keys[0]].state is StageState.COMPLETED
        assert interrupted_children[keys[1]].state is StageState.FAILED
        assert interrupted_progress.stages["tech-lead"].state is StageState.FAILED

        architect.queue(MockResponse(payload=plans[2]))
        reviewer.queue(MockResponse(payload=tech_lead_approval_response()))
        resumed_progress = _SeedOrderProgress(
            stream=StringIO(), attempt_history_limit=1
        )
        result = TechLeadLoop(
            repository,
            store.get_borg(borg.id),
            store,
            reviewer,
            architect_agent=architect,
            io=_io(),
            progress=resumed_progress,
        ).run()

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        architect_record = resumed_progress.stages["architect"]
        assert architect_record.state is StageState.COMPLETED
        assert architect_record.retained is True
        assert architect_record.started_at is None
        children = resumed_progress.stages["tech-lead"].children
        assert list(children) == keys
        assert children[keys[0]].state is StageState.COMPLETED
        assert children[keys[0]].retained is True
        assert children[keys[0]].started_at is None
        assert children[keys[1]].state is StageState.COMPLETED
        assert children[keys[1]].retained is False
        assert children[keys[1]].started_at is not None
        assert resumed_progress.stages["tech-lead"].state is StageState.COMPLETED
        assert resumed_progress.seed_parent_states == [StageState.PENDING]
        bounded = resumed_progress.child_render_state("tech-lead")
        assert [item.key for item in bounded.children] == [keys[1]]
        assert bounded.earlier_attempt_count == 1
        assert len(architect.calls) == 5
        assert len(reviewer.calls) == 3


@pytest.mark.parametrize(
    ("cancel_setup", "error_type", "expected_state"),
    [
        pytest.param(
            True,
            ArchitectCancelled,
            StageState.STOPPED,
            id="cancelled",
        ),
        pytest.param(False, RuntimeError, StageState.FAILED, id="failed"),
    ],
)
def test_revision_constructor_error_reconciles_child_before_parent(
    committed_git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    persist_planning_context,
    planning_plan_response,
    tech_lead_change_request_response,
    cancel_setup: bool,
    error_type: type[Exception],
    expected_state: StageState,
) -> None:
    architect = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    architect.queue(MockResponse(payload=planning_plan_response()))
    reviewer = MockAdapter(name="openai").queue(
        MockResponse(
            payload=tech_lead_change_request_response("Revise the rollout plan.")
        )
    )
    database = committed_git_repo.parent / "tech-lead-constructor-cancel.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "review-constructor-cancel"
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        cancel = CancellationToken()
        progress = RunProgress(stream=StringIO())
        loop = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            reviewer,
            architect_agent=architect,
            io=_io(),
            cancel=cancel,
            progress=progress,
        )

        def interrupt_revision_setup(_self, *_args, **_kwargs) -> None:
            if cancel_setup:
                cancel.cancel()
                progress.begin_cancellation()
            raise error_type("revision setup interrupted")

        monkeypatch.setattr(ArchitectLoop, "__init__", interrupt_revision_setup)
        with pytest.raises(error_type, match="setup interrupted"):
            loop.run()

        review = next(
            item
            for item in store.list_planning_attempts(borg.id)
            if item.phase == "tech_review"
        )
        child = progress.stages["tech-lead"].children[
            f"architect-revision:{review.id}"
        ]
        assert child.state is expected_state
        assert child.started_at is not None
        assert progress.stages["tech-lead"].state is expected_state
        progress.close()


def test_revalidates_architect_handoff_before_invoking_tech_lead(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_approval_response,
) -> None:
    invalid = planning_plan_response()
    invalid["phases"][0]["name"] = "02-release-workflow"
    reviewer = MockAdapter(name="openai").queue(
        MockResponse(payload=tech_lead_approval_response())
    )
    database = committed_git_repo.parent / "tech-lead-invalid.sqlite3"

    with SqliteStore.open(database) as store:
        repository, draft = persist_planning_context(
            committed_git_repo, store, "invalid-review-handoff"
        )
        working = store.compare_and_set_borg_state(
            draft.id,
            expected_state=BorgState.DRAFT,
            expected_version=draft.state_version,
            new_state=BorgState.ARCHITECT_WORKING,
        )
        attempt = PlanningAttempt(
            borg_id=draft.id,
            phase="architect_plan",
            round=1,
            adapter="openai",
            model="test-model",
            status=PlanningAttemptStatus.COMPLETED,
            result=invalid,
            started_at=working.created_at,
            finished_at=working.created_at,
        )
        store.append_planning_attempt(attempt)
        handoff = store.compare_and_set_borg_state(
            draft.id,
            expected_state=working.state,
            expected_version=working.state_version,
            new_state=BorgState.TECH_REVIEW_WORKING,
        )

        with pytest.raises(TechLeadError, match="deterministic validation"):
            TechLeadLoop(
                repository, handoff, store, reviewer, io=_io()
            ).run()

        assert store.get_borg(borg_id=draft.id).state is BorgState.TECH_REVIEW_WORKING
        assert reviewer.calls == []
        assert [item.phase for item in store.list_planning_attempts(draft.id)] == [
            "architect_plan"
        ]


def _current_plan(spec) -> dict:
    manifest = json.loads(
        (spec.cwd / ".betterborg/state/planning/context/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return json.loads((spec.cwd / manifest["current_plan"]).read_text(encoding="utf-8"))


def _findings(spec) -> list[dict]:
    return _published(spec, "findings.json")


def _assert_prior_finding_count(spec, expected: int) -> None:
    assert len(_findings(spec)) == expected


def _blocked_by_three_rejections(
    store,
    repository,
    borg,
    planning_plan_response,
    tech_lead_change_request_response,
):
    """Drive a Borg to BLOCKED on the default minimum and return its adapters.

    A budget of nothing is what asks for exactly the minimum's rounds and the
    block they reach, which is what this exercises.
    """
    architect = MockAdapter(name="openai").queue(
        MockResponse(payload={"decision": "ready_to_plan"})
    )
    for index in range(4):
        architect.queue(
            MockResponse(payload=planning_plan_response(summary=f"Revision {index}."))
        )
    reviewer = MockAdapter(name="openai")
    for index in range(3):
        reviewer.queue(
            MockResponse(payload=tech_lead_change_request_response(f"Fix {index}."))
        )
    handoff = ArchitectLoop(repository, borg, store, architect, io=_io()).run()
    result = TechLeadLoop(
        repository,
        handoff.borg,
        store,
        reviewer,
        architect_agent=architect,
        io=_io(),
        grant_budget=0,
    ).run()
    assert result.borg.state is BorgState.BLOCKED
    return architect, reviewer, result


def test_a_blocked_plan_reconstructs_its_progress_without_a_stranded_child(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_change_request_response,
) -> None:
    """The rejection that blocked never revises, so it declares no revision.

    A child declared for it stays pending, and a pending child refuses to let
    its parent be seeded, so re-entering the record raises instead of
    reporting what it holds.
    """
    database = committed_git_repo.parent / "tech-lead-blocked-progress.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "review-blocked-progress"
        )
        architect, reviewer, first = _blocked_by_three_rejections(
            store,
            repository,
            borg,
            planning_plan_response,
            tech_lead_change_request_response,
        )
        blocked = store.get_borg(borg.id)
        assert blocked is not None
        progress = RunProgress(stream=StringIO())

        again = TechLeadLoop(
            repository,
            blocked,
            store,
            reviewer,
            architect_agent=architect,
            io=_io(),
            progress=progress,
        ).run()

        assert again.borg.state is BorgState.BLOCKED
        assert again == first
        children = progress.stages["tech-lead"].children
        assert [child.state for child in children.values()] == [
            StageState.COMPLETED,
            StageState.COMPLETED,
        ]


def test_a_blocked_plan_stays_blocked_when_the_budget_is_raised_afterwards(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_change_request_response,
) -> None:
    """Whether a rejection blocked was settled when it completed.

    The budget bounds a run from its start. Counting the record against a
    number raised since would deny the plainly terminal record, and the caller
    would get an error naming a state rather than the result it asked for.
    """
    database = committed_git_repo.parent / "tech-lead-blocked-raised.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "review-blocked-raised"
        )
        architect, reviewer, first = _blocked_by_three_rejections(
            store,
            repository,
            borg,
            planning_plan_response,
            tech_lead_change_request_response,
        )
        blocked = store.get_borg(borg.id)
        assert blocked is not None
        reviews_before = len(reviewer.calls)

        again = TechLeadLoop(
            repository,
            blocked,
            store,
            reviewer,
            architect_agent=architect,
            io=_io(),
            review_rounds=5,
        ).run()

        assert again == first
        assert again.borg.state is BorgState.BLOCKED
        assert len(reviewer.calls) == reviews_before


def _io(answers: Iterator[str] | None = None) -> InteractiveIO:
    supplied_answers = answers or iter(())
    return InteractiveIO(
        prompt=lambda _message: next(supplied_answers, None),
        confirm=lambda _message, _default: False,
        write=lambda _message: None,
    )


_CONTEXT_DIR = Path(".betterborg/state/planning/context")


def _published(spec, name: str) -> list[dict]:
    """Read one published planning-context document from a turn's worktree."""
    return json.loads((spec.cwd / _CONTEXT_DIR / name).read_text(encoding="utf-8"))


def _review(
    decision: str,
    *,
    summary: str,
    findings: Sequence[dict] = (),
    resolved: Sequence[str] = (),
) -> dict:
    return {
        "decision": decision,
        "summary": summary,
        "findings": list(findings),
        "resolved": list(resolved),
    }


def _raised(
    message: str,
    *,
    severity: str = "major",
    repeats: str | None = None,
    suggestion: str | None = None,
) -> dict:
    item: dict = {"severity": severity, "message": message, "repeats": repeats}
    if suggestion is not None:
        item["suggestion"] = suggestion
    return item


def _reviewer(
    reviews: Sequence[Callable[[list[dict], list[dict]], dict]],
    handed: list[list[dict]],
) -> MockAdapter:
    """Build a reviewer whose rounds answer the open ledger they were given."""

    def answering(review):
        def respond(spec):
            ledger = _published(spec, "open-findings.json")
            handed.append(ledger)
            return review(ledger, _published(spec, "findings.json"))

        return respond

    reviewer = MockAdapter(name="openai")
    for review in reviews:
        reviewer.queue(MockResponse(dynamic=answering(review)))
    return reviewer


STEERING_ACTIVITY = AgentActivity(AgentActivityKind.READING, "open-findings.json")


def _steering(
    rounds: int,
    *,
    confidence: str = "high",
    activities: tuple[AgentActivity, ...] = (),
) -> MockAdapter:
    """Build a steering agent with a note for every round that can be steered.

    One per review, which is more than the loop can reach: the round a review
    leads into is steered at most once, and the last review leads into none.
    """
    agent = MockAdapter(name="openai")
    for index in range(rounds):
        agent.queue(
            MockResponse(
                payload={
                    "note": f"Steering note {index + 1}.",
                    "confidence": confidence,
                },
                activities=activities,
            )
        )
    return agent


def _architect(planning_plan_response, revisions: int) -> MockAdapter:
    architect = MockAdapter(name="openai")
    architect.queue(MockResponse(payload={"decision": "ready_to_plan"}))
    for index in range(revisions + 1):
        architect.queue(
            MockResponse(payload=planning_plan_response(summary=f"Plan {index}."))
        )
    return architect


def _run_reviews(
    repository,
    borg,
    store,
    planning_plan_response,
    reviews: Sequence[Callable[[list[dict], list[dict]], dict]],
    handed: list[list[dict]],
    *,
    review_rounds: int | None = None,
    grant_budget: int = 0,
    steering_agent: MockAdapter | None = None,
):
    """Drive one Tech Lead cycle over the reviews supplied.

    The minimum is the number of reviews and nothing is granted past them by
    default, so a rule about the ledger is read off the rounds the test wrote
    rather than off however far the loop would run.
    """
    architect = _architect(planning_plan_response, len(reviews) - 1)
    handoff = ArchitectLoop(repository, borg, store, architect, io=_io()).run()
    return TechLeadLoop(
        repository,
        handoff.borg,
        store,
        _reviewer(reviews, handed),
        architect_agent=architect,
        steering_agent=(
            _steering(len(reviews)) if steering_agent is None else steering_agent
        ),
        io=_io(),
        review_rounds=len(reviews) if review_rounds is None else review_rounds,
        grant_budget=grant_budget,
    ).run()


def _assessments(store, borg_id) -> list[tuple[int, int | None, bool | None, bool]]:
    """Read back each round's recorded verdict, snapshot and refund."""
    return [
        (row.round, row.open_findings, row.refunded, row.converging)
        for row in store.list_review_assessments(borg_id, loop="tech_review")
    ]


def _ledger(store, borg_id) -> dict[str, object]:
    return {
        row.message: row for row in store.list_planning_ledger_findings(borg_id)
    }


@pytest.mark.parametrize("budget", [-1, 1.5])
def test_a_grant_budget_that_is_not_a_whole_number_at_or_above_zero_is_refused(
    committed_git_repo: Path,
    persist_planning_context,
    budget: object,
) -> None:
    """Below zero would stop the loop short of the minimum it was told to run."""
    database = committed_git_repo.parent / "tech-lead-grant.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, f"grant-{str(budget).strip('-.')}"
        )
        with pytest.raises(TechLeadError, match="at least 0"):
            TechLeadLoop(
                repository,
                borg,
                store,
                MockAdapter(name="openai"),
                io=_io(),
                grant_budget=budget,
            )


def test_the_account_of_a_stopped_loop_is_read_off_its_record(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """A blocked plan explains itself the same way however it is reached.

    Re-running the command against it runs no loop at all, and a setting edited
    since does not move a plan that has already blocked, so the account of why
    it blocked comes from what its rounds recorded rather than from the numbers
    in force now. The rounds a loop charged as grants are what say where its
    minimum was.
    """
    handed: list[list[dict]] = []

    def repeat(ledger, _history):
        if not ledger:
            return _review(
                "request_changes",
                summary="The rollback is uncovered.",
                findings=[_raised("Cover a partial rollback.", severity="blocker")],
            )
        return _review(
            "request_changes",
            summary="The rollback is still uncovered.",
            findings=[
                _raised(
                    "Cover a partial rollback.",
                    severity="blocker",
                    repeats=ledger[0]["id"],
                )
            ],
        )

    database = committed_git_repo.parent / "grants-account.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "grants-account"
        )
        result = _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [repeat, repeat, repeat, repeat],
            handed,
            review_rounds=2,
            grant_budget=2,
        )

        assert result.borg.state is BorgState.BLOCKED
        account = tech_lead_grant_account(store, borg.id)
        assert (account.rounds, account.grants, account.charged) == (4, 2, 2)
        # Two of its four rounds were charged as grants, so the minimum those
        # rounds ran under was two, whatever the repository now configures.
        assert account.minimum == 2
        assert account.sentence() == (
            "The loop took 2 granted rounds past its minimum of 2, 2 of them "
            "closing nothing, and its last round was not converging."
        )


def test_a_repeated_objection_stays_one_row_the_loop_already_failed_to_close(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """Resolution, carry-forward and a repeat of a row already closed.

    A second row for the same objection would leave the first closed at the
    round that closed it and the repeat born in the current one, so the round
    that just heard an old objection again would read as one that fully
    drained its predecessor.
    """
    handed: list[list[dict]] = []

    def round_one(ledger, _history):
        assert ledger == []
        return _review(
            "request_changes",
            summary="The rollback is uncovered.",
            findings=[
                _raised("Cover a partial rollback.", severity="blocker"),
            ],
        )

    def round_two(ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback is covered; the checks are not.",
            resolved=[ledger[0]["id"]],
            findings=[_raised("Name the rollback checks.")],
        )

    def round_three(ledger, history):
        # The closed objection left the open ledger, and the history still
        # carries it under the id it was first recorded with.
        assert [row["message"] for row in ledger] == ["Name the rollback checks."]
        first = next(
            item for item in history if item["message"] == "Cover a partial rollback."
        )
        return _review(
            "request_changes",
            summary="The rollback regressed.",
            findings=[
                _raised("Rollback coverage is gone again.", repeats=first["id"])
            ],
        )

    database = committed_git_repo.parent / "ledger-repeat.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "ledger-repeat"
        )
        result = _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [round_one, round_two, round_three],
            handed,
        )

        assert result.borg.state is BorgState.BLOCKED
        # Each round was handed the ledger it answered, not a page of history.
        assert [
            [row["message"] for row in ledger] for ledger in handed
        ] == [
            [],
            ["Cover a partial rollback."],
            ["Name the rollback checks."],
        ]
        assert handed[1][0]["first_raised_in_round"] == 1
        assert handed[1][0]["severity"] == "blocker"

        rows = _ledger(store, borg.id)
        assert set(rows) == {"Cover a partial rollback.", "Name the rollback checks."}
        repeated = rows["Cover a partial rollback."]
        # The repeat moved the row instead of adding one, so it keeps the
        # severity and message it was first recorded with.
        assert repeated.status is FindingStatus.REGRESSED
        assert repeated.severity == "blocker"
        assert repeated.first_seen_round == 1
        assert repeated.last_seen_round == 3
        carried = rows["Name the rollback checks."]
        assert carried.status is FindingStatus.OPEN
        assert carried.first_seen_round == 2
        assert carried.last_seen_round == 3
        # The immutable record still holds one row per statement every round
        # made, under the round that made it.
        assert [
            (item.round, item.message)
            for item in store.list_planning_findings(borg.id)
        ] == [
            (1, "Cover a partial rollback."),
            (2, "Name the rollback checks."),
            (3, "Rollback coverage is gone again."),
        ]


def test_an_approval_closes_every_row_including_one_its_findings_regressed(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """A reviewer that approves has said the work is ready.

    An approval carrying minor findings is still an approval, so an objection
    its own findings raised again resolves with everything else.
    """
    handed: list[list[dict]] = []

    def round_one(ledger, _history):
        return _review(
            "request_changes",
            summary="Two things are missing.",
            findings=[_raised("Name the rollback checks.")],
        )

    def round_two(ledger, _history):
        return _review(
            "request_changes",
            summary="The checks are still missing.",
            findings=[
                _raised("The checks are still unnamed.", repeats=ledger[0]["id"]),
                _raised("Cover the failure path."),
            ],
        )

    def round_three(ledger, _history):
        repeated = next(
            row for row in ledger if row["message"] == "Name the rollback checks."
        )
        return _review(
            "approve",
            summary="The plan is ready, with one small thing left.",
            findings=[
                _raised(
                    "The check names could be tidier.",
                    severity="minor",
                    repeats=repeated["id"],
                )
            ],
        )

    database = committed_git_repo.parent / "ledger-approval.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "ledger-approval"
        )
        result = _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [round_one, round_two, round_three],
            handed,
        )

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        rows = _ledger(store, borg.id)
        assert set(rows) == {"Name the rollback checks.", "Cover the failure path."}
        assert all(
            row.status is FindingStatus.RESOLVED and row.last_seen_round == 3
            for row in rows.values()
        )
        # The regressed row kept what it was first recorded with throughout.
        assert rows["Name the rollback checks."].severity == "major"
        assert rows["Name the rollback checks."].first_seen_round == 1
        assert open_planning_findings(store, borg.id) == []


def test_a_review_closes_a_row_an_earlier_round_regressed(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """An objection raised again is still one a later round can close.

    A regressed row that only an approval could close would put a floor under
    the open count for the rest of the cycle, and leave the plan showing an
    objection its reviewer said the revision answered.
    """
    handed: list[list[dict]] = []

    def round_one(ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback is uncovered.",
            findings=[_raised("Cover a partial rollback.", severity="blocker")],
        )

    def round_two(ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback is uncovered again.",
            findings=[
                _raised(
                    "Rollback coverage is gone again.",
                    severity="blocker",
                    repeats=ledger[0]["id"],
                )
            ],
        )

    def round_three(ledger, _history):
        regressed = next(
            row for row in ledger if row["message"] == "Cover a partial rollback."
        )
        # The round a reviewer is told the objection started in is the round
        # that raised it, not the round that last re-established its status.
        assert regressed["first_raised_in_round"] == 1
        return _review(
            "request_changes",
            summary="The rollback is covered; the checks are not.",
            resolved=[regressed["id"]],
            findings=[_raised("Name the rollback checks.")],
        )

    database = committed_git_repo.parent / "ledger-regressed-close.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "ledger-regressed-close"
        )
        result = _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [round_one, round_two, round_three],
            handed,
        )

        assert result.borg.state is BorgState.BLOCKED
        rows = _ledger(store, borg.id)
        closed = rows["Cover a partial rollback."]
        assert closed.status is FindingStatus.RESOLVED
        assert closed.first_seen_round == 1
        assert closed.last_seen_round == 3
        assert closed.severity == "blocker"
        # What still stands is the one objection nobody has answered yet.
        assert [row.message for row in open_planning_findings(store, borg.id)] == [
            "Name the rollback checks."
        ]


def test_an_approval_closes_the_row_its_own_new_finding_opened(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """Approval closes the ledger it leaves behind, not the one it inherited.

    An approval may carry minor findings of its own, and a row born in the
    approving round is one of the rows that round has to close: a plan waiting
    for a human shows no objections at all.
    """
    handed: list[list[dict]] = []

    def round_one(ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback checks are unnamed.",
            findings=[_raised("Name the rollback checks.")],
        )

    def round_two(ledger, _history):
        return _review(
            "approve",
            summary="The plan is ready, with one small thing left.",
            resolved=[ledger[0]["id"]],
            findings=[_raised("Tidy the wording.", severity="minor")],
        )

    database = committed_git_repo.parent / "ledger-approval-new.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "ledger-approval-new"
        )
        result = _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [round_one, round_two],
            handed,
        )

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        rows = _ledger(store, borg.id)
        assert set(rows) == {"Name the rollback checks.", "Tidy the wording."}
        assert all(
            row.status is FindingStatus.RESOLVED and row.last_seen_round == 2
            for row in rows.values()
        )
        assert rows["Tidy the wording."].first_seen_round == 2
        assert open_planning_findings(store, borg.id) == []


def test_a_closed_id_the_ledger_cannot_place_leaves_its_row_open(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """An unrecognised id costs the loop a round, never the objection."""
    handed: list[list[dict]] = []

    def round_one(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback is uncovered.",
            findings=[_raised("Cover a partial rollback.", severity="blocker")],
        )

    def round_two(_ledger, _history):
        return _review(
            "request_changes",
            summary="Something else is missing.",
            resolved=["not-an-identifier", str(uuid4())],
            findings=[_raised("Name the rollback checks.")],
        )

    database = committed_git_repo.parent / "ledger-unknown-resolved.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "ledger-unknown-resolved"
        )
        _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [round_one, round_two],
            handed,
        )

        rows = _ledger(store, borg.id)
        assert rows["Cover a partial rollback."].status is FindingStatus.OPEN
        assert rows["Cover a partial rollback."].last_seen_round == 2


def test_a_repeat_the_ledger_cannot_place_still_records_a_regression(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """The pointer moves the right row; the claim stands without it.

    Recorded as a plain new row instead, a blocker the loop has already failed
    to close would read to the assessment as discovery on new surface.
    """
    handed: list[list[dict]] = []

    def round_one(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback is uncovered.",
            findings=[_raised("Cover a partial rollback.")],
        )

    def round_two(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback is uncovered again.",
            findings=[
                _raised(
                    "Rollback coverage is gone again.",
                    severity="blocker",
                    repeats=str(uuid4()),
                )
            ],
        )

    database = committed_git_repo.parent / "ledger-unknown-repeat.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "ledger-unknown-repeat"
        )
        _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [round_one, round_two],
            handed,
        )

        claimed = _ledger(store, borg.id)["Rollback coverage is gone again."]
        assert claimed.status is FindingStatus.REGRESSED
        assert claimed.first_seen_round == 2


def test_a_review_that_resolves_and_repeats_one_id_leaves_it_regressed(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """Only one of the two readings costs the loop nothing if it is wrong."""
    handed: list[list[dict]] = []

    def round_one(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback is uncovered.",
            findings=[_raised("Cover a partial rollback.")],
        )

    def round_two(ledger, _history):
        return _review(
            "request_changes",
            summary="Closed, and still there.",
            resolved=[ledger[0]["id"]],
            findings=[
                _raised("Rollback coverage is still missing.", repeats=ledger[0]["id"])
            ],
        )

    database = committed_git_repo.parent / "ledger-contradiction.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "ledger-contradiction"
        )
        _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [round_one, round_two],
            handed,
        )

        rows = store.list_planning_ledger_findings(borg.id)
        assert len(rows) == 1
        assert rows[0].status is FindingStatus.REGRESSED
        assert rows[0].message == "Cover a partial rollback."


def test_the_reconciliation_is_written_with_the_attempt_that_produced_it(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """Every row this round touched names the review that touched it.

    The immutable findings are untouched by any of it: they are the record of
    what each round said, and the ledger is the current view of what stands.
    """
    handed: list[list[dict]] = []

    def round_one(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback is uncovered.",
            findings=[_raised("Cover a partial rollback.", severity="blocker")],
        )

    def round_two(_ledger, _history):
        return _review(
            "request_changes",
            summary="The checks are missing too.",
            findings=[_raised("Name the rollback checks.")],
        )

    database = committed_git_repo.parent / "ledger-attempt.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "ledger-attempt"
        )
        _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [round_one, round_two],
            handed,
        )

        reviews = [
            item
            for item in store.list_planning_attempts(borg.id)
            if item.phase == "tech_review"
        ]
        assert len(reviews) == 2
        rows = store.list_planning_ledger_findings(borg.id)
        assert {row.attempt_id for row in rows} == {reviews[1].id}

        snapshots = store.list_planning_findings(borg.id)
        assert [(item.round, item.severity, item.message) for item in snapshots] == [
            (1, "blocker", "Cover a partial rollback."),
            (2, "major", "Name the rollback checks."),
        ]
        assert [item.attempt_id for item in snapshots] == [
            reviews[0].id,
            reviews[1].id,
        ]


def test_a_second_planning_cycle_inherits_none_of_the_first_cycle_ledger(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """Each cycle's ledger holds its own rows and answers for them alone.

    A cycle can only follow one that ended in an approval, and an approval
    resolves everything, so what an unscoped ledger would hand the new cycle is
    a block of resolved rows landing on its restarted round numbers: a cycle
    that raised one objection and repeated it would read as strictly draining
    on the strength of resolutions it inherited. The scope is a column, so a
    row of the closed cycle is left exactly as that cycle left it whatever its
    status was — which is what the seeded open row below reads, since no
    reachable run leaves one open in a cycle an approval closed.
    """
    handed: list[list[dict]] = []

    def reject(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback is uncovered.",
            findings=[_raised("Cover a partial rollback.")],
        )

    def approve(_ledger, _history):
        return _review("approve", summary="The plan is ready.")

    def second_cycle(_ledger, _history):
        return _review(
            "request_changes",
            summary="The staging is unspecified.",
            findings=[_raised("Stage the rollout explicitly.")],
        )

    database = committed_git_repo.parent / "ledger-cycles.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "ledger-cycles"
        )
        approved = _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [reject, approve],
            handed,
        )
        assert approved.borg.state is BorgState.PLAN_APPROVAL_PENDING

        stranded = PlanningLedgerFinding(
            borg_id=borg.id,
            cycle_id=INITIAL_PLANNING_CYCLE,
            attempt_id=approved.attempt.id,
            first_seen_round=1,
            last_seen_round=1,
            severity="blocker",
            message="An objection the closed cycle never answered.",
        )
        store.record_planning_ledger_findings([stranded])

        request = PlanChangeRequest(
            borg_id=borg.id, round=1, note="Stage the rollout."
        )
        with store.transaction():
            store.append_plan_change_request(request)
            changed = store.compare_and_set_borg_state(
                borg.id,
                expected_state=approved.borg.state,
                expected_version=approved.borg.state_version,
                new_state=BorgState.ARCHITECT_WORKING,
            )

        blocked = _run_reviews(
            repository,
            changed,
            store,
            planning_plan_response,
            [second_cycle],
            handed,
        )

        assert blocked.borg.state is BorgState.BLOCKED
        # The new cycle's first round starts from nothing, and the resolutions
        # that closed the first cycle stay in the first cycle.
        assert handed[-1] == []
        every_row = store.list_planning_ledger_findings(borg.id)
        assert len(every_row) == 3
        # The approval that let the cycle be changed credited its resolution
        # to the round that established it, and the row left open in that cycle
        # is where that cycle left it: the new cycle neither answered for it nor
        # carried it forward onto a round of its own.
        first_cycle = store.list_planning_ledger_findings(
            borg.id, cycle_id=INITIAL_PLANNING_CYCLE
        )
        assert [
            (row.status, row.first_seen_round, row.last_seen_round)
            for row in first_cycle
        ] == [
            (FindingStatus.RESOLVED, 1, 2),
            (FindingStatus.OPEN, 1, 1),
        ]
        assert first_cycle[1].attempt_id == approved.attempt.id
        current = store.list_planning_ledger_findings(
            borg.id, cycle_id=str(request.id)
        )
        assert [row.message for row in current] == ["Stage the rollout explicitly."]
        assert [
            row.message for row in open_planning_findings(store, borg.id)
        ] == ["Stage the rollout explicitly."]


@pytest.mark.parametrize(
    ("mutate", "missing"),
    [
        pytest.param(
            lambda payload: payload.pop("resolved"), "resolved", id="resolved"
        ),
        pytest.param(
            lambda payload: payload["findings"][0].pop("repeats"),
            "repeats",
            id="repeats",
        ),
    ],
)
def test_a_review_that_omits_a_ledger_declaration_fails_its_schema(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    tech_lead_change_request_response,
    mutate,
    missing: str,
) -> None:
    """Requiring both is what forces a reviewer to answer rather than omit."""
    payload = tech_lead_change_request_response("Cover a partial rollback.")
    mutate(payload)
    architect = _architect(planning_plan_response, 0)
    reviewer = MockAdapter(name="openai").queue(MockResponse(payload=payload))
    database = committed_git_repo.parent / f"ledger-schema-{missing}.sqlite3"

    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, f"ledger-schema-{missing}"
        )
        handoff = ArchitectLoop(repository, borg, store, architect, io=_io()).run()
        expected = f"missing required property '{missing}'"
        with pytest.raises(TechLeadError, match=expected):
            TechLeadLoop(
                repository,
                handoff.borg,
                store,
                reviewer,
                architect_agent=architect,
                io=_io(),
            ).run()

        assert store.list_planning_ledger_findings(borg.id) == []


def test_a_draining_loop_runs_past_its_minimum_and_finishes_on_approval(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """Every granted round that closed something is refunded, so none is spent.

    The loop terminates anyway: a refund needs a count that keeps falling, a
    review that continues always reports at least one finding, and a strictly
    decreasing sequence above a floor of one cannot run forever. It reaches
    agreement instead.
    """
    handed: list[list[dict]] = []

    def round_one(ledger, _history):
        assert ledger == []
        return _review(
            "request_changes",
            summary="Four things stand in the way.",
            findings=[_raised(f"Objection {index}.") for index in range(4)],
        )

    def round_two(ledger, _history):
        return _review(
            "request_changes",
            summary="Two are answered.",
            resolved=[row["id"] for row in ledger[:2]],
            findings=[_raised("Objection 4.")],
        )

    def round_three(ledger, _history):
        return _review(
            "request_changes",
            summary="Two more are answered.",
            resolved=[row["id"] for row in ledger[:2]],
            findings=[_raised("Objection 5.")],
        )

    def round_four(ledger, _history):
        return _review(
            "approve",
            summary="The plan is ready.",
            resolved=[row["id"] for row in ledger],
        )

    database = committed_git_repo.parent / "grants-draining.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "grants-draining"
        )
        result = _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [round_one, round_two, round_three, round_four],
            handed,
            review_rounds=1,
            grant_budget=10,
        )

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        assert len(handed) == 4
        # Round one is the minimum and carries no refund decision; the three
        # that follow are grants, and each of them closed more than it raised.
        assert _assessments(store, borg.id) == [
            (1, 4, None, False),
            (2, 3, True, True),
            (3, 2, True, True),
            (4, 0, True, True),
        ]


def test_a_loop_repeating_one_blocker_stops_at_the_minimum_plus_the_budget(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """A grant that closed nothing is charged, and the charges are capped."""
    handed: list[list[dict]] = []

    def raise_it(ledger, _history):
        assert ledger == []
        return _review(
            "request_changes",
            summary="The rollback is uncovered.",
            findings=[_raised("Cover a partial rollback.", severity="blocker")],
        )

    def repeat_it(ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback is still uncovered.",
            findings=[
                _raised(
                    "Cover a partial rollback.",
                    severity="blocker",
                    repeats=ledger[0]["id"],
                )
            ],
        )

    database = committed_git_repo.parent / "grants-repeating.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "grants-repeating"
        )
        result = _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [raise_it, repeat_it, repeat_it],
            handed,
            review_rounds=1,
            grant_budget=2,
        )

        assert result.borg.state is BorgState.BLOCKED
        assert _assessments(store, borg.id) == [
            (1, 1, None, False),
            (2, 1, False, False),
            (3, 1, False, False),
        ]
        # One objection is one row for as long as the loop argues about it.
        assert [row.status for row in store.list_planning_ledger_findings(borg.id)] == [
            FindingStatus.REGRESSED
        ]


def test_a_loop_alternating_two_objections_earns_no_refund_either(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """Closing one objection while reopening another leaves the count where it was."""
    handed: list[list[dict]] = []

    def raise_first(ledger, _history):
        assert ledger == []
        return _review(
            "request_changes",
            summary="The rollback is uncovered.",
            findings=[_raised("Cover a partial rollback.")],
        )

    def swap(ledger, _history):
        return _review(
            "request_changes",
            summary="One answered, one raised.",
            resolved=[ledger[0]["id"]],
            findings=[_raised("Name the rollback checks.")],
        )

    def swap_back(ledger, history):
        first = next(
            item for item in history if item["message"] == "Cover a partial rollback."
        )
        return _review(
            "request_changes",
            summary="The first objection is back.",
            resolved=[ledger[0]["id"]],
            findings=[
                _raised("Cover a partial rollback.", repeats=str(first["id"]))
            ],
        )

    database = committed_git_repo.parent / "grants-alternating.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "grants-alternating"
        )
        result = _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [raise_first, swap, swap_back],
            handed,
            review_rounds=1,
            grant_budget=2,
        )

        assert result.borg.state is BorgState.BLOCKED
        assert _assessments(store, borg.id) == [
            (1, 1, None, False),
            (2, 1, False, False),
            (3, 1, False, False),
        ]


def test_each_round_keeps_the_verdict_it_was_judged_by(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """A loop that converges and then stalls keeps both answers on the record.

    A later round's verdict pasted over the rounds before it would lose the
    one Stage 6 reads a round later, and the refund earned on the way would be
    unaccountable.
    """
    handed: list[list[dict]] = []

    def round_one(ledger, _history):
        assert ledger == []
        return _review(
            "request_changes",
            summary="Four things stand in the way.",
            findings=[_raised(f"Objection {index}.") for index in range(4)],
        )

    def round_two(ledger, _history):
        return _review(
            "request_changes",
            summary="Two are answered.",
            resolved=[row["id"] for row in ledger[:2]],
            findings=[_raised("Objection 4.")],
        )

    def round_three(_ledger, _history):
        return _review(
            "request_changes",
            summary="Nothing is answered and something else is wrong.",
            findings=[_raised("Objection 5.", severity="blocker")],
        )

    database = committed_git_repo.parent / "grants-verdicts.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "grants-verdicts"
        )
        result = _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [round_one, round_two, round_three],
            handed,
            review_rounds=1,
            grant_budget=1,
        )

        assert result.borg.state is BorgState.BLOCKED
        assert _assessments(store, borg.id) == [
            (1, 4, None, False),
            (2, 3, True, True),
            (3, 4, False, False),
        ]


def test_a_resumed_run_honours_the_refund_the_round_before_it_earned(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """Grants are read off the record, because a run is rebuilt on every entry.

    A resumed loop that assessed a round it had already granted would spend
    the budget twice and lose the bound the budget promises. Here the refunded
    round is what buys the fourth: a loop that forgot it would have stopped at
    the third.
    """
    handed: list[list[dict]] = []

    def round_one(ledger, _history):
        assert ledger == []
        return _review(
            "request_changes",
            summary="Three things stand in the way.",
            findings=[_raised(f"Objection {index}.") for index in range(3)],
        )

    def closing(ledger, _history):
        return _review(
            "request_changes",
            summary="Two are answered, one is new.",
            resolved=[row["id"] for row in ledger[:2]],
            findings=[_raised("Objection 3.")],
        )

    def standing_still(_ledger, _history):
        return _review(
            "request_changes",
            summary="Nothing moved.",
            findings=[_raised("Objection 4.")],
        )

    database = committed_git_repo.parent / "grants-resumed.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "grants-resumed"
        )
        architect = _architect(planning_plan_response, 1)
        reviewer = _reviewer([round_one, closing], handed)
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()

        # The second review asks for a revision, and the run dies before the
        # Architect can answer it.
        with pytest.raises((TechLeadError, ArchitectError)):
            TechLeadLoop(
                repository,
                handoff.borg,
                store,
                reviewer,
                architect_agent=architect,
                steering_agent=_steering(4),
                io=_io(),
                review_rounds=1,
                grant_budget=2,
            ).run()
        interrupted = store.get_borg(borg.id)
        assert interrupted is not None
        assert interrupted.state is BorgState.ARCHITECT_WORKING
        assert _assessments(store, borg.id) == [
            (1, 3, None, False),
            (2, 2, True, True),
        ]

        for _ in range(2):
            architect.queue(
                MockResponse(payload=planning_plan_response(summary="Resumed."))
            )
        resumed = TechLeadLoop(
            repository,
            interrupted,
            store,
            _reviewer([standing_still, standing_still], handed),
            architect_agent=architect,
            steering_agent=_steering(4),
            io=_io(),
            review_rounds=1,
            grant_budget=2,
        ).run()

        assert resumed.borg.state is BorgState.BLOCKED
        # Four rounds, not five: the granted round the interrupted run
        # assessed is counted once, and its refund is still on the record.
        assert _assessments(store, borg.id) == [
            (1, 3, None, False),
            (2, 2, True, True),
            (3, 3, False, False),
            (4, 4, False, False),
        ]


def test_a_budget_of_nothing_stops_a_draining_loop_where_its_counter_did(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """Zero asks for exactly today's rounds and today's terminal state.

    The loop is closing findings and the verdict says so, and it blocks all
    the same, because the first round past the minimum has no budget to come
    out of.
    """
    handed: list[list[dict]] = []

    def round_one(ledger, _history):
        assert ledger == []
        return _review(
            "request_changes",
            summary="Three things stand in the way.",
            findings=[_raised(f"Objection {index}.") for index in range(3)],
        )

    def draining(ledger, _history):
        return _review(
            "request_changes",
            summary="Two are answered, one is new.",
            resolved=[row["id"] for row in ledger[:2]],
            findings=[_raised("Objection 3.")],
        )

    database = committed_git_repo.parent / "grants-zero-budget.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "grants-zero-budget"
        )
        result = _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [round_one, draining],
            handed,
            review_rounds=2,
            grant_budget=0,
        )

        assert result.borg.state is BorgState.BLOCKED
        assert _assessments(store, borg.id) == [
            (1, 3, None, False),
            (2, 2, None, True),
        ]


def test_the_reviewer_is_told_its_round_and_nothing_about_the_budget(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """No loop knows whether a round is its last, so the sentence never says.

    A remaining-grants count would be no better than the total it replaced:
    any budget number in a reviewer's prompt reads as a deadline it is being
    held to.
    """
    handed: list[list[dict]] = []

    def asking(_ledger, _history):
        return _review(
            "request_changes",
            summary="Something is wrong.",
            findings=[_raised("Name the rollback checks.")],
        )

    def approving(ledger, _history):
        return _review(
            "approve",
            summary="The plan is ready.",
            resolved=[row["id"] for row in ledger],
        )

    database = committed_git_repo.parent / "grants-prompt.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "grants-prompt"
        )
        architect = _architect(planning_plan_response, 2)
        reviewer = _reviewer([asking, asking, approving], handed)
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        result = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            reviewer,
            architect_agent=architect,
            steering_agent=_steering(3),
            io=_io(),
            review_rounds=1,
            grant_budget=7,
        ).run()

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        prompts = [call.user_prompt for call in reviewer.calls]
        assert [
            f"This is Tech Lead review round {round_number}." in prompt
            for round_number, prompt in enumerate(prompts, start=1)
        ] == [True, True, True]
        for prompt in prompts:
            assert "final" not in prompt
            assert " of " not in prompt
            assert "7" not in prompt


@pytest.mark.parametrize("review_rounds", [1, 2])
def test_a_contract_slip_ends_a_granted_round_as_it_ends_one_inside_the_minimum(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    review_rounds: int,
) -> None:
    """The Tech Lead's own contract has no allowance, and this plan adds none.

    A longer loop reaches that bound more often, and a round it reaches on a
    grant ends the run needing a person exactly as one inside the minimum
    does, rather than blocking with its findings kept.
    """
    handed: list[list[dict]] = []

    def asking(_ledger, _history):
        return _review(
            "request_changes",
            summary="Something is wrong.",
            findings=[_raised("Name the rollback checks.")],
        )

    def silent(_ledger, _history):
        return _review("request_changes", summary="Something is still wrong.")

    database = committed_git_repo.parent / f"grants-slip-{review_rounds}.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, f"grants-slip-{review_rounds}"
        )
        architect = _architect(planning_plan_response, 1)
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()

        with pytest.raises(TechLeadError, match="must include findings"):
            TechLeadLoop(
                repository,
                handoff.borg,
                store,
                _reviewer([asking, silent], handed),
                architect_agent=architect,
                steering_agent=_steering(2),
                io=_io(),
                review_rounds=review_rounds,
                grant_budget=5,
            ).run()

        current = store.get_borg(borg.id)
        assert current is not None
        assert current.state is BorgState.TECH_REVIEW_WORKING
        # The failed round was never assessed, so it earned no refund.
        assert [row.round for row in store.list_review_assessments(borg.id)] == [1]


@pytest.mark.parametrize("review_rounds", [1, 2])
def test_the_architect_answer_budget_ends_a_granted_run_the_same_way(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    review_rounds: int,
) -> None:
    """The Architect may assume answers three times in a planning cycle.

    The allowance was sized when a loop could not exceed its configured
    rounds. Nothing here moves it, so a revision that raises a fourth round of
    questions ends the run rather than blocking, whether the revision was
    granted or inside the minimum.
    """
    handed: list[list[dict]] = []
    architect = MockAdapter(name="openai")
    for index in range(ARCHITECT_QUESTION_ROUND_CAP):
        architect.queue(
            MockResponse(
                payload={
                    "decision": "ask_more",
                    "questions": [
                        {"id": "q1", "question": f"Question {index + 1}?"}
                    ],
                }
            )
        )
        architect.queue(
            MockResponse(
                payload={
                    "answers": [
                        {"q_id": "q1", "answer": f"Assumption {index + 1}."}
                    ]
                }
            )
        )
    architect.queue(
        MockResponse(
            payload={
                **planning_plan_response(),
                # Named, so the plan is not sent back once to account for the
                # answers it assumed, which would spend the revision response.
                "assumptions": [
                    {
                        "question": f"Question {index + 1}?",
                        "assumption": f"Assumption {index + 1}.",
                    }
                    for index in range(ARCHITECT_QUESTION_ROUND_CAP)
                ],
            }
        )
    )
    architect.queue(
        MockResponse(
            payload={
                **planning_plan_response(summary="Revised."),
                "open_questions": ["Which release channel is the default?"],
            }
        )
    )

    def asking(_ledger, _history):
        return _review(
            "request_changes",
            summary="Something is wrong.",
            findings=[_raised("Name the rollback checks.")],
        )

    database = committed_git_repo.parent / f"grants-answers-{review_rounds}.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, f"grants-answers-{review_rounds}"
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io(), unattended=True
        ).run()

        with pytest.raises(ArchitectError, match="asked past question round"):
            TechLeadLoop(
                repository,
                handoff.borg,
                store,
                _reviewer([asking], handed),
                architect_agent=architect,
                io=_io(),
                unattended=True,
                review_rounds=review_rounds,
                grant_budget=5,
            ).run()

        current = store.get_borg(borg.id)
        assert current is not None
        assert current.state is BorgState.ARCHITECT_AWAITING_ANSWERS


@pytest.mark.parametrize(
    ("declaration", "converging"),
    [("declared", False), ("undeclared", True)],
)
def test_an_undeclared_repeat_of_a_closed_blocker_reads_as_fresh_discovery(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    declaration: str,
    converging: bool,
) -> None:
    """The declaration is what lets the veto see a loop that is stuck.

    Recorded as a plain new row, an objection the loop already failed to close
    reads as new discovery on new surface, which is the one shape that looks
    like convergence while the loop is going nowhere. So the round the omission
    bought runs unsteered, where a declared repeat would have steered it.
    """

    def round_one(ledger, _history):
        assert ledger == []
        return _review(
            "request_changes",
            summary="The rollback is uncovered.",
            findings=[_raised("Cover a partial rollback.", severity="blocker")],
        )

    def round_two(ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback is covered; the checks are not.",
            resolved=[ledger[0]["id"]],
            findings=[_raised("Name the rollback checks.")],
        )

    def declared(ledger, history):
        first = next(
            item for item in history if item["message"] == "Cover a partial rollback."
        )
        return _review(
            "request_changes",
            summary="The rollback is uncovered again.",
            resolved=[ledger[0]["id"]],
            findings=[
                _raised(
                    "Cover a partial rollback.",
                    severity="blocker",
                    repeats=str(first["id"]),
                )
            ],
        )

    def undeclared(ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback is uncovered again.",
            resolved=[ledger[0]["id"]],
            findings=[_raised("Cover a partial rollback.", severity="blocker")],
        )

    handed: list[list[dict]] = []
    third = declared if declaration == "declared" else undeclared
    database = committed_git_repo.parent / f"grants-{declaration}-repeat.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, f"grants-{declaration}-repeat"
        )
        result = _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [round_one, round_two, third],
            handed,
            review_rounds=1,
            grant_budget=2,
        )

        assert result.borg.state is BorgState.BLOCKED
        assert _assessments(store, borg.id)[-1] == (3, 1, False, converging)


@pytest.mark.parametrize(
    ("declaration", "state", "rounds"),
    [
        (
            "declared",
            BorgState.PLAN_APPROVAL_PENDING,
            [(1, 2, None, False), (2, 1, True, True), (3, 0, True, True)],
        ),
        (
            "undeclared",
            BorgState.BLOCKED,
            [(1, 2, None, False), (2, 2, False, False)],
        ),
    ],
)
def test_an_undeclared_repeat_costs_the_grant_a_declared_one_earns_back(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    declaration: str,
    state: BorgState,
    rounds: list[tuple[int, int | None, bool | None, bool]],
) -> None:
    """The omission adds a second row where the declaration moves the first.

    So the open count does not fall, the grant is charged, and the loop stops
    a round earlier than the one that said what it was repeating.
    """

    def round_one(ledger, _history):
        assert ledger == []
        return _review(
            "request_changes",
            summary="Two things stand in the way.",
            findings=[
                _raised("Cover a partial rollback."),
                _raised("Name the rollback checks."),
            ],
        )

    def declared(ledger, _history):
        rollback = next(
            row for row in ledger if row["message"] == "Cover a partial rollback."
        )
        checks = next(
            row for row in ledger if row["message"] == "Name the rollback checks."
        )
        return _review(
            "request_changes",
            summary="The checks are named; the rollback is not covered.",
            resolved=[checks["id"]],
            findings=[
                _raised("Cover a partial rollback.", repeats=rollback["id"])
            ],
        )

    def undeclared(ledger, _history):
        checks = next(
            row for row in ledger if row["message"] == "Name the rollback checks."
        )
        return _review(
            "request_changes",
            summary="The checks are named; the rollback is not covered.",
            resolved=[checks["id"]],
            findings=[_raised("Cover a partial rollback.")],
        )

    def approving(ledger, _history):
        return _review(
            "approve",
            summary="The plan is ready.",
            resolved=[row["id"] for row in ledger],
        )

    handed: list[list[dict]] = []
    second = declared if declaration == "declared" else undeclared
    database = committed_git_repo.parent / f"grants-{declaration}-cost.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, f"grants-{declaration}-cost"
        )
        result = _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [round_one, second, approving],
            handed,
            review_rounds=1,
            grant_budget=1,
        )

        assert result.borg.state is state
        assert _assessments(store, borg.id) == rounds


def test_the_recorded_snapshot_outlives_a_regression_that_moves_the_drain(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """A refund compares two recorded snapshots, never two rows of a drain.

    The drain is recomputed from current lifecycle state every time it runs, so
    an objection that regresses raises the counts of the rounds it was open
    through after the fact. A reader built on those rows would reopen a
    decision the round they belong to already made, and deny a refund the
    round it belonged to earned.
    """
    handed: list[list[dict]] = []

    def round_one(ledger, _history):
        assert ledger == []
        return _review(
            "request_changes",
            summary="Three things stand in the way.",
            findings=[
                _raised("Cover a partial rollback.", severity="blocker"),
                _raised("Name the rollback checks."),
                _raised("Document the release channel."),
            ],
        )

    def round_two(ledger, _history):
        closed = [
            row["id"]
            for row in ledger
            if row["message"]
            in {"Cover a partial rollback.", "Name the rollback checks."}
        ]
        return _review(
            "request_changes",
            summary="The rollback is covered and the checks are named.",
            resolved=closed,
            findings=[_raised("Name the release owner.")],
        )

    def round_three(_ledger, history):
        first = next(
            item for item in history if item["message"] == "Cover a partial rollback."
        )
        return _review(
            "request_changes",
            summary="The rollback regressed.",
            findings=[
                _raised(
                    "Cover a partial rollback.",
                    severity="blocker",
                    repeats=str(first["id"]),
                )
            ],
        )

    database = committed_git_repo.parent / "grants-drain-divergence.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "grants-drain-divergence"
        )
        result = _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [round_one, round_two, round_three],
            handed,
            review_rounds=1,
            grant_budget=1,
        )

        assert result.borg.state is BorgState.BLOCKED
        rows = store.list_review_assessments(borg.id, loop="tech_review")
        # Round two closed two of the three objections and raised one, which
        # is the refund it earned and keeps.
        assert [
            (row.round, row.open_findings, row.refunded) for row in rows
        ] == [(1, 3, None), (2, 2, True), (3, 3, False)]

        # The same round, recomputed after the regression, counts the objection
        # as open throughout: a reader comparing these rows would read round
        # two as having closed nothing and deny it the refund it earned.
        recomputed = {
            entry["round"]: entry["open_after"] for entry in rows[-1].evidence["drain"]
        }
        assert recomputed == {1: 3, 2: 3, 3: 3}
        assert recomputed[2] > rows[1].open_findings


def _blocked(message: str) -> dict:
    return _raised(message, severity="blocker")


def _approving(ledger, _history):
    return _review(
        "approve",
        summary="The plan is ready.",
        resolved=[row["id"] for row in ledger],
    )


def _steering_rows(store, borg_id) -> list[tuple]:
    """Read back every column a steered round records for itself."""
    return [
        (row.round, row.source, row.converging, row.note, row.cycle_id)
        for row in store.list_steering_notes(borg_id)
    ]


def _revision_prompts(architect: MockAdapter) -> list[str]:
    """Return the prompts of the Architect turns that wrote a plan."""
    return [
        call.user_prompt
        for call in architect.calls
        if "emit the implementation plan" in call.user_prompt
    ]


def test_a_granted_round_the_review_read_as_stuck_steers_the_architect(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """The round that decides is not the round that is steered.

    Round one is the minimum, so the revision it asks for is the first grant,
    and its verdict is what decides whether that grant is steered.
    """
    handed: list[list[dict]] = []

    def stuck(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback checks are still unnamed.",
            findings=[_blocked("Name the rollback checks.")],
        )

    database = committed_git_repo.parent / "steering-stuck.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "steering-stuck"
        )
        architect = _architect(planning_plan_response, 1)
        steering = _steering(1)
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        result = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            _reviewer([stuck, _approving], handed),
            architect_agent=architect,
            steering_agent=steering,
            io=_io(),
            review_rounds=1,
            grant_budget=5,
        ).run()

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        assert len(steering.calls) == 1
        assert (
            "Steering note for this revision:\n\nSteering note 1."
            in _revision_prompts(architect)[-1]
        )
        # Keyed to the round whose assessment asked for it, in the ledger's
        # numbering, and carrying the verdict and the note that was used.
        assert _steering_rows(store, borg.id) == [
            (1, "agent", False, "Steering note 1.", INITIAL_PLANNING_CYCLE)
        ]
        # And naming the turn that wrote it, which is what tells a note an
        # agent produced from one no attempt stands behind.
        assert [
            item.id
            for item in store.list_planning_attempts(borg.id)
            if item.phase == "steering"
        ] == [row.attempt_id for row in store.list_steering_notes(borg.id)]


def test_a_granted_round_the_review_read_as_closing_in_is_not_steered(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """Draining decides how a granted round is spent, not whether it happens.

    A converging loop already has the findings it has to answer in front of
    the agent answering them, so its revision gets the prompt it would have
    had.
    """
    handed: list[list[dict]] = []

    def opening(_ledger, _history):
        return _review(
            "request_changes",
            summary="Three things stand in the way.",
            findings=[_raised(f"Objection {index}.") for index in range(3)],
        )

    def closing(ledger, _history):
        return _review(
            "request_changes",
            summary="Two are answered.",
            resolved=[row["id"] for row in ledger[:2]],
            findings=[_raised("Objection 3.")],
        )

    database = committed_git_repo.parent / "steering-closing.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "steering-closing"
        )
        architect = _architect(planning_plan_response, 2)
        steering = _steering(2)
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        result = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            _reviewer([opening, closing, _approving], handed),
            architect_agent=architect,
            steering_agent=steering,
            io=_io(),
            review_rounds=2,
            grant_budget=5,
        ).run()

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        assert [
            row.converging
            for row in store.list_review_assessments(
                borg.id, loop="tech_review"
            )
        ] == [False, True, True]
        assert steering.calls == []
        assert store.list_steering_notes(borg.id) == []
        assert not any(
            "Steering note" in prompt for prompt in _revision_prompts(architect)
        )


def test_the_first_steered_round_is_the_one_the_minimum_ends_on(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """A round inside the minimum is not a grant, however stuck it reads.

    A refund records that a round was itself a grant, which is one round later
    than this test, so the refund is not what triggers a note.
    """
    handed: list[list[dict]] = []

    def stuck(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback checks are still unnamed.",
            findings=[_blocked("Name the rollback checks.")],
        )

    database = committed_git_repo.parent / "steering-minimum.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "steering-minimum"
        )
        architect = _architect(planning_plan_response, 3)
        steering = _steering(3)
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        result = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            _reviewer([stuck, stuck, stuck, _approving], handed),
            architect_agent=architect,
            steering_agent=steering,
            io=_io(),
            review_rounds=3,
            grant_budget=5,
        ).run()

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        # Every round read as stuck, and only the round the minimum ends on
        # led into a grant.
        assert [
            row.converging
            for row in store.list_review_assessments(
                borg.id, loop="tech_review"
            )
        ] == [
            False,
            False,
            False,
            True,
        ]
        assert len(steering.calls) == 1
        assert [row.round for row in store.list_steering_notes(borg.id)] == [3]
        assert [
            "Steering note" in prompt for prompt in _revision_prompts(architect)
        ] == [False, False, False, True]


def test_the_assembled_note_names_open_objections_and_how_long_each_stood() -> None:
    """The fallback is the second place the open-objections rule is applied.

    A note that lists what the reviewer already closed, or that says every
    objection is one round old, is the misdirection the confidence bar routes
    away from — arriving by the path it routes to.
    """
    raised_early = PlanningLedgerFinding(
        borg_id=uuid4(),
        cycle_id=INITIAL_PLANNING_CYCLE,
        attempt_id=uuid4(),
        first_seen_round=1,
        last_seen_round=4,
        severity="blocker",
        message="Name the rollback checks.",
    )
    raised_late = replace(
        raised_early,
        id=uuid4(),
        first_seen_round=4,
        message="Cover the partial rollback.",
    )
    closed = replace(
        raised_early,
        id=uuid4(),
        status=FindingStatus.RESOLVED,
        message="Say which region ships first.",
    )

    note = assembled_steering_note(
        [raised_early, closed, raised_late], round_number=4
    )

    assert (
        "Name the rollback checks. (blocker, first raised in round 1 and open "
        "for 4 rounds)"
    ) in note
    assert (
        "Cover the partial rollback. (blocker, first raised in round 4 and "
        "open for 1 round)"
    ) in note
    # Closed in an earlier round, so it is not one still open.
    assert "Say which region ships first." not in note


def test_a_note_and_a_prompt_with_nothing_open_say_so_rather_than_nothing() -> None:
    """Both renderings hold their shape when a section has nothing in it.

    Neither is reached with an empty section by a loop that steers — a note is
    written for a round the review left objections behind in — so what keeps
    the headings from standing over silence is only the arms themselves.
    """
    verdict = ReviewAssessment(
        borg_id=uuid4(),
        loop="tech_lead_review",
        round=2,
        minimum=1,
        converging=False,
    )

    note = assembled_steering_note([], round_number=2)
    prompt = render_steering_prompt(
        subject=SteeringSubject(answerer="The Architect", work="the plan"),
        verdict=verdict,
        ledger=[],
        summaries=[],
    )

    assert "- none the ledger still holds open." in note
    assert "- none the ledger still holds open." in prompt
    assert "- no rounds have been reconciled yet." in prompt
    assert "- no round recorded a summary." in prompt


def test_the_steering_prompt_states_the_rules_nothing_else_holds() -> None:
    """Three of its rules have no other enforcement anywhere in the stage.

    The confidence bar reads a number the turn reports about itself, so the
    sentence asking for it honestly is the only thing behind it; the note
    reaching the right agent is a matter of who it is addressed to; and the
    write tripwires block a turn that wrote without ever having told it not to.
    """
    assert (
        "Report how confident you are of the note as high, medium or low, and "
        "report it as you find it rather than to get the note used."
    ) in STEERING_SYSTEM_PROMPT
    assert (
        "Write the note to the agent that will answer the findings, not to "
        "the reviewer."
    ) in STEERING_SYSTEM_PROMPT
    assert "Do not modify any file." in STEERING_SYSTEM_PROMPT


def test_a_steering_scope_names_the_run_its_note_belongs_to() -> None:
    """A durable turn's context carries the scope its loop fills, and no other."""
    approval_id = uuid4()
    task_id = uuid4()

    assert SteeringScope(
        loop="tech_lead_review", cycle_id=INITIAL_PLANNING_CYCLE
    ).request_context() == {
        "loop": "tech_lead_review",
        "cycle_id": INITIAL_PLANNING_CYCLE,
    }
    assert SteeringScope(
        loop="supervisor_review", plan_approval_id=approval_id
    ).request_context() == {
        "loop": "supervisor_review",
        "plan_approval_id": str(approval_id),
    }
    assert SteeringScope(
        loop="task_review", task_id=task_id
    ).request_context() == {"loop": "task_review", "task_id": str(task_id)}


@pytest.mark.parametrize("confidence", ["medium", "low"])
def test_a_note_the_turn_is_not_confident_of_is_replaced_by_the_assembled_one(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    confidence: str,
) -> None:
    """The costs are not symmetric, so only a confident note is attached.

    A merely plausible note misdirects a round that was bought, where the
    assembled one leaves the Architect exactly where an unsteered round would
    have left it.
    """
    handed: list[list[dict]] = []

    def stuck(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback checks are still unnamed.",
            findings=[_blocked("Name the rollback checks.")],
        )

    database = committed_git_repo.parent / f"steering-{confidence}.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, f"steering-{confidence}"
        )
        architect = _architect(planning_plan_response, 1)
        steering = _steering(1, confidence=confidence)
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        TechLeadLoop(
            repository,
            handoff.borg,
            store,
            _reviewer([stuck, _approving], handed),
            architect_agent=architect,
            steering_agent=steering,
            io=_io(),
            review_rounds=1,
            grant_budget=5,
        ).run()

        rows = store.list_steering_notes(borg.id)
        assert [(row.round, row.source) for row in rows] == [(1, "assembled")]
        assert "Name the rollback checks." in rows[0].note
        assert "open for 1 round" in rows[0].note
        assert "Steering note 1." not in _revision_prompts(architect)[-1]
        assert rows[0].note in _revision_prompts(architect)[-1]


def test_a_steering_turn_that_fails_leaves_the_granted_round_intact(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """The round was already granted and the revision is still worth running."""
    handed: list[list[dict]] = []

    def stuck(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback checks are still unnamed.",
            findings=[_blocked("Name the rollback checks.")],
        )

    database = committed_git_repo.parent / "steering-raises.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "steering-raises"
        )
        architect = _architect(planning_plan_response, 1)
        steering = MockAdapter(name="openai").queue(
            MockResponse(raise_error=RuntimeError("the note turn fell over"))
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        result = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            _reviewer([stuck, _approving], handed),
            architect_agent=architect,
            steering_agent=steering,
            io=_io(),
            review_rounds=1,
            grant_budget=5,
        ).run()

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        rows = store.list_steering_notes(borg.id)
        assert [(row.round, row.source) for row in rows] == [(1, "assembled")]
        assert rows[0].note in _revision_prompts(architect)[-1]
        # The machinery does not close an attempt that crashed, so the turn
        # left its own open and nothing else. The round it steers is in its
        # request context, which is what stops the next round replaying it.
        assert [
            (item.status, item.request["steered_round"])
            for item in store.list_planning_attempts(borg.id)
            if item.phase == "steering"
        ] == [(PlanningAttemptStatus.RUNNING, 1)]


def test_an_operator_stopping_the_run_stops_it_from_the_steering_turn_too(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """An operator who stopped the run has not asked for an unsteered revision.

    The loop's cancellation error subclasses its general one, so a handler
    reaching for the assembled note through the general one would swallow the
    stop and revise anyway.
    """
    handed: list[list[dict]] = []
    cancel = CancellationToken()

    def stuck(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback checks are still unnamed.",
            findings=[_blocked("Name the rollback checks.")],
        )

    database = committed_git_repo.parent / "steering-cancelled.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "steering-cancelled"
        )
        architect = _architect(planning_plan_response, 1)
        steering = CancellingAgent(stops=True, cancel=cancel)
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()

        with pytest.raises(TechLeadCancelled):
            TechLeadLoop(
                repository,
                handoff.borg,
                store,
                _reviewer([stuck, _approving], handed),
                architect_agent=architect,
                steering_agent=steering,
                io=_io(),
                cancel=cancel,
                review_rounds=1,
                grant_budget=5,
            ).run()

        current = store.get_borg(borg.id)
        assert current is not None
        assert current.state is BorgState.ARCHITECT_WORKING
        # No note resolved, so no row was written and no revision ran.
        assert store.list_steering_notes(borg.id) == []
        assert len(_revision_prompts(architect)) == 1


def test_a_steering_turn_its_adapter_gave_up_on_falls_back_without_stopping(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """A cancelled status covers two things and only one of them is a stop.

    The adapters return it for an operator's stop and for bounded transient
    retries exhausted alike, and an optional turn must not end a run over a
    provider hiccup.
    """
    handed: list[list[dict]] = []

    def stuck(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback checks are still unnamed.",
            findings=[_blocked("Name the rollback checks.")],
        )

    database = committed_git_repo.parent / "steering-retries.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "steering-retries"
        )
        architect = _architect(planning_plan_response, 1)
        steering = CancellingAgent(stops=False)
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        result = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            _reviewer([stuck, _approving], handed),
            architect_agent=architect,
            steering_agent=steering,
            io=_io(),
            cancel=CancellationToken(),
            review_rounds=1,
            grant_budget=5,
        ).run()

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        rows = store.list_steering_notes(borg.id)
        assert [(row.round, row.source) for row in rows] == [(1, "assembled")]
        assert rows[0].note in _revision_prompts(architect)[-1]


def test_a_resumed_granted_round_reuses_the_note_its_row_already_holds(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """There is exactly one note per grant, however often the round is re-entered."""
    handed: list[list[dict]] = []

    def stuck(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback checks are still unnamed.",
            findings=[_blocked("Name the rollback checks.")],
        )

    database = committed_git_repo.parent / "steering-resumed.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "steering-resumed"
        )
        architect = _architect(planning_plan_response, 0)
        steering = _steering(1)
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()

        # The note is written, and the revision it was written for dies.
        with pytest.raises((TechLeadError, ArchitectError)):
            TechLeadLoop(
                repository,
                handoff.borg,
                store,
                _reviewer([stuck], handed),
                architect_agent=architect,
                steering_agent=steering,
                io=_io(),
                review_rounds=1,
                grant_budget=5,
            ).run()
        assert len(steering.calls) == 1

        architect.queue(
            MockResponse(payload=planning_plan_response(summary="Resumed."))
        )
        resumed = TechLeadLoop(
            repository,
            store.get_borg(borg.id),
            store,
            _reviewer([_approving], handed),
            architect_agent=architect,
            steering_agent=steering,
            io=_io(),
            review_rounds=1,
            grant_budget=5,
        ).run()

        assert resumed.borg.state is BorgState.PLAN_APPROVAL_PENDING
        # The resumed round paid for no second turn and ran on the same note.
        assert len(steering.calls) == 1
        assert [
            (row.round, row.note) for row in store.list_steering_notes(borg.id)
        ] == [(1, "Steering note 1.")]
        assert "Steering note 1." in _revision_prompts(architect)[-1]


def test_a_crashed_steering_turn_is_replaced_rather_than_replayed(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """The round in the request context is what keeps the pairing honest.

    A crash leaves the attempt open, and without the round it steers in its
    context the next granted round would replay the note written for the last.
    """
    handed: list[list[dict]] = []
    stale = committed_git_repo.parent / "stale-note.json"

    def crash(spec):
        spec.result_path.parent.mkdir(parents=True, exist_ok=True)
        spec.result_path.write_text(stale.read_text(encoding="utf-8"), "utf-8")
        raise RuntimeError("the note turn crashed after writing its result")

    def stuck(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback checks are still unnamed.",
            findings=[_blocked("Name the rollback checks.")],
        )

    stale.write_text(
        json.dumps({"note": "The stale note.", "confidence": "high"}),
        encoding="utf-8",
    )
    database = committed_git_repo.parent / "steering-crashed.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "steering-crashed"
        )
        architect = _architect(planning_plan_response, 2)
        steering = MockAdapter(name="openai")
        steering.queue(MockResponse(dynamic=crash))
        steering.queue(
            MockResponse(payload={"note": "The fresh note.", "confidence": "high"})
        )
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        result = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            _reviewer([stuck, stuck, _approving], handed),
            architect_agent=architect,
            steering_agent=steering,
            io=_io(),
            review_rounds=1,
            grant_budget=5,
        ).run()

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        attempts = [
            item
            for item in store.list_planning_attempts(borg.id)
            if item.phase == "steering"
        ]
        # The orphan the crash left is failed on the next round's entry rather
        # than recovered into it, and the round it steers is what says so.
        assert [item.status for item in attempts] == [
            PlanningAttemptStatus.FAILED,
            PlanningAttemptStatus.COMPLETED,
        ]
        # The whole context and not the round alone: both planning loops file
        # under one phase in one Borg, so the scope is what keeps a crashed
        # cycle's orphan from being replayed into the next one.
        assert [
            (
                item.request["loop"],
                item.request["cycle_id"],
                item.request["steered_round"],
            )
            for item in attempts
        ] == [
            ("tech_review", INITIAL_PLANNING_CYCLE, 1),
            ("tech_review", INITIAL_PLANNING_CYCLE, 2),
        ]
        rows = store.list_steering_notes(borg.id)
        assert [(row.round, row.source) for row in rows] == [
            (1, "assembled"),
            (2, "agent"),
        ]
        assert rows[1].note == "The fresh note."
        assert "The stale note." not in "".join(_revision_prompts(architect))


def test_a_second_steered_round_gets_its_own_runner_and_progress_child(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """A progress child is declared and started once, so two rounds need two.

    Reusing the loop's own runner would put the note on the reviewer's agent
    and ignore the steering configuration entirely, so each steered round
    builds its own.
    """
    handed: list[list[dict]] = []

    def stuck(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback checks are still unnamed.",
            findings=[_blocked("Name the rollback checks.")],
        )

    database = committed_git_repo.parent / "steering-twice.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "steering-twice"
        )
        architect = _architect(planning_plan_response, 2)
        reviewer = _reviewer([stuck, stuck, _approving], handed)
        steering = _steering(2, activities=(STEERING_ACTIVITY,))
        progress = RunProgress(stream=StringIO())
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        result = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            reviewer,
            architect_agent=architect,
            steering_agent=steering,
            io=_io(),
            review_rounds=1,
            grant_budget=5,
            progress=progress,
        ).run()

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        # The stage completed, which it refuses to do while a child of it is
        # still running: every steering child is reconciled on the success path.
        assert progress.stages["tech-lead"].state is StageState.COMPLETED
        children = progress.stages["tech-lead"].children
        assert [
            children[key].state
            for key in ("steering:1", "steering:2")
        ] == [StageState.COMPLETED, StageState.COMPLETED]
        assert len(steering.calls) == 2
        assert len(reviewer.calls) == 3
        assert [
            (row.round, row.note) for row in store.list_steering_notes(borg.id)
        ] == [(1, "Steering note 1."), (2, "Steering note 2.")]
        # Each runner is bound to its own child, which is what the child is
        # for: a turn reporting against the stage instead reads on screen as
        # the reviewer stalling for as long as the note takes.
        assert [
            children[key].activity for key in ("steering:1", "steering:2")
        ] == [STEERING_ACTIVITY, STEERING_ACTIVITY]
        assert progress.stages["tech-lead"].activity != STEERING_ACTIVITY
        progress.close()


def test_a_fallen_back_steered_round_still_reconciles_its_progress_child(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """Every path reconciles the child, not only the one that wrote a note.

    The turn machinery emits into a child and never touches its lifecycle, so
    the stage refuses to complete while any child of it is still running. A
    round that fell back returns through the same point as one that succeeded,
    so this cannot fail while its sibling passes — it is here against a
    reconciliation that one day reads the note before deciding. Over two
    rounds, because the note a round falls back to is assembled for the round
    it steers, and a loop that steers once steers exactly on its minimum.
    """
    handed: list[list[dict]] = []

    def stuck(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback checks are still unnamed.",
            findings=[_blocked("Name the rollback checks.")],
        )

    database = committed_git_repo.parent / "steering-fallback-progress.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "steering-fallback-progress"
        )
        architect = _architect(planning_plan_response, 2)
        reviewer = _reviewer([stuck, stuck, _approving], handed)
        steering = _steering(2, confidence="medium")
        progress = RunProgress(stream=StringIO())
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        result = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            reviewer,
            architect_agent=architect,
            steering_agent=steering,
            io=_io(),
            review_rounds=1,
            grant_budget=5,
            progress=progress,
        ).run()

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        assert progress.stages["tech-lead"].state is StageState.COMPLETED
        children = progress.stages["tech-lead"].children
        assert [
            children[key].state for key in ("steering:1", "steering:2")
        ] == [StageState.COMPLETED, StageState.COMPLETED]
        # Both rounds ran on the assembled note, which is what makes this the
        # fallback path rather than the success one.
        rows = store.list_steering_notes(borg.id)
        assert [(row.round, row.source) for row in rows] == [
            (1, "assembled"),
            (2, "assembled"),
        ]
        # And the second one is assembled for the round it steers, not for the
        # minimum that round passed.
        assert (
            "Name the rollback checks. (blocker, first raised in round 1 and "
            "open for 2 rounds)"
        ) in rows[1].note
        progress.close()


def test_the_note_rides_the_plan_prompts_contract_corrections(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """A contract retry is the same granted round arguing with itself."""
    handed: list[list[dict]] = []

    def stuck(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback checks are still unnamed.",
            findings=[_blocked("Name the rollback checks.")],
        )

    database = committed_git_repo.parent / "steering-correction.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "steering-correction"
        )
        architect = _architect(planning_plan_response, 0)
        invalid = planning_plan_response(summary="Invalid.")
        invalid["phases"][0]["name"] = "02-release-workflow"
        architect.queue(MockResponse(payload=invalid))
        architect.queue(
            MockResponse(payload=planning_plan_response(summary="Corrected."))
        )
        steering = _steering(1)
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        result = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            _reviewer([stuck, _approving], handed),
            architect_agent=architect,
            steering_agent=steering,
            io=_io(),
            review_rounds=1,
            grant_budget=5,
        ).run()

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        revisions = _revision_prompts(architect)[1:]
        assert len(revisions) == 2
        assert all("Steering note 1." in prompt for prompt in revisions)
        assert "Return the whole plan again." in revisions[-1]


def test_neither_the_reviewer_nor_the_question_turn_is_handed_the_note(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """A reviewer told what to conclude is not a reviewer.

    Nor does the note reach a turn about what the Architect needs to know. The
    question turn is unreachable once a note exists — a note follows a review,
    a review follows a plan, and a cycle that has planned is ready — so what is
    asserted here is that a steered revision opens no question round, and that
    the answering turn it does reach carries none.
    """
    handed: list[list[dict]] = []

    def stuck(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback checks are still unnamed.",
            findings=[_blocked("Name the rollback checks.")],
        )

    open_question = planning_plan_response(summary="Needs an answer.")
    open_question["open_questions"] = ["Which region ships first?"]
    database = committed_git_repo.parent / "steering-audience.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "steering-audience"
        )
        architect = _architect(planning_plan_response, 0)
        architect.queue(MockResponse(payload=open_question))
        architect.queue(
            MockResponse(payload={"answers": [{"q_id": "q1", "answer": "Europe."}]})
        )
        # Two: a plan that answered its own question and names no assumption
        # is corrected once before it stands.
        for summary in ("Answered.", "Answered with assumptions."):
            architect.queue(
                MockResponse(payload=planning_plan_response(summary=summary))
            )
        reviewer = _reviewer([stuck, _approving], handed)
        steering = _steering(1)
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io(), unattended=True
        ).run()
        result = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            reviewer,
            architect_agent=architect,
            steering_agent=steering,
            io=_io(),
            unattended=True,
            review_rounds=1,
            grant_budget=5,
        ).run()

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        assert not any("Steering note" in call.user_prompt for call in reviewer.calls)
        answering = [
            call.user_prompt
            for call in architect.calls
            if "decide them yourself" in call.user_prompt
        ]
        assert answering and not any("Steering note" in prompt for prompt in answering)
        # The steered revision opened no question round of its own, which is
        # why no prompt of that turn's can carry a note.
        asked = [
            attempt
            for attempt in store.list_planning_attempts(borg.id)
            if attempt.phase == "architect_questions"
        ]
        assert [attempt.round for attempt in asked] == [1]


def test_the_note_row_carries_the_ledger_round_and_not_the_attempts_own(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """Attempts are unique per phase and round, so the two cannot be one number.

    A second planning cycle restarts the ledger's rounds while the steering
    phase's own count carries on, and a row holding the attempt's number would
    say nothing about the round it steers.
    """
    handed: list[list[dict]] = []

    def stuck(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback checks are still unnamed.",
            findings=[_blocked("Name the rollback checks.")],
        )

    database = committed_git_repo.parent / "steering-cycles.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "steering-cycles"
        )
        first = _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [stuck, _approving],
            handed,
            review_rounds=1,
            grant_budget=5,
        )
        assert first.borg.state is BorgState.PLAN_APPROVAL_PENDING

        request = PlanChangeRequest(
            borg_id=borg.id, round=1, note="Cover the rollback path."
        )
        with store.transaction():
            store.append_plan_change_request(request)
            changed = store.compare_and_set_borg_state(
                borg.id,
                expected_state=first.borg.state,
                expected_version=first.borg.state_version,
                new_state=BorgState.ARCHITECT_WORKING,
            )
        second = _run_reviews(
            repository,
            changed,
            store,
            planning_plan_response,
            [stuck, _approving],
            handed,
            review_rounds=1,
            grant_budget=5,
        )

        assert second.borg.state is BorgState.PLAN_APPROVAL_PENDING
        attempts = [
            item
            for item in store.list_planning_attempts(borg.id)
            if item.phase == "steering"
        ]
        # The phase's own count carries across the cycles; the rows both name
        # the ledger round one their own cycle steered.
        assert [item.round for item in attempts] == [1, 2]
        assert [
            (row.round, row.cycle_id) for row in store.list_steering_notes(borg.id)
        ] == [(1, INITIAL_PLANNING_CYCLE), (1, str(request.id))]


def test_a_steered_cycle_reads_its_own_argument_and_not_the_one_before_it(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """A cycle that blocked leaves its objections open for the next one to see.

    Only an approval closes a ledger, so a cycle that spent its grants leaves
    every row standing. Read unscoped, the next cycle's steering turn is handed
    objections against a plan that no longer exists.
    """
    handed: list[list[dict]] = []

    def first_cycle(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback checks are still unnamed.",
            findings=[_blocked("Name the rollback checks.")],
        )

    def second_cycle(_ledger, _history):
        return _review(
            "request_changes",
            summary="The migration order is still unstated.",
            findings=[_blocked("State the migration order.")],
        )

    database = committed_git_repo.parent / "steering-stale-cycle.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "steering-stale-cycle"
        )
        blocked = _run_reviews(
            repository,
            borg,
            store,
            planning_plan_response,
            [first_cycle, first_cycle],
            handed,
            review_rounds=1,
            grant_budget=1,
        )
        # Spent its grant without closing anything, so its rows stay open.
        assert blocked.borg.state is BorgState.BLOCKED
        stale = store.list_planning_ledger_findings(
            borg.id, cycle_id=INITIAL_PLANNING_CYCLE
        )
        assert stale and all(row.status is FindingStatus.OPEN for row in stale)

        request = PlanChangeRequest(
            borg_id=borg.id, round=1, note="Cover the migration order."
        )
        with store.transaction():
            store.append_plan_change_request(request)
            changed = store.compare_and_set_borg_state(
                borg.id,
                expected_state=blocked.borg.state,
                expected_version=blocked.borg.state_version,
                new_state=BorgState.ARCHITECT_WORKING,
            )
        steering = _steering(1)
        second = _run_reviews(
            repository,
            changed,
            store,
            planning_plan_response,
            [second_cycle, _approving],
            handed,
            review_rounds=1,
            grant_budget=5,
            steering_agent=steering,
        )

        assert second.borg.state is BorgState.PLAN_APPROVAL_PENDING
        assert len(steering.calls) == 1
        prompt = steering.calls[0].user_prompt
        assert "State the migration order." in prompt
        assert "Name the rollback checks." not in prompt
        # The rounds argued are this cycle's, not both cycles' together.
        assert "Rounds argued so far: 1." in prompt
        decided = prompt.split("## What each round decided")[1].strip().splitlines()
        assert decided == ["- Round 1: The migration order is still unstated."]


def test_a_steering_agent_that_cannot_be_held_to_reading_is_refused(
    committed_git_repo: Path,
    persist_planning_context,
) -> None:
    """The steering agent passes the boundary check its siblings pass.

    It reads rows and returns prose, and an adapter that can enforce neither a
    tool allowlist nor a read-only sandbox cannot be held to that.
    """
    unbounded = MockAdapter(name="openai")
    unbounded.capabilities = replace(
        unbounded.capabilities, tool_allowlist=False, read_only_sandbox=False
    )
    database = committed_git_repo.parent / "steering-unbounded.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "steering-unbounded"
        )
        with pytest.raises(TechLeadError, match="Steering read-only"):
            TechLeadLoop(
                repository,
                borg,
                store,
                MockAdapter(name="openai"),
                steering_agent=unbounded,
                io=_io(),
            )


def test_the_note_turn_runs_on_the_steering_stage_and_reads_the_argument(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
) -> None:
    """It resolves its own agent, and what it is given is the argument itself.

    The open ledger, what each round raised and closed, and each round's own
    account of what it decided. Not which arm of the assessment fired, nor
    whether its veto is what stopped it: neither survives into the result the
    assessment returns, and the ledger shows the same thing anyway.
    """
    handed: list[list[dict]] = []

    def opening(_ledger, _history):
        return _review(
            "request_changes",
            summary="Two things stand in the way.",
            findings=[
                _blocked("Name the rollback checks."),
                _raised("Say which region ships first."),
            ],
        )

    def standing_still(ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback checks are still unnamed.",
            resolved=[ledger[1]["id"]],
            findings=[_raised("Cover the partial rollback.")],
        )

    database = committed_git_repo.parent / "steering-prompt.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "steering-prompt"
        )
        architect = _architect(planning_plan_response, 2)
        reviewer = _reviewer([opening, standing_still, _approving], handed)
        steering = _steering(2)
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()
        result = TechLeadLoop(
            repository,
            handoff.borg,
            store,
            reviewer,
            architect_agent=architect,
            steering_agent=steering,
            steering_model="steering-model",
            io=_io(),
            review_rounds=1,
            grant_budget=5,
        ).run()

        assert result.borg.state is BorgState.PLAN_APPROVAL_PENDING
        attempts = store.list_planning_attempts(borg.id)
        # Its own stage's model, and not the one the reviewer runs under.
        assert [
            item.model for item in attempts if item.phase == "steering"
        ] == ["steering-model", "steering-model"]
        assert all(
            item.model != "steering-model"
            for item in attempts
            if item.phase != "steering"
        )
        assert [
            item.summary for item in attempts if item.phase == "steering"
        ] == ["steering note reported high confidence"] * 2

        # Held to reading like every other turn the planning machinery runs,
        # which is all there is to assert: it cannot be given another set.
        assert all(
            call.allowed_tools == READ_ONLY_API_TOOLS for call in steering.calls
        )
        # And to its own instructions: the stage's constant, not the prompt of
        # whatever loop happens to be steering.
        assert all(
            call.system_prompt == STEERING_SYSTEM_PROMPT
            for call in steering.calls
        )
        # And to the shape it has to answer in: a permissive schema leaves the
        # confidence the bar reads absent on every turn.
        assert all(
            call.schema == STEERING_NOTE_SCHEMA for call in steering.calls
        )

        prompt = steering.calls[-1].user_prompt
        # Who is about to answer, and what the argument is about. The note is
        # written to the agent that can act on it, so naming another loop's
        # answerer here addresses it to an agent that never sees it.
        assert (
            "The Architect is about to answer these findings again, on a round "
            "granted because the review of the implementation plan is not "
            "closing in on agreement."
        ) in prompt
        # The round the argument has reached, which is what says how
        # entrenched it is — and not the minimum it passed to get here.
        assert "Rounds argued so far: 2." in prompt
        assert "Name the rollback checks. (blocker, first raised in round 1)" in prompt
        assert "Cover the partial rollback. (major, first raised in round 2)" in prompt
        assert "Say which region ships first." not in prompt
        assert "Round 1: raised 2, closed 0, 2 still open afterwards." in prompt
        assert "Round 2: raised 1, closed 1, 2 still open afterwards." in prompt
        assert "Round 1: Two things stand in the way." in prompt
        assert "Round 2: The rollback checks are still unnamed." in prompt
        # The verdict's own workings are not in the assessment's result and are
        # not here either.
        assert "veto" not in prompt and "converg" not in prompt


def test_the_note_row_and_its_attempt_are_written_in_one_durable_step(
    committed_git_repo: Path,
    persist_planning_context,
    planning_plan_response,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A round that recorded which note it used but never finished its turn is
    a round the next entry would re-run and charge twice.
    """
    handed: list[list[dict]] = []

    def stuck(_ledger, _history):
        return _review(
            "request_changes",
            summary="The rollback checks are still unnamed.",
            findings=[_blocked("Name the rollback checks.")],
        )

    database = committed_git_repo.parent / "steering-atomic.sqlite3"
    with SqliteStore.open(database) as store:
        repository, borg = persist_planning_context(
            committed_git_repo, store, "steering-atomic"
        )
        architect = _architect(planning_plan_response, 1)
        steering = _steering(1)
        handoff = ArchitectLoop(
            repository, borg, store, architect, io=_io()
        ).run()

        def refuse(_note):
            raise RuntimeError("the note row could not be written")

        monkeypatch.setattr(store, "record_steering_note", refuse)
        with pytest.raises(RuntimeError, match="note row"):
            TechLeadLoop(
                repository,
                handoff.borg,
                store,
                _reviewer([stuck, _approving], handed),
                architect_agent=architect,
                steering_agent=steering,
                io=_io(),
                review_rounds=1,
                grant_budget=5,
            ).run()
        monkeypatch.undo()

        assert store.list_steering_notes(borg.id) == []
        assert [
            item.status
            for item in store.list_planning_attempts(borg.id)
            if item.phase == "steering"
        ] == [PlanningAttemptStatus.RUNNING]
