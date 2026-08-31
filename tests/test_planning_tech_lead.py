"""Tech Lead findings, revisions, durability, and convergence contracts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import pytest

from betterborg_cli.agent_runtime.mock import MockAdapter, MockResponse
from betterborg_cli.planning import (
    ArchitectCancelled,
    ArchitectError,
    ArchitectLoop,
    TechLeadError,
    TechLeadLoop,
)
from betterborg_cli.prd_session import InteractiveIO
from betterborg_cli.progress import RunProgress, StageState
from betterborg_cli.store import (
    BorgState,
    PlanningAttempt,
    PlanningAttemptStatus,
    SqliteStore,
)


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
        resumed_progress = RunProgress(
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
        bounded = resumed_progress.child_render_state("tech-lead")
        assert [item.key for item in bounded.children] == [keys[1]]
        assert bounded.earlier_attempt_count == 1
        assert len(architect.calls) == 5
        assert len(reviewer.calls) == 3


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
        (spec.cwd / ".borg/state/planning/context/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return json.loads((spec.cwd / manifest["current_plan"]).read_text(encoding="utf-8"))


def _findings(spec) -> list[dict]:
    return json.loads(
        (
            spec.cwd / ".borg/state/planning/context/findings.json"
        ).read_text(encoding="utf-8")
    )


def _assert_prior_finding_count(spec, expected: int) -> None:
    assert len(_findings(spec)) == expected


def _io(answers: Iterator[str] | None = None) -> InteractiveIO:
    supplied_answers = answers or iter(())
    return InteractiveIO(
        prompt=lambda _message: next(supplied_answers, None),
        confirm=lambda _message, _default: False,
        write=lambda _message: None,
    )
