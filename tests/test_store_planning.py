"""Durable planning history and Borg state transition contracts."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from betterborg_cli.planning.cycles import INITIAL_PLANNING_CYCLE
from betterborg_cli.store import (
    Borg,
    BorgState,
    FindingStatus,
    PlanChangeRequest,
    PlanningAttempt,
    PlanningAttemptStatus,
    PlanningFinding,
    PlanningLedgerFinding,
    PlanningQuestion,
    Repository,
    ReviewAssessment,
    SqliteStore,
    StaleBorgStateError,
)


def test_migration_004_planning_history_survives_reopen(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"
    repository = Repository(root=tmp_path / "repository")
    borg = Borg(repository_id=repository.id, name="DurablePlanner")
    architect_attempt = PlanningAttempt(
        borg_id=borg.id,
        phase="architect_questions",
        round=1,
        adapter="mock",
        model="test-model",
        request={"prd_path": ".betterborg/prds/DurablePlanner.md"},
    )
    question = PlanningQuestion(
        borg_id=borg.id,
        attempt_id=architect_attempt.id,
        round=1,
        questions=[
            {
                "id": "scope",
                "question": "Which platforms are required?",
                "why": "The plan needs a compatibility boundary.",
            }
        ],
    )
    review_attempt = PlanningAttempt(
        borg_id=borg.id,
        phase="tech_review",
        round=1,
        adapter="mock",
        model="test-model",
        request={"plan_revision": 1},
    )
    finding = PlanningFinding(
        borg_id=borg.id,
        attempt_id=review_attempt.id,
        round=1,
        severity="major",
        message="The rollback behavior is unspecified.",
        suggestion="Describe the recovery path.",
    )
    change_request = PlanChangeRequest(
        borg_id=borg.id,
        round=1,
        note="Keep the migration forward-only.",
        decided_by="operator",
    )

    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.append_planning_attempt(architect_attempt)
        completed_architect_attempt = store.complete_planning_attempt(
            architect_attempt.id,
            status=PlanningAttemptStatus.COMPLETED,
            result={"status": "ask_more"},
            summary="One material question remains.",
        )
        store.append_planning_question(question)
        answered_question = store.answer_planning_question(
            question.id,
            [{"q_id": "scope", "answer": "Linux, macOS, and Windows."}],
        )
        store.append_planning_attempt(review_attempt)
        completed_review_attempt = store.complete_planning_attempt(
            review_attempt.id,
            status=PlanningAttemptStatus.COMPLETED,
            result={"status": "request_changes"},
        )
        store.append_planning_finding(finding)
        store.append_plan_change_request(change_request)
        applied_at = store.applied_migrations()

    with SqliteStore.open(database) as reopened:
        assert reopened.applied_migrations() == applied_at == tuple(range(1, 15))
        assert reopened.get_repository(repository.id) == repository
        assert reopened.get_borg(borg.id) == borg
        assert reopened.list_planning_attempts(borg.id) == [
            completed_architect_attempt,
            completed_review_attempt,
        ]
        assert reopened.list_planning_questions(borg.id) == [answered_question]
        assert reopened.list_planning_findings(borg.id) == [finding]
        assert reopened.list_plan_change_requests(borg.id) == [change_request]

        with pytest.raises(ValueError, match="already completed"):
            reopened.complete_planning_attempt(
                architect_attempt.id,
                status=PlanningAttemptStatus.COMPLETED,
                result={"status": "different-result"},
            )
        with pytest.raises(ValueError, match="already been answered"):
            reopened.answer_planning_question(
                question.id,
                [{"q_id": "scope", "answer": "A stale replacement."}],
            )


def test_compare_and_set_rejects_stale_writers_across_all_states(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.sqlite3"
    repository = Repository(root=tmp_path / "repository")
    borg = Borg(repository_id=repository.id, name="StatefulPlanner")

    with SqliteStore.open(database) as setup:
        setup.add_repository(repository)
        setup.add_borg(borg)

    first = SqliteStore.open(database)
    second = SqliteStore.open(database)
    try:
        current = borg
        original_snapshot = second.get_borg(borg.id)
        assert original_snapshot == current
        targets = [*list(BorgState)[1:], BorgState.DRAFT]
        for target in targets:
            stale_snapshot = second.get_borg(borg.id)
            assert stale_snapshot == current

            current = first.compare_and_set_borg_state(
                borg.id,
                expected_state=current.state,
                expected_version=current.state_version,
                new_state=target,
            )
            assert current.state is target

            with pytest.raises(StaleBorgStateError, match="state changed"):
                second.compare_and_set_borg_state(
                    borg.id,
                    expected_state=stale_snapshot.state,
                    expected_version=stale_snapshot.state_version,
                    new_state=BorgState.BLOCKED,
                )

        assert current.state is original_snapshot.state
        assert current.state_version > original_snapshot.state_version
        with pytest.raises(StaleBorgStateError, match="state changed"):
            second.compare_and_set_borg_state(
                borg.id,
                expected_state=original_snapshot.state,
                expected_version=original_snapshot.state_version,
                new_state=BorgState.BLOCKED,
            )
    finally:
        first.close()
        second.close()

    with SqliteStore.open(database) as reopened:
        persisted = reopened.get_borg(borg.id)
        assert persisted == current
        assert persisted.state_version == len(BorgState)


@pytest.mark.parametrize(
    "history_kind",
    ["completed attempt", "answered question", "finding", "change request"],
)
@pytest.mark.parametrize("statement", ["UPDATE", "DELETE", "REPLACE"])
def test_planning_history_rejects_raw_mutation_deletion_and_replacement(
    tmp_path: Path, history_kind: str, statement: str
) -> None:
    repository = Repository(root=tmp_path / "repository")
    borg = Borg(repository_id=repository.id, name="AppendOnlyPlanner")
    attempt = PlanningAttempt(
        borg_id=borg.id,
        phase="tech_review",
        round=1,
        adapter="mock",
        model="test-model",
    )
    question = PlanningQuestion(
        borg_id=borg.id,
        attempt_id=attempt.id,
        round=1,
        questions=[{"id": "scope", "question": "Which platforms?"}],
    )
    finding = PlanningFinding(
        borg_id=borg.id,
        attempt_id=attempt.id,
        round=1,
        severity="major",
        message="The platform scope is unclear.",
    )
    change_request = PlanChangeRequest(
        borg_id=borg.id,
        round=1,
        note="Clarify the platform scope.",
    )

    with SqliteStore.open(tmp_path / "state.sqlite3") as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.append_planning_attempt(attempt)
        completed_attempt = store.complete_planning_attempt(
            attempt.id,
            status=PlanningAttemptStatus.COMPLETED,
            result={"status": "request_changes"},
        )
        store.append_planning_question(question)
        answered_question = store.answer_planning_question(
            question.id,
            [{"q_id": "scope", "answer": "All desktop platforms."}],
        )
        store.append_planning_finding(finding)
        store.append_plan_change_request(change_request)

        table, column, record_id, read_history, expected_history = {
            "completed attempt": (
                "planning_attempts",
                "summary",
                completed_attempt.id,
                store.list_planning_attempts,
                [completed_attempt],
            ),
            "answered question": (
                "planning_questions",
                "questions_json",
                answered_question.id,
                store.list_planning_questions,
                [answered_question],
            ),
            "finding": (
                "planning_findings",
                "message",
                finding.id,
                store.list_planning_findings,
                [finding],
            ),
            "change request": (
                "plan_change_requests",
                "note",
                change_request.id,
                store.list_plan_change_requests,
                [change_request],
            ),
        }[history_kind]

        sql, parameters = {
            "UPDATE": (
                f"UPDATE {table} SET {column} = ? WHERE id = ?",
                ("changed", str(record_id)),
            ),
            "DELETE": (
                f"DELETE FROM {table} WHERE id = ?",
                (str(record_id),),
            ),
            "REPLACE": (
                f"INSERT OR REPLACE INTO {table} SELECT * FROM {table} WHERE id = ?",
                (str(record_id),),
            ),
        }[statement]

        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            with store.transaction() as connection:
                connection.execute(sql, parameters)

        assert read_history(borg.id) == expected_history


def test_migration_013_finding_ledger_updates_in_place_within_its_cycle(
    tmp_path: Path,
) -> None:
    """A ledger row is the current lifecycle view, so it is written again.

    The immutable findings beside it are the record of what each round said.
    Scoping the ledger by cycle is what stops a new cycle inheriting the block
    of resolved rows an approval left on the previous one.
    """
    database = tmp_path / "state.sqlite3"
    repository = Repository(root=tmp_path / "repository")
    borg = Borg(repository_id=repository.id, name="LedgerPlanner")
    review = PlanningAttempt(
        borg_id=borg.id,
        phase="tech_review",
        round=1,
        adapter="mock",
        model="test-model",
    )
    revision = PlanningAttempt(
        borg_id=borg.id,
        phase="tech_review",
        round=2,
        adapter="mock",
        model="test-model",
    )
    change_request = PlanChangeRequest(
        borg_id=borg.id, round=1, note="Stage the rollout."
    )
    raised = PlanningLedgerFinding(
        borg_id=borg.id,
        cycle_id=INITIAL_PLANNING_CYCLE,
        attempt_id=review.id,
        first_seen_round=1,
        last_seen_round=1,
        severity="blocker",
        message="Cover a partial rollback.",
        suggestion="Name the checks it runs.",
    )
    next_cycle = PlanningLedgerFinding(
        borg_id=borg.id,
        cycle_id=str(change_request.id),
        attempt_id=revision.id,
        first_seen_round=1,
        last_seen_round=1,
        severity="major",
        message="Stage the rollout explicitly.",
    )

    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.append_planning_attempt(review)
        store.append_planning_attempt(revision)
        store.append_plan_change_request(change_request)
        store.record_planning_ledger_findings([raised])
        store.record_planning_ledger_findings(
            [
                replace(
                    raised,
                    attempt_id=revision.id,
                    first_seen_round=2,
                    last_seen_round=2,
                    status=FindingStatus.RESOLVED,
                    severity="minor",
                    message="A later round said something else.",
                    suggestion="And suggested something else.",
                ),
                next_cycle,
            ]
        )

    with SqliteStore.open(database) as reopened:
        assert reopened.applied_migrations() == tuple(range(1, 15))
        rows = reopened.list_planning_ledger_findings(borg.id)
        assert len(rows) == 2
        closed = next(row for row in rows if row.id == raised.id)
        assert closed.status is FindingStatus.RESOLVED
        assert closed.last_seen_round == 2
        assert closed.attempt_id == revision.id
        # Only lifecycle moves. A later write offering a different severity,
        # message, suggestion or birth round changes none of them, which is
        # what keeps a repeat from escalating a minor into a blocker.
        assert closed.severity == "blocker"
        assert closed.message == "Cover a partial rollback."
        assert closed.suggestion == "Name the checks it runs."
        assert closed.first_seen_round == 1
        assert closed.round == 1
        assert reopened.list_planning_ledger_findings(
            borg.id, cycle_id=str(change_request.id)
        ) == [next_cycle]
        assert reopened.list_planning_ledger_findings(
            borg.id, cycle_id=INITIAL_PLANNING_CYCLE
        ) == [closed]


def test_migration_014_review_assessments_survive_reopen(tmp_path: Path) -> None:
    """One row per round, scoped by the run of rounds its loop compares.

    Loops are rebuilt from configuration on every entry, so a verdict held in
    memory is lost the moment a run is interrupted and the budget would be
    spent twice.
    """
    database = tmp_path / "state.sqlite3"
    repository = Repository(root=tmp_path / "repository")
    borg = Borg(repository_id=repository.id, name="AssessedPlanner")
    review = PlanningAttempt(
        borg_id=borg.id,
        phase="tech_review",
        round=1,
        adapter="mock",
        model="test-model",
    )
    change_request = PlanChangeRequest(
        borg_id=borg.id, round=1, note="Stage the rollout."
    )
    minimum_round = ReviewAssessment(
        borg_id=borg.id,
        loop="tech_review",
        cycle_id=INITIAL_PLANNING_CYCLE,
        attempt_id=review.id,
        round=1,
        minimum=1,
        converging=False,
        open_findings=2,
        evidence={"drain": [], "trend": None},
    )
    granted_round = ReviewAssessment(
        borg_id=borg.id,
        loop="tech_review",
        cycle_id=INITIAL_PLANNING_CYCLE,
        attempt_id=review.id,
        round=2,
        minimum=1,
        converging=True,
        open_findings=1,
        refunded=True,
        evidence={"trend": "improving"},
    )
    next_cycle = ReviewAssessment(
        borg_id=borg.id,
        loop="tech_review",
        cycle_id=str(change_request.id),
        round=1,
        minimum=1,
        converging=False,
        open_findings=1,
    )

    with SqliteStore.open(database) as store:
        store.add_repository(repository)
        store.add_borg(borg)
        store.append_planning_attempt(review)
        store.append_plan_change_request(change_request)
        for assessment in (minimum_round, granted_round, next_cycle):
            store.record_review_assessment(assessment)

    with SqliteStore.open(database) as reopened:
        assert reopened.applied_migrations() == tuple(range(1, 15))
        rows = reopened.list_review_assessments(borg.id, loop="tech_review")
        assert rows == [minimum_round, granted_round, next_cycle]
        # A round inside the minimum is neither charged nor refunded, so it
        # records no refund decision at all.
        assert rows[0].refunded is None
        assert reopened.list_review_assessments(
            borg.id, cycle_id=INITIAL_PLANNING_CYCLE
        ) == [minimum_round, granted_round]
        assert reopened.list_review_assessments(borg.id, loop="supervisor_review") == []

        # One assessment per round of one scope, enforced rather than relied on:
        # the budget is read by counting the rounds that earned no refund, so a
        # second row for a round that earned one lifts the refund count above
        # the number of grants and the loop never stops.
        with pytest.raises(sqlite3.IntegrityError):
            reopened.record_review_assessment(
                replace(granted_round, id=uuid4(), converging=False)
            )
        # The same round of another scope is a different round.
        reopened.record_review_assessment(
            replace(
                granted_round,
                id=uuid4(),
                cycle_id=str(change_request.id),
                refunded=False,
            )
        )
