"""Durable planning history and Borg state transition contracts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from betterborg_cli.store import (
    Borg,
    BorgState,
    PlanChangeRequest,
    PlanningAttempt,
    PlanningAttemptStatus,
    PlanningFinding,
    PlanningQuestion,
    Repository,
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
        request={"prd_path": ".borg/prds/DurablePlanner.md"},
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
        assert reopened.applied_migrations() == applied_at == (1, 2, 3, 4, 5)
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
