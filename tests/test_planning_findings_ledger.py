"""Ledger reconciliation rules a reviewer's declarations can reach."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid4

from betterborg_cli.planning.cycles import INITIAL_PLANNING_CYCLE
from betterborg_cli.planning.findings_ledger import reconcile_planning_ledger
from betterborg_cli.store import FindingStatus, PlanningFinding, PlanningLedgerFinding

_BORG = uuid4()


def _finding(message: str, *, severity: str = "major") -> PlanningFinding:
    return PlanningFinding(
        borg_id=_BORG,
        attempt_id=uuid4(),
        round=1,
        severity=severity,
        message=message,
    )


def _fold(
    existing: Sequence[PlanningLedgerFinding],
    findings: Sequence[PlanningFinding],
    *,
    repeats: dict[UUID, str | None] | None = None,
    resolved: Sequence[str] = (),
    review_round: int,
    approved: bool = False,
) -> list[PlanningLedgerFinding]:
    return reconcile_planning_ledger(
        existing,
        findings=[(item, (repeats or {}).get(item.id)) for item in findings],
        resolved=list(resolved),
        attempt_id=uuid4(),
        cycle_id=INITIAL_PLANNING_CYCLE,
        review_round=review_round,
        approved=approved,
    )


def _row(rows: Sequence[PlanningLedgerFinding], message: str):
    return next(row for row in rows if row.message == message)


def test_a_regressed_row_carried_in_silence_is_open_on_this_round() -> None:
    """Carrying a row forward re-establishes it as open in the round that did.

    A status left as regressed would say a round the reviewer never mentioned
    the objection in is the round the objection was raised again.
    """
    raised = _finding("Cover a partial rollback.", severity="blocker")
    first = _fold([], [raised], review_round=1)

    again = _finding("Rollback coverage is gone again.", severity="blocker")
    second = _fold(
        first, [again], repeats={again.id: str(raised.id)}, review_round=2
    )
    assert _row(second, "Cover a partial rollback.").status is (
        FindingStatus.REGRESSED
    )

    unrelated = _finding("Name the rollback checks.")
    third = _fold(second, [unrelated], review_round=3)
    carried = _row(third, "Cover a partial rollback.")
    assert carried.status is FindingStatus.OPEN
    assert carried.first_seen_round == 1
    assert carried.last_seen_round == 3
    assert carried.severity == "blocker"


def test_an_objection_repeated_twice_running_moves_with_each_round() -> None:
    """A row already regressed is regressed again by the round that says so.

    The round a repeat belongs to is what the blocker veto reads, so a row left
    standing at the round that first regressed it hides a blocker the loop has
    now failed to close twice.
    """
    raised = _finding("Cover a partial rollback.", severity="blocker")
    first = _fold([], [raised], review_round=1)

    second_round = _finding("Rollback coverage is gone again.", severity="blocker")
    second = _fold(
        first,
        [second_round],
        repeats={second_round.id: str(raised.id)},
        review_round=2,
    )
    assert second[0].status is FindingStatus.REGRESSED
    assert second[0].last_seen_round == 2

    third_round = _finding("Rollback coverage is still gone.", severity="blocker")
    third = _fold(
        second,
        [third_round],
        repeats={third_round.id: str(raised.id)},
        review_round=3,
    )
    assert [row.message for row in third] == ["Cover a partial rollback."]
    assert third[0].status is FindingStatus.REGRESSED
    assert third[0].last_seen_round == 3
    assert third[0].first_seen_round == 1


def test_closing_a_row_a_later_round_already_closed_changes_nothing() -> None:
    """The round that closed an objection keeps the credit for closing it.

    Re-establishing a resolution in a later round would credit that round with
    a drain it did not achieve.
    """
    raised = _finding("Cover a partial rollback.")
    first = _fold([], [raised], review_round=1)
    second = _fold(first, [], resolved=[str(raised.id)], review_round=2)
    assert second[0].status is FindingStatus.RESOLVED
    assert second[0].last_seen_round == 2
    attributed = second[0].attempt_id

    third = _fold(second, [], resolved=[str(raised.id)], review_round=3)
    assert third[0].status is FindingStatus.RESOLVED
    assert third[0].last_seen_round == 2
    assert third[0].attempt_id == attributed


def test_a_padded_id_moves_the_row_it_names_rather_than_adding_one() -> None:
    """An id is stripped before it is read.

    The schema's non-blank check is a search rather than an anchor, so a padded
    id reaches the ledger. Read literally it places no row, and the objection
    would then hold two rows for the rest of the cycle: the original nobody
    closed, and the restatement that could not find it.
    """
    raised = _finding("Cover a partial rollback.", severity="blocker")
    first = _fold([], [raised], review_round=1)

    again = _finding("Rollback coverage is gone again.")
    second = _fold(
        first,
        [again],
        repeats={again.id: f"  {raised.id}  "},
        review_round=2,
    )
    assert [row.message for row in second] == ["Cover a partial rollback."]
    moved = second[0]
    assert moved.status is FindingStatus.REGRESSED
    assert moved.severity == "blocker"
    assert moved.last_seen_round == 2


def test_two_findings_naming_one_row_move_it_once() -> None:
    """One objection is one row however many findings restate it.

    A round that says the same objection twice has still raised it once, so the
    ledger holds one regressed row rather than a row and two restatements of
    it. What the row was first recorded with is what the fixer answers.
    """
    raised = _finding("Cover a partial rollback.", severity="minor")
    first = _fold([], [raised], review_round=1)

    one = _finding("The rollback path is untested.")
    two = _finding("The rollback checks are unnamed.", severity="blocker")
    second = _fold(
        first,
        [one, two],
        repeats={one.id: str(raised.id), two.id: str(raised.id)},
        review_round=2,
    )

    assert [row.message for row in second] == ["Cover a partial rollback."]
    moved = second[0]
    assert moved.status is FindingStatus.REGRESSED
    # Neither restatement escalated it, and the later one did not overwrite the
    # earlier one's move either.
    assert moved.severity == "minor"
    assert moved.first_seen_round == 1
    assert moved.last_seen_round == 2
