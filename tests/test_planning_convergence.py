"""Convergence arms and the veto, each against a ledger that fires it."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

from betterborg_cli.planning.convergence import assess_convergence, drain_evidence
from betterborg_cli.planning.cycles import INITIAL_PLANNING_CYCLE
from betterborg_cli.store import FindingStatus, PlanningLedgerFinding

_BORG = uuid4()


def _row(
    first: int,
    last: int,
    *,
    status: FindingStatus = FindingStatus.OPEN,
    severity: str = "major",
    message: str | None = None,
) -> PlanningLedgerFinding:
    return PlanningLedgerFinding(
        borg_id=_BORG,
        cycle_id=INITIAL_PLANNING_CYCLE,
        attempt_id=uuid4(),
        first_seen_round=first,
        last_seen_round=last,
        status=status,
        severity=severity,
        message=message or f"Objection born in round {first}.",
    )


def _opens(ledger: Sequence[PlanningLedgerFinding]) -> list[int]:
    return [row.open_after for row in assess_convergence(ledger).drain]


def test_a_falling_open_count_converges_on_the_one_arm_that_can_read_it() -> None:
    """Two rounds is where the strictly-decreasing arm stands alone.

    Every strict decrease is a round that resolved more than it raised, and a
    first round never is, so from three rounds on the arm never fires without
    the outpacing arm firing beside it.
    """
    ledger = [
        _row(1, 2, status=FindingStatus.RESOLVED),
        _row(1, 2),
    ]

    result = assess_convergence(ledger)

    assert _opens(ledger) == [2, 1]
    assert result.converging is True
    assert result.trend == "improving"


def test_two_rounds_resolving_more_than_they_raise_converge_on_a_plateau() -> None:
    """The outpacing pair carries a loop whose open count is not falling.

    The last round raises more than it closes, so the count rises and the
    strictly-decreasing arm cannot read the pair before it.
    """
    ledger = [
        _row(1, 4, message="The objection nobody answers."),
        _row(1, 2, status=FindingStatus.RESOLVED),
        _row(1, 2, status=FindingStatus.RESOLVED),
        _row(1, 3, status=FindingStatus.RESOLVED),
        _row(4, 4),
    ]

    result = assess_convergence(ledger)

    assert _opens(ledger) == [4, 2, 1, 2]
    assert result.converging is True
    assert result.trend == "worsening"


def test_two_rounds_draining_their_predecessor_converge_while_the_count_holds() -> None:
    """Fresh discovery holds the count up while the loop stays productive.

    Each round closes its predecessor's whole backlog and finds one new thing,
    so nothing falls and nothing outpaces, and the loop is plainly working.
    """
    ledger = [
        _row(1, 2, status=FindingStatus.RESOLVED),
        _row(2, 3, status=FindingStatus.RESOLVED),
        _row(3, 3),
    ]

    result = assess_convergence(ledger)

    assert _opens(ledger) == [1, 1, 1]
    assert result.converging is True
    assert result.trend == "steady"


def test_a_last_round_that_drains_and_finds_less_converges_alone() -> None:
    """The single-round form of the pair arm, for a window that opens flat.

    Inside three rounds starting on a plateau neither pair arm can fire, and a
    round that emptied its predecessor's backlog and found less than the round
    before it is the strongest evidence one round can carry.
    """
    ledger = [
        _row(1, 3, status=FindingStatus.RESOLVED),
        _row(1, 3, status=FindingStatus.RESOLVED),
        _row(2, 3, status=FindingStatus.RESOLVED),
        _row(2, 3, status=FindingStatus.RESOLVED),
        _row(3, 3),
    ]

    result = assess_convergence(ledger)

    assert _opens(ledger) == [2, 4, 1]
    assert result.converging is True


def test_a_blocker_the_loop_regressed_vetoes_an_arm_that_fired() -> None:
    """Severity is what the veto keys on, against a ledger that converges.

    Read against a ledger no arm fires for, both severities would answer "not
    converging" and the test would pass for the wrong reason.
    """
    draining = [
        _row(1, 2, status=FindingStatus.RESOLVED),
        _row(2, 3, status=FindingStatus.RESOLVED),
        _row(3, 3),
    ]
    regressed = _row(3, 3, status=FindingStatus.REGRESSED, severity="blocker")
    demoted = _row(3, 3, status=FindingStatus.REGRESSED, severity="major")

    assert assess_convergence([*draining, regressed]).converging is False
    assert assess_convergence([*draining, demoted]).converging is True


def test_a_carried_blocker_the_loop_never_closed_vetoes_the_same_way() -> None:
    """A blocker born earlier and open in the last round is the stuck signal.

    One first raised in the last round is fresh discovery on new surface,
    which is what a converging loop does.
    """
    resolved_pair = [
        _row(1, 2, status=FindingStatus.RESOLVED),
        _row(1, 2, status=FindingStatus.RESOLVED),
        _row(1, 3, status=FindingStatus.RESOLVED),
        _row(4, 4),
    ]
    carried = _row(1, 4, severity="blocker")
    fresh = _row(4, 4, severity="blocker")

    assert assess_convergence([*resolved_pair, carried]).converging is False
    assert assess_convergence([*resolved_pair, fresh]).converging is True


def test_one_assessed_round_reads_as_not_converging() -> None:
    """The verdict is conservative on missing data."""
    result = assess_convergence([_row(1, 1)])

    assert result.converging is False
    assert result.trend is None
    assert [row.round for row in result.drain] == [1]


def test_an_empty_ledger_reads_as_not_converging() -> None:
    result = assess_convergence([])

    assert result.converging is False
    assert result.trend is None
    assert result.drain == ()


def test_a_finding_carried_in_silence_kills_the_full_drain_arms() -> None:
    """Silence puts a floor under the open count that identity arms cannot pass.

    A round carrying an earlier objection forward has not drained its
    predecessor, whatever its counts say, so the arms that read identity go
    quiet while the arms that read counts can still find the loop productive.
    """
    draining = [
        _row(1, 2, status=FindingStatus.RESOLVED),
        _row(2, 3, status=FindingStatus.RESOLVED),
        _row(3, 3),
    ]
    silent = _row(1, 3, message="The objection nobody mentions again.")

    assert assess_convergence(draining).converging is True
    assert assess_convergence([*draining, silent]).converging is False

    # The count arms read the same silence and still see the drain.
    outpacing = [
        _row(1, 4, message="The objection nobody mentions again."),
        _row(1, 2, status=FindingStatus.RESOLVED),
        _row(1, 2, status=FindingStatus.RESOLVED),
        _row(1, 3, status=FindingStatus.RESOLVED),
        _row(4, 4),
    ]
    assert assess_convergence(outpacing).converging is True


def test_a_regression_raises_the_counts_of_the_rounds_it_was_open_through() -> None:
    """The drain is recomputed from current lifecycle state every time.

    Which is why a refund compares two recorded snapshots and never two rows
    of this table: a reader built on the drain would revisit a decision an
    earlier round already made.
    """
    closed = [
        _row(1, 2, status=FindingStatus.RESOLVED, message="Name the checks."),
        _row(2, 2, message="The rollback is uncovered."),
    ]
    assert _opens(closed) == [1, 1]

    # The same two objections, one of them raised again two rounds later. Round
    # two closed nothing after all, and the count it is read as leaving open
    # rises from one to two after the round is over.
    reopened = [
        _row(1, 3, status=FindingStatus.REGRESSED, message="Name the checks."),
        _row(2, 2, message="The rollback is uncovered."),
    ]
    assert _opens(reopened) == [1, 2, 2]


def test_the_evidence_carries_every_round_the_drain_spans() -> None:
    result = assess_convergence(
        [
            _row(1, 2, status=FindingStatus.RESOLVED),
            _row(2, 2),
        ]
    )

    assert drain_evidence(result) == {
        "drain": [
            {"new": 1, "open_after": 1, "resolved": 0, "round": 1},
            {"new": 1, "open_after": 1, "resolved": 1, "round": 2},
        ],
        "trend": "steady",
    }


def test_a_blocker_spelled_differently_still_vetoes() -> None:
    """The veto turns on one severity, so it reads one folded.

    A row whose severity was recorded in another case would otherwise slip
    past the one check that reads a loop as stuck.
    """
    converging = [
        _row(1, 2, status=FindingStatus.RESOLVED, message="Closed at round two."),
        _row(1, 1, message="Still open."),
    ]
    assert assess_convergence(converging).converging is True

    for spelling in ("blocker", "Blocker", "BLOCKER"):
        vetoed = [
            *converging,
            _row(
                1,
                2,
                status=FindingStatus.REGRESSED,
                severity=spelling,
                message="Raised again.",
            ),
        ]
        assert assess_convergence(vetoed).converging is False, spelling
