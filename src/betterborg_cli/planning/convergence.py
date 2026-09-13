"""Whether a review loop is closing in on agreement, read from its ledger.

Pure over the lifecycle rows one loop has built: it returns the verdict, the
per-round drain the verdict was read from, and the direction of the last
transition. It is the only place ledger rows become counts, which is what lets
it be tested against ledgers built to fire one arm at a time.

The verdict decides how a granted round is spent, never whether the loop gets
one. Every arm is a positive signal, because reviewers find issues in waves as
they move through the work, so an arm that does not fire never vetoes another.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from betterborg_cli.store import (
    FindingStatus,
    PlanningLedgerFinding,
    TaskLedgerFinding,
)

#: Rounds of drain history the verdict reads, which is the tail of the loop.
CONVERGENCE_WINDOW = 3

#: The states an objection still costs its loop in.
_OPEN_STATUSES = frozenset({FindingStatus.OPEN, FindingStatus.REGRESSED})

#: Resolution is the only close this product has, so one set serves both
#: readings the drain needs: the event a round is credited with, and the state
#: that stops the row counting as open in every round after it.
_CLOSED_STATUSES = frozenset({FindingStatus.RESOLVED})

LedgerRow = PlanningLedgerFinding | TaskLedgerFinding

#: One row's ``(first_seen_round, last_seen_round, status, severity)``, which
#: is everything the drain and the veto read.
_Fields = tuple[int, int, FindingStatus, str]


@dataclass(frozen=True, slots=True)
class DrainRow:
    """One round's drain.

    ``new`` are the objections first raised that round, ``resolved`` the ones
    whose resolution that round established, and ``open_after`` the ones born
    by that round and still standing after its reconciliation.
    """

    round: int
    resolved: int
    new: int
    open_after: int


@dataclass(frozen=True, slots=True)
class ConvergenceResult:
    """A deterministic read of whether a loop was closing in.

    ``trend`` is the direction of the last transition, and ``drain`` carries
    the per-round counts the verdict was read from.
    """

    converging: bool
    trend: str | None
    drain: tuple[DrainRow, ...]


def assess_convergence(
    ledger: Iterable[LedgerRow], *, window: int = CONVERGENCE_WINDOW
) -> ConvergenceResult:
    """Decide from a loop's ledger whether its argument is closing in.

    Converging when, over the last ``window`` rounds, any of these holds:
    ``open_after`` strictly decreases; two consecutive rounds resolve more
    than they raise; two consecutive rounds each fully drain their
    predecessor's non-empty backlog, which keeps a loop productive while fresh
    discovery holds ``open_after`` up; or the last round both fully drains its
    predecessor and raises strictly fewer findings than the round before it.

    One veto overrides all of them: an open blocker in the last round that the
    loop already tried and failed to close. Fewer than two assessed rounds
    reads as not converging, because the verdict is conservative on missing
    data.
    """

    fields = [
        (
            row.first_seen_round,
            row.last_seen_round,
            row.status,
            # Folded before it is compared, as the source folds it: the veto
            # turns on one severity, and a row that spelled it differently
            # would slip past the one check that reads a loop as stuck.
            row.severity.lower(),
        )
        for row in ledger
    ]
    drain = _compute_drain(fields)
    if len(drain) < 2:
        return ConvergenceResult(False, None, tuple(drain))

    last_round = drain[-1].round
    # Only a blocker the loop already tried and failed to close vetoes: one
    # born before the last round and still open there, or one regressed, which
    # is a fix the reviewer says did not hold. A blocker first raised in the
    # last round is fresh discovery on new surface, which is what a converging
    # loop does.
    repeated_blockers = sum(
        1
        for first_seen, last_seen, status, severity in fields
        if severity == "blocker"
        and status in _OPEN_STATUSES
        and last_seen == last_round
        and (first_seen < last_round or status is FindingStatus.REGRESSED)
    )
    trend = (
        "improving"
        if drain[-1].open_after < drain[-2].open_after
        else "worsening"
        if drain[-1].open_after > drain[-2].open_after
        else "steady"
    )

    tail = drain[-window:] if window > 0 else drain
    opens = [row.open_after for row in tail]
    # A window of one round has no transition in it, so it decreases in the
    # same empty way an unassessed loop converges: not at all.
    strictly_decreasing = len(opens) >= 2 and all(
        later < earlier
        for earlier, later in zip(opens, opens[1:], strict=False)
    )
    outpacing = [row.resolved > row.new for row in tail]
    two_consecutive_outpacing = any(
        earlier and later
        for earlier, later in zip(outpacing, outpacing[1:], strict=False)
    )
    # The full-drain arms read identity, not count: an equal count of findings
    # raised and closed inside one round must not stand in for its
    # predecessor's actual backlog, or a loop stuck on a fixed set of
    # objections while churning same-round discoveries would read as
    # productive. The flag belongs to the later round of each pair, and it is
    # computed over the whole drain before the window is taken so that the
    # window's leading pair, whose predecessor sits outside the slice,
    # survives the restriction.
    full_drain = [False] + [
        drain[index - 1].open_after > 0
        and not _carries_open(fields, drain[index].round)
        for index in range(1, len(drain))
    ]
    tail_full_drain = full_drain[len(drain) - len(tail) :]
    two_consecutive_full_drain = any(
        earlier and later
        for earlier, later in zip(
            tail_full_drain, tail_full_drain[1:], strict=False
        )
    )
    # The single-round form of the pair arm. Inside a three-round window that
    # opens on a plateau neither pair arm can fire, and a last round that both
    # emptied its predecessor's backlog and found less to raise is the
    # strongest evidence one round can carry.
    last_drains_and_shrinks = full_drain[-1] and drain[-1].new < drain[-2].new
    converging = (
        strictly_decreasing
        or two_consecutive_outpacing
        or two_consecutive_full_drain
        or last_drains_and_shrinks
    ) and repeated_blockers == 0
    return ConvergenceResult(converging, trend, tuple(drain))


def drain_evidence(result: ConvergenceResult) -> dict[str, Any]:
    """Render one verdict as the evidence its durable record carries."""

    return {
        "drain": [
            {
                "new": row.new,
                "open_after": row.open_after,
                "resolved": row.resolved,
                "round": row.round,
            }
            for row in result.drain
        ],
        "trend": result.trend,
    }


def _compute_drain(fields: Sequence[_Fields]) -> list[DrainRow]:
    """Project the per-round drain over the rounds the ledger spans.

    Read from current lifecycle state, so an objection that regressed reads as
    open throughout, which is the conservative reconstruction — and the reason
    a refund compares two recorded snapshots rather than two of these rows.
    """

    if not fields:
        return []
    first = min(first_seen for first_seen, _, _, _ in fields)
    last = max(last_seen for _, last_seen, _, _ in fields)
    drain: list[DrainRow] = []
    for number in range(first, last + 1):
        new = sum(1 for first_seen, _, _, _ in fields if first_seen == number)
        resolved = sum(
            1
            for _, last_seen, status, _ in fields
            if last_seen == number and status in _CLOSED_STATUSES
        )
        open_after = sum(
            1
            for first_seen, last_seen, status, _ in fields
            if first_seen <= number
            and not (status in _CLOSED_STATUSES and last_seen <= number)
        )
        drain.append(
            DrainRow(
                round=number, resolved=resolved, new=new, open_after=open_after
            )
        )
    return drain


def _carries_open(fields: Sequence[_Fields], round_number: int) -> bool:
    """Return whether any earlier round's objection survives this round."""

    return any(
        first_seen < round_number
        and not (status in _CLOSED_STATUSES and last_seen <= round_number)
        for first_seen, last_seen, status, _ in fields
    )


__all__ = [
    "CONVERGENCE_WINDOW",
    "ConvergenceResult",
    "DrainRow",
    "LedgerRow",
    "assess_convergence",
    "drain_evidence",
]
