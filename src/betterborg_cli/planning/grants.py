"""How long a review loop may grind without closing anything.

A loop's configured rounds are a minimum. Every round past it is a grant, and
a grant is issued whenever the budget holds, whether or not the findings are
draining: draining decides how a granted round is spent, not whether it
happens. A grant that closed nothing is charged against the budget and one
that closed something is refunded, so the budget bounds how long a loop may
grind, never how long it may run.

The loop stops when its charged grants reach the budget. Counting from the
other end gives the same number and is the shape the loops' own checks
already had: attempts, minus refunds, against the minimum plus the budget.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from betterborg_cli.store import ReviewAssessment, SqliteStore

#: Grants a planning loop gets where its repository configures no budget. Ten,
#: because a loop oscillating near the finish line parks on a technicality for
#: several rounds while staying blocker-free, and a budget of three ends it
#: there: the counts 1, 2, 2, 3 spend a budget of three without the loop ever
#: being in real trouble. Zero is legal and asks for exactly the minimum and
#: today's terminal states, since the first round past the minimum then has no
#: budget to come out of.
PLANNING_GRANT_BUDGET = 10

#: Grants a task's review gets where its repository configures no budget. Ten,
#: for the reason ``PLANNING_GRANT_BUDGET`` gives: one number bounds every loop
#: in the same way, and zero is legal here on the same terms.
EXECUTION_GRANT_BUDGET = 10

#: The loop a task's review rounds record themselves under. Named rather than
#: spelled at each reader, because the account and the round that writes it have
#: to agree on it or the budget reads an empty history.
TASK_REVIEW_LOOP = "task_review"


@dataclass(frozen=True, slots=True)
class GrantDecision:
    """What one assessed round did to its loop's budget.

    ``refunded`` is ``None`` on a round inside the minimum, which is neither
    charged nor refunded because it is not a grant.
    """

    refunded: bool | None
    continues: bool


@dataclass(frozen=True, slots=True)
class GrantAccount:
    """What a loop's rounds cost it, for a gate with no store left to read.

    Every number here is read off the record and none off configuration, which
    is what makes a re-run explain a stopped loop the same way the run that
    stopped it did. A setting edited afterwards does not move a plan that has
    already blocked, so it must not move the account of why it blocked either —
    which is why the minimum is the one the last round ran under rather than the
    one in force now.
    """

    rounds: int
    minimum: int
    grants: int
    charged: int
    converging: bool | None

    def sentence(self) -> str:
        """Say in one line what the rounds cost and how the last one read.

        A loop with one round has nothing to compare it against, so it makes no
        claim about which way its argument was going: the verdict on that round
        is the conservative default rather than a reading of anything.
        """

        if self.grants == 0:
            spent = f"took no rounds past its minimum of {self.minimum}"
        else:
            rounds = "round" if self.grants == 1 else "rounds"
            spent = (
                f"took {self.grants} granted {rounds} past its minimum of "
                f"{self.minimum}, {self.charged} of them closing nothing"
            )
        if self.rounds < 2:
            return f"The loop {spent}."
        verdict = "was converging" if self.converging else "was not converging"
        return f"The loop {spent}, and its last round {verdict}."


def assess_grant(
    *,
    review_round: int,
    minimum: int,
    budget: int,
    snapshot: int,
    recorded: Sequence[ReviewAssessment],
) -> GrantDecision:
    """Decide what this round cost its loop and whether another one follows.

    ``snapshot`` is the count of objections still open after this round's
    reconciliation, and a granted round earns its refund by coming in strictly
    below its predecessor's recorded count. The comparison is between two
    snapshots and never between two rows of a drain: a drain is recomputed from
    current lifecycle state every time it is read, so an objection that
    regresses raises its earlier rounds' counts after the fact.

    A round whose predecessor recorded no answer earned no refund. Progress is
    proven, never presumed, so a loop whose bookkeeping went missing must not
    become one that runs for free.
    """

    history = [item for item in recorded if item.round < review_round]
    refunded: bool | None = None
    if review_round > minimum:
        previous = next(
            (item for item in history if item.round == review_round - 1), None
        )
        refunded = (
            previous is not None
            and previous.open_findings is not None
            and snapshot < previous.open_findings
        )
    refunds = sum(1 for item in history if item.refunded) + bool(refunded)
    return GrantDecision(
        refunded=refunded,
        continues=review_round - refunds < minimum + budget,
    )


def planning_grant_account(
    store: SqliteStore,
    borg_id: UUID,
    *,
    loop: str,
    cycle_id: str | None = None,
    plan_approval_id: UUID | None = None,
) -> GrantAccount:
    """Account for what one planning loop's rounds cost it."""

    return grant_account(
        store.list_review_assessments(
            borg_id,
            loop=loop,
            cycle_id=cycle_id,
            plan_approval_id=plan_approval_id,
        )
    )


def execution_grant_account(
    store: SqliteStore, borg_id: UUID, *, task_id: UUID
) -> GrantAccount:
    """Account for what one task's review passes cost it.

    Scoped to the task as well as the loop, because the configured minimum is
    shared by every task in the run and cannot carry one task's grants.
    """

    return grant_account(
        store.list_review_assessments(
            borg_id, loop=TASK_REVIEW_LOOP, task_id=task_id
        )
    )


def grant_account(recorded: Sequence[ReviewAssessment]) -> GrantAccount:
    """Total up the rounds one loop has assessed, and nothing but them.

    The one place assessments become an account, so a reader that wants to know
    what a loop's rounds cost never counts them itself. A round that has been
    assessed but not yet written is one of its own rounds too: the reason a loop
    gives for stopping accounts for the round it stopped on.
    """

    grants = [item for item in recorded if item.refunded is not None]
    latest = max(recorded, key=lambda item: item.round, default=None)
    return GrantAccount(
        rounds=len(recorded),
        minimum=0 if latest is None else latest.minimum,
        grants=len(grants),
        charged=sum(1 for item in grants if not item.refunded),
        converging=None if latest is None else latest.converging,
    )


__all__ = [
    "EXECUTION_GRANT_BUDGET",
    "PLANNING_GRANT_BUDGET",
    "TASK_REVIEW_LOOP",
    "GrantAccount",
    "GrantDecision",
    "assess_grant",
    "execution_grant_account",
    "grant_account",
    "planning_grant_account",
]
