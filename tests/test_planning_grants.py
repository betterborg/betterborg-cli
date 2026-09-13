"""What a round costs its loop, and what the account says it bought."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from betterborg_cli.planning.grants import (
    PLANNING_GRANT_BUDGET,
    GrantAccount,
    assess_grant,
    planning_grant_account,
)
from betterborg_cli.repository_config import PlanningLimits
from betterborg_cli.store import Borg, Repository, ReviewAssessment, SqliteStore

_BORG = uuid4()


def _recorded(
    *rounds: tuple[int, int | None, bool | None],
    minimum: int = 1,
) -> list[ReviewAssessment]:
    return [
        ReviewAssessment(
            borg_id=_BORG,
            loop="tech_review",
            round=number,
            minimum=minimum,
            converging=False,
            open_findings=snapshot,
            refunded=refunded,
        )
        for number, snapshot, refunded in rounds
    ]


def test_the_unconfigured_budget_is_the_one_the_loops_default_to() -> None:
    assert PlanningLimits().grant_budget == PLANNING_GRANT_BUDGET


def test_a_round_inside_the_minimum_is_neither_charged_nor_refunded() -> None:
    decision = assess_grant(
        review_round=3,
        minimum=3,
        budget=2,
        snapshot=4,
        recorded=_recorded((1, 1, None), (2, 2, None)),
    )

    assert decision.refunded is None
    assert decision.continues is True


def test_a_granted_round_that_closed_something_is_refunded() -> None:
    decision = assess_grant(
        review_round=2,
        minimum=1,
        budget=1,
        snapshot=1,
        recorded=_recorded((1, 2, None)),
    )

    assert decision.refunded is True
    # Refunded, so the budget it did not spend buys the round after it.
    assert decision.continues is True


def test_a_granted_round_that_closed_nothing_spends_the_budget() -> None:
    decision = assess_grant(
        review_round=2,
        minimum=1,
        budget=1,
        snapshot=2,
        recorded=_recorded((1, 2, None)),
    )

    assert decision.refunded is False
    assert decision.continues is False


def test_a_round_whose_predecessor_recorded_nothing_earns_no_refund() -> None:
    """Progress is proven, never presumed.

    A loop whose bookkeeping went missing must not become one that runs for
    free, so the round is counted against the budget rather than refunded.
    """
    decision = assess_grant(
        review_round=3,
        minimum=1,
        budget=2,
        snapshot=1,
        recorded=_recorded((1, 5, None)),
    )

    assert decision.refunded is False
    assert decision.continues is False


def test_a_predecessor_whose_test_was_not_a_count_earns_no_refund() -> None:
    """An empty snapshot is a loop judged some other way, not a count of none."""
    decision = assess_grant(
        review_round=2,
        minimum=1,
        budget=5,
        snapshot=0,
        recorded=_recorded((1, None, None)),
    )

    assert decision.refunded is False


def test_the_budget_counts_the_grants_that_bought_nothing() -> None:
    """Charges and refunds interleave, so the count is of charges alone."""
    recorded = _recorded(
        (1, 4, None),
        (2, 3, True),
        (3, 4, False),
        (4, 3, True),
        (5, 4, False),
    )

    spent = assess_grant(
        review_round=6, minimum=1, budget=3, snapshot=5, recorded=recorded
    )
    assert spent.refunded is False
    assert spent.continues is False

    # The same six rounds with the sixth closing something leave the budget
    # where the fifth round left it, with one charge still to spend.
    earned = assess_grant(
        review_round=6, minimum=1, budget=3, snapshot=3, recorded=recorded
    )
    assert earned.refunded is True
    assert earned.continues is True


def test_a_budget_of_nothing_stops_the_loop_at_its_minimum() -> None:
    """Zero asks for exactly today's rounds and today's terminal states."""
    decision = assess_grant(
        review_round=2,
        minimum=2,
        budget=0,
        snapshot=1,
        recorded=_recorded((1, 3, None)),
    )

    assert decision.refunded is None
    assert decision.continues is False


def test_a_round_already_recorded_is_not_counted_against_itself() -> None:
    """The history a round reads is the rounds before it, never its own row."""
    decision = assess_grant(
        review_round=2,
        minimum=1,
        budget=1,
        snapshot=1,
        recorded=_recorded((1, 2, None), (2, 1, False)),
    )

    assert decision.refunded is True
    assert decision.continues is True


def test_the_account_says_what_the_grants_bought() -> None:
    account = GrantAccount(
        rounds=13, minimum=3, grants=10, charged=10, converging=False
    )

    assert account.sentence() == (
        "The loop took 10 granted rounds past its minimum of 3, 10 of them "
        "closing nothing, and its last round was not converging."
    )


def test_the_account_of_a_loop_that_took_no_grants_says_so() -> None:
    account = GrantAccount(
        rounds=3, minimum=3, grants=0, charged=0, converging=True
    )

    assert account.sentence() == (
        "The loop took no rounds past its minimum of 3, and its last round "
        "was converging."
    )


def test_one_grant_reads_as_one_round() -> None:
    account = GrantAccount(
        rounds=2, minimum=1, grants=1, charged=1, converging=False
    )

    assert "took 1 granted round past its minimum of 1" in account.sentence()


def test_a_loop_too_short_to_judge_claims_nothing_about_its_argument() -> None:
    """One round has nothing to compare against.

    The verdict on it is the assessment being conservative about missing data
    rather than a reading of an argument, so the account does not report it as
    one.
    """
    account = GrantAccount(
        rounds=1, minimum=1, grants=0, charged=0, converging=False
    )

    assert account.sentence() == "The loop took no rounds past its minimum of 1."


def test_the_minimum_an_account_reports_is_the_one_its_rounds_recorded() -> None:
    """A setting edited after a loop stopped does not move what it did.

    Nor does inferring the minimum from the shape of the refunds: a minimum
    raised part-way through a run turns rounds already granted back into rounds
    inside it, so the grants stop being a contiguous tail and the arithmetic
    reports a number nobody ever configured. The rounds recorded what they ran
    under.
    """
    account = GrantAccount(
        rounds=8, minimum=6, grants=3, charged=3, converging=False
    )

    assert account.rounds - account.grants == 5
    assert account.minimum == 6
    assert "past its minimum of 6" in account.sentence()


def test_the_account_reads_the_minimum_its_last_round_recorded(
    tmp_path: Path,
) -> None:
    """A minimum raised part-way through a run leaves the grants discontiguous.

    Rounds already granted become rounds inside the minimum again, so counting
    back from the grants reports a number nobody configured. Each round
    recorded the minimum it ran under, and the last one is the one that stopped.
    """
    repository = Repository(root=tmp_path / "repository")
    borg = Borg(repository_id=repository.id, name="RaisedMinimum")
    with SqliteStore.open(tmp_path / "state.sqlite3") as store:
        with store.transaction():
            store.add_repository(repository)
            store.add_borg(borg)
        for number, minimum, refunded in (
            (1, 3, None),
            (2, 3, None),
            (3, 3, None),
            (4, 3, False),
            (5, 6, None),
            (6, 6, None),
            (7, 6, False),
            (8, 6, False),
        ):
            store.record_review_assessment(
                ReviewAssessment(
                    borg_id=borg.id,
                    loop="tech_review",
                    cycle_id="initial",
                    round=number,
                    minimum=minimum,
                    converging=False,
                    open_findings=2,
                    refunded=refunded,
                )
            )

        account = planning_grant_account(
            store, borg.id, loop="tech_review", cycle_id="initial"
        )

    assert (account.rounds, account.grants, account.charged) == (8, 3, 3)
    # Counting back from the grants would report five, which was never a
    # minimum of anything.
    assert account.rounds - account.grants == 5
    assert account.minimum == 6
    assert "past its minimum of 6" in account.sentence()
