"""Which planning cycle a Borg is in, and what names it.

A cycle is not a row anywhere: it is the run of attempts since the latest human
request to change the plan. The attempts a cycle holds and the name its scoped
rows carry are both read off that one request, so they are read in one place.
"""

from __future__ import annotations

from uuid import UUID

from betterborg_cli.store import PlanChangeRequest, SqliteStore

#: The cycle a scoped row carries before any plan change request opens a
#: second one. A null does not compare equal to itself, so a lookup keyed on
#: the scope would miss the commonest case there is.
INITIAL_PLANNING_CYCLE = "initial"


def current_planning_cycle(
    store: SqliteStore, borg_id: UUID
) -> PlanChangeRequest | None:
    """Return the change request that opened the current cycle, where one did."""

    change_requests = store.list_plan_change_requests(borg_id)
    return change_requests[-1] if change_requests else None


def current_planning_cycle_id(store: SqliteStore, borg_id: UUID) -> str:
    """Name the cycle a scoped row belongs to."""

    opened_by = current_planning_cycle(store, borg_id)
    return INITIAL_PLANNING_CYCLE if opened_by is None else str(opened_by.id)
