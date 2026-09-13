"""Reviewer objections whose lifecycle outlives the round that raised them.

A finding table is an immutable per-round snapshot of what a reviewer said.
The ledger beside it is the current answer to a different question: which
objections the work under review still has to answer. One objection is one
row for as long as its loop argues about it, so a later round can close it,
carry it, or raise it again without the loop losing track of which of its
rounds have already tried.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any, TypeVar
from uuid import UUID

from betterborg_cli.planning.cycles import current_planning_cycle_id
from betterborg_cli.store import (
    ExecutionLedgerFinding,
    FindingStatus,
    PlanningFinding,
    PlanningLedgerFinding,
    SqliteStore,
    TaskFinding,
    TaskLedgerFinding,
)

#: A finding's declaration that it raises an objection the ledger already
#: holds, or nothing. Required but nullable, and expressed without a
#: composition keyword on purpose: one provider transport normalizes schemas
#: into a strict subset and falls back to a schema described in the prompt on
#: any of them, losing provider-side enforcement of every field at once.
REPEATS_SCHEMA: dict[str, Any] = {
    "type": ["string", "null"],
    "minLength": 1,
    "pattern": r"\S",
}

#: The ids a review considers closed. Required unconditionally, which is what
#: forces a reviewer to answer rather than omit: one that closed nothing sends
#: an empty list.
RESOLVED_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {"type": "string", "minLength": 1, "pattern": r"\S"},
}

_LedgerRow = TypeVar(
    "_LedgerRow",
    ExecutionLedgerFinding,
    PlanningLedgerFinding,
    TaskLedgerFinding,
)


def open_findings(rows: Iterable[_LedgerRow]) -> list[_LedgerRow]:
    """Return the rows a review loop still has to answer for."""

    return [row for row in rows if row.status is not FindingStatus.RESOLVED]


def open_planning_findings(
    store: SqliteStore, borg_id: UUID
) -> list[PlanningLedgerFinding]:
    """Return the Tech Lead objections the current plan still has to answer."""

    return open_findings(
        store.list_planning_ledger_findings(
            borg_id, cycle_id=current_planning_cycle_id(store, borg_id)
        )
    )


def open_task_findings(
    store: SqliteStore, borg_id: UUID, plan_approval_id: UUID
) -> list[TaskLedgerFinding]:
    """Return the Supervisor objections this plan approval still has to answer.

    One ledger spans every batch of the approval. A per-batch one would arrive
    empty on every round, because each revision mints a batch of its own.
    """

    return open_findings(
        store.list_task_ledger_findings(
            borg_id, plan_approval_id=plan_approval_id
        )
    )


def open_execution_findings(
    store: SqliteStore, task_id: UUID
) -> list[ExecutionLedgerFinding]:
    """Return the review objections this task's commit still has to answer.

    One ledger spans the task's review rounds, so a finding an earlier round
    raised and no later round repeated is still in front of whoever answers it.
    """

    return open_findings(store.list_execution_ledger_findings(task_id))


def planning_ledger_json(row: PlanningLedgerFinding) -> dict[str, Any]:
    """Render one open Tech Lead objection as context an agent can act on."""

    return {
        "first_raised_in_round": row.first_seen_round,
        "id": str(row.id),
        "message": row.message,
        "severity": row.severity,
        "suggestion": row.suggestion,
    }


def task_ledger_json(row: TaskLedgerFinding) -> dict[str, Any]:
    """Render one open Supervisor objection with its reference as a label.

    Every Project Manager revision mints fresh task references, so a carried
    row's reference names where the objection started and never a task in the
    batch in hand. A reviewer that copies it into a finding of its own trips
    the refusal that guards the batch under review, so it is named for what it
    is rather than as a reference to reuse.
    """

    return {
        "first_raised_in_round": row.first_seen_round,
        "id": str(row.id),
        "message": row.message,
        "raised_against_batch_id": str(row.batch_id),
        "raised_against_task_ref": row.task_ref,
        "severity": row.severity,
        "suggestion": row.suggestion,
    }


def reconcile_planning_ledger(
    existing: Sequence[PlanningLedgerFinding],
    *,
    findings: Sequence[tuple[PlanningFinding, str | None]],
    resolved: Sequence[str],
    attempt_id: UUID,
    cycle_id: str,
    review_round: int,
    approved: bool,
) -> list[PlanningLedgerFinding]:
    """Fold one Tech Lead review into the ledger its cycle has been building."""

    def build(
        finding: PlanningFinding, status: FindingStatus
    ) -> PlanningLedgerFinding:
        return PlanningLedgerFinding(
            id=finding.id,
            borg_id=finding.borg_id,
            cycle_id=cycle_id,
            attempt_id=attempt_id,
            first_seen_round=review_round,
            last_seen_round=review_round,
            status=status,
            severity=finding.severity,
            message=finding.message,
            suggestion=finding.suggestion,
            created_at=finding.created_at,
        )

    return _reconcile(
        existing,
        findings=findings,
        resolved=resolved,
        attempt_id=attempt_id,
        review_round=review_round,
        approved=approved,
        build=build,
    )


def reconcile_task_ledger(
    existing: Sequence[TaskLedgerFinding],
    *,
    findings: Sequence[tuple[TaskFinding, str | None]],
    resolved: Sequence[str],
    attempt_id: UUID,
    plan_approval_id: UUID,
    review_round: int,
    approved: bool,
) -> list[TaskLedgerFinding]:
    """Fold one Supervisor review into the ledger its plan approval is building."""

    def build(finding: TaskFinding, status: FindingStatus) -> TaskLedgerFinding:
        return TaskLedgerFinding(
            id=finding.id,
            borg_id=finding.borg_id,
            plan_approval_id=plan_approval_id,
            batch_id=finding.batch_id,
            attempt_id=attempt_id,
            first_seen_round=review_round,
            last_seen_round=review_round,
            status=status,
            severity=finding.severity,
            message=finding.message,
            suggestion=finding.suggestion,
            task_ref=finding.task_ref,
            created_at=finding.created_at,
        )

    return _reconcile(
        existing,
        findings=findings,
        resolved=resolved,
        attempt_id=attempt_id,
        review_round=review_round,
        approved=approved,
        build=build,
    )


def reconcile_execution_ledger(
    existing: Sequence[ExecutionLedgerFinding],
    *,
    findings: Sequence[tuple[ExecutionLedgerFinding, str | None]],
    resolved: Sequence[str],
    attempt_id: UUID,
    review_round: int,
    approved: bool,
) -> list[ExecutionLedgerFinding]:
    """Fold one execution review into the ledger its task has been building.

    A declared finding arrives as the row this round would record, because the
    review attempt's own payload is the immutable record here and there is no
    snapshot row to build one from. What the round it belongs to and the attempt
    that produced it are is still the reconciler's to say, as it is for the two
    ledgers whose rows it builds itself: one owner for the numbers a later round
    reads back.
    """

    def build(
        finding: ExecutionLedgerFinding, status: FindingStatus
    ) -> ExecutionLedgerFinding:
        return replace(
            finding,
            status=status,
            first_seen_round=review_round,
            last_seen_round=review_round,
            attempt_id=attempt_id,
        )

    return _reconcile(
        existing,
        findings=findings,
        resolved=resolved,
        attempt_id=attempt_id,
        review_round=review_round,
        approved=approved,
        build=build,
    )


def _reconcile(
    existing: Sequence[_LedgerRow],
    *,
    findings: Sequence[
        tuple[ExecutionLedgerFinding | PlanningFinding | TaskFinding, str | None]
    ],
    resolved: Sequence[str],
    attempt_id: UUID,
    review_round: int,
    approved: bool,
    build: Callable[[Any, FindingStatus], _LedgerRow],
) -> list[_LedgerRow]:
    """Return the whole ledger this round leaves behind, in stable order."""

    held = {row.id: row for row in existing}
    rows: dict[UUID, _LedgerRow] = dict(held)
    touched: set[UUID] = set()

    def establish(row_id: UUID, status: FindingStatus) -> None:
        rows[row_id] = replace(
            rows[row_id],
            status=status,
            last_seen_round=review_round,
            attempt_id=attempt_id,
        )
        touched.add(row_id)

    # A review closes what it says it closed. An id the ledger cannot place
    # leaves the row it meant open, which costs the loop a grant and lets no
    # blocker through. An id naming a row that is already closed re-establishes
    # nothing, so the round that first closed it keeps the credit.
    for declaration in resolved:
        target = _held(held, declaration)
        if target is not None and target.status is not FindingStatus.RESOLVED:
            establish(target.id, FindingStatus.RESOLVED)

    for finding, declaration in findings:
        target = _held(held, declaration)
        if target is not None:
            # One objection is one row for as long as the loop argues about
            # it, so the row keeps the severity and message it was first
            # recorded with and a repeat cannot escalate a minor into a
            # blocker. Two findings naming one row are two restatements of the
            # objection it holds, and they move it once. A review that resolves
            # and repeats the same id has contradicted itself, and the repeat
            # wins.
            establish(target.id, FindingStatus.REGRESSED)
            continue
        # The pointer moves the right row; the claim stands without it. A
        # repeat the ledger cannot place is still the assertion that this
        # objection has been raised before and not closed, which is what the
        # blocker veto reads.
        rows[finding.id] = build(
            finding,
            FindingStatus.REGRESSED
            if declaration is not None
            else FindingStatus.OPEN,
        )
        touched.add(finding.id)

    # Silence is not agreement. An objection nobody mentioned again is carried
    # forward as open, which re-establishes its status in this round: a row the
    # loop regressed earlier and has argued about since is open on the strength
    # of this round, not of the round that regressed it.
    for row_id, row in list(rows.items()):
        if row_id not in touched and row.status is not FindingStatus.RESOLVED:
            establish(row_id, FindingStatus.OPEN)

    # Approval closes the ledger: a reviewer that says the work is ready has
    # answered every objection still standing, including one its own findings
    # regressed.
    if approved:
        for row_id, row in list(rows.items()):
            if row.status is not FindingStatus.RESOLVED:
                establish(row_id, FindingStatus.RESOLVED)

    return list(rows.values())


def _held(
    held: Mapping[UUID, _LedgerRow], declaration: str | None
) -> _LedgerRow | None:
    """Return the row an id names, or nothing when the ledger cannot place it."""

    if declaration is None:
        return None
    try:
        # Stripped before it is parsed: a padded id reads as one the ledger
        # cannot place, which would leave the row it names open and add a
        # second row for the objection it already holds.
        identifier = UUID(declaration.strip())
    except ValueError:
        return None
    return held.get(identifier)
