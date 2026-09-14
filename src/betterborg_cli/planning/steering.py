"""The paragraph a stalled review loop's answerer is handed.

Every stage before this one buys a loop more rounds. This is what makes an
extra round worth having: a loop that has circled the same objection three
times does not need a fourth turn at the same prompt, it needs telling which
objection is the one that matters, or that what the reviewer keeps asking for
is not what the work was for.

The note goes to the agent that can act on it and never to the reviewer, which
meets the result rather than the instruction. Writing it is an agent turn like
any other, resolved through the steering stage's own configuration, and what it
is given is the argument itself: the open ledger, what each round raised and
closed, and each round's own account of what it decided.

Two things can go wrong with it and only one is the turn failing. A turn that
gives back nothing usable falls back to a note assembled from the ledger
without an agent. So does a turn that succeeds without being confident of its
own answer: the costs are not symmetric, because a merely plausible note
misdirects a round that was bought, where no note at all leaves the answerer
exactly where an unsteered round would have left it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from betterborg_cli.agent_runtime.base import AgentAdapter, CancellationToken
from betterborg_cli.agent_runtime.selection import SelectedAgent
from betterborg_cli.planning.convergence import LedgerRow, assess_convergence
from betterborg_cli.planning.findings_ledger import open_findings
from betterborg_cli.planning.turns import DurablePlanningTurns, PlanningProgress
from betterborg_cli.store import (
    Borg,
    PlanningAttemptStatus,
    Repository,
    ReviewAssessment,
    SqliteStore,
    SteeringNote,
    SteeringNoteSource,
)

#: The phase a steering turn's attempt is filed under, in planning and in
#: execution alike. Its own name in both halves, because the execution driver
#: selects on the phase string to replay a completed attempt as a round's
#: outcome and to decide which attempts declare a commit: filed under "review"
#: or "fix" a steering attempt would be replayed as that round's result or read
#: as declaring the fix's commit, and under its own name both ignore it.
STEERING_PHASE = "steering"

#: Confidences a steering turn can report. Only the first is attached, because
#: a confident note about the wrong objection spends the round it was granted
#: arguing the loop further off course.
_STEERING_CONFIDENCES = ("high", "medium", "low")

STEERING_NOTE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["note", "confidence"],
    "properties": {
        "note": {"type": "string", "minLength": 1, "pattern": r"\S"},
        "confidence": {"type": "string", "enum": list(_STEERING_CONFIDENCES)},
    },
}

#: Betterborg's own subject rather than the repository's, so the instructions
#: are a constant here rather than a generated per-repository prompt: what this
#: turn needs to know is the shape of a stalled argument.
STEERING_SYSTEM_PROMPT = """You are reading a review loop that is not closing \
in on agreement, and writing the one paragraph a person watching the argument \
would have handed to whoever answers the findings next.

You are given the objections still open, what each round raised and closed, \
and each round's own account of what it decided. You have no other evidence \
and you need none: the argument is the subject. Do not modify any file.

Write the note to the agent that will answer the findings, not to the \
reviewer. Say which open objection is the one that actually matters, or say \
that what the reviewer keeps asking for is not what the work was for and what \
should be done about that. Be specific about objections you name, and keep it \
to a short paragraph: it is read alongside the findings themselves, so repeat \
none of them.

Report how confident you are of the note as high, medium or low, and report it \
as you find it rather than to get the note used. A note that confidently names \
the wrong objection spends the round it was written for; a note the loop can \
tell is uncertain costs nothing. Say medium or low whenever the argument does \
not clearly point at one answer. Return only the required JSON object.
"""


@dataclass(frozen=True, slots=True)
class SteeringScope:
    """The run of rounds one note belongs to.

    Carried the way the assessments it is read beside carry theirs: a field per
    kind of scope, filled only where its loop has one.
    """

    loop: str
    cycle_id: str | None = None
    plan_approval_id: UUID | None = None
    task_id: UUID | None = None

    def request_context(self) -> dict[str, Any]:
        """Name this scope in a durable turn's request context."""

        context: dict[str, Any] = {"loop": self.loop}
        if self.cycle_id is not None:
            context["cycle_id"] = self.cycle_id
        if self.plan_approval_id is not None:
            context["plan_approval_id"] = str(self.plan_approval_id)
        if self.task_id is not None:
            context["task_id"] = str(self.task_id)
        return context


@dataclass(frozen=True, slots=True)
class SteeringSubject:
    """Who the note is written for, and what their loop is arguing about."""

    answerer: str
    work: str


def steering_verdict(
    recorded: Sequence[ReviewAssessment],
) -> ReviewAssessment | None:
    """Return the assessment asking for the round after it to be steered.

    The round that decides is not the round that is steered. A round is steered
    when its predecessor called the loop stuck and the round it leads into is a
    grant, which is one round earlier than the test a refund records: ``refunded``
    is set where a round is itself past the minimum, and the round after the
    minimum's last round is the first grant there is.

    The verdict and the minimum it is measured against come off the recorded row
    rather than from configuration in force now, as every other reader of these
    rows does, so a setting edited since cannot steer a round that has already
    run.
    """

    latest = max(recorded, key=lambda item: item.round, default=None)
    if latest is None or latest.converging or latest.round < latest.minimum:
        return None
    return latest


def recorded_steering_note(
    store: SqliteStore,
    borg_id: UUID,
    *,
    scope: SteeringScope,
    round_number: int,
) -> SteeringNote | None:
    """Return the note already written for one granted round.

    What makes a resumed round cheap: a run interrupted inside a steered
    revision comes back into the same round, finds its note, and runs on that
    rather than paying for a second turn.
    """

    return next(
        (
            item
            for item in store.list_steering_notes(
                borg_id,
                loop=scope.loop,
                cycle_id=scope.cycle_id,
                plan_approval_id=scope.plan_approval_id,
                task_id=scope.task_id,
            )
            if item.round == round_number
        ),
        None,
    )


def build_steering_note(
    *,
    borg_id: UUID,
    scope: SteeringScope,
    verdict: ReviewAssessment,
    note: str,
    source: SteeringNoteSource,
    attempt_id: UUID | None = None,
) -> SteeringNote:
    """Build the row a steered round records, keyed to the round it steers.

    ``converging`` is the verdict that asked for the note, and today only one
    verdict ever does: the trigger returns nothing where a loop is closing in,
    so no row this writes can hold ``True``. It is read off the assessment
    rather than written as a constant because what the row says it was steered
    for should stay true of a trigger that one day steers on other terms.
    """

    return SteeringNote(
        borg_id=borg_id,
        loop=scope.loop,
        cycle_id=scope.cycle_id,
        plan_approval_id=scope.plan_approval_id,
        task_id=scope.task_id,
        round=verdict.round,
        attempt_id=attempt_id,
        note=note,
        source=source,
        converging=verdict.converging,
    )


def confident_steering_note(payload: Mapping[str, Any]) -> str | None:
    """Return a steering turn's note where it is confident of it."""

    if str(payload.get("confidence")) != "high":
        return None
    note = str(payload.get("note") or "").strip()
    return note or None


def assembled_steering_note(
    ledger: Sequence[LedgerRow], *, round_number: int
) -> str:
    """Assemble the note a round falls back to, from the ledger alone.

    Every answerer already has this ledger — the fixer in its prompt, the
    Project Manager and the Architect in files their worktrees publish — so
    this adds little to a prompt and is not meant to. What it is for is the
    record: the answering turn is no worse off than an unsteered one either
    way, and a round that attached something can say which note it used, where
    a round that attached nothing cannot be told from one that was never
    steered.
    """

    rows = open_findings(ledger)
    lines = [
        "This round was granted because the review is not closing its "
        "findings, and no steering note could be written for it. These "
        "objections are still open:",
        "",
    ]
    if rows:
        lines.extend(
            f"- {row.message} ({row.severity}, first raised in round "
            f"{row.first_seen_round} and open for "
            f"{_rounds(round_number - row.first_seen_round + 1)})"
            for row in rows
        )
    else:
        lines.append("- none the ledger still holds open.")
    return "\n".join(lines)


def _rounds(count: int) -> str:
    return f"{count} round" if count == 1 else f"{count} rounds"


def render_steering_prompt(
    *,
    subject: SteeringSubject,
    verdict: ReviewAssessment,
    ledger: Sequence[LedgerRow],
    summaries: Sequence[tuple[int, str]],
) -> str:
    """Render the stalled argument a steering turn reads and nothing else.

    Which arm of the assessment fired, and whether its veto is what stopped it,
    are not here: neither survives into the result the assessment returns. The
    ledger shows the same thing anyway, because a blocker the loop raised early
    and has not closed is visible as a row.
    """

    drain = assess_convergence(ledger).drain
    sections = [
        f"{subject.answerer} is about to answer these findings again, on a "
        f"round granted because the review of {subject.work} is not closing "
        "in on agreement.",
        "",
        f"Rounds argued so far: {verdict.round}.",
        "",
        "## Objections still open",
        "",
    ]
    rows = open_findings(ledger)
    if rows:
        sections.extend(
            f"- {row.message} ({row.severity}, first raised in round "
            f"{row.first_seen_round})"
            + (f" (suggestion: {row.suggestion})" if row.suggestion else "")
            for row in rows
        )
    else:
        sections.append("- none the ledger still holds open.")
    sections.extend(["", "## What each round raised and closed", ""])
    if drain:
        sections.extend(
            f"- Round {row.round}: raised {row.new}, closed {row.resolved}, "
            f"{row.open_after} still open afterwards."
            for row in drain
        )
    else:
        sections.append("- no rounds have been reconciled yet.")
    sections.extend(["", "## What each round decided", ""])
    if summaries:
        sections.extend(
            f"- Round {number}: {summary}" for number, summary in summaries
        )
    else:
        sections.append("- no round recorded a summary.")
    return "\n".join(sections).rstrip() + "\n"


class PlanningSteeringTurn:
    """One steered planning round's note, written through a turn of its own.

    Built once per steered round and never once per loop: the durable-turn
    machinery binds its agent, its model and the progress child it emits into
    at construction, so a loop reusing the runner it already has would write
    the note on the reviewer's own agent and ignore the steering configuration
    entirely.
    """

    def __init__(
        self,
        repository: Repository,
        borg: Borg,
        store: SqliteStore,
        agent: AgentAdapter | SelectedAgent,
        *,
        scope: SteeringScope,
        subject: SteeringSubject,
        model: str,
        artifact_dir: Path,
        error_factory: type[Exception],
        cancelled_error_factory: type[Exception],
        cancel: CancellationToken | None = None,
        progress: PlanningProgress | None = None,
        stage_key: str | None = None,
        child_key: str | None = None,
        dirty_borg_documents: Sequence[Path] = (),
        worktrees_root: Path | None = None,
    ) -> None:
        self.store = store
        self.borg_id = borg.id
        self.scope = scope
        self.subject = subject
        self.cancel = cancel
        self._error = error_factory
        self._cancelled_error = cancelled_error_factory
        self._turns = DurablePlanningTurns(
            repository,
            borg,
            store,
            agent,
            role="Steering",
            model=model,
            artifact_dir=artifact_dir,
            error_factory=error_factory,
            cancelled_error_factory=cancelled_error_factory,
            cancel=cancel,
            progress=progress,
            stage_key=stage_key,
            child_key=child_key,
            dirty_borg_documents=dirty_borg_documents,
            worktrees_root=worktrees_root,
        )

    def note(
        self,
        *,
        verdict: ReviewAssessment,
        ledger: Sequence[LedgerRow],
        summaries: Sequence[tuple[int, str]],
    ) -> str:
        """Write the note one granted round runs on, and return it.

        For a round that has none: the loop reads ``recorded_steering_note``
        before it builds one of these, because the same answer decides whether
        a progress child is worth declaring at all. Reaching here twice for one
        round is refused by the row's own uniqueness rather than quietly
        doubled.
        """

        assembled = assembled_steering_note(ledger, round_number=verdict.round)
        try:
            attempt, payload = self._turns.run(
                phase=STEERING_PHASE,
                # The steering phase's own running count, and never the round
                # the note steers: attempts are unique per phase and round, so
                # a ledger round passed for one would collide on a second
                # planning cycle.
                round_number=self._turns.next_round(STEERING_PHASE),
                schema=STEERING_NOTE_SCHEMA,
                system_prompt=STEERING_SYSTEM_PROMPT,
                user_prompt=render_steering_prompt(
                    subject=self.subject,
                    verdict=verdict,
                    ledger=ledger,
                    summaries=summaries,
                ),
                turn_name="steering note",
                # Naming the round it steers is what keeps the pairing honest:
                # an open attempt whose context disagrees with the caller's is
                # failed and replaced rather than replayed, so an orphan left
                # by a crashed round cannot hand the next round the note
                # written for the last one.
                request_context={
                    **self.scope.request_context(),
                    "steered_round": verdict.round,
                },
            )
        # Cancellation is caught first, or not caught at all: each loop's
        # cancellation error subclasses its general one, so a handler reaching
        # for the assembled note through the general one would swallow an
        # operator's stop and revise anyway.
        except self._cancelled_error:
            # A cancelled status covers two things and only one of them is an
            # operator's stop: the adapters return it for bounded transient
            # retries exhausted too, and an optional turn must not end a run
            # over a provider hiccup. The token is what tells them apart, and
            # an operator who stopped the run has not asked for the round to
            # continue unsteered.
            if self._stopped():
                raise
            return self._fallback(verdict, assembled)
        except self._error:
            # Every other way a turn can give back nothing usable: it raised,
            # it returned a status the machinery cannot use, the workspace
            # refused it. The round was already granted and the revision is
            # still worth running.
            return self._fallback(verdict, assembled)

        note = confident_steering_note(payload)
        source = (
            SteeringNoteSource.AGENT
            if note is not None
            else SteeringNoteSource.ASSEMBLED
        )
        text = note if note is not None else assembled
        # The attempt and the row in one durable step, which is what lets
        # the row record which note was used. The machinery closes an attempt
        # itself when a turn is cancelled, returns a bad status, or returns a
        # result it cannot use, and completing one twice raises, so a fallback
        # note's row is written on its own. It leaves open the one it crashed
        # on, which the request context's round is what recovers from.
        with self.store.transaction():
            self.store.complete_planning_attempt(
                attempt.id,
                status=PlanningAttemptStatus.COMPLETED,
                result=payload,
                summary=(
                    f"steering note reported {payload['confidence']} confidence"
                ),
            )
            self.store.record_steering_note(
                build_steering_note(
                    borg_id=self.borg_id,
                    scope=self.scope,
                    verdict=verdict,
                    note=text,
                    source=source,
                    attempt_id=attempt.id,
                )
            )
        return text

    def _fallback(self, verdict: ReviewAssessment, assembled: str) -> str:
        self.store.record_steering_note(
            build_steering_note(
                borg_id=self.borg_id,
                scope=self.scope,
                verdict=verdict,
                note=assembled,
                source=SteeringNoteSource.ASSEMBLED,
            )
        )
        return assembled

    def _stopped(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()


__all__ = [
    "STEERING_NOTE_SCHEMA",
    "STEERING_PHASE",
    "STEERING_SYSTEM_PROMPT",
    "PlanningSteeringTurn",
    "SteeringScope",
    "SteeringSubject",
    "assembled_steering_note",
    "build_steering_note",
    "confident_steering_note",
    "recorded_steering_note",
    "render_steering_prompt",
    "steering_verdict",
]
